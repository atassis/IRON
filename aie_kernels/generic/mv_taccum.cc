// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Transposed-A matvec: the reduction runs DOWN the rows of a row-major matrix.
//
//     c[j] = sum over p of  w[p] * a[p][j],    a stored [rows][DIM_N], j in [0, DIM_N)
//
// mv.cc's matvec is the other contraction: one output per row, reducing ALONG a row. Attention's
// context step wants this one -- out[d] = sum_p softmax[p] * V[p][d] against a V cache stored
// [S][head_dim] -- and expressing it as a dot product is what forces a physical transpose of the
// whole cache first. Reducing down the rows removes that op entirely: each core takes a DIM_N-wide
// COLUMN SLICE of the cache, which is a contiguous run per row and therefore a legal shim BD,
// where a per-element transposed read is not (a 2-byte stride against a 4-byte address granule).
//
// KNOWN GAP, not yet measured: the inner loop issues one `vmac.f` per row against a SCALAR load of
// w[p] and a `vbcst.16`, which serialise ahead of it -- 5 bundles per row for 16 MACs. At M=1 decode
// this op is movement-bound, so the first version leaves it; loading w in vector chunks and
// broadcasting per lane is the obvious fix if it ever gates.
//
// The accumulator is f32 and lives in L2/L1 ACROSS calls, because a chunk of rows contributes to
// every output: unlike mv.cc, successive calls do not write disjoint outputs, they add to the same
// ones. Hence zero/accumulate/finish rather than one call per output tile.

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef DIM_N
#define DIM_N 16
#endif

// Chunk width for taccum_rows_g4 below: the native bf16 vmac lane count, not model-dependent (see
// docs/superpowers/specs/2026-09-25-flash-multirow-mmul-design.md section 1's VL=64).
#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif

// DEFAULT OFF because it LOSES at DIM_N=256 (+9.8% bundles/element) while winning ~70-76% at
// every other width swept. 256 is where the baseline already wins: the accumulator decomposition
// has broken the recurrence and a second chain only spills. Gemma-4-12B is 256 at every geometry;
// Qwen3 is 128, where this pays. Sweep and mechanism:
// kb/two-accumulators-invert-at-the-width-the-baseline-already-wins.
#ifndef TACC_UNROLL
#define TACC_UNROLL 1   // 2 = two accumulators over even/odd rows; see the sweep above
#endif

template <uint32_t N>
static inline void taccum_rows(uint32_t rows,
                               const bfloat16 *__restrict a,
                               const bfloat16 *__restrict w,
                               float *__restrict acc)
{
    // aie_api never sets the rounding register and the documented default is floor; every kernel
    // here that converts to bf16 has to say so explicitly.
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    // `rows` is a runtime argument, so the odd case takes a scalar tail rather than the
    // if-constexpr fallback mv_quant.cc's compile-time chunk count allows. Reassociation only,
    // and not bit-identical: gate on token parity, never a logits byte-compare. See TACC_UNROLL.
    if constexpr (TACC_UNROLL == 2) {
        aie::accum<accfloat, N> ac0 = ::aie::zeros<accfloat, N>();
        aie::accum<accfloat, N> ac1 = ::aie::zeros<accfloat, N>();
        const uint32_t rows2 = rows & ~1u;
        const bfloat16 *__restrict a_cur = a;
        for (uint32_t p = 0; p < rows2; p += 2) {
            aie::vector<bfloat16, N> av0 = aie::load_v<N>(a_cur);
            aie::vector<bfloat16, N> wv0 = aie::broadcast<bfloat16, N>(w[p]);
            ac0 = aie::mac(ac0, av0, wv0);
            a_cur += N;

            aie::vector<bfloat16, N> av1 = aie::load_v<N>(a_cur);
            aie::vector<bfloat16, N> wv1 = aie::broadcast<bfloat16, N>(w[p + 1]);
            ac1 = aie::mac(ac1, av1, wv1);
            a_cur += N;
        }
        aie::accum<accfloat, N> ac = ::aie::add(ac0, ac1);
        if (rows & 1u) {
            aie::vector<bfloat16, N> avt = aie::load_v<N>(a_cur);
            aie::vector<bfloat16, N> wvt = aie::broadcast<bfloat16, N>(w[rows2]);
            ac = aie::mac(ac, avt, wvt);
        }
        aie::accum<accfloat, N> base;
        base.from_vector(aie::load_v<N>(acc));
        ac = ::aie::add(ac, base);
        aie::store_v(acc, ac.template to_vector<float>());
    } else {
        aie::accum<accfloat, N> ac;
        ac.from_vector(aie::load_v<N>(acc));
        for (uint32_t p = 0; p < rows; p++, a += N) {
            aie::vector<bfloat16, N> av = aie::load_v<N>(a);
            aie::vector<bfloat16, N> wv = aie::broadcast<bfloat16, N>(w[p]);
            ac = aie::mac(ac, av, wv);
        }
        aie::store_v(acc, ac.template to_vector<float>());
    }
}

// docs/superpowers/specs/2026-09-25-flash-multirow-mmul-design.md section 1, P4b/P3: the 4 groups'
// accumulator chunks are held together across `rows` V rows so their per-row MACs are independent
// chains, instead of finishing one group's row before starting the next. `groups` is fixed at 4
// below (the caller is attn_global_flash's hpc4 worker only); kept as a runtime argument only for
// ABI parity with taccum_rows_bf16_f32. At rows=1 this is P3's body: load, one mac, store per
// group per DIM_N-wide chunk, same order taccum_rows_bf16_f32/DIM_N=HD already produces.
static inline void taccum_rows_g4(uint32_t rows,
                                  uint32_t w_stride, uint32_t w_off,
                                  const bfloat16 *__restrict a,
                                  const bfloat16 *__restrict w,
                                  float *__restrict acc)
{
    static_assert(DIM_N % VEC_SIZE == 0, "DIM_N must be a whole number of VEC_SIZE-wide chunks");
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *w0 = w + w_off, *w1 = w0 + w_stride, *w2 = w1 + w_stride, *w3 = w2 + w_stride;
    for (uint32_t c = 0; c < DIM_N; c += VEC_SIZE) {
        aie::accum<accfloat, VEC_SIZE> a0, a1, a2, a3;
        a0.from_vector(aie::load_v<VEC_SIZE>(acc + 0 * DIM_N + c));
        a1.from_vector(aie::load_v<VEC_SIZE>(acc + 1 * DIM_N + c));
        a2.from_vector(aie::load_v<VEC_SIZE>(acc + 2 * DIM_N + c));
        a3.from_vector(aie::load_v<VEC_SIZE>(acc + 3 * DIM_N + c));
        const bfloat16 *__restrict ap = a + c;
        // Step 1 calls this with rows=1; Step 2 will call it with rows=8. The pragma below must
        // stay true at every call site -- P4b's own AIE_LOOP_MIN_ITERATION_COUNT(8) is only valid
        // once rows is fixed at 8, and claiming it at rows=1 miscompiles the loop (device hang,
        // not a CPU-visible error).
        AIE_LOOP_MIN_ITERATION_COUNT(1)
        for (uint32_t p = 0; p < rows; p++, ap += DIM_N) {
            aie::vector<bfloat16, VEC_SIZE> av = aie::load_v<VEC_SIZE>(ap);
            a0 = aie::mac(a0, av, aie::broadcast<bfloat16, VEC_SIZE>(w0[p]));
            a1 = aie::mac(a1, av, aie::broadcast<bfloat16, VEC_SIZE>(w1[p]));
            a2 = aie::mac(a2, av, aie::broadcast<bfloat16, VEC_SIZE>(w2[p]));
            a3 = aie::mac(a3, av, aie::broadcast<bfloat16, VEC_SIZE>(w3[p]));
        }
        aie::store_v(acc + 0 * DIM_N + c, a0.to_vector<float>());
        aie::store_v(acc + 1 * DIM_N + c, a1.to_vector<float>());
        aie::store_v(acc + 2 * DIM_N + c, a2.to_vector<float>());
        aie::store_v(acc + 3 * DIM_N + c, a3.to_vector<float>());
    }
}

extern "C" {

void taccum_rows_g4_bf16_f32(uint32_t rows,
                             uint32_t groups,
                             uint32_t w_stride,
                             uint32_t w_off,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict w_in,
                             float *__restrict acc)
{
    taccum_rows_g4(rows, w_stride, w_off, a_in, w_in, acc);
}

/* All three take `groups`: one core owns a GQA group -- batch_group query heads that share this
 * core's kv head. A is read once and applied to each group member's w, which is the whole point of
 * the mapping, so the group loop belongs inside the kernel rather than in the runtime sequence.
 *
 * Layouts: a is [rows][DIM_N], w is [groups][w_stride] read at w_off, acc and c are [groups][DIM_N].
 */

void taccum_zero_f32(uint32_t groups, float *__restrict acc)
{
    for (uint32_t g = 0; g < groups; g++)
        aie::store_v(acc + g * DIM_N, aie::zeros<float, DIM_N>());
}

void taccum_rows_bf16_f32(uint32_t rows,
                          uint32_t groups,
                          uint32_t w_stride,
                          uint32_t w_off,
                          const bfloat16 *__restrict a_in,
                          const bfloat16 *__restrict w_in,
                          float *__restrict acc)
{
    for (uint32_t g = 0; g < groups; g++)
        taccum_rows<DIM_N>(rows, a_in, w_in + g * w_stride + w_off, acc + g * DIM_N);
}

void taccum_finish_bf16(uint32_t groups,
                        const float *__restrict acc,
                        bfloat16 *__restrict c_out)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    for (uint32_t g = 0; g < groups; g++) {
        aie::accum<accfloat, DIM_N> ac;
        ac.from_vector(aie::load_v<DIM_N>(acc + g * DIM_N));
        aie::store_v(c_out + g * DIM_N, ac.template to_vector<bfloat16>());
    }
}

/* Split-K's finish: the same convert, preceded by the normalisation the softmax deferred.
 *
 * A SEPARATE function rather than an extra argument on the one above, because tmatvec/design.py
 * binds taccum_finish_bf16 too (with three arguments) and a func.func symbol is keyed by NAME --
 * changing the shared signature would break that operator silently at MLIR generation.
 */
void taccum_finish_scaled_bf16(uint32_t groups,
                               const float *__restrict state,
                               const float *__restrict acc,
                               bfloat16 *__restrict c_out)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    // partial_softmax_f32state_bf16 writes raw exp2 and grows a running sum in state[1]: the
    // denominator is not known until the last segment has been seen, so the one divide the whole
    // attention does lands here.
    const float inv_l = aie::inv(state[1]);
    for (uint32_t g = 0; g < groups; g++) {
        aie::accum<accfloat, DIM_N> ac;
        ac.from_vector(aie::mul(aie::load_v<DIM_N>(acc + g * DIM_N), inv_l).template to_vector<float>());
        aie::store_v(c_out + g * DIM_N, ac.template to_vector<bfloat16>());
    }
}

} // extern "C"
