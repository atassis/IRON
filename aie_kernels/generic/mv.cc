// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif

// Independent accumulator count for matvec_vectorized's k-reduction. 1 reproduces the original
// single-accumulator kernel byte-for-byte (the self-feedback `acc = mac(acc, ...)` chain measured
// at 18.3 MAC/cyc, a 7-cycle body bound by the mac's own latency, not by any resource). >1 breaks
// that recurrence into GEMV_NACC independent chains -- modelled on mv_taccum.cc's live
// multi-accumulator pattern -- so the pipeliner can overlap them instead of serializing on one
// feedback edge. Reordering the sum changes the floating-point result, so NACC>1 is NOT
// bit-identical to the NACC=1 path; see the gemv op's parity note before shipping it as a default.
#ifndef GEMV_NACC
#define GEMV_NACC 1
#endif

void matvec_scalar(uint32_t m,
                   uint32_t k,
                   const bfloat16 *__restrict a,
                   const bfloat16 *__restrict b,
                   bfloat16 *__restrict c)
{
    for (uint32_t row = 0; row < m; row++) {
        float acc = 0;
        for (uint32_t i = 0; i < k; i++) {
            acc += a[row * k + i] * b[i];
        }
        c[row] = static_cast<bfloat16>(acc);
    }
}

/*
Matrix-vector multiplication kernel

 - m: Number of output rows == number of rows in the input matrix
 - k: Number of columns in the input matrix == length of the input vector
 - a: Pointer to the input matrix, stored in row-major order
 - b: Pointer to the input vector
 - c: Pointer to the output vector
 - r: Vector size; data from the matrix and vector will be loaded in and processed in chunks of this size
 - NACC: number of independent accumulators the k-reduction is split across (see GEMV_NACC above)

The k-loop stays the ONLY hardware loop (zero-overhead, innermost) regardless of NACC: the
NACC-wide fan-out below is unrolled at compile time (AIE_LOOP_UNROLL_FULL, trip count is the
template constant NACC), so it never becomes a second runtime loop nested under the ZOL. Nesting
the ZOL is the mistake this kernel's own history already paid for -- the last 6.5x on this exact
loop came from UNDOING a nested-loop version (kb: narrow-weight-formats-are-closed-at-m1-decode).
*/
template <uint32_t r, uint32_t k, uint32_t NACC = GEMV_NACC>
void matvec_vectorized(uint32_t m, const bfloat16 *__restrict a, const bfloat16 *__restrict b, bfloat16 *__restrict c)
{
    static_assert(NACC == 1 || NACC == 2 || NACC == 4 || NACC == 8,
                  "matvec_vectorized supports 1/2/4/8 independent accumulators");
    // K007: the shape is picked here (stride = r * NACC), so the divisibility it requires is
    // asserted here too, not left for aiecc's tile-allocation error to name later.
    static_assert(k % (r * NACC) == 0,
                  "k must be a whole number of NACC-wide r-chunks per zero-overhead-loop iteration");
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    constexpr uint32_t stride = r * NACC;
    bfloat16 *c_end = c + m;
    const bfloat16 *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum<accfloat, r> acc[NACC];
        AIE_LOOP_UNROLL_FULL
        for (uint32_t j = 0; j < NACC; j++)
            acc[j] = aie::zeros<accfloat, r>();

        // The following two pragmas enable pipelining the zero-overhead loop, but they do assume that there are at
        // least two iterations of the loop, i.e. k >= 2*r. This pragma will break the code if that is not the case!
        AIE_LOOP_MIN_ITERATION_COUNT(k / stride)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += stride, a += stride) {
            AIE_LOOP_UNROLL_FULL
            for (uint32_t j = 0; j < NACC; j++) {
                aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a + j * r);
                aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur + j * r);
                acc[j] = aie::mac(acc[j], a_vec, b_vec);
            }
        }

        // Exactly one reduce_add per output row, same as the NACC=1 path: NACC independent
        // accumulators are combined into a single vector by plain (non-reducing) adds first.
        aie::vector<float, r> sum = acc[0].template to_vector<float>();
        AIE_LOOP_UNROLL_FULL
        for (uint32_t j = 1; j < NACC; j++)
            sum = aie::add(sum, acc[j].template to_vector<float>());
        *c = static_cast<bfloat16>(aie::reduce_add(sum));
    }
}

// T4: batches ROWS output rows per reduce, instead of splitting one row's K-reduction (T1/NACC
// above). Measured motivation (K=128, r=64, scores shape): the per-row block outside the ZOL is
// ~55 of ~61 static bundles -- mostly ONE row's reduce_add tree (log2(r) shift/add steps) -- and
// splitting the 1-2-iteration K loop into more accumulators left that block's cost flat or worse
// (NACC=8 measured 88 bundles vs ~61). That fixed cost is per ROW, not per K-chunk, so amortise it
// across ROWS rows instead: aie::reduce_add_v folds up to 4 independent r-wide accumulators into
// one packed-sum vector in a SINGLE call, in place of ROWS separate O(log r) trees. Each row still
// keeps its own single accumulator (no K-split) -- T3 found that lever dry at this shape.
#ifndef GEMV_ROWBATCH
#define GEMV_ROWBATCH 1
#endif

template <uint32_t r, uint32_t k, uint32_t ROWS>
void matvec_vectorized_rowbatch(uint32_t m, const bfloat16 *__restrict a, const bfloat16 *__restrict b, bfloat16 *__restrict c)
{
    static_assert(ROWS >= 1 && ROWS <= 4, "aie::reduce_add_v folds at most 4 vectors per call");
    static_assert(k % r == 0, "k must be a whole number of r-wide chunks");
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *b_end = b + k;
    bfloat16 *c_end = c + m;
    // K008: m must be a whole number of ROWS-row groups. No tail path in this prototype --
    // ship-blocking if ROWS ever stops dividing the caller's m_input evenly.
    for (; c < c_end; c += ROWS, a += ROWS * k) {
        aie::accum<accfloat, r> acc[ROWS];
        const bfloat16 *__restrict a_row[ROWS];
        AIE_LOOP_UNROLL_FULL
        for (uint32_t j = 0; j < ROWS; j++) {
            acc[j] = aie::zeros<accfloat, r>();
            a_row[j] = a + j * k;
        }

        AIE_LOOP_MIN_ITERATION_COUNT(k / r)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r) {
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            AIE_LOOP_UNROLL_FULL
            for (uint32_t j = 0; j < ROWS; j++) {
                aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a_row[j]);
                acc[j] = aie::mac(acc[j], a_vec, b_vec);
                a_row[j] += r;
            }
        }

        if constexpr (ROWS == 1) {
            c[0] = static_cast<bfloat16>(aie::reduce_add(acc[0].template to_vector<float>()));
        } else if constexpr (ROWS == 2) {
            auto packed = aie::reduce_add_v(acc[0].template to_vector<float>(),
                                             acc[1].template to_vector<float>());
            c[0] = static_cast<bfloat16>(packed[0]);
            c[1] = static_cast<bfloat16>(packed[1]);
        } else if constexpr (ROWS == 3) {
            auto packed = aie::reduce_add_v(acc[0].template to_vector<float>(),
                                             acc[1].template to_vector<float>(),
                                             acc[2].template to_vector<float>());
            c[0] = static_cast<bfloat16>(packed[0]);
            c[1] = static_cast<bfloat16>(packed[1]);
            c[2] = static_cast<bfloat16>(packed[2]);
        } else if constexpr (ROWS == 4) {
            auto packed = aie::reduce_add_v(acc[0].template to_vector<float>(),
                                             acc[1].template to_vector<float>(),
                                             acc[2].template to_vector<float>(),
                                             acc[3].template to_vector<float>());
            AIE_LOOP_UNROLL_FULL
            for (uint32_t j = 0; j < 4; j++)
                c[j] = static_cast<bfloat16>(packed[j]);
        }
    }
}

// Runtime-K form. DIM_K is a template constant above, so the projection (K=d_model) and the scores
// (K=head_dim) otherwise need two compiled bodies; here K is an argument. Alias mechanics: see
// rms_norm.cc's weighted_rms_norm_cols. The cost is the pipelining hint -- with K compile-time it is
// k/stride (16 at d_model, 2 at head_dim); here it can only assert the minimum both callers satisfy.
template <uint32_t r>
void matvec_vectorized_rtk(uint32_t m, uint32_t k,
                           const bfloat16 *__restrict a, const bfloat16 *__restrict b,
                           bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *c_end = c + m;
    const bfloat16 *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

// G-batched form of matvec_vectorized_rtk: G query rows (`a`, contiguous [G][k]) dotted against
// the shared vector `b`, writing row_off within each head's out_stride-wide segment of `c`. See
// iron/operators/attn_global_dp/design.py's flash_worker_l1_bytes for why (combined per-head
// buffers, kb: attn-global-flash-per-row-call-overhead).
template <uint32_t r>
void matvec_vectorized_rtk_gstride(uint32_t G, uint32_t row_off, uint32_t k, uint32_t out_stride,
                                   const bfloat16 *__restrict a, const bfloat16 *__restrict b,
                                   bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *b_end = b + k;
    for (uint32_t g = 0; g < G; g++, a += k, c += out_stride) {
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
        const bfloat16 *__restrict a_cur = a;
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a_cur += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a_cur);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        c[row_off] = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

// docs/superpowers/specs/2026-09-25-flash-multirow-mmul-design.md section 1, P2: the 4 heads'
// accumulator chains and reduce_add_v trees are interleaved instead of run one head at a time, so
// they overlap. G is fixed at 4 (the reduce_add_v call below), not a template/runtime parameter --
// the caller-side ABI parity with sc_matvec_g_rtk_bf16_bf16 is kept at the extern "C" wrapper.
// `kblk` is `rows` contiguous K rows [rows][k]; `q_in` is G contiguous k-wide query rows, resident
// across the whole call.
template <uint32_t r>
void matvec_g4_rows_rtk_gstride(uint32_t rows, uint32_t row_off, uint32_t k, uint32_t out_stride,
                                const bfloat16 *__restrict q_in,
                                const bfloat16 *__restrict kblk,
                                bfloat16 *__restrict sc)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *__restrict krow = kblk;
    for (uint32_t p = 0; p < rows; p++, krow += k) {
        aie::accum<accfloat, r> acc[4];
        AIE_LOOP_UNROLL_FULL
        for (uint32_t g = 0; g < 4; g++) acc[g] = aie::zeros<accfloat, r>();
        const bfloat16 *__restrict k_end = krow + k;
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (const bfloat16 *__restrict k_cur = krow; k_cur < k_end; k_cur += r) {
            aie::vector<bfloat16, r> kv = aie::load_v<r>(k_cur);
            uint32_t off = (uint32_t)(k_cur - krow);
            AIE_LOOP_UNROLL_FULL
            for (uint32_t g = 0; g < 4; g++)
                acc[g] = aie::mac(acc[g], aie::load_v<r>(q_in + g * k + off), kv);
        }
        auto s = aie::reduce_add_v(acc[0].template to_vector<float>(), acc[1].template to_vector<float>(),
                                   acc[2].template to_vector<float>(), acc[3].template to_vector<float>());
        AIE_LOOP_UNROLL_FULL
        for (uint32_t g = 0; g < 4; g++)
            sc[g * out_stride + row_off + p] = static_cast<bfloat16>(s[g]);
    }
}

extern "C" {

/* The row offset parameter in the functions below is a workaround. The output will be written to c + row_offset * m.
 * This is simpler than to do pointer arithmetic in the calling MLIR code, but that's all this is for -- an offset into
 * `c`.  */

void matvec_scalar_bf16_bf16(uint32_t m,
                             uint32_t row_offset,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_scalar(m, DIM_K, a_in, b_in, c_out);
}

void matvec_rtk_bf16_bf16(uint32_t m,
                          uint32_t row_offset,
                          uint32_t k,
                          const bfloat16 *__restrict a_in,
                          const bfloat16 *__restrict b_in,
                          bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_vectorized_rtk<VEC_SIZE>(m, k, a_in, b_in, c_out);
}
#ifdef GEMV_ALIAS_SC
// The scores caller's second name for the body above -- see rms_norm.cc. Opt-in because the
// row-batched path below takes k as a TEMPLATE parameter and so cannot share this body.
void sc_matvec_rtk_bf16_bf16(uint32_t m,
                             uint32_t row_offset,
                             uint32_t k,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             bfloat16 *__restrict c_out)
    __attribute__((alias("matvec_rtk_bf16_bf16")));
#endif

#ifdef GEMV_ALIAS_SC
// attn_global_flash's per-row score call, G resident heads in one: `a_in` is G contiguous
// HD-wide query rows, `b_in` is the shared K row, `c_out` is [G][out_stride] with row `row_off`
// written per head.
void sc_matvec_g_rtk_bf16_bf16(uint32_t G, uint32_t row_off, uint32_t k, uint32_t out_stride,
                               const bfloat16 *__restrict a_in,
                               const bfloat16 *__restrict b_in,
                               bfloat16 *__restrict c_out)
{
    matvec_vectorized_rtk_gstride<VEC_SIZE>(G, row_off, k, out_stride, a_in, b_in, c_out);
}
#endif

#ifdef GEMV_ALIAS_SC
void sc_matvec_g_rows_bf16_bf16(uint32_t rows, uint32_t G, uint32_t row_off, uint32_t k,
                                uint32_t out_stride,
                                const bfloat16 *__restrict q_in,
                                const bfloat16 *__restrict kblk,
                                bfloat16 *__restrict sc)
{
    matvec_g4_rows_rtk_gstride<VEC_SIZE>(rows, row_off, k, out_stride, q_in, kblk, sc);
}
#endif

void matvec_vectorized_bf16_bf16(uint32_t m,
                                 uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out)
{
    c_out += row_offset;
#if GEMV_ROWBATCH > 1
    // T4 path: same (m, row_offset, a, b, c) contract as the T1/NACC path below -- row-batching
    // is an internal restructuring of how `m` rows get reduced, not a layout change -- so the
    // existing MLIR caller (design.py) needs no change to opt in via -DGEMV_ROWBATCH=N.
    matvec_vectorized_rowbatch<VEC_SIZE, DIM_K, GEMV_ROWBATCH>(m, a_in, b_in, c_out);
#else
    matvec_vectorized<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
#endif
}

} // extern "C"