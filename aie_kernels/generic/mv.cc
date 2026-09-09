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