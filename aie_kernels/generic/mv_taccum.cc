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

template <uint32_t N>
static inline void taccum_rows(uint32_t rows,
                               const bfloat16 *__restrict a,
                               const bfloat16 *__restrict w,
                               float *__restrict acc)
{
    // aie_api never sets the rounding register and the documented default is floor; every kernel
    // here that converts to bf16 has to say so explicitly.
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, N> ac;
    ac.from_vector(aie::load_v<N>(acc));
    for (uint32_t p = 0; p < rows; p++, a += N) {
        aie::vector<bfloat16, N> av = aie::load_v<N>(a);
        aie::vector<bfloat16, N> wv = aie::broadcast<bfloat16, N>(w[p]);
        ac = aie::mac(ac, av, wv);
    }
    aie::store_v(acc, ac.template to_vector<float>());
}

extern "C" {

void taccum_zero_f32(float *__restrict acc)
{
    aie::store_v(acc, aie::zeros<float, DIM_N>());
}

void taccum_rows_bf16_f32(uint32_t rows,
                          const bfloat16 *__restrict a_in,
                          const bfloat16 *__restrict w_in,
                          float *__restrict acc)
{
    taccum_rows<DIM_N>(rows, a_in, w_in, acc);
}

void taccum_finish_bf16(const float *__restrict acc, bfloat16 *__restrict c_out)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    aie::accum<accfloat, DIM_N> ac;
    ac.from_vector(aie::load_v<DIM_N>(acc));
    aie::store_v(c_out, ac.template to_vector<bfloat16>());
}

} // extern "C"
