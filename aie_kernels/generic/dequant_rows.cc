// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GEMV's quantized weight rows (int4, per-group f32 scale, quant_row_layout.h) expanded to bf16
// rows, with mv_quant.cc's arithmetic: bf16(q) * bf16(scale), rounded to nearest even. A GEMM
// then reads the weights GEMV multiplies, without a second packed copy.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "quant_row_layout.h"

#ifndef DIM_K
#define DIM_K 2560
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif
#ifndef DQ_ROWS
#define DQ_ROWS 2
#endif

static constexpr uint32_t R = GROUP_SIZE < 64 ? GROUP_SIZE : 64;
static constexpr uint32_t N_GROUPS = DIM_K / GROUP_SIZE;
static constexpr uint32_t HEADER = N_GROUPS * sizeof(float);
static constexpr uint32_t PAYLOAD = DIM_K / 2;
static constexpr uint32_t STRIDE = HEADER + PAYLOAD;
static_assert(DIM_K % GROUP_SIZE == 0 && GROUP_SIZE % R == 0, "whole groups, whole chunks");
static_assert(QUANT_ALIGN_OK(R / 2, HEADER, PAYLOAD, STRIDE), "int4 payload load is not aligned");
static_assert((DIM_K * 2) % 64 == 0, "bf16 output rows must stay vector-aligned");

extern "C" {

void dequant_rows_int4(int8_t *a, bfloat16 *c)
{
    event0();
    const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(a);
    for (uint32_t row = 0; row < DQ_ROWS; ++row) {
        const quant_row_offsets off = quant_row_at<STRIDE, HEADER, PAYLOAD>(row);
        const float *scale = reinterpret_cast<const float *>(bytes + off.header);
        const int8_t *packed = reinterpret_cast<const int8_t *>(bytes + off.payload);
        bfloat16 *out = c + row * DIM_K;
        for (uint32_t chunk = 0; chunk < DIM_K / R; ++chunk) {
            ::aie::vector<int8, R> q8 =
                ::aie::unpack(::aie::vector_cast<int4>(::aie::load_v<R / 2>(packed + chunk * (R / 2))));
            ::aie::vector<bfloat16, R> sv =
                ::aie::broadcast<bfloat16, R>((bfloat16)scale[(chunk * R) / GROUP_SIZE]);
            ::aie::store_v(out + chunk * R,
                           ::aie::mul(::aie::to_float<bfloat16>(q8, 0), sv).template to_vector<bfloat16>());
        }
    }
    ::aie::set_rounding(saved);
    event1();
}

} // extern "C"
