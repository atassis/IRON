// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Local copy of aie_kernels/aie2p/rms_norm.cc's weighted_rms_norm, exposed under two distinct
// extern "C" names (qkv_rms_norm_d for the D-wide hn norm, qkv_rms_norm_hd for the HD-wide
// per-head qk-norm). The fused qkv_head core calls both shapes from ONE aie.core body; two
// IRON Kernel() wrappers pointing at the SAME symbol name with different memref shapes would
// each emit their own `func.func private @weighted_rms_norm` declaration, which is a duplicate
// MLIR symbol -- hence two names instead of one shared with the vendored kernel.

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, int N>
void rms_norm_general(const T *restrict input,
                      const T *restrict input2,
                      T *restrict output,
                      int32_t cols,
                      float epsilon)
{
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();

    int vector_chunks = cols / N;
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    float sum_sq = ::aie::reduce_add(add_res);

    int remaining = cols % N;
    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            float square = static_cast<float>(val) * static_cast<float>(val);
            sum_sq += square;
        }
    }

    float rms = sum_sq / cols + epsilon;
    float inv_rms = aie::invsqrt(rms);
    ::aie::accum<accfloat, N> inv_rms_v;
    inv_rms_v.from_vector(::aie::broadcast<float, N>(inv_rms), 0);

    for (int i = 0; i < vector_chunks; i++) {
        ::aie::accum<accfloat, N> reg_a;
        reg_a.from_vector(::aie::load_v<N>(input + i * N), 0);
        reg_a = ::aie::mul(reg_a.template to_vector<float>(), inv_rms_v.template to_vector<float>());
        if (input2) {
            ::aie::accum<accfloat, N> reg_b;
            reg_b.from_vector(::aie::load_v<N>(input2 + i * N), 0);
            reg_a = ::aie::mul(reg_a.template to_vector<float>(), reg_b.template to_vector<float>());
        }
        ::aie::store_v(output + i * N, reg_a.template to_vector<T>());
    }

    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            T norm_val = static_cast<T>(static_cast<float>(val) * inv_rms);
            if (input2) {
                T mul_val = input2[start_idx + i];
                output[start_idx + i] = static_cast<T>(static_cast<float>(norm_val) * static_cast<float>(mul_val));
            } else {
                output[start_idx + i] = norm_val;
            }
        }
    }
}

extern "C" {

void qkv_rms_norm_d(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int32_t size, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    rms_norm_general<bfloat16, 32>(a_in, b_in, c_out, size, epsilon);
}

void qkv_rms_norm_hd(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int32_t size, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    rms_norm_general<bfloat16, 32>(a_in, b_in, c_out, size, epsilon);
}

} // extern "C"
