// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T_in, typename T_out> void eltwise_add(T_in *a, T_in *b, T_out *c, int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

template <typename T_in, typename T_out> void eltwise_vadd(T_in *a, T_in *b, T_out *c, int size)
{

    constexpr int vec_factor = 32;
    event0();
    T_in *__restrict pA1 = a;
    T_in *__restrict pB1 = b;
    T_out *__restrict pC1 = c;
    const int F = size / vec_factor;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++) {
        aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        pA1 += vec_factor;
        aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
        pB1 += vec_factor;
        aie::vector<T_out, vec_factor> cout = aie::add(A0, B0);
        aie::store_v(pC1, cout);
        pC1 += vec_factor;
    }
    // Tail. `size` is a per-core tile size chosen by the caller and nothing upstream requires it to
    // be a multiple of vec_factor, so without this the last `size % 32` outputs are NEVER WRITTEN
    // and the consumer reads whatever the buffer held. Silent, not a crash, and invisible to any
    // model whose tile happens to divide 32: Qwen3-0.6B's residual tile is 1024/8 = 128 = 32*4 and
    // is correct, while Gemma-3-270M's is 640/8 = 80 = 32*2 + 16, dropping 20% of every residual
    // add and turning full-depth decode into garbage tokens.
    const int tail = size - F * vec_factor;   // pA1/pB1/pC1 already point past the vector body
    for (int i = 0; i < tail; i++) {
        pC1[i] = pA1[i] + pB1[i];
    }
    event1();
}

extern "C" {

void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_add<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_vadd<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

} // extern "C"
