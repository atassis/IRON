// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// softmax.cc::softmax_simple_bf16's three passes, each streaming its own tile instead of walking
// one row-length L1 buffer. Same passes, same order, same values -- so the split is bit-identical
// rather than close (tests/test_chunked_softmax_golden.py).
//
// Its own TU: softmax.cc links into every shipped decode arm, and new functions would move their
// ELFs.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "flash_contract.h"

#define SM_VEC_LEN FLASH_SM_VEC_LEN
#define log2e 1.4453125 // softmax.cc's own constant; they must agree or the split is not a split

using namespace aie;

// -inf over [from, total). softmax.cc::mask_bf16 is the scalar original and stays untouched; the
// callers below reach this only for the ONE segment straddling the mask boundary.
static inline void mask_tail_ninf(bfloat16 *inout, int32_t from, int32_t total)
{
    const bfloat16 ninf = (bfloat16)(-INFINITY);
    int32_t i = from;
    for (; i < total && (i & (SM_VEC_LEN - 1)); i++)
        inout[i] = ninf;
    aie::vector<bfloat16, SM_VEC_LEN> v = aie::broadcast<bfloat16, SM_VEC_LEN>(ninf);
    for (; i + SM_VEC_LEN <= total; i += SM_VEC_LEN)
        aie::store_v(inout + i, v);
    for (; i < total; i++)
        inout[i] = ninf;
}

extern "C" {

// Two state buffers rather than one [row][max, lanes]: a 64-lane f32 accumulator needs 512-bit
// alignment and a 65-float row stride is 260 B. The max starts at 0, not -inf, because
// softmax_simple_bf16 does -- a contract here, not a choice.
void softmax_segment_init_f32(float *restrict lanes, float *restrict mx, const int32_t row)
{
    aie::store_v(lanes + row * SM_VEC_LEN, aie::zeros<float, SM_VEC_LEN>());
    mx[row] = 0.0f;
}

// Pass 1 of 3. `unmasked` is this segment's valid width, clamped into [0, vector_size] by the
// caller; a wholly masked segment leaves the state alone, since -inf cannot raise a max of 0.
void softmax_segment_max_bf16(bfloat16 *restrict input_vector, float *restrict mx,
                              const int32_t row, const int32_t vector_size, const int32_t unmasked)
{
    event0();
    ::aie::set_rounding(FLASH_ROUNDING_MODE);
    if (unmasked <= 0) {
        event1();
        return;
    }
    if (unmasked < vector_size)
        mask_tail_ninf(input_vector, unmasked, vector_size);

    auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    aie::vector<bfloat16, SM_VEC_LEN> log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);
    // Running max as a VECTOR, for the reason partial_softmax_f32state_bf16 states: a scalar f32
    // compare per iteration is a __gtsf2 libcall on AIE2P.
    aie::vector<bfloat16, SM_VEC_LEN> max_acc = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)mx[row]);
    aie::accum<accfloat, SM_VEC_LEN> scaled_accum;

    const int elem_iters = vector_size / SM_VEC_LEN;
    for (int i = 0; i < elem_iters; i++) {
        scaled_accum = aie::mul(*it_in++, log2e_vec);
        max_acc = aie::max(max_acc, scaled_accum.to_vector<bfloat16>());
    }
    mx[row] = aie::reduce_max(max_acc);
    event1();
}

// Pass 2 of 3. The accumulator is carried through memory because it has to survive the next
// segment's DMA; accfloat is f32, so the round trip is exact and the lane order stays the
// unchunked kernel's -- which is what makes pass 3's reduce_add agree.
void softmax_segment_sum_bf16(bfloat16 *restrict input_vector, float *restrict lanes,
                              const float *restrict mx, const int32_t row,
                              const int32_t vector_size, const int32_t unmasked)
{
    event0();
    ::aie::set_rounding(FLASH_ROUNDING_MODE);
    if (unmasked <= 0) {
        event1();
        return;
    }
    if (unmasked < vector_size)
        mask_tail_ninf(input_vector, unmasked, vector_size);

    float *restrict acc_mem = lanes + row * SM_VEC_LEN;
    auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    aie::vector<bfloat16, SM_VEC_LEN> log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);
    aie::vector<bfloat16, SM_VEC_LEN> max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)mx[row]);
    aie::accum<accfloat, SM_VEC_LEN> exp_val_accum(aie::load_v<SM_VEC_LEN>(acc_mem));
    aie::accum<accfloat, SM_VEC_LEN> scaled_accum, exp_in_accum;
    aie::vector<bfloat16, SM_VEC_LEN> exp_val;

    const int elem_iters = vector_size / SM_VEC_LEN;
    for (int i = 0; i < elem_iters; i++) {
        scaled_accum = aie::mul(*it_in++, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
        exp_val_accum = add(exp_val_accum, exp_val);
    }
    aie::store_v(acc_mem, exp_val_accum.to_vector<float>());
    event1();
}

// Pass 3 of 3, softmax_simple_bf16's second and final passes fused. Recomputing the exponentials
// is cheaper than a second segment buffer and yields the same bf16 vector.
void softmax_segment_apply_bf16(bfloat16 *restrict input_vector, bfloat16 *restrict output_vector,
                                const float *restrict lanes, const float *restrict mx,
                                const int32_t row, const int32_t vector_size, const int32_t unmasked)
{
    event0();
    ::aie::set_rounding(FLASH_ROUNDING_MODE);
    auto it_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
    const int elem_iters = vector_size / SM_VEC_LEN;

    if (unmasked <= 0) {
        aie::vector<bfloat16, SM_VEC_LEN> z = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)0.0f);
        for (int i = 0; i < elem_iters; i++)
            *it_out++ = z;
        event1();
        return;
    }
    if (unmasked < vector_size)
        mask_tail_ninf(input_vector, unmasked, vector_size);

    const float accum_exp_val = aie::reduce_add(aie::load_v<SM_VEC_LEN>(lanes + row * SM_VEC_LEN));
    const bfloat16 col_sum_inv = (bfloat16)aie::inv(accum_exp_val);
    auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    aie::vector<bfloat16, SM_VEC_LEN> log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);
    aie::vector<bfloat16, SM_VEC_LEN> max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)mx[row]);
    aie::accum<accfloat, SM_VEC_LEN> scaled_accum, exp_in_accum, out_vals;
    aie::vector<bfloat16, SM_VEC_LEN> exp_val;

    for (int i = 0; i < elem_iters; i++) {
        scaled_accum = aie::mul(*it_in++, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
        out_vals = aie::mul(exp_val, col_sum_inv);
        *it_out++ = out_vals.to_vector<bfloat16>();
    }
    event1();
}

} // extern "C"
