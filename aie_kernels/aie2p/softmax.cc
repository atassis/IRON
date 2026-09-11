// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "flash_contract.h"

#define SM_VEC_LEN 64   // 32
#define log2e 1.4453125 // 1.44269504089

using namespace aie;

void softmax_simple_bf16(bfloat16 *restrict input_vector, bfloat16 *restrict output_vector, const int32_t vector_size)
{
    event0();
    // Match partial_softmax_alias_bf16 below, which has always set this. Without it the bf16
    // conversions here inherit the core's default rounding mode, which is toward zero, so every
    // exp value is biased low and the softmax row sums to less than 1. MEASURED on the fused
    // decode: 0.995138 over the valid width, and CONSTANT across widths 2..32 -- a per-element
    // bias does not care how many elements are summed, which is what rules out the reduction.
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // VJUNG: We do 3 passes on the vector:
    // 1. Find the max value scaled by log2e in the vector
    // 2. Calculate the exponentials of the scaled values minus the maximum
    // 3. Calculate the softmax by dividing each exponential by the sum of all exponentials
    // Note: The multiplication by log2e is very sensitive, casting it to bf16 before exponentiation leads to wrong
    // output.

    auto it_log_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_log_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
    auto it_scale = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
    auto it_soft_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

    aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec, max_val_vec;
    aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

    float max_val = 0;
    float accum_exp_val = 0;
    float running_max = 0;
    bfloat16 col_sum_inv;
    const int elem_iters = vector_size / SM_VEC_LEN;

    exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

    log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);

    // First pass
    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_log_in++;
        scaled_accum = aie::mul(input_bf16, log2e_vec);
        running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
        if (running_max > max_val) {
            max_val = running_max;
        }
    }
    max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

    // Second pass
    for (int i = 0; i < elem_iters; i++) {

        input_bf16 = *it_exp_in++;

        scaled_accum = aie::mul(input_bf16, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
        exp_val_accum = add(exp_val_accum, exp_val);

        *it_exp_out++ = exp_val;
    }

    // Final pass
    aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
    accum_exp_val = aie::reduce_add(reduce);
    col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

    for (int c = 0; c < elem_iters; c++) {
        in_elems = *it_scale++;
        out_vals = aie::mul(in_elems, col_sum_inv);
        *it_soft_out++ = out_vals.to_vector<bfloat16>();
    }

    event1();

    return;
}

void partial_softmax_alias_bf16(bfloat16 *restrict input_vector,
                                bfloat16 *restrict output_vector,
                                bfloat16 *restrict scale_buffer,
                                const int32_t vector_size,
                                const int32_t row_idx,
                                const int32_t num_rows,
                                const bfloat16 scale)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // VJUNG: We do 3 passes on the vector:
    // 1. Find the max value scaled by log2e in the vector
    // 2. Calculate the exponentials of the scaled values minus the maximum
    // 3. Calculate the softmax by dividing each exponential by the sum of all exponentials
    // Note: The multiplication by log2e is very sensitive, casting it to bf16 before exponentiation leads to wrong
    // output.

    auto it_log_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_log_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_in = aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

    aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec, max_val_vec;
    aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

    float max_val = 0;
    float accum_exp_val = 0;
    float running_max = 0;
    float col_sum_inv;
    const int elem_iters = vector_size / SM_VEC_LEN;

    exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

    log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)scale);

    // First pass
    for (int i = 0; i < elem_iters; i++) {
        input_bf16 = *it_log_in++;
        scaled_accum = aie::mul(input_bf16, log2e_vec);
        running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
        if (running_max > max_val) {
            max_val = running_max;
        }
    }

    // Compute m_{i}
    if (max_val > scale_buffer[row_idx]) {
        scale_buffer[num_rows + row_idx] = max_val;
    } else {
        scale_buffer[num_rows + row_idx] = scale_buffer[row_idx];
        max_val = scale_buffer[row_idx];
    }

    max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

    // Second pass
    for (int i = 0; i < elem_iters; i++) {

        input_bf16 = *it_exp_in++;

        scaled_accum = aie::mul(input_bf16, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
        exp_val_accum = add(exp_val_accum, exp_val);

        *it_exp_out++ = exp_val;
    }

    aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
    accum_exp_val = aie::reduce_add(reduce);

    scale_buffer[3 * num_rows + row_idx] = accum_exp_val;

    event1();

    return;
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output, const int32_t input_size)
{
    softmax_simple_bf16(input, output, input_size);
}

void partial_softmax_bf16(bfloat16 *restrict input,
                          bfloat16 *restrict output,
                          bfloat16 *restrict scale_buffer,
                          const int32_t input_size,
                          const int32_t row_idx,
                          const int32_t num_rows,
                          const bfloat16 scale)
{
    partial_softmax_alias_bf16(input, output, scale_buffer, input_size, row_idx, num_rows, scale);
}

/* ---- split-K flash attention ------------------------------------------------------------------
 *
 * partial_softmax_alias_bf16 above keeps its running state in a bf16 scale_buffer. That is flat in
 * accuracy at mha's block sizes and degrades with the window -- measured on the CPU golden
 * (tests/test_split_k_golden.py), rel-L2 1.440e-03 at S=4096 rising to 5.944e-03 at 32768, against
 * a f32 state that stays at 1.5-2.2e-03 with no trend. attn_block_dp's split-K needs the 32k end
 * of that, so it carries two f32 words per group instead.
 *
 * Everything ELSE stays bf16 on purpose: the segment max comes off a bf16 reduce_max and the
 * correction off aie::exp2<bfloat16>, and a f32 correction was measured worth 1.723e-03 against
 * bf16's 1.718e-03 at S=32768, i.e. nothing. No scalar exp2 is needed on the core.
 */

// One segment of an online softmax. Reads/updates `state` = {running max, running sum}, writes
// UNNORMALISED exp2 to `output` -- the divide by the running sum happens once, in
// taccum_finish_bf16, after the last segment. Returns the factor the caller must apply to the f32
// context accumulator before adding this segment's contribution.
float partial_softmax_f32state_bf16(bfloat16 *restrict input_vector,
                                    bfloat16 *restrict output_vector,
                                    float *restrict state,
                                    const int32_t vector_size,
                                    const bfloat16 scale)
{
    event0();
    ::aie::set_rounding(FLASH_ROUNDING_MODE);

    const int elem_iters = vector_size / FLASH_SM_VEC_LEN;
    auto it_max_in = aie::cbegin_restrict_vector<FLASH_SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_in = aie::cbegin_restrict_vector<FLASH_SM_VEC_LEN>((bfloat16 *)input_vector);
    auto it_exp_out = aie::begin_restrict_vector<FLASH_SM_VEC_LEN>((bfloat16 *)output_vector);

    aie::vector<bfloat16, FLASH_SM_VEC_LEN> log2e_vec, max_val_vec, exp_val;
    aie::accum<accfloat, FLASH_SM_VEC_LEN> scaled_accum, exp_in_accum, exp_val_accum;

    log2e_vec = aie::broadcast<bfloat16, FLASH_SM_VEC_LEN>(scale);

    // Pass 1 -- this segment's max over the SCALED values. Starts at -inf, not at 0 the way
    // softmax_simple_bf16 does: a full-row softmax is shift-invariant so 0 is harmless there, but
    // a max that has to COMPOSE with the previous segment's must be the real one.
    float seg_max = -INFINITY;
    for (int i = 0; i < elem_iters; i++) {
        scaled_accum = aie::mul(*it_max_in++, log2e_vec);
        float r = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
        if (r > seg_max)
            seg_max = r;
    }

    const float m_prev = state[0];

    // A wholly masked segment contributes nothing and must not reach exp2, where -inf minus -inf
    // is NaN. attn_block_dp's ceil-divided segment count never produces one, so this is a guard
    // against a caller's arithmetic rather than an expected path -- but a NaN here is silent and
    // poisons the whole token, which is the asymmetry that earns the branch.
    if (seg_max == -INFINITY) {
        aie::vector<bfloat16, FLASH_SM_VEC_LEN> z =
            aie::broadcast<bfloat16, FLASH_SM_VEC_LEN>((bfloat16)0.0f);
        for (int i = 0; i < elem_iters; i++)
            *it_exp_out++ = z;
        event1();
        return 1.0f;
    }

    const float m_new = (seg_max > m_prev) ? seg_max : m_prev;

    // exp2 of a VECTOR, lane 0 taken: the scalar form would pull in libm on the core. m_prev is
    // -inf only on the first segment, where the accumulator is still zero and the factor is
    // irrelevant -- take 0 rather than feeding -inf to exp2.
    float corr;
    if (m_prev == -INFINITY) {
        corr = 0.0f;
    } else {
        aie::vector<float, FLASH_SM_VEC_LEN> d =
            aie::broadcast<float, FLASH_SM_VEC_LEN>(m_prev - m_new);
        corr = (float)aie::exp2<bfloat16>(d)[0];
    }

    // Pass 2 -- exp2 into the output and sum this segment. Mirrors softmax_simple_bf16's second
    // pass exactly, including keeping the log2e product in the accumulator: casting it to bf16
    // before the exponent is what that kernel's own comment warns produces wrong output.
    max_val_vec = aie::broadcast<bfloat16, FLASH_SM_VEC_LEN>((bfloat16)m_new);
    exp_val_accum = aie::zeros<accfloat, FLASH_SM_VEC_LEN>();
    for (int i = 0; i < elem_iters; i++) {
        scaled_accum = aie::mul(*it_exp_in++, log2e_vec);
        exp_in_accum = aie::sub(scaled_accum, max_val_vec);
        exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
        exp_val_accum = add(exp_val_accum, exp_val);
        *it_exp_out++ = exp_val;
    }
    const float seg_sum = aie::reduce_add(exp_val_accum.to_vector<float>());

    state[0] = m_new;
    state[1] = state[1] * corr + seg_sum;

    event1();
    return corr;
}

// {running max, running sum} for one group, before any segment has run.
void flash_state_init(float *restrict state)
{
    state[0] = -INFINITY;
    state[1] = 0.0f;
}

// Scale an f32 context accumulator in place by the factor partial_softmax_f32state_bf16 returned.
// mha.cc::rescale_O does the same job for a 64x64 bf16 tile with the layout hardcoded into three
// fixed loops; this one takes a length and a dtype that match attn_block_dp's [gqa][head_dim] f32.
void acc_rescale_f32(uint32_t n, float factor, float *restrict acc)
{
    aie::vector<float, 16> f = aie::broadcast<float, 16>(factor);
    for (uint32_t i = 0; i < n; i += 16)
        aie::store_v(acc + i, aie::mul(aie::load_v<16>(acc + i), f).to_vector<float>());
}

void mask_bf16(bfloat16 *inout, const int32 unmasked_size, const int32 total_size)
{
    // TODO: Optimize this to use vector code
    for (int32 i = unmasked_size; i < total_size; i++) {
        inout[i] = (bfloat16)(-INFINITY);
    }
}

} // extern "C"