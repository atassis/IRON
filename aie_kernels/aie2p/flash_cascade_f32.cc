// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Online-softmax merge of per-core flash partials along a cascade chain
// (put -> fold_put -> ... -> fold_put -> fold_finish), one head per call.
//
// Cascade layout of one head's partial, 1 + n/16 words of v16float:
//   word 0      {max, sum, 0 x 14}, max in the log2e-scaled domain of partial_softmax_f32state_bf16
//   words 1..   the unnormalised f32 context, 16 lanes per word, in order
// `state` and `acc` are that function's {max, sum, correction} and [n] f32; `n` is a multiple of 16.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "flash_contract.h"

// Weights of the two sides of a merge. A side whose max is -inf is empty and weighs zero, never
// exp2(-inf - -inf). Same exp2 as flash_merge_column.
static inline void flash_fold_weights(float m_own, float m_in, float &m_new, float &c_own,
                                      float &c_in)
{
    m_new = (m_in > m_own) ? m_in : m_own;
    aie::vector<float, FLASH_SM_VEC_LEN> d_own = aie::broadcast<float, FLASH_SM_VEC_LEN>(m_own - m_new);
    aie::vector<float, FLASH_SM_VEC_LEN> d_in = aie::broadcast<float, FLASH_SM_VEC_LEN>(m_in - m_new);
    c_own = (m_own == -INFINITY) ? 0.0f : (float)aie::exp2<bfloat16>(d_own)[0];
    c_in = (m_in == -INFINITY) ? 0.0f : (float)aie::exp2<bfloat16>(d_in)[0];
}

static inline aie::vector<float, 16> flash_header(float m, float l)
{
    aie::vector<float, 16> h = aie::zeros<float, 16>();
    h.set(m, 0);
    h.set(l, 1);
    return h;
}

extern "C" {

// `acc_off` indexes into a caller's combined [heads][n] accumulator. See
// iron/operators/attn_global_dp/design.py's flash_worker_l1_bytes.
void flash_cascade_put_f32(uint32_t n, const float *__restrict state, const float *__restrict acc,
                           int32_t acc_off)
{
    acc += acc_off;
    put_mcd(flash_header(state[0], state[1]).to_native());
    for (uint32_t i = 0; i < n; i += 16)
        put_mcd(aie::load_v<16>(acc + i).to_native());
}

void flash_cascade_fold_put_f32(uint32_t n, const float *__restrict state,
                                const float *__restrict acc, int32_t acc_off)
{
    acc += acc_off;
    ::aie::set_rounding(FLASH_ROUNDING_MODE);
    aie::vector<float, 16> hdr(get_scd_v16float());
    float m_new, c_own, c_in;
    flash_fold_weights(state[0], hdr[0], m_new, c_own, c_in);
    put_mcd(flash_header(m_new, state[1] * c_own + hdr[1] * c_in).to_native());

    aie::vector<float, 16> w_own = aie::broadcast<float, 16>(c_own);
    aie::vector<float, 16> w_in = aie::broadcast<float, 16>(c_in);
    for (uint32_t i = 0; i < n; i += 16) {
        aie::vector<float, 16> in(get_scd_v16float());
        aie::accum<accfloat, 16> a = aie::mul(aie::load_v<16>(acc + i), w_own);
        put_mcd(aie::add(a, aie::mul(in, w_in)).to_vector<float>().to_native());
    }
}

// The chain end: fold, divide by the merged sum, narrow to bf16 (taccum_finish_scaled_bf16's math).
void flash_cascade_fold_finish_bf16(uint32_t n, const float *__restrict state,
                                    const float *__restrict acc, int32_t acc_off,
                                    bfloat16 *__restrict out)
{
    acc += acc_off;
    ::aie::set_rounding(FLASH_ROUNDING_MODE);
    aie::vector<float, 16> hdr(get_scd_v16float());
    float m_new, c_own, c_in;
    flash_fold_weights(state[0], hdr[0], m_new, c_own, c_in);
    const float inv_l = aie::inv(state[1] * c_own + hdr[1] * c_in);

    aie::vector<float, 16> w_own = aie::broadcast<float, 16>(c_own * inv_l);
    aie::vector<float, 16> w_in = aie::broadcast<float, 16>(c_in * inv_l);
    for (uint32_t i = 0; i < n; i += 16) {
        aie::vector<float, 16> in(get_scd_v16float());
        aie::accum<accfloat, 16> a = aie::mul(aie::load_v<16>(acc + i), w_own);
        a = aie::add(a, aie::mul(in, w_in));
        aie::store_v(out + i, a.to_vector<bfloat16>());
    }
}

} // extern "C"
