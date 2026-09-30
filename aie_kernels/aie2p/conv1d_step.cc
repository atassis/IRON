// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One decode step of a depthwise causal conv on a [K, C] window tile: rows 0..K-2 are the input
// history (oldest first), row K-1 the new input. The output tile holds the history shifted by one
// in rows 0..K-2 and the conv result in row K-1, so the op can run in place on the window.
// bf16 operands, f32 accumulate; C and K are compile-time so every loop bound is a literal.
// With CS_T > 1 the window is [K-1+T, C]: T new inputs after the history, and the output holds the
// last K-1 inputs as the next history followed by the T conv results.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef CS_C
#define CS_C 256
#endif
#ifndef CS_K
#define CS_K 4
#endif
#ifndef CS_T
#define CS_T 1
#endif

static constexpr unsigned L = 32;
static_assert(CS_C % L == 0, "tile channels must be a multiple of the 32-lane bf16 vector");
static_assert(CS_K >= 2, "a one-tap conv has no history to carry");

template <unsigned C, unsigned K>
static inline void conv1d_step_window_core(const bfloat16 *__restrict win,
                                           const bfloat16 *__restrict w,
                                           bfloat16 *__restrict out)
{
    event0();
    const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    for (unsigned c = 0; c < C; c += L) {
        aie::accum<accfloat, L> acc = aie::mul(aie::load_v<L>(&w[c]), aie::load_v<L>(&win[c]));
        for (unsigned j = 1; j < K; ++j)
            acc = aie::mac(acc, aie::load_v<L>(&w[j * C + c]), aie::load_v<L>(&win[j * C + c]));
        for (unsigned j = 0; j + 1 < K; ++j)
            aie::store_v(&out[j * C + c], aie::load_v<L>(&win[(j + 1) * C + c]));
        aie::store_v(&out[(K - 1) * C + c], acc.template to_vector<bfloat16>());
    }
    ::aie::set_rounding(saved);
    event1();
}

template <unsigned C, unsigned K, unsigned T>
static inline void conv1d_seq_window_core(const bfloat16 *__restrict win,
                                          const bfloat16 *__restrict w,
                                          bfloat16 *__restrict out)
{
    event0();
    const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    for (unsigned c = 0; c < C; c += L) {
        for (unsigned t = 0; t < T; ++t) {
            aie::accum<accfloat, L> acc =
                aie::mul(aie::load_v<L>(&w[c]), aie::load_v<L>(&win[t * C + c]));
            for (unsigned j = 1; j < K; ++j)
                acc = aie::mac(acc, aie::load_v<L>(&w[j * C + c]),
                               aie::load_v<L>(&win[(t + j) * C + c]));
            aie::store_v(&out[(K - 1 + t) * C + c], acc.template to_vector<bfloat16>());
        }
        for (unsigned j = 0; j + 1 < K; ++j)
            aie::store_v(&out[j * C + c], aie::load_v<L>(&win[(T + j) * C + c]));
    }
    ::aie::set_rounding(saved);
    event1();
}

extern "C" {

void conv1d_step_window(bfloat16 *win, bfloat16 *w, bfloat16 *out)
{
#if CS_T == 1
    conv1d_step_window_core<CS_C, CS_K>(win, w, out);
#else
    conv1d_seq_window_core<CS_C, CS_K, CS_T>(win, w, out);
#endif
}

} // extern "C"
