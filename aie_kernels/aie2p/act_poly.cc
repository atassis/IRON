// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// sigmoid and SiLU in f32 polynomial math, bf16 in and out: sigmoid(x) = exp(-softplus(-x)), no
// reciprocal and no SFU LUT. Drop-in for silu.cc / sigmoid.cc where their tanh LUT error is visible.

#include "../aie_kernel_utils.h"
#include "poly_exp.h"

#include <stdint.h>

template <bool Silu> static inline vf act(vf x)
{
    vf sig = expv(negv(softplusv(negv(x))));
    return Silu ? vmul(x, sig) : sig;
}

template <bool Silu>
static inline void act_poly_bf16(bfloat16 *restrict in, bfloat16 *restrict out, const int32_t n)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    auto it_in = aie::begin_restrict_vector<32>(in);
    auto it_out = aie::begin_restrict_vector<32>(out);
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (int i = 0; i < n; i += 32) {
        aie::accum<accfloat, 32> acc;
        acc.from_vector(*it_in++);
        aie::vector<float, 32> x = acc.template to_vector<float>();
        acc.from_vector(aie::concat(act<Silu>(x.extract<16>(0)), act<Silu>(x.extract<16>(1))));
        *it_out++ = acc.template to_vector<bfloat16>();
    }
    event1();
}

// The same two functions at bf16 output accuracy in native bf16 x bf16 -> accfloat MACs: every
// f32 operand is carried as two bf16 limbs (16 mantissa bits), so a product costs three native MACs
// where an emulated f32 product costs nine. sigmoid(x) = 1 / (1 + 2^(-x log2 e)).
using lacc = aie::accum<accfloat, 16>;
using lbf = aie::vector<bfloat16, 16>;

static inline void limbs(lacc a, lbf &hi, lbf &lo)
{
    hi = a.template to_vector<bfloat16>();
    lacc h;
    h.from_vector(hi);
    lo = aie::sub(a, h).template to_vector<bfloat16>();
}

static inline lacc mul2(lbf ah, lbf al, lbf bh, lbf bl)
{
    lacc r = aie::mul(ah, bh);
    r = aie::mac(r, ah, bl);
    return aie::mac(r, al, bh);
}

static inline lacc acc_of(vf v)
{
    lacc a;
    a.from_vector(v);
    return a;
}

static inline lacc acc_of(lbf v)
{
    lacc a;
    a.from_vector(v);
    return a;
}

static inline lbf bcast(float c) { return aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(c)); }

template <bool Silu> static inline lbf act_fast(lbf x)
{
    const float nl2e = -1.4426950408889634f;
    const bfloat16 l_hi = static_cast<bfloat16>(nl2e);
    lacc t = aie::mul(x, aie::broadcast<bfloat16, 16>(l_hi));
    t = aie::mac(t, x, bcast(nl2e - float(l_hi)));
    vf tv = aie::min(aie::max(t.template to_vector<float>(), aie::broadcast<float, V>(-126.0f)),
                     aie::broadcast<float, V>(126.0f));
    aie::vector<int32_t, V> ki = aie::to_fixed<int32_t>(tv);
    ki = aie::sub(ki, aie::select(aie::broadcast<int32_t, V>(0), aie::broadcast<int32_t, V>(1),
                                  aie::lt(tv, aie::to_float<float>(ki))));
    lbf fh, fl;
    limbs(aie::sub(acc_of(tv), acc_of(aie::to_float<float>(ki))), fh, fl);
    // 2^f on [0, 1): the minimax quintic exp2v uses, Horner with two-limb p and f.
    static constexpr float c[6] = {1.0f, 0.6931471805f, 0.2402265069f, 0.0555041087f,
                                   0.0096181291f, 0.0013333558f};
    lbf ph = bcast(c[5]), pl = bcast(c[5] - float(static_cast<bfloat16>(c[5])));
    lacc p;
    for (int j = 4; j >= 0; --j) {
        p = aie::add(mul2(ph, pl, fh, fl), acc_of(aie::broadcast<float, V>(c[j])));
        limbs(p, ph, pl);
    }
    aie::vector<int32_t, V> e = aie::add(p.template to_vector<float>().template cast_to<int32_t>(),
                                         aie::upshift(ki, 23));
    // d = 1 + 2^t, r = 1/d refined twice by Newton: r += r (1 - d r).
    const lacc d = aie::add(acc_of(e.template cast_to<float>()), acc_of(aie::broadcast<float, V>(1.0f)));
    lbf dh, dl, rh, rl;
    limbs(d, dh, dl);
    limbs(acc_of(aie::inv(d.template to_vector<float>())), rh, rl);
    for (int it = 0; it < 2; ++it) {
        lacc err = aie::sub(acc_of(aie::broadcast<float, V>(1.0f)), mul2(dh, dl, rh, rl));
        lbf eh, el;
        limbs(err, eh, el);
        lacc r = aie::add(mul2(rh, rl, eh, el), acc_of(aie::add(acc_of(rh), acc_of(rl)).template to_vector<float>()));
        limbs(r, rh, rl);
    }
    if (!Silu)
        return aie::add(acc_of(rh), acc_of(rl)).template to_vector<bfloat16>();
    lacc y = aie::mul(x, rh);
    return aie::mac(y, x, rl).template to_vector<bfloat16>();
}

template <bool Silu>
static inline void act_fast_bf16(bfloat16 *restrict in, bfloat16 *restrict out, const int32_t n)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    for (int i = 0; i < n; i += 16)
        aie::store_v(out + i, act_fast<Silu>(aie::load_v<16>(in + i)));
    event1();
}

extern "C" {

void silu_poly_bf16(bfloat16 *in, bfloat16 *out, int32_t n) { act_poly_bf16<true>(in, out, n); }

void sigmoid_poly_bf16(bfloat16 *in, bfloat16 *out, int32_t n) { act_poly_bf16<false>(in, out, n); }

void silu_fast_bf16(bfloat16 *in, bfloat16 *out, int32_t n) { act_fast_bf16<true>(in, out, n); }

void sigmoid_fast_bf16(bfloat16 *in, bfloat16 *out, int32_t n) { act_fast_bf16<false>(in, out, n); }

} // extern "C"
