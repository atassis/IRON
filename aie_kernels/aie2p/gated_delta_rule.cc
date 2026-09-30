// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One decode step of the gated delta rule (Gated DeltaNet: Qwen3.5 / Qwen3-Next linear attention).
//   gdn_gates_all  alpha = exp(neg_a * softplus(a + dt_bias)), beta = sigmoid(b), every head, in f32
//                  polynomial math -- no SFU LUT, because alpha compounds through the state.
//   gdn_unit       one (head, value chunk): S <- alpha*S; err = (v - S^T k)*beta; S += k (x) err;
//                  o = S^T q, with S [DK, DVC] f32 read from s_in and written to s_out.
//   gdn_seq        gdn_unit over GDR_T consecutive tokens, the state held in L1 throughout.
// aie2p emulates f32 vector multiplies through bf16, so the gates land at ~8.5e-5 relative, not 2e-7.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GDR_DK
#define GDR_DK 128
#endif
#ifndef GDR_DVC
#define GDR_DVC 16
#endif
#ifndef GDR_HEADS
#define GDR_HEADS 32
#endif
#ifndef GDR_T
#define GDR_T 1
#endif
#ifndef GDR_REP
#define GDR_REP 2
#endif
#ifndef GDR_LIMBS
#define GDR_LIMBS 0
#endif

#include "poly_exp.h"

static_assert(GDR_HEADS % V == 0, "head count must be a multiple of the 16-lane f32 vector");
static_assert(GDR_DVC % V == 0 && GDR_DK % 32 == 0, "chunk and key dims must be vector multiples");

template <unsigned N> static inline aie::vector<float, N> bf16_to_f32(const bfloat16 *p)
{
    aie::accum<accfloat, N> acc;
    acc.from_vector(aie::load_v<N>(p));
    return acc.template to_vector<float>();
}

template <unsigned H>
static inline void gates_core(const bfloat16 *__restrict ab, const float *__restrict params,
                              float *__restrict gates)
{
    for (unsigned h = 0; h < H; h += V) {
        vf a = bf16_to_f32<V>(&ab[h]);
        vf b = bf16_to_f32<V>(&ab[H + h]);
        vf alpha = expv(vmul(aie::load_v<V>(&params[h]),
                             softplusv(aie::add(a, aie::load_v<V>(&params[H + h])))));
        vf beta = expv(negv(softplusv(negv(b))));
        auto [lo, hi] = aie::interleave_zip(alpha, beta, 1);
        aie::store_v(&gates[2 * h], lo);
        aie::store_v(&gates[2 * h + V], hi);
    }
}

// out[0..DVC) = sum_i x[i] * S[i, :]
template <unsigned DK, unsigned DVC>
static inline aie::vector<float, DVC> state_read(const float *__restrict S, const float *__restrict x)
{
    aie::accum<accfloat, DVC> acc;
    acc.from_vector(aie::broadcast<float, DVC>(0.0f));
    for (unsigned i = 0; i < DK; ++i)
        acc = aie::mac(acc, aie::load_v<DVC>(&S[i * DVC]), aie::broadcast<float, DVC>(x[i]));
    return acc.template to_vector<float>();
}

template <unsigned DK, unsigned DVC>
static inline void unit_core(const bfloat16 *__restrict k, const bfloat16 *__restrict q,
                             const bfloat16 *__restrict v, const float *__restrict gate,
                             const float *__restrict s_in, float *__restrict s_out,
                             bfloat16 *__restrict o)
{
    // k and q as f32 in L1 so the row loops broadcast from memory, not from a DK-wide register.
    alignas(64) float kf[DK];
    alignas(64) float qf[DK];
    for (unsigned i = 0; i < DK; i += 32) {
        aie::store_v(&kf[i], bf16_to_f32<32>(&k[i]));
        aie::store_v(&qf[i], bf16_to_f32<32>(&q[i]));
    }
    const aie::vector<float, DVC> alpha = aie::broadcast<float, DVC>(gate[0]);
    const aie::vector<float, DVC> beta = aie::broadcast<float, DVC>(gate[1]);

    // err = (v - alpha * S^T k) * beta: the read sees the DECAYED state.
    aie::vector<float, DVC> pred = aie::mul(state_read<DK, DVC>(s_in, kf), alpha).template to_vector<float>();
    aie::vector<float, DVC> err =
        aie::mul(aie::sub(bf16_to_f32<DVC>(v), pred), beta).template to_vector<float>();
    for (unsigned i = 0; i < DK; ++i) {
        aie::accum<accfloat, DVC> acc = aie::mul(aie::load_v<DVC>(&s_in[i * DVC]), alpha);
        acc = aie::mac(acc, err, aie::broadcast<float, DVC>(kf[i]));
        aie::store_v(&s_out[i * DVC], acc.template to_vector<float>());
    }
    aie::accum<accfloat, DVC> oacc;
    oacc.from_vector(state_read<DK, DVC>(s_out, qf));
    aie::store_v(o, oacc.template to_vector<bfloat16>());
}

#if GDR_T > 1 || GDR_LIMBS
// The limb path keeps each state row as two bf16 limbs [hi | lo] (the same 64 bytes as its f32
// form) so every product is a native bf16 x bf16 -> accfloat MAC: k, q and v are bf16 already, and
// an f32 x f32 product is emulated as nine of them. hi + lo carries 16 mantissa bits.
using accv = aie::accum<accfloat, GDR_DVC>;
using bfv = aie::vector<bfloat16, GDR_DVC>;

static inline void to_limbs(accv a, bfloat16 *dst)
{
    const bfv hi = a.template to_vector<bfloat16>();
    accv h;
    h.from_vector(hi);
    aie::store_v(dst, hi);
    aie::store_v(dst + GDR_DVC, aie::sub(a, h).template to_vector<bfloat16>());
}

static inline accv from_limbs(const bfloat16 *src)
{
    accv a, b;
    a.from_vector(aie::load_v<GDR_DVC>(src));
    b.from_vector(aie::load_v<GDR_DVC>(src + GDR_DVC));
    return aie::add(a, b);
}

// One token over S held as limbs in place: err = (v - alpha S^T k) beta, S = alpha S + k err^T,
// o = S^T q, the output accumulated in the update's own pass over S.
template <unsigned DK>
static inline void limb_step(const bfloat16 *__restrict k, const bfloat16 *__restrict q,
                             const bfloat16 *__restrict v, const float *__restrict gate,
                             bfloat16 *__restrict S, bfloat16 *__restrict o)
{
    accv p0 = aie::zeros<accfloat, GDR_DVC>(), p1 = aie::zeros<accfloat, GDR_DVC>();
    for (unsigned i = 0; i < DK; i += 2) {
        const bfloat16 *r0 = S + i * 2 * GDR_DVC, *r1 = r0 + 2 * GDR_DVC;
        const bfv k0 = aie::broadcast<bfloat16, GDR_DVC>(k[i]), k1 = aie::broadcast<bfloat16, GDR_DVC>(k[i + 1]);
        p0 = aie::mac(p0, aie::load_v<GDR_DVC>(r0), k0);
        p1 = aie::mac(p1, aie::load_v<GDR_DVC>(r1), k1);
        p0 = aie::mac(p0, aie::load_v<GDR_DVC>(r0 + GDR_DVC), k0);
        p1 = aie::mac(p1, aie::load_v<GDR_DVC>(r1 + GDR_DVC), k1);
    }
    const aie::vector<float, GDR_DVC> alpha = aie::broadcast<float, GDR_DVC>(gate[0]);
    const aie::vector<float, GDR_DVC> beta = aie::broadcast<float, GDR_DVC>(gate[1]);
    const aie::vector<float, GDR_DVC> pred =
        aie::mul(aie::add(p0, p1).template to_vector<float>(), alpha).template to_vector<float>();
    accv e;
    e.from_vector(aie::mul(aie::sub(bf16_to_f32<GDR_DVC>(v), pred), beta).template to_vector<float>());
    const bfv e_hi = e.template to_vector<bfloat16>();
    accv eh;
    eh.from_vector(e_hi);
    const bfv e_lo = aie::sub(e, eh).template to_vector<bfloat16>();
    const bfloat16 a_hi_s = static_cast<bfloat16>(gate[0]);
    const bfv a_hi = aie::broadcast<bfloat16, GDR_DVC>(a_hi_s);
    const bfv a_lo = aie::broadcast<bfloat16, GDR_DVC>(static_cast<bfloat16>(gate[0] - float(a_hi_s)));
    accv o0 = aie::zeros<accfloat, GDR_DVC>(), o1 = aie::zeros<accfloat, GDR_DVC>();
    // Two rows a trip: each row's update is one serial MAC chain, and two independent ones overlap.
    for (unsigned i = 0; i < DK; i += 2) {
        bfloat16 *r0 = S + i * 2 * GDR_DVC, *r1 = r0 + 2 * GDR_DVC;
        const bfv hi0 = aie::load_v<GDR_DVC>(r0), lo0 = aie::load_v<GDR_DVC>(r0 + GDR_DVC);
        const bfv hi1 = aie::load_v<GDR_DVC>(r1), lo1 = aie::load_v<GDR_DVC>(r1 + GDR_DVC);
        const bfv k0 = aie::broadcast<bfloat16, GDR_DVC>(k[i]), k1 = aie::broadcast<bfloat16, GDR_DVC>(k[i + 1]);
        accv a0 = aie::mul(hi0, a_hi), a1 = aie::mul(hi1, a_hi);
        a0 = aie::mac(a0, hi0, a_lo);
        a1 = aie::mac(a1, hi1, a_lo);
        a0 = aie::mac(a0, lo0, a_hi);
        a1 = aie::mac(a1, lo1, a_hi);
        a0 = aie::mac(a0, e_hi, k0);
        a1 = aie::mac(a1, e_hi, k1);
        a0 = aie::mac(a0, e_lo, k0);
        a1 = aie::mac(a1, e_lo, k1);
        to_limbs(a0, r0);
        to_limbs(a1, r1);
        const bfv q0 = aie::broadcast<bfloat16, GDR_DVC>(q[i]), q1 = aie::broadcast<bfloat16, GDR_DVC>(q[i + 1]);
        o0 = aie::mac(o0, aie::load_v<GDR_DVC>(r0), q0);
        o1 = aie::mac(o1, aie::load_v<GDR_DVC>(r1), q1);
        o0 = aie::mac(o0, aie::load_v<GDR_DVC>(r0 + GDR_DVC), q0);
        o1 = aie::mac(o1, aie::load_v<GDR_DVC>(r1 + GDR_DVC), q1);
    }
    aie::store_v(o, aie::add(o0, o1).template to_vector<bfloat16>());
}

// seq_core's contract on the limb form: s_in's f32 rows become limbs in s_out, n tokens step in
// place, and s_out goes back to f32 rows.
template <unsigned DK, unsigned T>
static inline void limb_seq(const bfloat16 *q, const bfloat16 *k, const bfloat16 *v, const float *gates,
                            const float *s_in, float *s_out, bfloat16 *o, unsigned o_stride, unsigned n)
{
    bfloat16 *S = reinterpret_cast<bfloat16 *>(s_out);
    for (unsigned i = 0; i < DK; ++i) {
        accv a;
        a.from_vector(aie::load_v<GDR_DVC>(s_in + i * GDR_DVC));
        to_limbs(a, S + i * 2 * GDR_DVC);
    }
    for (unsigned t = 0; t < n; ++t)
        limb_step<DK>(k + t * DK, q + t * DK, v + t * DK, gates + t * 2 * GDR_HEADS, S, o + t * o_stride);
    for (unsigned t = n; t < T; ++t)
        aie::store_v(o + t * o_stride, aie::zeros<bfloat16, GDR_DVC>());
    for (unsigned i = 0; i < DK; ++i)
        aie::store_v(s_out + i * GDR_DVC, from_limbs(S + i * 2 * GDR_DVC).template to_vector<float>());
}
#endif

// Ping-pongs between s_in and s_out so neither restrict pointer aliases the other. Steps the first
// n of the T tokens; the rest leave the state untouched and their o rows zero.
template <unsigned DK, unsigned DVC, unsigned T>
static inline void seq_core(const bfloat16 *q, const bfloat16 *k, const bfloat16 *v, const float *gates,
                            float *s_in, float *s_out, bfloat16 *o, unsigned o_stride, unsigned n = T)
{
    float *src = s_in;
    float *dst = s_out;
    for (unsigned t = 0; t < n; ++t) {
        unit_core<DK, DVC>(k + t * DK, q + t * DK, v + t * DK, gates + t * 2 * GDR_HEADS, src, dst,
                           o + t * o_stride);
        float *tmp = src;
        src = dst;
        dst = tmp;
    }
    for (unsigned t = n; t < T; ++t)
        aie::store_v(o + t * o_stride, aie::zeros<bfloat16, DVC>());
    if (src != s_out)
        for (unsigned i = 0; i < DK * DVC; i += 16)
            aie::store_v(&s_out[i], aie::load_v<16>(&src[i]));
}

extern "C" {

#if GDR_T == 1
// ab: [a | b] bf16 per head; params: [neg_a | dt_bias] f32 carried in a bf16-typed stream element.
void gdn_gates_all(bfloat16 *ab, bfloat16 *params, float *gates)
{
    event0();
    gates_core<GDR_HEADS>(ab, reinterpret_cast<const float *>(params), gates);
    event1();
}
#endif

// Stream element -> core-local buffer, so the stream fifo stays two deep: a compute tile has 16
// BDs and every fifo slot takes one.
void gdn_copy(bfloat16 *el, bfloat16 *dst, int32_t offset)
{
    for (unsigned i = 0; i < GDR_DK; i += 32)
        aie::store_v(&dst[offset + i], aie::load_v<32>(&el[i]));
}

#if GDR_T == 1
// loc: the core's q, k and v heads (dk each, dv == dk) at element offsets q_off, k_off, v_off.
void gdn_unit(bfloat16 *loc, float *gates, int32_t head, int32_t q_off, int32_t k_off,
              int32_t v_off, int32_t chunk, float *s_in, float *s_out, bfloat16 *o)
{
    event0();
    const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
#if GDR_LIMBS
    limb_seq<GDR_DK, 1>(loc + q_off, loc + k_off, loc + v_off + chunk * GDR_DVC, gates + 2 * head,
                        s_in, s_out, o + chunk * GDR_DVC, 0, 1);
#else
    unit_core<GDR_DK, GDR_DVC>(loc + k_off, loc + q_off, loc + v_off + chunk * GDR_DVC,
                               gates + 2 * head, s_in, s_out, o + chunk * GDR_DVC);
#endif
    ::aie::set_rounding(saved);
    event1();
}
#else
// gdn_gates_all for token t, into row t of a [GDR_T, 2 * GDR_HEADS] gate table.
void gdn_gates_t(bfloat16 *ab, bfloat16 *params, float *gates, int32_t t)
{
    gates_core<GDR_HEADS>(ab, reinterpret_cast<const float *>(params), gates + t * 2 * GDR_HEADS);
}

// Row `idx` of region `region` of loc, which holds one k-head group as [q | k | v_0 .. v_REP-1],
// GDR_T dk-wide rows each.
void gdn_copy_at(bfloat16 *el, bfloat16 *loc, int32_t region, int32_t idx)
{
    gdn_copy(el, loc, (region * GDR_T + idx) * GDR_DK);
}

// In place on loc's q and k rows: x / sqrt(sum x^2 + 1e-6), q also by dk^-0.5 -- HF's l2norm,
// which decode applies as a separate RMSNorm op before the delta rule.
void gdn_l2qk(bfloat16 *loc)
{
    const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    for (unsigned r = 0; r < 2 * GDR_T; ++r) {
        bfloat16 *x = loc + r * GDR_DK;
        aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
        for (unsigned i = 0; i < GDR_DK; i += 32) {
            aie::vector<bfloat16, 32> v = aie::load_v<32>(&x[i]);
            acc = aie::mac(acc, v, v);
        }
        const float inv = aie::invsqrt(aie::reduce_add(acc.template to_vector<float>()) + 1e-6f);
        const float scale = r < GDR_T ? aie::invsqrt(float(GDR_DK)) : 1.0f;  // q's dk^-0.5
        const aie::vector<float, V> invv = aie::mul(aie::broadcast<float, V>(inv),
                                                    aie::broadcast<float, V>(scale))
                                               .template to_vector<float>();
        for (unsigned i = 0; i < GDR_DK; i += V) {
            aie::accum<accfloat, V> y = aie::mul(bf16_to_f32<V>(&x[i]), invv);
            aie::store_v(&x[i], y.template to_vector<bfloat16>());
        }
    }
    ::aie::set_rounding(saved);
}

// Value head head0 + grp * GDR_REP + slot over GDR_T tokens; o is [GDR_T, dv] for that head, and
// chunk selects the dvc columns of v, S and o.
void gdn_seq(bfloat16 *loc, float *gates, int32_t head0, int32_t grp, int32_t slot, int32_t chunk,
             float *s_in, float *s_out, bfloat16 *o)
{
    event0();
    const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    seq_core<GDR_DK, GDR_DVC, GDR_T>(loc, loc + GDR_T * GDR_DK,
                                     loc + (2 + slot) * GDR_T * GDR_DK + chunk * GDR_DVC,
                                     gates + 2 * (head0 + grp * GDR_REP + slot), s_in, s_out,
                                     o + chunk * GDR_DVC, GDR_DK);
    ::aie::set_rounding(saved);
    event1();
}

// gdn_seq over the first cnt[0] tokens only (an int32 in a bf16-typed stream element), so a
// padded chunk's trailing rows never reach the state.
void gdn_seq_n(bfloat16 *loc, float *gates, int32_t head0, int32_t grp, int32_t slot, int32_t chunk,
               bfloat16 *cnt, float *s_in, float *s_out, bfloat16 *o)
{
    event0();
    const int32_t n = *reinterpret_cast<const int32_t *>(cnt);
    const unsigned steps = n < 0 ? 0u : (n > int32_t(GDR_T) ? GDR_T : unsigned(n));
    const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
#if GDR_LIMBS
    limb_seq<GDR_DK, GDR_T>(loc, loc + GDR_T * GDR_DK, loc + (2 + slot) * GDR_T * GDR_DK + chunk * GDR_DVC,
                            gates + 2 * (head0 + grp * GDR_REP + slot), s_in, s_out,
                            o + chunk * GDR_DVC, GDR_DK, steps);
#else
    seq_core<GDR_DK, GDR_DVC, GDR_T>(loc, loc + GDR_T * GDR_DK,
                                     loc + (2 + slot) * GDR_T * GDR_DK + chunk * GDR_DVC,
                                     gates + 2 * (head0 + grp * GDR_REP + slot), s_in, s_out,
                                     o + chunk * GDR_DVC, GDR_DK, steps);
#endif
    ::aie::set_rounding(saved);
    event1();
}
#endif

} // extern "C"
