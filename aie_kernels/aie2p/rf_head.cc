// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Resident-forward head pass: per-head RMSNorm (q and k with their gains, v gainless) and RoPE
// (rotate-half) on q and k, for one MemTile's four heads [q 2m, q 2m+1, k m, v m] x 256, four
// rows per call. Rows arrive as the head of an 8640-B activation buffer; the rope unit carries
// 16 rows of [cos 128 | sin 128] bf16. Rounding matches the host model (rf_intmath.h, K048/K049).
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "rf_intmath.h"

namespace {

using namespace rf;

constexpr unsigned HD = 256, HALF = 128, HEADS = 4, ROWS = 4;
constexpr unsigned ROW = HEADS * HD;                 // one row of the MemTile's q/k/v
constexpr unsigned ROPE_OFF = 0, OUT_OFF = 8192, G_OFF = 30720;
constexpr float EPS = 1e-6f;

using v32b = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
using A16 = aie::accum<accfloat, 16>;

inline v32b bf(const A32 &a) { return a.to_vector<bfloat16>(); }

// Sum of squares of one 256-wide head row: 32 lanes over 8 chunks, the two lane halves, then the
// 16 lanes left to right (the adds run on the vector adder: rf_norm's order, 8 chunks not 15).
template <bool ALIGNED = true>
inline v32b ld(const bfloat16 *p) {
  if constexpr (ALIGNED)
    return aie::load_v<32>(p);
  else
    return aie::load_unaligned_v<32>(p, 8);
}

template <bool ALIGNED = true>
inline float head_ss(const bfloat16 *x) {
  alignas(64) float lanes[16];
  v32b v = ld<ALIGNED>(x);
  A32 a = aie::mul(v, v);
  for (unsigned j = 1; j < HD / 32; ++j) {
    v32b w = ld<ALIGNED>(x + 32 * j);
    a = aie::add(a, aie::mul(w, w));
  }
  aie::store_v(lanes, aie::add(a.extract<16>(0), a.extract<16>(1)).to_vector<float>());
  float t = lanes[0];
  for (unsigned k = 1; k < 16; ++k)
    t += lanes[k];
  return t;
}

// Rows 4u .. 4u+3 of the 16-row block: normalise the 16 head rows, rope q and k, write bf16 rows
// to scr + OUT_OFF for the write-back DMA. Head h row i sits at x + h * XH + i * XR on input and
// at out + h * OH + i * OR on output. An XH off the 64-B grid is read with unaligned loads (8-lane
// alignment, 16 B).
template <unsigned XH, unsigned XR, unsigned OH, unsigned OR>
void head_unit(const uint8_t *__restrict ab, uint8_t *__restrict scr, int32_t u) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *x = reinterpret_cast<const bfloat16 *>(ab);
  bfloat16 *o = reinterpret_cast<bfloat16 *>(scr + OUT_OFF);
  const bfloat16 *rope = reinterpret_cast<const bfloat16 *>(scr + ROPE_OFF);
  const bfloat16 *gq = reinterpret_cast<const bfloat16 *>(scr + G_OFF);
  const bfloat16 *gk = gq + HD;
  alignas(64) uint32_t ss[16];
  for (unsigned i = 0; i < ROWS; ++i)
    for (unsigned h = 0; h < HEADS; ++h)
      ss[i * HEADS + h] = bits(head_ss<XH % 32 == 0>(x + h * XH + i * XR));
  aie::set_rounding(aie::rounding_mode::floor);
  V q = div_rn(aie::load_v<16>(reinterpret_cast<const int32_t *>(ss)), bc((int32_t)bits(256.0f)));
  aie::set_rounding(aie::rounding_mode::conv_even);
  A16 m, eps;
  m.from_vector(aie::vector_cast<float>(q));
  eps.from_vector(aie::broadcast<float, 16>(EPS));
  V me = aie::vector_cast<int32_t>(aie::add(m, eps).to_vector<float>());
  aie::set_rounding(aie::rounding_mode::floor);
  V r = div_rn(bc((int32_t)bits(1.0f)), sqrt_rn(me));
  aie::set_rounding(aie::rounding_mode::conv_even);
  A16 ra;
  ra.from_vector(aie::vector_cast<float>(r));
  aie::vector<bfloat16, 16> rb = ra.to_vector<bfloat16>();
  alignas(64) bfloat16 n[HD];
  for (unsigned i = 0; i < ROWS; ++i) {
    const bfloat16 *c = rope + (4 * u + i) * HD;
    const bfloat16 *s = c + HALF;
    for (unsigned h = 0; h < HEADS; ++h) {
      const bfloat16 *xh = x + h * XH + i * XR;
      bfloat16 *oh = o + h * OH + i * OR;
      v32b sv = aie::broadcast<bfloat16, 32>(rb[i * HEADS + h]);
      const bfloat16 *g = h < 2 ? gq : gk;
      for (unsigned j = 0; j < HD / 32; ++j) {
        v32b t = bf(aie::mul(ld<XH % 32 == 0>(xh + 32 * j), sv));
        aie::store_v((h == 3 ? oh : n) + 32 * j, h == 3 ? t : bf(aie::mul(t, aie::load_v<32>(g + 32 * j))));
      }
      if (h == 3)
        continue;
      // out1 = x1 c - x2 s, out2 = x2 c + x1 s, each from exact products, one rounding to bf16
      for (unsigned j = 0; j < HALF / 32; ++j) {
        v32b x1 = aie::load_v<32>(n + 32 * j), x2 = aie::load_v<32>(n + HALF + 32 * j);
        v32b cj = aie::load_v<32>(c + 32 * j), sj = aie::load_v<32>(s + 32 * j);
        aie::store_v(oh + 32 * j, bf(aie::sub(aie::mul(x1, cj), aie::mul(x2, sj))));
        aie::store_v(oh + HALF + 32 * j, bf(aie::add(aie::mul(x2, cj), aie::mul(x1, sj))));
      }
    }
  }
}

// One unit of either layer type at run time, so a core holds one copy: sliding (glob 0) is four
// 256-wide heads [q, q, k, v] read 1080 elements apart, out [4 heads][4 rows x 256]; global (glob 1)
// is one 512-wide q head, out [half 2][4 rows x 256] (the attention's Q units are half heads). The
// rope table rows are [cos nreal | sin nreal]; dims past nreal in each half rotate with cos 1,
// sin 0 (proportional RoPE: 64 real frequencies of 256 at head 512).
__attribute__((noinline)) void head_unit_rt(const uint8_t *__restrict ab, uint8_t *__restrict scr,
                                            int32_t u, int32_t glob) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const unsigned hd = glob ? 512 : 256, half = hd / 2, heads = glob ? 1 : 4;
  const unsigned xh = glob ? 2160 : 1080, oh = ROWS * hd, nreal = glob ? 64 : HALF;
  const bfloat16 *x = reinterpret_cast<const bfloat16 *>(ab);
  bfloat16 *o = reinterpret_cast<bfloat16 *>(scr + OUT_OFF);
  const bfloat16 *rope = reinterpret_cast<const bfloat16 *>(scr + ROPE_OFF);
  const bfloat16 *gq = reinterpret_cast<const bfloat16 *>(scr + G_OFF);
  const bfloat16 *gk = gq + HD;
  alignas(64) uint32_t ss[16] = {};
  for (unsigned i = 0; i < ROWS; ++i)
    for (unsigned h = 0; h < heads; ++h) {
      const bfloat16 *r = x + h * xh + i * hd;
      alignas(64) float lanes[16];
      A32 a = aie::mul(ld<false>(r), ld<false>(r));
      for (unsigned j = 1; j < hd / 32; ++j) {
        v32b w = ld<false>(r + 32 * j);
        a = aie::add(a, aie::mul(w, w));
      }
      aie::store_v(lanes, aie::add(a.extract<16>(0), a.extract<16>(1)).to_vector<float>());
      float t = lanes[0];
      for (unsigned k = 1; k < 16; ++k)
        t += lanes[k];
      ss[i * heads + h] = bits(t);
    }
#ifdef RF_FAST_RSQRT
  aie::vector<float, 16> ms =
      aie::mul(aie::vector_cast<float>(aie::load_v<16>(reinterpret_cast<const int32_t *>(ss))),
               glob ? 1.0f / 512.0f : 1.0f / 256.0f).to_vector<float>();
  V r = aie::vector_cast<int32_t>(aie::invsqrt(aie::add(ms, aie::broadcast<float, 16>(EPS))));
#else
  aie::set_rounding(aie::rounding_mode::floor);
  V q = div_rn(aie::load_v<16>(reinterpret_cast<const int32_t *>(ss)),
               bc((int32_t)bits(glob ? 512.0f : 256.0f)));
  aie::set_rounding(aie::rounding_mode::conv_even);
  A16 m, eps;
  m.from_vector(aie::vector_cast<float>(q));
  eps.from_vector(aie::broadcast<float, 16>(EPS));
  V me = aie::vector_cast<int32_t>(aie::add(m, eps).to_vector<float>());
  aie::set_rounding(aie::rounding_mode::floor);
  V r = div_rn(bc((int32_t)bits(1.0f)), sqrt_rn(me));
#endif
  aie::set_rounding(aie::rounding_mode::conv_even);
  A16 ra;
  ra.from_vector(aie::vector_cast<float>(r));
  aie::vector<bfloat16, 16> rb = ra.to_vector<bfloat16>();
  const v32b one = aie::broadcast<bfloat16, 32>(1.0f), zero = aie::zeros<bfloat16, 32>();
  alignas(64) bfloat16 n[512];
  for (unsigned i = 0; i < ROWS; ++i) {
    const bfloat16 *c = rope + (4 * u + i) * 2 * nreal;
    for (unsigned h = 0; h < heads; ++h) {
      const bfloat16 *xh_ = x + h * xh + i * hd;
      bfloat16 *oh_ = glob ? o + i * half : o + h * oh + i * hd;
      bfloat16 *oh2 = glob ? oh_ + ROWS * half : oh_ + half;
      v32b sv = aie::broadcast<bfloat16, 32>(rb[i * heads + h]);
      const bool v = !glob && h == 3;
      const bfloat16 *g = (glob || h < 2) ? gq : gk;
      for (unsigned j = 0; j < hd / 32; ++j) {
        v32b t = bf(aie::mul(ld<false>(xh_ + 32 * j), sv));
        aie::store_v((v ? oh_ : n) + 32 * j, v ? t : bf(aie::mul(t, aie::load_v<32>(g + 32 * j))));
      }
      if (v)
        continue;
      for (unsigned j = 0; j < half / 32; ++j) {
        const bool re = 32 * j < nreal;
        v32b x1 = aie::load_v<32>(n + 32 * j), x2 = aie::load_v<32>(n + half + 32 * j);
        v32b cj = re ? aie::load_v<32>(c + 32 * j) : one, sj = re ? aie::load_v<32>(c + nreal + 32 * j) : zero;
        aie::store_v(oh_ + 32 * j, bf(aie::sub(aie::mul(x1, cj), aie::mul(x2, sj))));
        aie::store_v(oh2 + 32 * j, bf(aie::add(aie::mul(x2, cj), aie::mul(x1, sj))));
      }
    }
  }
}

}  // namespace

extern "C" {

// One copy routine for the scratch's tables (rope, q and k gains): len bytes, a multiple of 64,
// to scr + off. rf_scr_copy_e is the same code for a weight element source.
void rf_scr_copy(const uint8_t *__restrict src, uint8_t *__restrict scr, int32_t off, int32_t len) {
  for (int32_t i = 0; i < len; i += 64)
    aie::store_v(scr + off + i, aie::load_v<64>(src + i));
}
void rf_scr_copy_e(const uint8_t *__restrict src, uint8_t *__restrict scr, int32_t off, int32_t len)
    __attribute__((alias("rf_scr_copy")));

// Global: the k_norm gain element (512 bf16) beside the q gain.
void rf_head_kgain(const uint8_t *__restrict elem, uint8_t *__restrict scr) {
  for (unsigned i = 0; i < 1024; i += 64)
    aie::store_v(scr + 32768 + i, aie::load_v<64>(elem + i));
}

void rf_head_unit_rt(const uint8_t *__restrict ab, uint8_t *__restrict scr, int32_t u, int32_t glob) {
  head_unit_rt(ab, scr, u, glob);
}

// Global K: column c's 64 dims of one 16-row block, [16][32c .. 32c+31 | 256+32c ..] (a RoPE pair
// per lane), with the cross-column rstd (rf_norm's rf_rstd_recv slot). V = bf16(k * bf16(rstd)),
// K = bf16(V * bf16(k_norm)) roped; out [K 16 x 64][V 16 x 64] at OUT_OFF.
void rf_head_k_g(const uint8_t *__restrict ab, uint8_t *__restrict scr, int32_t c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned KW = 64, KG_OFF = 32768, RSTD = 31744;
  const bfloat16 *x = reinterpret_cast<const bfloat16 *>(ab);
  bfloat16 *ok = reinterpret_cast<bfloat16 *>(scr + OUT_OFF), *ov = ok + 16 * KW;
  const bfloat16 *rope = reinterpret_cast<const bfloat16 *>(scr + ROPE_OFF);
  const bfloat16 *g1 = reinterpret_cast<const bfloat16 *>(scr + KG_OFF) + 32 * c, *g2 = g1 + 256;
  const float *rs = reinterpret_cast<const float *>(scr + RSTD);
  const bool re = 32 * c < 64;
  A16 ra;
  ra.from_vector(aie::load_v<16>(rs));
  aie::vector<bfloat16, 16> rb = ra.to_vector<bfloat16>();
  const v32b one = aie::broadcast<bfloat16, 32>(1.0f), zero = aie::zeros<bfloat16, 32>();
  for (unsigned i = 0; i < 16; ++i) {
    v32b sv = aie::broadcast<bfloat16, 32>(rb[i]);
    v32b v1 = bf(aie::mul(aie::load_v<32>(x + i * KW), sv)), v2 = bf(aie::mul(aie::load_v<32>(x + i * KW + 32), sv));
    aie::store_v(ov + i * KW, v1);
    aie::store_v(ov + i * KW + 32, v2);
    v32b x1 = bf(aie::mul(v1, aie::load_v<32>(g1))), x2 = bf(aie::mul(v2, aie::load_v<32>(g2)));
    const bfloat16 *cr = rope + i * 128 + 32 * c;
    v32b cj = re ? aie::load_v<32>(cr) : one, sj = re ? aie::load_v<32>(cr + 64) : zero;
    aie::store_v(ok + i * KW, bf(aie::sub(aie::mul(x1, cj), aie::mul(x2, sj))));
    aie::store_v(ok + i * KW + 32, bf(aie::add(aie::mul(x2, cj), aie::mul(x1, sj))));
  }
}


void rf_head_rope(const uint8_t *__restrict ab, uint8_t *__restrict scr) {
  for (unsigned i = 0; i < 8192; i += 64)
    aie::store_v(scr + ROPE_OFF + i, aie::load_v<64>(ab + i));
}

// Row-major unit: [4 rows][4 heads x 256], in and out.
void rf_head_unit(const uint8_t *__restrict ab, uint8_t *__restrict scr, int32_t u) {
  head_unit<HD, ROW, HD, ROW>(ab, scr, u);
}

// Head-major unit: in, 4 heads of [4 rows x 256] read 1080 elements apart (each an 8640-B unit's
// quarter); out, [4 heads][4 rows x 256].
void rf_head_unit_hm(const uint8_t *__restrict ab, uint8_t *__restrict scr, int32_t u) {
  head_unit<1080, HD, ROWS * HD, HD>(ab, scr, u);
}

}  // extern "C"
