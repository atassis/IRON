// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Resident-forward RMSNorm and residual on one column's 480-wide slice of 16 rows, in a scratch
// region (the idle gate/up W block). Rows arrive as 8-row units of 7680 B at the head of an
// activation buffer. Per-row sums of squares reduce across the 8 columns on the row-0 cascade;
// the last column computes rstd and sends it on its core stream to the columns that scale.
// Every rounding is one the host model reproduces: exact bf16 products, accfloat adds, scalar f32
// ops and a correctly rounded integer square root.
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "rf_intmath.h"

namespace {

using namespace rf;

constexpr unsigned W = 480;              // columns per slice
constexpr unsigned ROWS = 16;
constexpr unsigned UNIT = 8 * W * 2;     // one 8-row bf16 unit
constexpr unsigned X_SLOT = 0, Y_SLOT = 2, OUT = 2 * UNIT;   // OUT reuses the y slots (pre pass)
constexpr unsigned G_OFF = 4 * UNIT;     // gain slice (W bf16), then the layer scalar
constexpr unsigned RSTD_OFF = G_OFF + 1024;
constexpr float EPS = 1e-6f;
static_assert(RSTD_OFF + ROWS * 4 <= 34560, "scratch is the gate/up W block");

using v32b = aie::vector<bfloat16, 32>;
using A32 = aie::accum<accfloat, 32>;
using A16 = aie::accum<accfloat, 16>;
using A64 = aie::accum<accfloat, 64>;

inline void conv_even() { aie::set_rounding(aie::rounding_mode::conv_even); }

inline bfloat16 *rows(uint8_t *scr, unsigned slot) {
  return reinterpret_cast<bfloat16 *>(scr + slot * UNIT);
}
inline float *rstd(uint8_t *scr) { return reinterpret_cast<float *>(scr + RSTD_OFF); }

// Per-row sum of squares of 16 rows of w columns: 32 lanes over the w / 32 column chunks, the two
// lane halves, then the 16 lanes left to right in scalar f32.
__attribute__((noinline)) A16 own_sums(const bfloat16 *x, unsigned w) {
  alignas(64) float lanes[16], s[16];
  for (unsigned i = 0; i < ROWS; ++i) {
    const bfloat16 *r = x + i * w;
    v32b v = aie::load_v<32>(r);
    A32 a = aie::mul(v, v);
    for (unsigned j = 1; j < w / 32; ++j) {
      v32b w = aie::load_v<32>(r + 32 * j);
      a = aie::add(a, aie::mul(w, w));
    }
    A16 h = aie::add(a.extract<16>(0), a.extract<16>(1));
    aie::store_v(lanes, h.to_vector<float>());
    float t = lanes[0];
    for (unsigned k = 1; k < 16; ++k)
      t += lanes[k];
    s[i] = t;
  }
  A16 o;
  o.from_vector(aie::load_v<16>(s));
  return o;
}

inline A16 from_west() { return A16(get_scd_v16accfloat()); }

inline aie::vector<bfloat16, 32> bf(const A32 &a) { return a.to_vector<bfloat16>(); }

}  // namespace

extern "C" {

void rf_gain(const uint8_t *__restrict elem, uint8_t *__restrict scr) {
  for (unsigned i = 0; i < 1024; i += 64)
    aie::store_v(scr + G_OFF + i, aie::load_v<64>(elem + i));
}

void rf_unit(const uint8_t *__restrict ab, uint8_t *__restrict scr, int32_t slot) {
  const bfloat16 *s = reinterpret_cast<const bfloat16 *>(ab);
  bfloat16 *d = rows(scr, slot);
  for (unsigned i = 0; i < UNIT / 2; i += 64)
    aie::store_v(d + i, aie::load_v<64>(s + i));
}

// The _w entries take the rows, their width w (a multiple of 32) and the bits of the f32 row
// length n the chain end divides by; the slot entries are the 480-wide, n = 3840 case.
void rf_ss_first_w(const bfloat16 *x, int32_t w) {
  conv_even();
  put_mcd(own_sums(x, w));
}

void rf_ss_mid_w(const bfloat16 *x, int32_t w) {
  conv_even();
  put_mcd(aie::add(from_west(), own_sums(x, w)));
}

// The chain end: rstd = 1 / sqrt(sum / n + eps) per row, kept and sent to the scaling cores.
__attribute__((noinline)) void rf_ss_last_w(uint8_t *scr, const bfloat16 *x, int32_t w, int32_t nbits) {
  conv_even();
#ifdef RF_FAST_RSQRT
  // nbits is the bits of 1 / n here: the SFU inverse square root, not correctly rounded
  aie::vector<float, 16> ms = aie::mul(aie::add(from_west(), own_sums(x, w)).to_vector<float>(), from_bits(nbits))
                                  .to_vector<float>();
  aie::vector<float, 16> rv = aie::invsqrt(aie::add(ms, aie::broadcast<float, 16>(EPS)));
  V r = aie::vector_cast<int32_t>(rv);
#else
  V tot = aie::vector_cast<int32_t>(aie::add(from_west(), own_sums(x, w)).to_vector<float>());
  // vector downshift rounds by the rounding register; the integer routines need truncation
  aie::set_rounding(aie::rounding_mode::floor);
  V q = div_rn(tot, bc(nbits));
  conv_even();
  A16 m, eps;
  m.from_vector(aie::vector_cast<float>(q));
  eps.from_vector(aie::broadcast<float, 16>(EPS));
  V me = aie::vector_cast<int32_t>(aie::add(m, eps).to_vector<float>());
  aie::set_rounding(aie::rounding_mode::floor);
  V r = div_rn(bc((int32_t)bits(1.0f)), sqrt_rn(me));
  conv_even();
#endif
  aie::store_v(reinterpret_cast<int32_t *>(rstd(scr)), r);
  for (unsigned i = 0; i < ROWS; ++i)
    put_ms((int)r[i]);
}

void rf_ss_first(uint8_t *scr, int32_t slot) { rf_ss_first_w(rows(scr, slot), W); }
void rf_ss_mid(uint8_t *scr, int32_t slot) { rf_ss_mid_w(rows(scr, slot), W); }
#ifdef RF_FAST_RSQRT
void rf_ss_last(uint8_t *scr, int32_t slot) { rf_ss_last_w(scr, rows(scr, slot), W, bits(1.0f / 3840.0f)); }
#else
void rf_ss_last(uint8_t *scr, int32_t slot) { rf_ss_last_w(scr, rows(scr, slot), W, bits(3840.0f)); }
#endif

void rf_rstd_recv(uint8_t *scr) {
  float *r = rstd(scr);
  for (unsigned i = 0; i < ROWS; ++i) {
    r[i] = from_bits((uint32_t)get_ss());
  }
}

// xn = bf16(bf16(x * bf16(rstd)) * g) as the next GEMM's A block: [kb 60][8-row half 2] subtiles.
// Row-major in place first, then each 8 x 8 subtile gathered from eight 16-byte row pieces.
void rf_pre_scale(uint8_t *scr) {
  conv_even();
  bfloat16 *x = rows(scr, X_SLOT);
  const bfloat16 *g = reinterpret_cast<const bfloat16 *>(scr + G_OFF);
  A16 ra;
  ra.from_vector(aie::load_v<16>(rstd(scr)));
  aie::vector<bfloat16, 16> rb = ra.to_vector<bfloat16>();
  for (unsigned i = 0; i < ROWS; ++i) {
    v32b s = aie::broadcast<bfloat16, 32>(rb[i]);
    AIE_LOOP_MIN_ITERATION_COUNT(W / 32)
    for (unsigned j = 0; j < W / 32; ++j) {
      bfloat16 *xp = x + i * W + 32 * j;
      aie::store_v(xp, bf(aie::mul(bf(aie::mul(aie::load_v<32>(xp), s)), aie::load_v<32>(g + 32 * j))));
    }
  }
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> so(reinterpret_cast<bfp16ebs8 *>(scr + OUT));
  for (unsigned kb = 0; kb < W / 8; ++kb)
    for (unsigned h = 0; h < 2; ++h) {
      const bfloat16 *b = x + 8 * h * W + 8 * kb;
      A64 ua;
      ua.from_vector(aie::concat(aie::load_v<8>(b), aie::load_v<8>(b + W), aie::load_v<8>(b + 2 * W),
                                 aie::load_v<8>(b + 3 * W), aie::load_v<8>(b + 4 * W), aie::load_v<8>(b + 5 * W),
                                 aie::load_v<8>(b + 6 * W), aie::load_v<8>(b + 7 * W)));
      so << ua.to_vector<bfp16ebs8>();
    }
}

// x_out = bf16(bf16(x + bf16(bf16(y * bf16(rstd)) * g)) * layer_scalar), in place of x.
void rf_post_scale(uint8_t *scr) {
  conv_even();
  bfloat16 *x = rows(scr, X_SLOT);
  const bfloat16 *y = rows(scr, Y_SLOT);
  const bfloat16 *g = reinterpret_cast<const bfloat16 *>(scr + G_OFF);
  const bfloat16 ls = g[W];
  A16 ra;
  ra.from_vector(aie::load_v<16>(rstd(scr)));
  aie::vector<bfloat16, 16> rb = ra.to_vector<bfloat16>();
  for (unsigned i = 0; i < ROWS; ++i) {
    v32b s = aie::broadcast<bfloat16, 32>(rb[i]);
    for (unsigned j = 0; j < W / 32; ++j) {
      bfloat16 *xp = x + i * W + 32 * j;
      v32b yn = bf(aie::mul(bf(aie::mul(aie::load_v<32>(y + i * W + 32 * j), s)),
                            aie::load_v<32>(g + 32 * j)));
      A32 xa, na;
      xa.from_vector(aie::load_v<32>(xp));
      na.from_vector(yn);
      v32b xo = bf(aie::add(xa, na));
      aie::store_v(xp, bf(aie::mul(xo, aie::broadcast<bfloat16, 32>(ls))));
    }
  }
}

}  // extern "C"
