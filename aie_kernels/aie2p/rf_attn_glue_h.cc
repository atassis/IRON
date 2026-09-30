// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Resident-forward attention glue, 16 query rows per head (fused_attn.cc at FA_ROWS=16): a pass
// holds H heads that share one K/V stream, each head_dim S x 64. A sliding layer runs H=2, S=4 (two
// q heads per kv head); a global layer H=1, S=8. Streams arrive in the layer image's static rings
// (1728-B private units, 8640-B broadcast units); a K or V block arrives as sub-blocks of 4 (K) or
// 2 (V) slices, each padded to whole units. Per-pass layouts, head h:
//   qT   [h][S][16 x 64]           sT  [h][64 x 16] f32      o (x V worker) [h][S/2][16 x 64] f32
//   pc   [h][P 16 x 64 bf16] then [h][corr 16, l 16] f32, then [h] u32 partial-row masks
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void fa_qkT_slice(bfloat16 *k, bfloat16 *qT, float *sT, int32_t slice);
void fa_sm_round_init_lo(float *st, int32_t *widths);
void fa_smT_block_x2_lo(float *sT, bfloat16 *p0, bfloat16 *p1, float *cl0, float *cl1, float *st,
                        int32_t *widths, int32_t mask);
void fa_pv_slice(bfloat16 *p, bfloat16 *v, float *o, int32_t slice);
void fa_o_rescale(float *o, float *cl, int32_t n_slices);
void fa_inv_l(float *cl, float *inv);
void rf_pv_native_slice(bfloat16 *p, bfloat16 *v, float *o, int32_t slice);
}

namespace {

constexpr unsigned T = 8, R = 16, SW = 64, KEYS = 64;
constexpr unsigned UNIT = 1728;               // the private ring's BD
constexpr unsigned SLICE_B = KEYS * SW * 2;   // one [64][64] bf16 slice
constexpr unsigned SLICE_E = SW * R;          // one head's slice of qT or o, elements
constexpr unsigned P_H = R * KEYS * 2;        // one head's P
constexpr unsigned CL_H = 2 * R * 4;          // one head's corr and l
constexpr unsigned ST_H = 48;                 // one head's softmax state (2R + 4 floats), 64-B aligned
constexpr float LOG2E = 1.4453125f;

inline unsigned cl_off(unsigned H, unsigned h) { return H * P_H + h * CL_H; }
inline unsigned mask_off(unsigned H) { return H * (P_H + CL_H); }

// n half rows of one slice from h0 on (a half row: 32 elements, 4 column tiles' worth of one
// row), from src into buf ([8 row tiles][8 col tiles][8][8], mm.cc's operand layout).
__attribute__((noinline)) void retile_half_rows(const bfloat16 *__restrict src, bfloat16 *__restrict buf,
                                                unsigned h0, unsigned n) {
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned i = 0; i < n; ++i) {
    unsigned h = h0 + i, row = h / 2;
    bfloat16 *__restrict d = buf + ((row / T) * T + (h % 2) * 4) * 64 + (row % T) * T;
    aie::vector<bfloat16, 32> v = aie::load_v<32>(src);
    src += 32;
    aie::store_v(d, v.extract<8>(0));
    aie::store_v(d + 64, v.extract<8>(1));
    aie::store_v(d + 128, v.extract<8>(2));
    aie::store_v(d + 192, v.extract<8>(3));
  }
}

// Stream bytes [u*UNIT, u*UNIT + UNIT) of a slice-major [n][64][64] bf16 stream (stream_b bytes;
// the rest of the last unit is padding), re-tiled into buf. done(slice) runs as soon as a slice
// is complete, before the unit's next bytes (the next slice's) overwrite buf.
template <typename F>
inline void retile_unit(const uint8_t *__restrict unit, bfloat16 *__restrict buf, int32_t u,
                        unsigned stream_b, F done) {
  unsigned o = u * UNIT, end = o + UNIT < stream_b ? o + UNIT : stream_b;
  const bfloat16 *src = reinterpret_cast<const bfloat16 *>(unit);
  while (o < end) {
    unsigned s = o / SLICE_B, so = o % SLICE_B;
    unsigned seg = end - o < SLICE_B - so ? end - o : SLICE_B - so;
    retile_half_rows(src, buf, so / 64, seg / 64);
    src += seg / 2;
    o += seg;
    if (so + seg == SLICE_B)
      done(s);
  }
}

// Stream bytes of unit u that fall in [0, len), copied to dst.
inline void copy_unit(const uint8_t *__restrict unit, uint8_t *__restrict dst, int32_t u, unsigned len) {
  unsigned o = u * UNIT, n = o >= len ? 0 : (len - o < UNIT ? len - o : UNIT) / 64;
  AIE_PREPARE_FOR_PIPELINING
  for (unsigned p = 0; p < n; ++p)
    aie::store_v(dst + o + p * 64, aie::load_v<64>(unit + p * 64));
}

}  // namespace

extern "C" {

#ifdef RF_GLUE_QK
// 16 rows x 256 dims, row-major at the head of a broadcast unit, into 4 slices of Q^T from element
// qoff on: fa_qT_slice's scaling and layout, read from rows.
void rf_q_rows(const uint8_t *__restrict ab, bfloat16 *__restrict qT, int32_t qoff) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *q = reinterpret_cast<const bfloat16 *>(ab);
  const aie::vector<bfloat16, 64> l2e = aie::broadcast<bfloat16, 64>((bfloat16)LOG2E);
  constexpr unsigned W = 4 * SW;
  for (unsigned sl = 0; sl < 4; ++sl)
    for (unsigned kt = 0; kt < T; ++kt)
      for (unsigned rt = 0; rt < R / T; ++rt) {
        const bfloat16 *b = q + rt * T * W + sl * SW + kt * T;
        aie::vector<bfloat16, 64> v;
        AIE_LOOP_UNROLL_FULL
        for (unsigned i = 0; i < 8; ++i)
          v.insert(i, aie::load_v<8>(b + i * W));
        aie::vector<bfloat16, 64> s = aie::mul(v, l2e).to_vector<bfloat16>();
        aie::store_v(qT + qoff + sl * SLICE_E + (kt * (R / T) + rt) * 64, aie::transpose(s, T, T));
      }
}

// Unit u of a K sub-block (4 slices from slice s0, 32768 B); QK^T for each of the H heads as each
// slice completes. geo = H | S << 4 | s0 << 8.
void rf_k_unit(const uint8_t *__restrict wb, bfloat16 *__restrict kbuf, bfloat16 *__restrict qT,
               float *__restrict sT, int32_t u, int32_t geo) {
  unsigned H = geo & 15, S = (geo >> 4) & 15, s0 = geo >> 8;
  retile_unit(wb, kbuf, u, 4 * SLICE_B, [&](int s) {
    for (unsigned h = 0; h < H; ++h)
      fa_qkT_slice(kbuf, qT + h * S * SLICE_E, sT + h * KEYS * R, s0 + s);
  });
}

void rf_zero_sT(float *sT, int32_t H) {
  for (int32_t i = 0; i < H * (int32_t)(KEYS * R); i += 16)
    aie::store_v(sT + i, aie::zeros<float, 16>());
}

void rf_s_unit(const uint8_t *__restrict wb, uint8_t *__restrict s, int32_t u, int32_t len) {
  copy_unit(wb, s, u, len);
}

void rf_w_unit(const uint8_t *__restrict wb, uint8_t *__restrict w, int32_t u) {
  copy_unit(wb, w, u, 2 * UNIT);
}

// Softmax state for the H heads of pass record tb (per head: hi[16] then lo[16]).
void rf_sm_init(float *st, uint8_t *w, int32_t tb, int32_t H) {
  for (int32_t h = 0; h < H; ++h)
    fa_sm_round_init_lo(st + h * ST_H, reinterpret_cast<int32_t *>(w + 256 * tb + 128 * h));
}

// One key block for the H heads: P and cl into pc, then per head the rows the block is partially
// visible to (the x V core runs those on the exact per-key path, K055).
void rf_sm_block(float *__restrict s, uint8_t *__restrict pc, float *__restrict st, uint8_t *w,
                 int32_t tb, int32_t H) {
  for (int32_t h = 0; h < H; ++h) {
    bfloat16 *p = reinterpret_cast<bfloat16 *>(pc + h * P_H);
    float *cl = reinterpret_cast<float *>(pc + cl_off(H, h));
    int32_t *wd = reinterpret_cast<int32_t *>(w + 256 * tb + 128 * h);
    float *sth = st + h * ST_H;
    fa_smT_block_x2_lo(s + h * KEYS * R, p, p, cl, cl, sth, wd, 1);
    const int32_t base = (reinterpret_cast<const int32_t *>(sth + 2 * R)[0] - 1) * KEYS;
    uint32_t m = 0;
    for (unsigned r = 0; r < R; ++r) {
      int32_t a = wd[R + r] > base ? wd[R + r] : base, b = wd[r] < base + KEYS ? wd[r] : base + KEYS;
      if (b > a && b - a < (int32_t)KEYS)
        m |= 1u << r;
    }
    reinterpret_cast<uint32_t *>(pc + mask_off(H))[h] = m;
  }
}
#endif

#ifdef RF_GLUE_PV
namespace {
// fa_o_rescale's arithmetic for n slices of one head's o (built with the glue, for code size).
void o_rescale(float *o, const float *cl, unsigned n) {
  for (unsigned rt = 0; rt < R / T; ++rt) {
    bool unit = true;
    for (unsigned r = 0; r < T; ++r)
      unit = unit && __builtin_bit_cast(uint32_t, cl[rt * T + r]) == 0x3f800000u;
    if (unit)
      continue;
    const float *c = cl + rt * T;
    const aie::vector<float, 64> f = aie::concat(
        aie::broadcast<float, 8>(c[0]), aie::broadcast<float, 8>(c[1]), aie::broadcast<float, 8>(c[2]),
        aie::broadcast<float, 8>(c[3]), aie::broadcast<float, 8>(c[4]), aie::broadcast<float, 8>(c[5]),
        aie::broadcast<float, 8>(c[6]), aie::broadcast<float, 8>(c[7]));
    for (unsigned sl = 0; sl < n; ++sl)
      for (unsigned nt = 0; nt < 8; ++nt) {
        float *t = o + sl * SLICE_E + (rt * 8 + nt) * 64;
        aie::store_v(t, aie::mul(aie::load_v<64>(t), f).to_vector<float>());
      }
  }
}

// Rows of mask in one head's P ([2 row tiles][8 col tiles][8][8]) zeroed in dst, the rest copied
// (src == dst zeroes in place).
void p_rows(const bfloat16 *src, bfloat16 *dst, uint32_t mask) {
  for (unsigned r = 0; r < R; ++r) {
    bool z = (mask >> r) & 1;
    for (unsigned ct = 0; ct < 8; ++ct) {
      unsigned o = ((r / T) * 8 + ct) * 64 + (r % T) * T;
      aie::store_v(dst + o, z ? aie::zeros<bfloat16, 8>() : aie::load_v<8>(src + o));
    }
  }
}
}  // namespace

void rf_pc_unit(const uint8_t *__restrict wb, uint8_t *__restrict pc, int32_t u, int32_t len) {
  copy_unit(wb, pc, u, len);
}

// Block start: rescale O by each head's correction; split each head's P into its partial rows
// (into pn, the finished-O buffer, whose lock the caller holds over the block) and the rest.
// geo = H | hb << 4 (hb: this worker's slices of a head).
void rf_pv_begin(float *o, uint8_t *pc, bfloat16 *pn, int32_t geo) {
  unsigned H = geo & 15, hb = (geo >> 4) & 15;
  const uint32_t *mk = reinterpret_cast<const uint32_t *>(pc + mask_off(H));
  for (unsigned h = 0; h < H; ++h) {
    o_rescale(o + h * hb * SLICE_E, reinterpret_cast<const float *>(pc + cl_off(H, h)), hb);
    if (mk[h]) {
      bfloat16 *p = reinterpret_cast<bfloat16 *>(pc + h * P_H);
      p_rows(p, pn + h * R * KEYS, ~mk[h]);
      p_rows(p, p, mk[h]);
    }
  }
}

// Unit u of a V sub-block (2 slices from slice s0 of this worker's share, 16384 B); P . V per
// head as each slice completes. geo = H | hb << 4 | s0 << 8.
void rf_v_unit(const uint8_t *__restrict wb, bfloat16 *__restrict vbuf, uint8_t *__restrict pc,
               float *__restrict o, bfloat16 *__restrict pn, int32_t u, int32_t geo) {
  unsigned H = geo & 15, hb = (geo >> 4) & 15, s0 = geo >> 8;
  const uint32_t *mk = reinterpret_cast<const uint32_t *>(pc + mask_off(H));
  retile_unit(wb, vbuf, u, 2 * SLICE_B, [&](int s) {
    for (unsigned h = 0; h < H; ++h) {
      float *oh = o + h * hb * SLICE_E;
      fa_pv_slice(reinterpret_cast<bfloat16 *>(pc + h * P_H), vbuf, oh, s0 + s);
      if (mk[h])
        rf_pv_native_slice(pn + h * R * KEYS, vbuf, oh, s0 + s);
    }
  });
}

void rf_pv_end(uint8_t *pc, float *inv, int32_t H) {
  for (int32_t h = 0; h < H; ++h)
    fa_inv_l(reinterpret_cast<float *>(pc + cl_off(H, h)), inv + h * R);
}

// The finished O share as the O projection's A block for its K sub-pass: rows are the 16 positions
// of the pass, K is each head's hb x 64 dims in turn, then zeros to 288, bfp16 in the chain's
// layout [k block 36][8-row half 2][8 rows x 9 B] (5184 B). Each 8 x 8 subtile is one tile of o,
// finished as fa_o_finish_slice rounds it, then blocked. geo = H | hb << 4.
void rf_pv_finish_a(const float *__restrict o, const float *__restrict inv, uint8_t *__restrict ablk,
                    int32_t geo) {
  unsigned H = geo & 15, hb = (geo >> 4) & 15;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> so(reinterpret_cast<bfp16ebs8 *>(ablk));
  for (unsigned h = 0; h < H; ++h)
    for (unsigned kb = 0; kb < hb * 8; ++kb)
      for (unsigned rt = 0; rt < 2; ++rt) {
        const float *iv = inv + h * R + rt * T;
        unsigned sl = kb / T, ct = kb % T;
        aie::vector<float, 64> f = aie::concat(
            aie::broadcast<float, 8>(iv[0]), aie::broadcast<float, 8>(iv[1]), aie::broadcast<float, 8>(iv[2]),
            aie::broadcast<float, 8>(iv[3]), aie::broadcast<float, 8>(iv[4]), aie::broadcast<float, 8>(iv[5]),
            aie::broadcast<float, 8>(iv[6]), aie::broadcast<float, 8>(iv[7]));
        aie::vector<bfloat16, 64> b =
            aie::mul(aie::load_v<64>(o + h * hb * SLICE_E + sl * SLICE_E + (rt * 8 + ct) * 64), f).to_vector<bfloat16>();
        aie::accum<accfloat, 64> a;
        a.from_vector(b);
        so << a.to_vector<bfp16ebs8>();
      }
  aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  for (unsigned i = H * hb * 8 * 2; i < 36 * 2; ++i)
    so << z.to_vector<bfp16ebs8>();
}
#endif

}  // extern "C"
