// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Resident-forward attention glue: fused_attn.cc's QK, softmax and x V kernels fed through the
// layer image's static core rings (a 1728-B private ring and an 8640-B broadcast ring) instead of
// their own objectFIFOs. Streams arrive in fixed-size units; these functions assemble or re-tile
// them into the buffers the kernels expect and call the kernels. Compiled in one translation unit
// after fused_attn.cc, once per core kind: RF_GLUE_QK (QK and softmax) or RF_GLUE_PV (x V).
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
void fa_o_finish_slice(float *o, float *inv, bfloat16 *out, int32_t slice);
}

namespace {

constexpr unsigned T = 8, ROWS = 32, NJ = ROWS / T, HD = 256, SW = 64;
constexpr unsigned UNIT = 1728;            // the private ring's BD
constexpr unsigned SLICE_B = 64 * SW * 2;  // one [64][64] bf16 slice
constexpr unsigned P_B = ROWS * 64 * 2;    // P, then cl (2 x ROWS f32) behind it
#ifdef RF_PV_EDGE
constexpr unsigned PREC_B = P_B + 2 * ROWS * 4 + 64;   // then the rows' partial-block mask (K055)
#else
constexpr unsigned PREC_B = P_B + 2 * ROWS * 4;
#endif
constexpr float LOG2E = 1.4453125f;

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

// Stream bytes [u*UNIT, u*UNIT + UNIT) of a slice-major [n][64][64] bf16 stream (stream_b bytes; the
// rest of the last unit is padding), re-tiled into buf. done(slice) runs as soon as a slice is
// complete, before the unit's next bytes (the next slice's) overwrite buf.
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
// 16 rows of one head, row-major [16][256] at the head of a broadcast unit, into rows 16*hh .. of
// Q^T: fa_qT_slice's scaling and layout, read from rows instead of delivered tiles.
void rf_q_rows(const uint8_t *__restrict ab, bfloat16 *__restrict qT, int32_t hh) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *q = reinterpret_cast<const bfloat16 *>(ab);
  const aie::vector<bfloat16, 64> l2e = aie::broadcast<bfloat16, 64>((bfloat16)LOG2E);
  for (unsigned sl = 0; sl < HD / SW; ++sl)
    for (unsigned kt = 0; kt < T; ++kt)
      for (unsigned rt = 0; rt < 2; ++rt) {
        const bfloat16 *b = q + rt * T * HD + sl * SW + kt * T;
        aie::vector<bfloat16, 64> v = aie::concat(
            aie::load_v<8>(b), aie::load_v<8>(b + HD), aie::load_v<8>(b + 2 * HD), aie::load_v<8>(b + 3 * HD),
            aie::load_v<8>(b + 4 * HD), aie::load_v<8>(b + 5 * HD), aie::load_v<8>(b + 6 * HD), aie::load_v<8>(b + 7 * HD));
        aie::vector<bfloat16, 64> s = aie::mul(v, l2e).to_vector<bfloat16>();
        aie::store_v(qT + sl * ROWS * 64 + (kt * NJ + hh * 2 + rt) * 64, aie::transpose(s, T, T));
      }
}

// Unit u of a K block ([4 slices][64 keys][64], 32768 B) into kbuf; QK^T on each slice completed
// (fa_qkT_slice).
void rf_k_unit(const uint8_t *__restrict wb, bfloat16 *__restrict kbuf, bfloat16 *__restrict qT,
               float *__restrict sT, int32_t u) {
  retile_unit(wb, kbuf, u, 4 * SLICE_B, [&](int s) { fa_qkT_slice(kbuf, qT, sT, s); });
}

void rf_s_unit(const uint8_t *__restrict wb, uint8_t *__restrict s, int32_t u) {
  copy_unit(wb, s, u, 64 * ROWS * 4);
}

void rf_w_unit(const uint8_t *__restrict wb, uint8_t *__restrict w, int32_t u) {
  copy_unit(wb, w, u, 2 * UNIT);
}

// Softmax on pass tb's widths (hi[32], lo[32] at w + 256 tb); P and cl into pc, one copy.
void rf_sm_init(float *st, uint8_t *w, int32_t tb) {
  fa_sm_round_init_lo(st, reinterpret_cast<int32_t *>(w + 256 * tb));
}

void rf_sm_block(float *__restrict s, uint8_t *__restrict pc, float *__restrict st, uint8_t *w,
                 int32_t tb) {
  bfloat16 *p = reinterpret_cast<bfloat16 *>(pc);
  float *cl = reinterpret_cast<float *>(pc + P_B);
  fa_smT_block_x2_lo(s, p, p, cl, cl, st, reinterpret_cast<int32_t *>(w + 256 * tb), 1);
#ifdef RF_PV_EDGE
  // bit r: the block is partially visible to row r (the softmax has advanced its block counter)
  const int32_t *hi = reinterpret_cast<const int32_t *>(w + 256 * tb), *lo = hi + ROWS;
  const int32_t base = (reinterpret_cast<const int32_t *>(st + 2 * ROWS)[0] - 1) * 64;
  uint32_t m = 0;
  for (unsigned r = 0; r < ROWS; ++r) {
    int32_t a = lo[r] > base ? lo[r] : base, b = hi[r] < base + 64 ? hi[r] : base + 64;
    if (b > a && b - a < 64)
      m |= 1u << r;
  }
  *reinterpret_cast<uint32_t *>(pc + P_B + 2 * ROWS * 4) = m;
#endif
}

#endif

#ifdef RF_GLUE_PV
// Unit u of one x V worker's V block ([2 slices][64 keys][64], 16384 B); P . V on each slice
// (fa_pv_slice).
#ifdef RF_PV_EDGE
void rf_pv_native_slice(bfloat16 *p, bfloat16 *v, float *o, int32_t slice);
}
namespace {
// Rows of mask in P ([4 row tiles][8 col tiles][8][8]) zeroed in dst, the rest copied (src ==
// dst zeroes in place).
void p_rows(const bfloat16 *src, bfloat16 *dst, uint32_t mask) {
  for (unsigned r = 0; r < ROWS; ++r) {
    bool z = (mask >> r) & 1;
    for (unsigned ct = 0; ct < 8; ++ct) {
      unsigned o = ((r / T) * 8 + ct) * 64 + (r % T) * T;
      aie::store_v(dst + o, z ? aie::zeros<bfloat16, 8>() : aie::load_v<8>(src + o));
    }
  }
}
}  // namespace
extern "C" {
#endif

#ifdef RF_PV_EDGE
// pn: the partial rows' P, in the finished-O buffer (the caller holds its lock over the block).
void rf_v_unit(const uint8_t *__restrict wb, bfloat16 *__restrict vbuf, uint8_t *__restrict pc,
               float *__restrict o, bfloat16 *__restrict pn, int32_t u) {
  // a row's partially visible block runs on the native bf16 path, its other blocks on bfp16 (K055)
  bfloat16 *p = reinterpret_cast<bfloat16 *>(pc);
  uint32_t m = *reinterpret_cast<const uint32_t *>(pc + P_B + 2 * ROWS * 4);
  retile_unit(wb, vbuf, u, 2 * SLICE_B, [&](int s) {
    fa_pv_slice(p, vbuf, o, s);
    if (m)
      rf_pv_native_slice(pn, vbuf, o, s);
  });
}
#else
void rf_v_unit(const uint8_t *__restrict wb, bfloat16 *__restrict vbuf, uint8_t *__restrict pc,
               float *__restrict o, int32_t u) {
  retile_unit(wb, vbuf, u, 2 * SLICE_B,
              [&](int s) { fa_pv_slice(reinterpret_cast<bfloat16 *>(pc), vbuf, o, s); });
}
#endif

void rf_pc_unit(const uint8_t *__restrict wb, uint8_t *__restrict pc, int32_t u) {
  copy_unit(wb, pc, u, PREC_B);
}

#ifdef RF_PV_EDGE
void rf_pv_begin(float *o, uint8_t *pc, bfloat16 *pn) {
  fa_o_rescale(o, reinterpret_cast<float *>(pc + P_B), 2);
  uint32_t m = *reinterpret_cast<const uint32_t *>(pc + P_B + 2 * ROWS * 4);
  if (m) {       // pn: the partial rows only; p: the rest
    bfloat16 *p = reinterpret_cast<bfloat16 *>(pc);
    p_rows(p, pn, ~m);
    p_rows(p, p, m);
  }
}
#else
void rf_pv_begin(float *o, uint8_t *pc) {
  fa_o_rescale(o, reinterpret_cast<float *>(pc + P_B), 2);
}
#endif

void rf_pv_end(uint8_t *pc, float *inv) {
  fa_inv_l(reinterpret_cast<float *>(pc + P_B), inv);
}

// Both slices of the finished O half, [2][32][64] bf16, into ch.
void rf_pv_finish(float *o, float *inv, bfloat16 *ch) {
  fa_o_finish_slice(o, inv, ch, 0);
  fa_o_finish_slice(o, inv, ch + ROWS * SW, 1);
}

// The finished O half as the O projection's A block for its K sub-pass: rows are the 16 positions
// of the pass, K is [head 0: 128][head 1: 128][32 zero], bfp16 in the chain's layout
// [k block 36][8-row half 2][8 rows x 9 B] (5184 B). Each 8 x 8 subtile is one tile of o, finished
// exactly as fa_o_finish_slice rounds it, then blocked.
void rf_pv_finish_a(const float *__restrict o, const float *__restrict inv, uint8_t *__restrict ablk) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> so(reinterpret_cast<bfp16ebs8 *>(ablk));
  for (unsigned hh = 0; hh < 2; ++hh)
    for (unsigned kb = 0; kb < 16; ++kb)
      for (unsigned r2 = 0; r2 < 2; ++r2) {
        unsigned rt = hh * 2 + r2, sl = kb / T, ct = kb % T;
        aie::vector<float, 64> f = aie::concat(
            aie::broadcast<float, 8>(inv[rt * T]), aie::broadcast<float, 8>(inv[rt * T + 1]),
            aie::broadcast<float, 8>(inv[rt * T + 2]), aie::broadcast<float, 8>(inv[rt * T + 3]),
            aie::broadcast<float, 8>(inv[rt * T + 4]), aie::broadcast<float, 8>(inv[rt * T + 5]),
            aie::broadcast<float, 8>(inv[rt * T + 6]), aie::broadcast<float, 8>(inv[rt * T + 7]));
        aie::vector<bfloat16, 64> b =
            aie::mul(aie::load_v<64>(o + sl * ROWS * 64 + (rt * 8 + ct) * 64), f).to_vector<bfloat16>();
        aie::accum<accfloat, 64> a;
        a.from_vector(b);
        so << a.to_vector<bfp16ebs8>();
      }
  aie::accum<accfloat, 64> z = aie::zeros<accfloat, 64>();
  for (unsigned i = 0; i < 8; ++i)
    so << z.to_vector<bfp16ebs8>();
}

#endif

}  // extern "C"
