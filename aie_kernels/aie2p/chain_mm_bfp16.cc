// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Chain GEMM position: a [KC x NB] weight block times RB 8-row blocks on the native bfp16 mmul;
// accumulators enter from the west (or zero), leave east; the chain end runs an epilogue.
// Layouts, all bfp16 in 72-byte 8x8 subtiles (block = 8 consecutive K, or N for outputs):
//   A [KC/8][RB]   W [NB/16][KC/8][2]   out [NB/8][RB] (the next GEMM's A)
//   int4 sub-block: [SUB_K/32][NB] bf16 scales, [NB/8][SUB_K/8] 32-byte subtiles,
//                   nibble i = (n = i/8, k = i%8), low nibble first
// A subtile load streams through one of two load-fill buffers: one stream each for A and W.
// Every entry sets conv_even: the default rounding is floor (K001).
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef KC
#define KC 480
#endif
#ifndef NB
#define NB 64
#endif
#ifndef SUB_K
#define SUB_K 96
#endif

using bfp = aie::block_vector<bfp16ebs8, 64>;
using Acc = aie::accum<accfloat, 64>;
// aie::mmul's front end admits only aie::vector operands; this is its bfp16 x bfp16 body.
using MM = aie::detail::mmul<8, 8, 8, bfp16ebs8, bfp16ebs8, 32>;

constexpr unsigned BLK = bfp::memory_bytes();
constexpr unsigned KB = KC / 8;
constexpr unsigned NS = NB / 8;
constexpr unsigned GROUP = 32;
constexpr unsigned SUB_SCALE_BYTES = SUB_K / GROUP * NB * 2;
constexpr unsigned SUB_BYTES = SUB_SCALE_BYTES + SUB_K * NB / 2;

static_assert(BLK == 72, "a bfp16ebs8 8x8 subtile is 64 mantissas + 8 exponents");
static_assert(sizeof(bfp16ebs8) != BLK, "size bfp16 buffers by memory_bytes(), never sizeof");
static_assert(KC % 8 == 0 && NB % 16 == 0, "KC is whole subtiles, NB whole subtile pairs");
static_assert(KC % SUB_K == 0 && SUB_K % GROUP == 0, "sub-blocks tile KC and hold whole groups");
static_assert(KB * NS * BLK + 2 * SUB_BYTES + 2 * 2 * KB * BLK <= 65536, "L1");

namespace {

using Stream = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;
using OStream = aie::block_vector_output_buffer_stream<bfp16ebs8, 64>;

inline const bfp16ebs8 *bp(const uint8_t *p, unsigned subtile) {
  return reinterpret_cast<const bfp16ebs8 *>(p + subtile * BLK);
}

inline Acc get_acc() {
  using A16 = aie::accum<accfloat, 16>;
  A16 a0(get_scd_v16accfloat()), a1(get_scd_v16accfloat());
  A16 a2(get_scd_v16accfloat()), a3(get_scd_v16accfloat());
  return aie::concat(a0, a1, a2, a3);
}

inline void put_acc(const Acc &a) {
  put_mcd(a.extract<16>(0));
  put_mcd(a.extract<16>(1));
  put_mcd(a.extract<16>(2));
  put_mcd(a.extract<16>(3));
}

inline Acc from_f32(const float *p) {
  Acc a;
  a.from_vector(aie::load_v<64>(p));
  return a;
}

enum Seed { ZERO, CASCADE, MEM };

// Accumulators of one pass: c[r][j] is row block r x n-subtile 2p+j, output subtile (2p+j)*RB+r.
template <unsigned RB> struct Pass {
  Acc c[RB][2];
};

template <unsigned RB> inline unsigned sub_of(unsigned p, unsigned r, unsigned j) {
  return (2 * p + j) * RB + r;
}

// One pass per n-subtile pair. The II hint is what pipelines the 16-row loop: 5 bundles per 4
// VMACs, 8 without it. Seeds are read in the order the west neighbour put them.
template <unsigned RB, Seed S, typename Sink, unsigned NBW = NB>
inline void chain_core(const uint8_t *__restrict a, const uint8_t *__restrict w,
                       const float *__restrict seed, Sink sink, unsigned kb = KB) {
  for (unsigned p = 0; p < NBW / 16; ++p) {
    MM m[RB][2];
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r)
      AIE_LOOP_UNROLL_FULL
      for (unsigned j = 0; j < 2; ++j) {
        if constexpr (S == CASCADE)
          m[r][j] = MM(get_acc());
        else if constexpr (S == MEM)
          m[r][j] = MM(from_f32(seed + sub_of<RB>(p, r, j) * 64));
      }
    Stream sa(bp(a, 0));
    Stream sw(bp(w, p * kb * 2));
    AIE_TRY_INITIATION_INTERVAL(RB == 2 ? 5 : 4)
    AIE_LOOP_MIN_ITERATION_COUNT(KB)
    for (unsigned k = 0; k < kb; ++k) {
      bfp x0 = sa.pop();
      bfp x1;
      if constexpr (RB == 2)
        x1 = sa.pop();
      bfp b0 = sw.pop(), b1 = sw.pop();
      m[0][0].mac(x0, true, b0, true);
      m[0][1].mac(x0, true, b1, true);
      if constexpr (RB == 2) {
        m[1][0].mac(x1, true, b0, true);
        m[1][1].mac(x1, true, b1, true);
      }
    }
    Pass<RB> out;
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r)
      AIE_LOOP_UNROLL_FULL
      for (unsigned j = 0; j < 2; ++j)
        out.c[r][j] = m[r][j].to_accum();
    sink(p, out);
  }
}

struct ToCascade {
  template <unsigned RB> void operator()(unsigned, const Pass<RB> &o) const {
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r)
      AIE_LOOP_UNROLL_FULL
      for (unsigned j = 0; j < 2; ++j)
        put_acc(o.c[r][j]);
  }
};

struct ToF32 {
  float *out;
  template <unsigned RB> void operator()(unsigned p, const Pass<RB> &o) const {
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r)
      AIE_LOOP_UNROLL_FULL
      for (unsigned j = 0; j < 2; ++j)
        aie::store_v(out + sub_of<RB>(p, r, j) * 64, o.c[r][j].template to_vector<float>());
  }
};

// The down projection's K sub-passes accumulate into the chain end's resident f32 tile.
struct AddF32 {
  float *out;
  template <unsigned RB> void operator()(unsigned p, const Pass<RB> &o) const {
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r)
      AIE_LOOP_UNROLL_FULL
      for (unsigned j = 0; j < 2; ++j) {
        float *d = out + sub_of<RB>(p, r, j) * 64;
        aie::store_v(d, aie::add(o.c[r][j], aie::load_v<64>(d)).template to_vector<float>());
      }
  }
};

struct ToBf16 {
  bfloat16 *out;
  template <unsigned RB> void operator()(unsigned p, const Pass<RB> &o) const {
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r)
      AIE_LOOP_UNROLL_FULL
      for (unsigned j = 0; j < 2; ++j)
        aie::store_v(out + sub_of<RB>(p, r, j) * 64, o.c[r][j].template to_vector<bfloat16>());
  }
};

// bf16 rows [8 * RB][NB] of one row block (the MemTile keeps q/k/v row-major per head).
struct ToBf16Rows {
  bfloat16 *out;
  template <unsigned RB> void operator()(unsigned p, const Pass<RB> &o) const {
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r)
      AIE_LOOP_UNROLL_FULL
      for (unsigned j = 0; j < 2; ++j) {
        aie::vector<bfloat16, 64> v = o.c[r][j].template to_vector<bfloat16>();
        bfloat16 *d = out + (r * 8) * NB + (2 * p + j) * 8;
        AIE_LOOP_UNROLL_FULL
        for (unsigned row = 0; row < 8; ++row)
          aie::store_v(d + row * NB, v.extract<8>(row));
      }
  }
};

struct ToBfp16 {
  uint8_t *out;
  template <unsigned RB> void operator()(unsigned p, const Pass<RB> &o) const {
    AIE_LOOP_UNROLL_FULL
    for (unsigned j = 0; j < 2; ++j) {
      OStream so(reinterpret_cast<bfp16ebs8 *>(out + sub_of<RB>(p, 0, j) * BLK));
      AIE_LOOP_UNROLL_FULL
      for (unsigned r = 0; r < RB; ++r)
        so << o.c[r][j].template to_vector<bfp16ebs8>();
    }
  }
};

// GELU-tanh = x * s(|x|) with s = sigmoid(2 sqrt(2/pi)(a + 0.044715 a^3)), s(-a) = 1 - s(a):
// a degree-8 polynomial in min(a, 4) - 2 (least squares; s(4) = 1 - 3e-5) in bf16 Horner on
// native bf16 MACs. Max error 2 bf16 ulps of GELU, the same as today's bf16 GELU output rounding.
using v32b = aie::vector<bfloat16, 32>;
using a32 = aie::accum<accfloat, 32>;

constexpr float GELU_C[9] = {9.773047566e-01f, 5.426859856e-02f, -5.423979461e-02f,
                             2.708265744e-02f, -4.168297164e-03f, -2.585965674e-03f,
                             1.167817041e-03f, 5.876845535e-05f, -7.190783072e-05f};

inline v32b bf(const a32 &a) { return a.to_vector<bfloat16>(); }
inline a32 acc_of(float c) {
  a32 a;
  a.from_vector(aie::broadcast<float, 32>(c));
  return a;
}

inline v32b gelu32(v32b x) {
  v32b a = aie::min(aie::abs(x), aie::broadcast<bfloat16, 32>(4.0f));
  a32 aa;
  aa.from_vector(a);
  v32b y = bf(aie::sub(aa, a32(aie::broadcast<float, 32>(2.0f))));
  v32b p = aie::broadcast<bfloat16, 32>(GELU_C[8]);
  AIE_LOOP_UNROLL_FULL
  for (int i = 7; i >= 0; --i)
    p = bf(aie::mac(acc_of(GELU_C[i]), y, p));
  a32 pa;
  pa.from_vector(p);
  v32b pn = bf(aie::sub(a32(aie::broadcast<float, 32>(1.0f)), pa));
  v32b s = aie::select(p, pn, aie::lt(x, aie::zeros<bfloat16, 32>()));
  return aie::mul(x, s).to_vector<bfloat16>();
}

// Gate and up interleaved per 8 columns: n-subtile pair (gate, up) of pass p is h subtile p. The
// sink parks them as bf16; gelu_up runs once per call, four vectors per iteration for ILP.
struct GeluUp {
  bfloat16 *scratch;
  template <unsigned RB> void operator()(unsigned p, const Pass<RB> &o) const {
    AIE_LOOP_UNROLL_FULL
    for (unsigned r = 0; r < RB; ++r) {
      bfloat16 *d = scratch + (p * RB + r) * 128;
      aie::store_v(d, o.c[r][0].template to_vector<bfloat16>());
      aie::store_v(d + 64, o.c[r][1].template to_vector<bfloat16>());
    }
  }
};

template <unsigned RB>
__attribute__((noinline)) void gelu_up(const bfloat16 *__restrict scratch, uint8_t *__restrict h) {
  constexpr unsigned N = NB / 16 * RB;
  static_assert(N % 2 == 0, "two h subtiles per iteration");
  OStream so(reinterpret_cast<bfp16ebs8 *>(h));
  AIE_LOOP_MIN_ITERATION_COUNT(N / 2)
  for (unsigned i = 0; i < N; i += 2) {
    const bfloat16 *s = scratch + i * 128;
    aie::vector<bfloat16, 64> g0 = aie::load_v<64>(s), u0 = aie::load_v<64>(s + 64);
    aie::vector<bfloat16, 64> g1 = aie::load_v<64>(s + 128), u1 = aie::load_v<64>(s + 192);
    v32b r0 = gelu32(g0.extract<32>(0)), r1 = gelu32(g0.extract<32>(1));
    v32b r2 = gelu32(g1.extract<32>(0)), r3 = gelu32(g1.extract<32>(1));
    so << aie::mul(aie::concat(r0, r1), u0).to_vector<bfp16ebs8>();
    so << aie::mul(aie::concat(r2, r3), u1).to_vector<bfp16ebs8>();
  }
}

inline aie::vector<bfloat16, 64> scale_vec(const bfloat16 *__restrict sc) {
  aie::vector<bfloat16, 64> sv;
  AIE_LOOP_UNROLL_FULL
  for (unsigned n = 0; n < 8; ++n)
    sv.insert(n, aie::broadcast<bfloat16, 8>(sc[n]));
  return sv;
}

// One sub-block of SUB_K K-rows for SNB columns, into the W block at K offset sub*SUB_K and
// subtile pair offset pair0. A subtile pair's destinations are adjacent, so each (pair, group) is
// one output stream.
template <unsigned SNB>
inline void convert_sub(const uint8_t *__restrict src, uint8_t *__restrict wblk, unsigned sub,
                        unsigned pair0, unsigned kb = KB) {
  const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(src);
  const uint8_t *q = src + SUB_K / GROUP * SNB * 2;
  constexpr unsigned SKB = SUB_K / 8, GKB = GROUP / 8;
  for (unsigned pr = 0; pr < SNB / 16; ++pr) {
    for (unsigned gi = 0; gi < SUB_K / GROUP; ++gi) {
      aie::vector<bfloat16, 64> s0 = scale_vec(scales + gi * SNB + pr * 16);
      aie::vector<bfloat16, 64> s1 = scale_vec(scales + gi * SNB + pr * 16 + 8);
      const uint8_t *q0 = q + ((2 * pr) * SKB + gi * GKB) * 32;
      const uint8_t *q1 = q0 + SKB * 32;
      OStream so(reinterpret_cast<bfp16ebs8 *>(
          wblk + (((pair0 + pr) * kb + sub * SKB + gi * GKB) * 2) * BLK));
      AIE_TRY_INITIATION_INTERVAL(4)
      AIE_LOOP_MIN_ITERATION_COUNT(GKB)
      for (unsigned kb = 0; kb < GKB; ++kb) {
        aie::vector<int8, 64> i0 = aie::unpack(aie::vector_cast<int4>(aie::load_v<32>(q0)));
        aie::vector<int8, 64> i1 = aie::unpack(aie::vector_cast<int4>(aie::load_v<32>(q1)));
        q0 += 32;
        q1 += 32;
        so << aie::mul(aie::to_float<bfloat16>(i0, 0), s0).to_vector<bfp16ebs8>();
        so << aie::mul(aie::to_float<bfloat16>(i1, 0), s1).to_vector<bfp16ebs8>();
      }
    }
  }
}

inline void conv_even() { aie::set_rounding(aie::rounding_mode::conv_even); }

// Down projection, Nb = 32: the W block fills the first DOWN_W bytes of the gate/up block, and the
// chain end keeps its [nt][16 x 32] f32 accumulator in the rest.
constexpr unsigned DOWN_NB = 32;
constexpr unsigned DOWN_W = KB * DOWN_NB / 8 * BLK;
constexpr unsigned DOWN_ACC = DOWN_NB * 16;
static_assert(DOWN_W % 64 == 0, "the accumulator is vector aligned");

inline float *down_acc(uint8_t *wblk, int32_t tb, unsigned kb = KB) {
  return reinterpret_cast<float *>(wblk + kb * DOWN_NB / 8 * BLK) + tb * DOWN_ACC;
}

}  // namespace

#define CHAIN_ENTRIES(RB)                                                                          \
  void chain_mm_first_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w) {           \
    conv_even();                                                                                   \
    chain_core<RB, ZERO>(a, w, nullptr, ToCascade{});                                              \
  }                                                                                                \
  void chain_mm_mid_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w) {             \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, ToCascade{});                                           \
  }                                                                                                \
  void chain_mm_last_f32_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,          \
                               float *__restrict out) {                                            \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, ToF32{out});                                            \
  }                                                                                                \
  void chain_mm_last_bf16_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,         \
                                bfloat16 *__restrict out) {                                        \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, ToBf16{out});                                           \
  }                                                                                                \
  void chain_mm_last_bfp16_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,        \
                                 uint8_t *__restrict out) {                                        \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, ToBfp16{out});                                          \
  }                                                                                                \
  void chain_mm_last_geluup_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,       \
                                  uint8_t *__restrict h, bfloat16 *__restrict scratch) {           \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, GeluUp{scratch});                                       \
    gelu_up<RB>(scratch, h);                                                                       \
  }                                                                                                \
  /* gate/up parked for an epilogue core that runs chain_gelu_up. */                               \
  void chain_mm_last_park_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,         \
                                bfloat16 *__restrict scratch) {                                    \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, GeluUp{scratch});                                       \
  }                                                                                                \
  void chain_gelu_up_r##RB(const bfloat16 *__restrict scratch, uint8_t *__restrict h) {           \
    conv_even();                                                                                   \
    gelu_up<RB>(scratch, h);                                                                       \
  }                                                                                                \
  void chain_mm_last_bf16rows_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,     \
                                    bfloat16 *__restrict out) {                                    \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, ToBf16Rows{out});                                       \
  }                                                                                                \
  void chain_mm_last_acc_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,          \
                               float *__restrict acc) {                                            \
    conv_even();                                                                                   \
    chain_core<RB, CASCADE>(a, w, nullptr, AddF32{acc});                                           \
  }                                                                                                \
  /* One core in place of a chain position: seed from io (zero when first), result to io. */      \
  void chain_mm_solo_r##RB(const uint8_t *__restrict a, const uint8_t *__restrict w,              \
                           float *__restrict io, int32_t first) {                                  \
    conv_even();                                                                                   \
    if (first)                                                                                     \
      chain_core<RB, ZERO>(a, w, nullptr, ToF32{io});                                              \
    else                                                                                           \
      chain_core<RB, MEM>(a, w, io, ToF32{io});                                                    \
  }

extern "C" {

void chain_convert_w(const uint8_t *__restrict sub_block, uint8_t *__restrict wblk, int32_t sub) {
  conv_even();
  convert_sub<NB>(sub_block, wblk, sub, 0);
}

// A half-width (32-column) sub-block into subtile pairs pair0, pair0 + 1 of the W block.
void chain_convert_half(const uint8_t *__restrict sub_block, uint8_t *__restrict wblk, int32_t sub,
                        int32_t pair0) {
  conv_even();
  convert_sub<32>(sub_block, wblk, sub, pair0);
}

CHAIN_ENTRIES(1)
CHAIN_ENTRIES(2)

void chain_down_first_r2(const uint8_t *__restrict a, const uint8_t *__restrict wblk) {
  conv_even();
  chain_core<2, ZERO, ToCascade, DOWN_NB>(a, wblk, nullptr, ToCascade{});
}

void chain_down_mid_r2(const uint8_t *__restrict a, const uint8_t *__restrict wblk) {
  conv_even();
  chain_core<2, CASCADE, ToCascade, DOWN_NB>(a, wblk, nullptr, ToCascade{});
}

// K sub-pass s of row block tb: s = 0 stores the chain result, later sub-passes add to it.
void chain_down_last_r2(const uint8_t *__restrict a, uint8_t *wblk, int32_t tb, int32_t s) {
  conv_even();
  float *acc = down_acc(wblk, tb);
  if (s == 0)
    chain_core<2, CASCADE, ToF32, DOWN_NB>(a, wblk, nullptr, ToF32{acc});
  else
    chain_core<2, CASCADE, AddF32, DOWN_NB>(a, wblk, nullptr, AddF32{acc});
}

// Row block tb of the accumulator as bf16 rows [16][32] into buffer par of out (the MemTile keeps
// y_d row-major, beside the residual).
void chain_down_emit_rows(uint8_t *wblk, bfloat16 *out, int32_t tb, int32_t par) {
  conv_even();
  const float *acc = down_acc(wblk, tb);
  bfloat16 *o = out + par * DOWN_ACC;
  for (unsigned r = 0; r < 2; ++r)
    for (unsigned row = 0; row < 8; ++row) {
      aie::vector<float, 32> v;
      for (unsigned cs = 0; cs < 4; ++cs)
        v.insert(cs, aie::load_v<8>(acc + (cs * 2 + r) * 64 + row * 8));
      aie::accum<accfloat, 32> a;
      a.from_vector(v);
      aie::store_v(o + (r * 8 + row) * 32, a.to_vector<bfloat16>());
    }
}

// The accumulator to bf16 in place (element i moves from byte 4i to 2i), [nt][1024] B for the DMA.
void chain_down_emit(uint8_t *wblk, int32_t nt) {
  conv_even();
  float *acc = down_acc(wblk, 0);
  bfloat16 *out = reinterpret_cast<bfloat16 *>(acc);
  for (int32_t i = 0; i < nt * (int32_t)DOWN_ACC; i += 64) {
    Acc v;
    v.from_vector(aie::load_v<64>(acc + i));
    aie::store_v(out + i, v.to_vector<bfloat16>());
  }
}

// The down path with K per chain position (kb 8-row K blocks, even, <= KB) at run time, so one
// build serves every projection that shares the core.
void chain_convert_half_k(const uint8_t *__restrict sub_block, uint8_t *__restrict wblk,
                          int32_t sub, int32_t pair0, int32_t kb) {
  conv_even();
  convert_sub<32>(sub_block, wblk, sub, pair0, kb);
}

void chain_down_first_k(const uint8_t *__restrict a, const uint8_t *__restrict wblk, int32_t kb) {
  conv_even();
  chain_core<2, ZERO, ToCascade, DOWN_NB>(a, wblk, nullptr, ToCascade{}, kb);
}

void chain_down_mid_k(const uint8_t *__restrict a, const uint8_t *__restrict wblk, int32_t kb) {
  conv_even();
  chain_core<2, CASCADE, ToCascade, DOWN_NB>(a, wblk, nullptr, ToCascade{}, kb);
}

// One chain instantiation: sub-pass 0 adds to a zeroed accumulator (program memory).
void chain_down_last_k(const uint8_t *__restrict a, uint8_t *wblk, int32_t tb, int32_t s,
                       int32_t kb) {
  conv_even();
  float *acc = down_acc(wblk, tb, kb);
  if (s == 0)
    for (unsigned i = 0; i < DOWN_ACC; i += 16)
      aie::store_v(acc + i, aie::zeros<float, 16>());
  chain_core<2, CASCADE, AddF32, DOWN_NB>(a, wblk, nullptr, AddF32{acc}, kb);
}

void chain_down_emit_rows_k(uint8_t *wblk, bfloat16 *out, int32_t tb, int32_t par, int32_t kb) {
  conv_even();
  const float *acc = down_acc(wblk, tb, kb);
  bfloat16 *o = out + par * DOWN_ACC;
  for (unsigned r = 0; r < 2; ++r)
    for (unsigned row = 0; row < 8; ++row) {
      aie::vector<float, 32> v;
      for (unsigned cs = 0; cs < 4; ++cs)
        v.insert(cs, aie::load_v<8>(acc + (cs * 2 + r) * 64 + row * 8));
      aie::accum<accfloat, 32> a;
      a.from_vector(v);
      aie::store_v(o + (r * 8 + row) * 32, a.to_vector<bfloat16>());
    }
}

}  // extern "C"
