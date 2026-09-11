// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Weight-quantized sibling of mv.cc: A (the GEMV's MxK matrix operand) streams as group-quantized
// int4 or int8 with a per-row, per-group scale (f32 or bf16, see SCALE_BF16 below), dequantized
// ON-CORE right before the same bf16xbf16 MAC mv.cc already uses. B (the vector) and C (the
// output) are unchanged bf16 -- only the WEIGHT-STREAM byte format is an axis here
// (iron/operators/gemv `weight_dtype`), never the arithmetic: narrow arithmetic buys nothing at
// M=1 decode (nothing to speed up, pack/unpack only adds ops), so this kernel still does a bf16
// MAC. The lever is DDR/L2 bytes for the weight, not FLOPs.
//
// Two families, four exported symbols. The SYMMETRIC pair (matvec_vectorized_int4/int8_bf16)
// dequantizes w = q*s; the AFFINE pair (..._int4a/int8a_bf16, further down) dequantizes
// w = q*s + m and is documented at its own definition. Row layout (one row = one output
// feature), K elements, GROUP_SIZE-wide quant groups, n_groups = DIM_K / GROUP_SIZE:
//   symmetric: [n_groups x scale][ payload ]
//   affine:    [n_groups x bf16 scale][n_groups x bf16 min][ payload ]
// payload is K/2 nibble-packed bytes (int4, low nibble = even column, high nibble = odd column --
// same convention as dequant_int4_group_row.cc) or K int8 bytes (int8, one byte per element).
// Packing the scale into the SAME buffer as the weight (rather than a 3rd input FIFO) is forced by
// the AIE2P tile DMA budget: a core has only 2 input channels, both already spent on A and B (see
// gemm_int8xint4_dequant.cc's identical constraint). A ROW-granularity header (not a
// tile-granularity one) keeps the row stride uniform, so the existing per-column contiguous TAP
// arithmetic in gemv/design.py needs no tiling-aware special case.
//
// Dequant unpacks the nibble/byte, vector-multiplies by the broadcast group scale and narrows to
// bf16 through an explicit accum with conv_even rounding rather than a raw cast (the banked WER
// lesson: default truncation biases toward zero).
//
// PROVENANCE, stated honestly because it was overstated here before: this is NOT the same idiom as
// dequant_int4_group.cc, which sign-extends with the scalar sext4() below, nor as
// gemm_int8xint4_dequant.cc, which feeds a native int4 vector to aie::mmul and carries its own
// "numerical correctness is UNVERIFIED" note. Neither validates `vector_cast<int4>` + `unpack`.
// For the SYMMETRIC path that gap never mattered: it clips to [-7,7], so nibble 0x8 is never
// emitted. The AFFINE path emits it by construction -- m = wmin - lo*s puts each group's own
// minimum at exactly q = -8 -- so the signed unpack is load-bearing here for the first time.
// It reads correct at the source (int4 is registered signed, and vector<T,N>::unpack() forwards
// is_signed() to unpack_sign), and it is UNTESTED ON DEVICE. A device gate must force a group
// containing its own minimum, which every real weight group does.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif
// Per-group scale storage width: 0 (default) = f32, matching quant.py's default
// scale_dtype="f32" and today's row layout byte-for-byte. 1 = bf16 (quant.py's
// scale_dtype="bf16"): the header shrinks from 4 to 2 bytes/group -- e.g. 544 -> 528B at
// K=1024,GROUP_SIZE=128, 2.9% of the row -- and the scalar `(bfloat16)scale[...]` cast below
// becomes a no-op read instead of a narrowing one, since the value is already bf16 in memory. The
// two sides (this macro and quant.py's scale_dtype) must be set together; mismatched, the payload
// offset is wrong and every dequant reads garbage.
#ifndef SCALE_BF16
#define SCALE_BF16 0
#endif

namespace {

// PAYLOAD ALIGNMENT, and it is a CONTRACT WITH THE PACKER, not a local detail. `aie::load_v<N>`
// on AIE2P requires the pointer aligned to the access width -- 32 B for a 256-bit load, 64 B for
// a 512-bit one -- and aie_api documents an unaligned pointer as UNDEFINED BEHAVIOUR, not as a
// slow path. The payload starts `header_bytes` into each row and every later row starts at
// `row*row_stride`, so BOTH have to clear the load width.
//
// int8 at r=64 loads 512 bits and the natural header is n_groups*4 = 32 B at k=1024,g=128: every
// even row is misaligned, and the arm computed garbage on device (ppl 4.25e9, top-1 0.00%) while
// compiling, linking, passing the numpy contract test and running bit-identically 5/5. int4
// escapes only because two nibbles per byte halve the load to 256 bits.
//
// PADDING THE HEADER DOES NOT WORK, and the reason is worth keeping. It breaks
// swiglu_mlp_dp's shared weight tile: gate/up (row width D) and down (row width FF) ride ONE
// ObjectFifo and the design asserts `TSI_GU * WROW_D == TSI_D * WROW_FF`, which holds because
// an unpadded row is affine in K with a proportional header (3*1056 == 3168). A pad is a
// ROUND-UP, so it is not proportional: 3*1088 != 3200. A pad that preserved it would have to
// know the operator's FF/D tile ratio, which the packer must not.
//
// So the width is chosen instead of the layout: `max_legal_vec_size` in
// iron/operators/gemv/quant.py picks the largest VEC_SIZE whose load clears every K the design
// builds. int4 keeps 64 (its load is r/2 = 32 B and a 4-byte-per-group header always clears
// that); int8 drops to 32 at this model's shapes. The static_asserts below are what make an
// illegal combination a COMPILE ERROR rather than a device result -- refuse, do not lie.
//
// The general fix is planar scales: a payload row of exactly K bytes starts at offset 0, is
// aligned for every width, and keeps the shared tile affine in K with no intercept.

#if SCALE_BF16
using scale_t = bfloat16;
#else
using scale_t = float;
#endif

inline int8_t sext4(uint8_t nibble) {
  return (int8_t)(((int8_t)(nibble << 4)) >> 4);
}

// r: vector chunk width (VEC_SIZE); k: full row length (DIM_K); g: quant group width
// (GROUP_SIZE). A vector chunk must never straddle a group boundary, so g must be a multiple of r.
template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int4_dequant(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                         bfloat16 *__restrict c) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t header_bytes = n_groups * sizeof(scale_t);
  constexpr uint32_t row_stride = header_bytes + k / 2;
  static_assert(header_bytes % (r / 2) == 0, "int4 payload start must clear the load width");
  static_assert(row_stride % (r / 2) == 0, "int4 row stride must clear the load width");
  // %4 is the shared bf16-granule arena's alignment (iron/common/sequence.py), not scale_t's own
  // (2-byte for bf16 would only need 2-byte alignment by itself) -- see quant.py's
  // row_stride_bytes for the full derivation, including the odd-n_groups bf16 case this
  // static_assert alone does not distinguish.
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row scale read)");
  constexpr uint32_t chunks_per_group = g / r;

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const scale_t *scale = reinterpret_cast<const scale_t *>(rowp);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + header_bytes);
    // ONE flat loop and ONE reduce per row. Hoisting the scale per GROUP is arithmetically nicer
    // but costs a reduce_add per group (8 per row at k=1024,g=128) against the 16 vector muls it
    // saves, and a 64-lane reduce is a log-depth shuffle chain -- far more than a vector mul.
    // It also nests the loops, and hardware loops are innermost-only (contract K013).
    ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
    uint32_t chunk = 0;
    for (const bfloat16 *__restrict b_cur = b; b_cur < b + k; b_cur += r, chunk++) {
      ::aie::vector<int8, r / 2> raw = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
      ::aie::vector<int8, r> q8 = ::aie::unpack(::aie::vector_cast<int4>(raw));
      ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
      ::aie::vector<bfloat16, r> sv =
          ::aie::broadcast<bfloat16, r>((bfloat16)scale[(chunk * r) / g]);
      acc = ::aie::mac(acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                       ::aie::load_v<r>(b_cur));
    }
    c[row] = static_cast<bfloat16>(::aie::reduce_add(acc.template to_vector<float>()));
  }
  ::aie::set_rounding(saved_rounding);
}

template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int8_dequant(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                         bfloat16 *__restrict c) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t header_bytes = n_groups * sizeof(scale_t);
  constexpr uint32_t row_stride = header_bytes + k;
  static_assert(header_bytes % r == 0, "int8 payload start must clear the load width");
  static_assert(row_stride % r == 0, "int8 row stride must clear the load width");

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const scale_t *scale = reinterpret_cast<const scale_t *>(rowp);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + header_bytes);
    // ONE flat loop and ONE reduce per row, the same shape the int4 path uses and for the same
    // reason: a per-GROUP scale hoist costs a reduce_add per group (8 per row at k=1024,g=128)
    // plus a dependent scalar FMA chain, and a 64-lane reduce is a log-depth shuffle chain. It
    // also nests the loops, and hardware loops are innermost-only (contract K013) -- at
    // g/r = 2 the inner trip count cannot carry one at all.
    //
    // The scale meets each product in bf16 here rather than meeting an f32 partial once per
    // group, so this rounds more. It rounds no more than BF16 WEIGHTS do: q is an exact integer
    // in bf16 (|q| <= 127, and bf16 carries integers to 256), so q*s has a bf16 weight's
    // precision exactly, accumulated in the same f32 accumulator.
    ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
    uint32_t chunk = 0;
    for (const bfloat16 *__restrict b_cur = b; b_cur < b + k; b_cur += r, chunk++) {
      ::aie::vector<int8, r> q8 = ::aie::load_v<r>(packed + chunk * r);
      ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
      ::aie::vector<bfloat16, r> sv =
          ::aie::broadcast<bfloat16, r>((bfloat16)scale[(chunk * r) / g]);
      acc = ::aie::mac(acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                       ::aie::load_v<r>(b_cur));
    }
    c[row] = static_cast<bfloat16>(::aie::reduce_add(acc.template to_vector<float>()));
  }
  ::aie::set_rounding(saved_rounding);
}

// ---------------------------------------------------------------------------------------------
// AFFINE forms: dequant = q*s + m, with a bf16 scale AND a bf16 min per (row, group), q signed.
//
// Row layout, n_groups = DIM_K / GROUP_SIZE:
//   [n_groups x bf16 scale][n_groups x bf16 min][ payload ]
// The header is 4 B/group, so the payload offset is always 4-byte aligned and is 16-byte aligned
// whenever n_groups % 4 == 0 -- which is why this format needs no tile-planar [Q][S][Z] layout.
//
// The min NEVER touches the inner loop. Its contribution to a row's dot product is
//   sum_k m_g(k) * b_k  =  sum_g m_g * (sum_{k in g} b_k)
// so it collapses to one per-group sum of B, computed ONCE for all `m` rows before the row loop,
// and n_groups scalar FMAs per row afterwards. That is the same factorisation FastFlowLM's own
// codec uses (`c_accum = aie::mac(c_accum, a_mins, reduce_b)`, mlir-air-q4nx
// programming_examples/fused_decode/kernels/q4_k.h:229). It is what lets the affine form keep the
// symmetric kernel's ONE flat loop and ONE reduce per row: a per-element `sub` against a zero
// point -- the other way to spell an asymmetric quantizer -- would add a broadcast and a vector
// subtract to every chunk instead, K/r times per row rather than n_groups times.
template <uint32_t r, uint32_t k, uint32_t g>
void group_sums_of_b(const bfloat16 *__restrict b, float *__restrict bsum) {
  constexpr uint32_t chunks_per_group = g / r;
  const ::aie::vector<bfloat16, r> ones = ::aie::broadcast<bfloat16, r>((bfloat16)1.0f);
  for (uint32_t gi = 0; gi < k / g; gi++) {
    ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
    for (uint32_t ci = 0; ci < chunks_per_group; ci++)
      acc = ::aie::mac(acc, ::aie::load_v<r>(b + gi * g + ci * r), ones);
    bsum[gi] = ::aie::reduce_add(acc.template to_vector<float>());
  }
}

template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int4_affine(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                        bfloat16 *__restrict c) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t header_bytes = 4 * n_groups;
  constexpr uint32_t row_stride = header_bytes + k / 2;
  static_assert(header_bytes % (r / 2) == 0, "int4a payload start must clear the load width");
  static_assert(row_stride % (r / 2) == 0, "int4a row stride must clear the load width");

  float bsum[n_groups];
  group_sums_of_b<r, k, g>(b, bsum);

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const bfloat16 *scale = reinterpret_cast<const bfloat16 *>(rowp);
    const bfloat16 *mins = reinterpret_cast<const bfloat16 *>(rowp + 2 * n_groups);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + header_bytes);
    ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
    uint32_t chunk = 0;
    for (const bfloat16 *__restrict b_cur = b; b_cur < b + k; b_cur += r, chunk++) {
      ::aie::vector<int8, r / 2> raw = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
      ::aie::vector<int8, r> q8 = ::aie::unpack(::aie::vector_cast<int4>(raw));
      ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
      ::aie::vector<bfloat16, r> sv = ::aie::broadcast<bfloat16, r>(scale[(chunk * r) / g]);
      acc = ::aie::mac(acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                       ::aie::load_v<r>(b_cur));
    }
    float row_sum = ::aie::reduce_add(acc.template to_vector<float>());
    for (uint32_t gi = 0; gi < n_groups; gi++)
      row_sum += (float)mins[gi] * bsum[gi];
    c[row] = static_cast<bfloat16>(row_sum);
  }
  ::aie::set_rounding(saved_rounding);
}

template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int8_affine(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                        bfloat16 *__restrict c) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t header_bytes = 4 * n_groups;
  constexpr uint32_t row_stride = header_bytes + k;
  static_assert(header_bytes % r == 0, "int8a payload start must clear the load width");
  static_assert(row_stride % r == 0, "int8a row stride must clear the load width");
  constexpr uint32_t chunks_per_group = g / r;

  float bsum[n_groups];
  group_sums_of_b<r, k, g>(b, bsum);

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const bfloat16 *scale = reinterpret_cast<const bfloat16 *>(rowp);
    const bfloat16 *mins = reinterpret_cast<const bfloat16 *>(rowp + 2 * n_groups);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + header_bytes);
    const bfloat16 *__restrict b_cur = b;
    float row_sum = 0.0f;
    for (uint32_t gi = 0; gi < n_groups; gi++) {
      ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
      for (uint32_t ci = 0; ci < chunks_per_group; ci++) {
        ::aie::vector<int8, r> q8 = ::aie::load_v<r>(packed + (gi * chunks_per_group + ci) * r);
        ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
        acc = ::aie::mac(acc, qbf, ::aie::load_v<r>(b_cur));
        b_cur += r;
      }
      row_sum += (float)scale[gi] * ::aie::reduce_add(acc.template to_vector<float>()) +
                 (float)mins[gi] * bsum[gi];
    }
    c[row] = static_cast<bfloat16>(row_sum);
  }
  ::aie::set_rounding(saved_rounding);
}

}  // namespace

extern "C" {

// Naming matches mv.cc's `matvec_{func_type}_{dtype_in}_{dtype_out}` convention
// (iron/operators/gemv/design.py builds the Kernel() name from the same template).
void matvec_vectorized_int4_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                 const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int4_dequant<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}

void matvec_vectorized_int8_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                 const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int8_dequant<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}

void matvec_vectorized_int4a_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                  const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int4_affine<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}

void matvec_vectorized_int8a_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                  const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int8_affine<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}

}  // extern "C"
