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

#include "quant_row_layout.h"

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

// Row layout: see quant_row_layout.h. PLANAR=1 selects the row-group form.

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

// Agrees with quant.py's _SCALE_BYTES only by assertion: sizeof on a vendor type is not
// self-evident here (bfp16ebs8 is 1 byte under Peano, 9 under Chess), and a disagreement would
// shift every payload pointer silently.
static_assert(sizeof(scale_t) == (SCALE_BF16 ? 2 : 4),
              "scale_t width disagrees with quant.py's _SCALE_BYTES");

inline int8_t sext4(uint8_t nibble) {
  return (int8_t)(((int8_t)(nibble << 4)) >> 4);
}

// The r-lane scale vector for a chunk spanning r/g groups. At r <= g that is the single broadcast
// this loop has always used. At r == 2g it is two half-broadcasts concatenated, which is what
// stops a group NARROWER than the vector from capping the vector: at g=32 the chunk was 32 lanes
// wide against a 64-lane machine, so every per-iteration cost -- loop control, the scale multiply,
// the bf16 convert -- was amortised over half the elements it could have been.
template <uint32_t r, uint32_t g>
inline ::aie::vector<bfloat16, r> chunk_scales(const scale_t *scale, uint32_t gi) {
  if constexpr (r <= g) {
    return ::aie::broadcast<bfloat16, r>((bfloat16)scale[gi]);
  } else {
    return ::aie::concat(::aie::broadcast<bfloat16, g>((bfloat16)scale[gi]),
                         ::aie::broadcast<bfloat16, g>((bfloat16)scale[gi + 1]));
  }
}


#ifndef MVQ_UNROLL
#define MVQ_UNROLL 2   // two accumulators; 1 restores the single-accumulator loop
#endif

// r: vector chunk width (VEC_SIZE); k: full row length (DIM_K); g: quant group width
// (GROUP_SIZE). A vector chunk must never straddle a group boundary, so g must be a multiple of r.
template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int4_dequant(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                         bfloat16 *__restrict c) {
  static_assert(g % r == 0 || r == 2 * g,
                "a chunk must sit inside one group, or span exactly two (chunk_scales)");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t header_bytes = n_groups * sizeof(scale_t);
  constexpr uint32_t payload_bytes = k / 2;
  constexpr uint32_t row_stride = header_bytes + payload_bytes;
  static_assert(QUANT_ALIGN_OK(r / 2, header_bytes, payload_bytes, row_stride),
                "int4 payload load is not aligned under this layout");
  // %4 is the shared bf16-granule arena's alignment (iron/common/sequence.py), not scale_t's own
  // (2-byte for bf16 would only need 2-byte alignment by itself) -- see quant.py's
  // row_stride_bytes for the full derivation, including the odd-n_groups bf16 case this
  // static_assert alone does not distinguish.
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row scale read)");
  constexpr uint32_t chunks_per_group = g / r;

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const quant_row_offsets off = quant_row_at<row_stride, header_bytes, payload_bytes>(row);
    const scale_t *scale = reinterpret_cast<const scale_t *>(a_bytes + off.header);
    const int8_t *packed = reinterpret_cast<const int8_t *>(a_bytes + off.payload);
    // ONE flat loop and ONE reduce per row. Hoisting the scale per GROUP is arithmetically nicer
    // but costs a reduce_add per group (8 per row at k=1024,g=128) against the 16 vector muls it
    // saves, and a 64-lane reduce is a log-depth shuffle chain -- far more than a vector mul.
    // It also nests the loops, and hardware loops are innermost-only (contract K013).
    // One accumulator makes chunk c+1's mac wait on chunk c's: the loop is recurrence-bound, not
    // slot-bound, measured at 15 bundles with 1 idle. MVQ_UNROLL=2 runs two independent
    // accumulators over even/odd chunks and sums once at the end -- 0.2344 -> 0.1328
    // bundles/element. Changes summation order (shallower tree, slightly MORE accurate), so it is
    // not bit-identical: gate on token parity, never a logits byte-compare. Needs an even chunk
    // count; odd falls back. See kb/breaking-the-accumulator-recurrence-halves-the-decode-gemv-loop.
    ::aie::accum<accfloat, r> acc;
    if constexpr (MVQ_UNROLL == 2 && (k / r) % 2 == 0) {
      ::aie::accum<accfloat, r> acc0 = ::aie::zeros<accfloat, r>();
      ::aie::accum<accfloat, r> acc1 = ::aie::zeros<accfloat, r>();
      const bfloat16 *__restrict b_cur = b;
      for (uint32_t chunk = 0; chunk < k / r; chunk += 2) {
        ::aie::vector<int8, r / 2> raw0 = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
        ::aie::vector<int8, r> q8_0 = ::aie::unpack(::aie::vector_cast<int4>(raw0));
        ::aie::vector<bfloat16, r> qbf0 = ::aie::to_float<bfloat16>(q8_0, 0);
        ::aie::vector<bfloat16, r> sv0 = chunk_scales<r, g>(scale, (chunk * r) / g);
        acc0 = ::aie::mac(acc0, ::aie::mul(qbf0, sv0).template to_vector<bfloat16>(),
                          ::aie::load_v<r>(b_cur));
        b_cur += r;
        ::aie::vector<int8, r / 2> raw1 = ::aie::load_v<r / 2>(packed + (chunk + 1) * (r / 2));
        ::aie::vector<int8, r> q8_1 = ::aie::unpack(::aie::vector_cast<int4>(raw1));
        ::aie::vector<bfloat16, r> qbf1 = ::aie::to_float<bfloat16>(q8_1, 0);
        ::aie::vector<bfloat16, r> sv1 = chunk_scales<r, g>(scale, ((chunk + 1) * r) / g);
        acc1 = ::aie::mac(acc1, ::aie::mul(qbf1, sv1).template to_vector<bfloat16>(),
                          ::aie::load_v<r>(b_cur));
        b_cur += r;
      }
      acc = ::aie::add(acc0, acc1);
    } else {
      acc = ::aie::zeros<accfloat, r>();
      uint32_t chunk = 0;
      for (const bfloat16 *__restrict b_cur = b; b_cur < b + k; b_cur += r, chunk++) {
        ::aie::vector<int8, r / 2> raw = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
        ::aie::vector<int8, r> q8 = ::aie::unpack(::aie::vector_cast<int4>(raw));
        ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
        ::aie::vector<bfloat16, r> sv = chunk_scales<r, g>(scale, (chunk * r) / g);
        acc = ::aie::mac(acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                         ::aie::load_v<r>(b_cur));
      }
    }
    c[row] = static_cast<bfloat16>(::aie::reduce_add(acc.template to_vector<float>()));
  }
  ::aie::set_rounding(saved_rounding);
}

// RUNTIME-K sibling of matvec_int4_dequant: k is a function argument, not a template constant, so
// ONE compiled object serves every K a caller names at generator time (gemma4-w-device-runtime-
// k-unsplit) -- merge_devices' body-equality check needs identical .o's, and -DDIM_K forces a new
// one per K. ROW_GROUP stays the compile-time macro; this family's K's all derive it as 1 (K022).
// Priced free on this loop by mv-quant-runtime-k-row-group-cost; built here for the first time.
// `if (n_chunks % 2 == 0)` below is an ordinary runtime branch, not `if constexpr` -- k is not
// compile-time -- so both loop bodies are compiled in, unlike the template form above.
template <uint32_t r, uint32_t g>
void matvec_int4_dequant_rtk(uint32_t m, uint32_t k, const int8_t *__restrict a,
                             const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  const uint32_t n_groups = k / g;
  const uint32_t header_bytes = n_groups * sizeof(scale_t);
  const uint32_t payload_bytes = k / 2;
  const uint32_t row_stride = header_bytes + payload_bytes;

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  const uint32_t n_chunks = k / r;
  for (uint32_t row = 0; row < m; row++) {
    const quant_row_offsets off = quant_row_at_rt(row, row_stride, header_bytes, payload_bytes);
    const scale_t *scale = reinterpret_cast<const scale_t *>(a_bytes + off.header);
    const int8_t *packed = reinterpret_cast<const int8_t *>(a_bytes + off.payload);
    ::aie::accum<accfloat, r> acc;
    if (MVQ_UNROLL == 2 && (n_chunks % 2) == 0) {
      ::aie::accum<accfloat, r> acc0 = ::aie::zeros<accfloat, r>();
      ::aie::accum<accfloat, r> acc1 = ::aie::zeros<accfloat, r>();
      const bfloat16 *__restrict b_cur = b;
      for (uint32_t chunk = 0; chunk < n_chunks; chunk += 2) {
        ::aie::vector<int8, r / 2> raw0 = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
        ::aie::vector<int8, r> q8_0 = ::aie::unpack(::aie::vector_cast<int4>(raw0));
        ::aie::vector<bfloat16, r> qbf0 = ::aie::to_float<bfloat16>(q8_0, 0);
        ::aie::vector<bfloat16, r> sv0 = chunk_scales<r, g>(scale, (chunk * r) / g);
        acc0 = ::aie::mac(acc0, ::aie::mul(qbf0, sv0).template to_vector<bfloat16>(),
                          ::aie::load_v<r>(b_cur));
        b_cur += r;
        ::aie::vector<int8, r / 2> raw1 = ::aie::load_v<r / 2>(packed + (chunk + 1) * (r / 2));
        ::aie::vector<int8, r> q8_1 = ::aie::unpack(::aie::vector_cast<int4>(raw1));
        ::aie::vector<bfloat16, r> qbf1 = ::aie::to_float<bfloat16>(q8_1, 0);
        ::aie::vector<bfloat16, r> sv1 = chunk_scales<r, g>(scale, ((chunk + 1) * r) / g);
        acc1 = ::aie::mac(acc1, ::aie::mul(qbf1, sv1).template to_vector<bfloat16>(),
                          ::aie::load_v<r>(b_cur));
        b_cur += r;
      }
      acc = ::aie::add(acc0, acc1);
    } else {
      acc = ::aie::zeros<accfloat, r>();
      const bfloat16 *__restrict b_cur = b;
      for (uint32_t chunk = 0; chunk < n_chunks; chunk++, b_cur += r) {
        ::aie::vector<int8, r / 2> raw = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
        ::aie::vector<int8, r> q8 = ::aie::unpack(::aie::vector_cast<int4>(raw));
        ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
        ::aie::vector<bfloat16, r> sv = chunk_scales<r, g>(scale, (chunk * r) / g);
        acc = ::aie::mac(acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                         ::aie::load_v<r>(b_cur));
      }
    }
    c[row] = static_cast<bfloat16>(::aie::reduce_add(acc.template to_vector<float>()));
  }
  ::aie::set_rounding(saved_rounding);
}

template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int8_dequant(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                         bfloat16 *__restrict c) {
  static_assert(g % r == 0 || r == 2 * g,
                "a chunk must sit inside one group, or span exactly two (chunk_scales)");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t header_bytes = n_groups * sizeof(scale_t);
  constexpr uint32_t payload_bytes = k;
  constexpr uint32_t row_stride = header_bytes + payload_bytes;
  static_assert(QUANT_ALIGN_OK(r, header_bytes, payload_bytes, row_stride),
                "int8 payload load is not aligned under this layout");

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const quant_row_offsets off = quant_row_at<row_stride, header_bytes, payload_bytes>(row);
    const scale_t *scale = reinterpret_cast<const scale_t *>(a_bytes + off.header);
    const int8_t *packed = reinterpret_cast<const int8_t *>(a_bytes + off.payload);
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
      ::aie::vector<bfloat16, r> sv = chunk_scales<r, g>(scale, (chunk * r) / g);
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
  constexpr uint32_t payload_bytes = k / 2;
  constexpr uint32_t row_stride = header_bytes + payload_bytes;
  static_assert(QUANT_ALIGN_OK(r / 2, header_bytes, payload_bytes, row_stride),
                "int4a payload load is not aligned under this layout");

  float bsum[n_groups];
  group_sums_of_b<r, k, g>(b, bsum);

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const quant_row_offsets off = quant_row_at<row_stride, header_bytes, payload_bytes>(row);
    const bfloat16 *scale = reinterpret_cast<const bfloat16 *>(a_bytes + off.header);
    const bfloat16 *mins = reinterpret_cast<const bfloat16 *>(a_bytes + off.header + 2 * n_groups);
    const int8_t *packed = reinterpret_cast<const int8_t *>(a_bytes + off.payload);
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
  constexpr uint32_t payload_bytes = k;
  constexpr uint32_t row_stride = header_bytes + payload_bytes;
  static_assert(QUANT_ALIGN_OK(r, header_bytes, payload_bytes, row_stride),
                "int8a payload load is not aligned under this layout");
  constexpr uint32_t chunks_per_group = g / r;

  float bsum[n_groups];
  group_sums_of_b<r, k, g>(b, bsum);

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const quant_row_offsets off = quant_row_at<row_stride, header_bytes, payload_bytes>(row);
    const bfloat16 *scale = reinterpret_cast<const bfloat16 *>(a_bytes + off.header);
    const bfloat16 *mins = reinterpret_cast<const bfloat16 *>(a_bytes + off.header + 2 * n_groups);
    const int8_t *packed = reinterpret_cast<const int8_t *>(a_bytes + off.payload);
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

// WHICH WRAPPER THIS OBJECT EMITS. Every caller compiles this source once per weight dtype and
// names the object after it, so only one wrapper is ever linked -- but all four used to be
// INSTANTIATED, and a template's static_asserts fire on instantiation whether or not anything
// calls it. That forced one VEC_SIZE to be legal for all four dtypes at once, which would drag
// int4 down to int8's alignment constraint and cost it the 64-wide load it is measured at parity
// with. It also put four kernels' .text in every object, against 16 KB of program memory.
//
// Unset, all four are emitted -- the behaviour any caller that predates this flag expects.
#if !defined(QUANT_EMIT_INT4) && !defined(QUANT_EMIT_INT8) && \
    !defined(QUANT_EMIT_INT4A) && !defined(QUANT_EMIT_INT8A) && !defined(QUANT_EMIT_INT4_RTK)
#define QUANT_EMIT_INT4 1
#define QUANT_EMIT_INT8 1
#define QUANT_EMIT_INT4A 1
#define QUANT_EMIT_INT8A 1
#endif

extern "C" {

// Naming matches mv.cc's `matvec_{func_type}_{dtype_in}_{dtype_out}` convention
// (iron/operators/gemv/design.py builds the Kernel() name from the same template).
#if defined(QUANT_EMIT_INT4) && QUANT_EMIT_INT4
void matvec_vectorized_int4_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                 const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int4_dequant<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}
#endif

#if defined(QUANT_EMIT_INT8) && QUANT_EMIT_INT8
void matvec_vectorized_int8_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                 const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int8_dequant<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}
#endif

#if defined(QUANT_EMIT_INT4A) && QUANT_EMIT_INT4A
void matvec_vectorized_int4a_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                  const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int4_affine<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}
#endif

#if defined(QUANT_EMIT_INT8A) && QUANT_EMIT_INT8A
void matvec_vectorized_int8a_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                  const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int8_affine<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}
#endif

// Wrapper for matvec_int4_dequant_rtk above. Standalone (no DIM_K needed), so it is in the "any
// QUANT_EMIT_* set" gate above, not just its own #if.
#if defined(QUANT_EMIT_INT4_RTK) && QUANT_EMIT_INT4_RTK
void matvec_vectorized_int4_bf16_rtk(uint32_t m, uint32_t row_offset, uint32_t k,
                                     const int8_t *__restrict a_in, const bfloat16 *__restrict b_in,
                                     bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int4_dequant_rtk<VEC_SIZE, GROUP_SIZE>(m, k, a_in, b_in, c_out);
}
#endif

}  // extern "C"
