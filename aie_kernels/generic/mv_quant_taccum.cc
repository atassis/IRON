// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Chunked-K accumulating sibling of mv_quant.cc's symmetric (int4/int8) dequant matvec:
// mv_taccum.cc's zero/accumulate/finish idiom (f32 accumulator persists across calls, one bf16
// rounding at the end) applied to mv_quant.cc's row-major dequant.
//
// LOCAL addressing, not global k_off: each accumulate call's `a` is a SELF-CONTAINED packed block
// for exactly DIM_K (the chunk width), indexed from 0 like mv_quant.cc's own rows. This is what
// lets the weight-tile ObjectFifo for one round hold only one chunk, not the full row -- see
// swiglu_mlp_dp/design.py's module docstring for the loop nesting this is built for.
// `row_offset` places a call's `m` rows in `acc`, same convention as mv_quant.cc's `c_out`.
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif
#ifndef SCALE_BF16
#define SCALE_BF16 0
#endif

namespace {

#if SCALE_BF16
using scale_t = bfloat16;
#else
using scale_t = float;
#endif

// r: VEC_SIZE; k: DIM_K (the chunk width); g: GROUP_SIZE. One flat loop, one reduce per row per
// call (K015) -- same form mv_quant.cc's int4 body already uses, applied here to int8 too since a
// chunk's own group count is small enough that the per-group-hoist form mv_quant.cc's int8 body
// uses buys nothing extra.
template <uint32_t r, uint32_t k, uint32_t g>
void taccum_chunk_int4(uint32_t m, uint32_t row_offset, const int8_t *__restrict a,
                       const bfloat16 *__restrict b, float *__restrict acc) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "chunk width (DIM_K) must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t row_stride = n_groups * sizeof(scale_t) + k / 2;
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row scale read)");

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const scale_t *scale = reinterpret_cast<const scale_t *>(rowp);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + n_groups * sizeof(scale_t));
    ::aie::accum<accfloat, r> chunk_acc = ::aie::zeros<accfloat, r>();
    uint32_t chunk = 0;
    for (const bfloat16 *__restrict b_cur = b; b_cur < b + k; b_cur += r, chunk++) {
      ::aie::vector<int8, r / 2> raw = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
      ::aie::vector<int8, r> q8 = ::aie::unpack(::aie::vector_cast<int4>(raw));
      ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
      ::aie::vector<bfloat16, r> sv = ::aie::broadcast<bfloat16, r>((bfloat16)scale[chunk * r / g]);
      chunk_acc = ::aie::mac(chunk_acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                             ::aie::load_v<r>(b_cur));
    }
    acc[row_offset + row] += ::aie::reduce_add(chunk_acc.template to_vector<float>());
  }
  ::aie::set_rounding(saved_rounding);
}

template <uint32_t r, uint32_t k, uint32_t g>
void taccum_chunk_int8(uint32_t m, uint32_t row_offset, const int8_t *__restrict a,
                       const bfloat16 *__restrict b, float *__restrict acc) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "chunk width (DIM_K) must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t row_stride = n_groups * sizeof(scale_t) + k;
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row scale read)");

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const scale_t *scale = reinterpret_cast<const scale_t *>(rowp);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + n_groups * sizeof(scale_t));
    ::aie::accum<accfloat, r> chunk_acc = ::aie::zeros<accfloat, r>();
    uint32_t chunk = 0;
    for (const bfloat16 *__restrict b_cur = b; b_cur < b + k; b_cur += r, chunk++) {
      ::aie::vector<int8, r> q8 = ::aie::load_v<r>(packed + chunk * r);
      ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
      ::aie::vector<bfloat16, r> sv = ::aie::broadcast<bfloat16, r>((bfloat16)scale[chunk * r / g]);
      chunk_acc = ::aie::mac(chunk_acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                             ::aie::load_v<r>(b_cur));
    }
    acc[row_offset + row] += ::aie::reduce_add(chunk_acc.template to_vector<float>());
  }
  ::aie::set_rounding(saved_rounding);
}

}  // namespace

// Emit only the requested dtype's wrapper, mirroring mv_quant.cc: all templates instantiate
// otherwise, and one VEC_SIZE would have to be legal for both int4 and int8's alignment at once
// (a `load_v<r/2>` at r=16 fails to even compile for int4's 8-wide native vector on this target).
#if !defined(QUANT_EMIT_INT4) && !defined(QUANT_EMIT_INT8)
#define QUANT_EMIT_INT4 1
#define QUANT_EMIT_INT8 1
#endif

extern "C" {

void mv_taccum_zero_f32(uint32_t m, float *__restrict acc) {
  for (uint32_t row = 0; row < m; row++) acc[row] = 0.0f;
}

#if defined(QUANT_EMIT_INT4) && QUANT_EMIT_INT4
void matvec_taccum_int4_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                             const bfloat16 *__restrict b_in, float *__restrict acc) {
  taccum_chunk_int4<VEC_SIZE, DIM_K, GROUP_SIZE>(m, row_offset, a_in, b_in, acc);
}
#endif

#if defined(QUANT_EMIT_INT8) && QUANT_EMIT_INT8
void matvec_taccum_int8_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                             const bfloat16 *__restrict b_in, float *__restrict acc) {
  taccum_chunk_int8<VEC_SIZE, DIM_K, GROUP_SIZE>(m, row_offset, a_in, b_in, acc);
}
#endif

// c_out: this core's plain D_PER_CORE-wide output buffer (row 0 of `acc`/`c_out` is this core's
// own first output row -- unlike mv_quant.cc's `c_out`, nothing here offsets into a larger shared
// buffer, since the caller already owns its own acc/c_out pair start to finish).
void mv_taccum_finish_bf16(uint32_t m, const float *__restrict acc, bfloat16 *__restrict c_out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (uint32_t row = 0; row < m; row++)
    c_out[row] = static_cast<bfloat16>(acc[row]);
}

}  // extern "C"
