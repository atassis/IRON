// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Quantized sibling of mv_taccum.cc: same broadcast-accumulate-down-rows loop, fed a
// mv_quant.cc-packed row. A row here is one CACHED POSITION, DIM_N wide -- not an output
// feature, which is what mv_quant.cc's row is -- so GROUP_SIZE divides DIM_N and the scale is
// per-token, which is what a V cache wants. See iron/operators/tmatvec/design.py for the K/V
// split. One dequant per row serves every `groups` member.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "quant_row_layout.h"

#ifndef DIM_N
#define DIM_N 16
#endif
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
static_assert(sizeof(scale_t) == (SCALE_BF16 ? 2 : 4),
              "scale_t width disagrees with quant.py's _SCALE_BYTES");

// Dequantize one DIM_N-wide row into `out`, chunk by chunk -- mv_quant.cc's per-chunk dequant,
// minus the dot-product reduce (this contraction doesn't reduce along the row at all).
template <uint32_t r, uint32_t n, uint32_t g>
static inline void dequant_row_int8(const int8_t *__restrict packed,
                                    const scale_t *__restrict scale,
                                    bfloat16 *__restrict out) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(n % g == 0, "DIM_N must be a whole number of groups");
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (uint32_t c = 0, chunks = n / r; c < chunks; c++) {
    ::aie::vector<int8, r> q8 = ::aie::load_v<r>(packed + c * r);
    ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
    ::aie::vector<bfloat16, r> sv = ::aie::broadcast<bfloat16, r>((bfloat16)scale[(c * r) / g]);
    ::aie::store_v(out + c * r, ::aie::mul(qbf, sv).template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
}

template <uint32_t r, uint32_t n, uint32_t g>
static inline void dequant_row_int4(const int8_t *__restrict packed,
                                    const scale_t *__restrict scale,
                                    bfloat16 *__restrict out) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(n % g == 0, "DIM_N must be a whole number of groups");
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (uint32_t c = 0, chunks = n / r; c < chunks; c++) {
    ::aie::vector<int8, r / 2> raw = ::aie::load_v<r / 2>(packed + c * (r / 2));
    ::aie::vector<int8, r> q8 = ::aie::unpack(::aie::vector_cast<int4>(raw));
    ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
    ::aie::vector<bfloat16, r> sv = ::aie::broadcast<bfloat16, r>((bfloat16)scale[(c * r) / g]);
    ::aie::store_v(out + c * r, ::aie::mul(qbf, sv).template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
}

// mv_taccum.cc's taccum_rows<N> body, against an already-dequantized row instead of a loaded one.
template <uint32_t n>
static inline void bcast_mac_row(const bfloat16 *__restrict row, bfloat16 w_p,
                                 float *__restrict acc) {
  ::aie::accum<accfloat, n> ac;
  ac.from_vector(::aie::load_v<n>(acc));
  ac = ::aie::mac(ac, ::aie::load_v<n>(row), ::aie::broadcast<bfloat16, n>(w_p));
  ::aie::store_v(acc, ac.template to_vector<float>());
}

}  // namespace

#if !defined(QUANT_EMIT_INT4) && !defined(QUANT_EMIT_INT8)
#define QUANT_EMIT_INT4 1
#define QUANT_EMIT_INT8 1
#endif

extern "C" {

/* a is DIM_N-wide packed rows (mv_quant.cc / quant_row_layout.h layout), one per cached position.
 * w is [groups][w_stride] read at w_off, same convention as mv_taccum.cc. acc/c are
 * [groups][DIM_N] -- taccum_zero_f32/taccum_finish_bf16 (mv_taccum.cc) are unchanged and reused
 * from the same archive; only the accumulate step depends on A's dtype. */

#if defined(QUANT_EMIT_INT8) && QUANT_EMIT_INT8
void taccum_rows_int8_f32(uint32_t rows, uint32_t groups, uint32_t w_stride, uint32_t w_off,
                          const int8_t *__restrict a_in, const bfloat16 *__restrict w_in,
                          float *__restrict acc) {
  constexpr uint32_t n_groups = DIM_N / GROUP_SIZE;
  constexpr uint32_t header = n_groups * sizeof(scale_t);
  constexpr uint32_t payload = DIM_N;
  constexpr uint32_t row_stride = header + payload;
  static_assert(QUANT_ALIGN_OK(VEC_SIZE, header, payload, row_stride),
                "int8 payload load is not aligned under this layout");
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a_in);
  bfloat16 deq[DIM_N];
  for (uint32_t p = 0; p < rows; p++) {
    const quant_row_offsets off = quant_row_at<row_stride, header, payload>(p);
    const scale_t *scale = reinterpret_cast<const scale_t *>(a_bytes + off.header);
    const int8_t *packed = reinterpret_cast<const int8_t *>(a_bytes + off.payload);
    dequant_row_int8<VEC_SIZE, DIM_N, GROUP_SIZE>(packed, scale, deq);
    for (uint32_t g = 0; g < groups; g++)
      bcast_mac_row<DIM_N>(deq, w_in[g * w_stride + w_off + p], acc + g * DIM_N);
  }
}
#endif

#if defined(QUANT_EMIT_INT4) && QUANT_EMIT_INT4
void taccum_rows_int4_f32(uint32_t rows, uint32_t groups, uint32_t w_stride, uint32_t w_off,
                          const int8_t *__restrict a_in, const bfloat16 *__restrict w_in,
                          float *__restrict acc) {
  constexpr uint32_t n_groups = DIM_N / GROUP_SIZE;
  constexpr uint32_t header = n_groups * sizeof(scale_t);
  constexpr uint32_t payload = DIM_N / 2;
  constexpr uint32_t row_stride = header + payload;
  static_assert(QUANT_ALIGN_OK(VEC_SIZE / 2, header, payload, row_stride),
                "int4 payload load is not aligned under this layout");
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a_in);
  bfloat16 deq[DIM_N];
  for (uint32_t p = 0; p < rows; p++) {
    const quant_row_offsets off = quant_row_at<row_stride, header, payload>(p);
    const scale_t *scale = reinterpret_cast<const scale_t *>(a_bytes + off.header);
    const int8_t *packed = reinterpret_cast<const int8_t *>(a_bytes + off.payload);
    dequant_row_int4<VEC_SIZE, DIM_N, GROUP_SIZE>(packed, scale, deq);
    for (uint32_t g = 0; g < groups; g++)
      bcast_mac_row<DIM_N>(deq, w_in[g * w_stride + w_off + p], acc + g * DIM_N);
  }
}
#endif

}  // extern "C"
