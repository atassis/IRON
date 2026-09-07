// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Weight-quantized sibling of mv.cc: A (the GEMV's MxK matrix operand) streams as group-quantized
// int4 or int8 with a per-row, per-group f32 scale, dequantized ON-CORE right before the same
// bf16xbf16 MAC mv.cc already uses. B (the vector) and C (the output) are unchanged bf16 -- only
// the WEIGHT-STREAM byte format is an axis here (iron/operators/gemv `weight_dtype`), never the
// arithmetic: narrow arithmetic buys nothing at M=1 decode (nothing to speed up, pack/unpack only
// adds ops), so this kernel still does a bf16 MAC. The lever is DDR/L2 bytes for the weight, not
// FLOPs.
//
// Row layout (one row = one output feature), K elements, GROUP_SIZE-wide quant groups,
// n_groups = DIM_K / GROUP_SIZE:
//   [n_groups x f32 scale][ payload ]
// payload is K/2 nibble-packed bytes (int4, low nibble = even column, high nibble = odd column --
// same convention as dequant_int4_group_row.cc) or K int8 bytes (int8, one byte per element).
// Packing the scale into the SAME buffer as the weight (rather than a 3rd input FIFO) is forced by
// the AIE2P tile DMA budget: a core has only 2 input channels, both already spent on A and B (see
// gemm_int8xint4_dequant.cc's identical constraint). A ROW-granularity header (not a
// tile-granularity one) keeps the row stride uniform, so the existing per-column contiguous TAP
// arithmetic in gemv/design.py needs no tiling-aware special case.
//
// Dequant follows the established, device-gated idiom (dequant_int4_group.cc /
// gemm_int8xint4_dequant.cc): scalar nibble/byte unpack into a float buffer, vector multiply by
// the (broadcast) group scale, narrow to bf16 via an explicit accum with conv_even rounding rather
// than a raw cast (the banked WER lesson: default truncation biases toward zero).
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif

namespace {

// r: vector chunk width (VEC_SIZE); k: full row length (DIM_K); g: quant group width
// (GROUP_SIZE). A vector chunk must never straddle a group boundary, so g must be a multiple of r.
template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int4_dequant(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                         bfloat16 *__restrict c) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t row_stride = n_groups * sizeof(float) + k / 2;
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row f32 scale read)");

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  // The group scale is applied ONCE per group to the group's reduced partial, not once per
  // element. Per element it forced a vector<float,r> materialisation inside the inner loop and
  // cost 89 of the loop's 97 bundles (contract K013); the arithmetic here is the same sum in
  // exact arithmetic, and rounds less, because a_i is an exact small integer in bf16 and the
  // scale meets the partial in f32 rather than every product in bf16.
  // Dequant follows aie_kernels/generic/expand.cc: the scale is broadcast and applied in BF16
  // (one 512-bit register at r=32), and the widening stays narrow -- 4 -> 8 -> 16 -> bf16 -- so an
  // f32 vector is never materialised. The earlier form multiplied in f32, which at r=64 is FOUR
  // vector registers and measured 89 of the loop's 97 bundles (contract K013).
  constexpr uint32_t blocks_per_group = g / r;
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const float *__restrict scale = reinterpret_cast<const float *>(rowp);
    const uint8_t *packed = rowp + n_groups * sizeof(float);
    ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
    AIE_LOOP_MIN_ITERATION_COUNT(n_groups)
    for (uint32_t gi = 0; gi < n_groups; gi++) {
      // Scale enters ONCE per group as the scalar multiplier of a MAC on the group's
      // accumulator -- the form mlir-air's int4_awq/mv_int4_bf16.cc uses. Multiplying each
      // block by a broadcast scale instead costs blocks_per_group vector multiplies per group.
      const bfloat16 sa = static_cast<bfloat16>(scale[gi]);
      const bfloat16 *__restrict b_cur = b + gi * g;
      const uint8_t *__restrict gp = packed + (gi * g) / 2;
      ::aie::accum<accfloat, r> g_acc;
      g_acc.from_vector(::aie::zeros<float, r>());
      AIE_LOOP_UNROLL_FULL
      for (uint32_t ci = 0; ci < blocks_per_group; ci++) {
        ::aie::vector<int8_t, r> q8 =
            ::aie::unpack(::aie::load_v<r>(reinterpret_cast<const int4 *>(gp + (ci * r) / 2)));
        ::aie::vector<bfloat16, r> a_vec =
            ::aie::to_float<bfloat16>(::aie::unpack(q8), 0);
        g_acc = ::aie::mac(g_acc, a_vec, ::aie::load_v<r>(b_cur + ci * r));
      }
      acc = ::aie::mac(acc, g_acc.template to_vector<bfloat16>(), sa);
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
  constexpr uint32_t row_stride = n_groups * sizeof(float) + k;
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row f32 scale read)");

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  const bfloat16 *b_end = b + k;
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const float *scale = reinterpret_cast<const float *>(rowp);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + n_groups * sizeof(float));
    aie::accum acc = aie::zeros<accfloat, r>();
    uint32_t chunk = 0;
    // Same pipelining hint mv.cc carries on its own inner loop. Without it this loop is not
    // converted to a zero-overhead loop: it compiles to a branchy software loop that reloads
    // state from the stack every iteration.
    AIE_LOOP_MIN_ITERATION_COUNT(k / r)
    for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, chunk++) {
      const float s = scale[(chunk * r) / g];
      const int8_t *chunk_packed = packed + chunk * r;
      // Not vectorised: the int8 payload starts n_groups*4 bytes into the row, so a
      // VEC_SIZE-wide load is misaligned. int4's load is half as wide and is not.
      float unpacked[r];
      for (uint32_t i = 0; i < r; i++) unpacked[i] = (float)chunk_packed[i];
      ::aie::vector<float, r> qv = ::aie::load_v<r>(unpacked);
      // aie::mul already yields an accumulator; the accum -> vector<float> -> accum round trip
      // this replaced forced a 4-register float materialisation inside the loop (K013 census:
      // 97-bundle body against 8 without the multiply).
      ::aie::vector<bfloat16, r> a_vec =
          ::aie::mul(qv, ::aie::broadcast<float, r>(s)).template to_vector<bfloat16>();
      ::aie::vector<bfloat16, r> b_vec = ::aie::load_v<r>(b_cur);
      acc = ::aie::mac(acc, a_vec, b_vec);
    }
    c[row] = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
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

}  // extern "C"
