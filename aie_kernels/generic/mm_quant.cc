// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Weight-quantized front half of mm.cc: B (the GEMM's KxN weight) streams as a group-quantized
// int4/int8 TILE SLAB and is expanded on-core into the bf16 tile mm.cc already multiplies. A and C
// stay bf16 -- the axis is the weight's bytes (iron/operators/gemm `weight_dtype`), never the
// arithmetic. Slab layout: iron/common/quant.py, written by `pack_gemm_weight`.
//
// The payload arrives ALREADY in mm.cc's destination order, so this streams rather than scatters:
// one t*s block is one load, one convert, one multiply, one store. A scale is constant over a
// block's MMUL_S columns, so an (n-block, group) pair's MMUL_T scales hoist out of the inner loop.
// Rounding is conv_even, restored on exit -- the default is floor and biases toward zero (K001).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif
#ifndef MMUL_S
#define MMUL_S 8
#endif
#ifndef MMUL_T
#define MMUL_T 8
#endif
// Byte offset of the payload, passed in rather than derived, so quant.py owns the number and the
// static_assert below turns a disagreement into a compile error instead of a device result.
#ifndef SCALE_REGION_BYTES
#define SCALE_REGION_BYTES 256
#endif

namespace {

template <uint32_t k, uint32_t n, uint32_t g, uint32_t s, uint32_t t, uint32_t scale_region,
          bool is_int4>
void dequant_b_tile(const int8_t *__restrict packed, bfloat16 *__restrict out) {
  static_assert(k % g == 0, "DIM_K must be a whole number of quant groups");
  static_assert(g % s == 0, "GROUP_SIZE must be a multiple of MMUL_S: a block needs one scale");
  static_assert(n % t == 0, "DIM_N must be a multiple of MMUL_T");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t blk = t * s;
  constexpr uint32_t blk_bytes = is_int4 ? blk / 2 : blk;
  constexpr uint32_t want = ((4 * n * n_groups + blk_bytes - 1) / blk_bytes) * blk_bytes;
  static_assert(scale_region == want,
                "scale region disagrees with iron/common/quant.py's gemm_tile_scale_region_bytes");
  constexpr uint32_t k_blocks = k / s;
  constexpr uint32_t blocks_per_group = g / s;

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const float *__restrict scales = reinterpret_cast<const float *>(packed);
  const int8_t *__restrict payload = packed + scale_region;

  for (uint32_t j = 0; j < n / t; j++) {
    for (uint32_t gi = 0; gi < n_groups; gi++) {
      ::aie::vector<bfloat16, blk> sv = ::aie::zeros<bfloat16, blk>();
      for (uint32_t r = 0; r < t; r++)
        sv.template insert<s>(r, ::aie::broadcast<bfloat16, s>(
                                     (bfloat16)scales[(j * t + r) * n_groups + gi]));
      for (uint32_t bi = 0; bi < blocks_per_group; bi++) {
        const uint32_t p = (j * k_blocks + gi * blocks_per_group + bi) * blk;
        ::aie::vector<int8, blk> q;
        if constexpr (is_int4)
          q = ::aie::unpack(::aie::vector_cast<int4>(::aie::load_v<blk / 2>(payload + p / 2)));
        else
          q = ::aie::load_v<blk>(payload + p);
        ::aie::vector<bfloat16, blk> qbf = ::aie::to_float<bfloat16>(q, 0);
        ::aie::store_v(out + p, ::aie::mul(qbf, sv).template to_vector<bfloat16>());
      }
    }
  }
  ::aie::set_rounding(saved_rounding);
}

}  // namespace

// Only the dtype the caller asked for is emitted: a template's static_asserts fire on
// instantiation, so emitting both would force one GROUP_SIZE/tile to be legal for both at once.
#if !defined(QUANT_EMIT_INT4) && !defined(QUANT_EMIT_INT8)
#define QUANT_EMIT_INT4 1
#define QUANT_EMIT_INT8 1
#endif

extern "C" {

#if defined(QUANT_EMIT_INT8) && QUANT_EMIT_INT8
void dequant_b_int8_bf16(const int8_t *__restrict b_in, bfloat16 *__restrict b_out) {
  dequant_b_tile<DIM_K, DIM_N, GROUP_SIZE, MMUL_S, MMUL_T, SCALE_REGION_BYTES, false>(b_in, b_out);
}
#endif

#if defined(QUANT_EMIT_INT4) && QUANT_EMIT_INT4
void dequant_b_int4_bf16(const int8_t *__restrict b_in, bfloat16 *__restrict b_out) {
  dequant_b_tile<DIM_K, DIM_N, GROUP_SIZE, MMUL_S, MMUL_T, SCALE_REGION_BYTES, true>(b_in, b_out);
}
#endif

}  // extern "C"
