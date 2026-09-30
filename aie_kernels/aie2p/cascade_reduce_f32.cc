// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Core-to-core cascade sum over an f32 buffer, one v16float cascade transfer (16 lanes) at a
// time. Raw put_mcd/get_scd intrinsics, not aie_api's cascade accessor (aie_api/cascade.hpp): that
// header lives in a fork not wired into this tree's include path, and mlir-aie's own shipped
// cascade_mm.cc kernel reaches the same intrinsics directly for the identical reason. aie2p only.
//
// Three roles chain core-to-core (put_only -> put_get -> ... -> put_get -> get_only), each adding
// its own `local` contribution to the running total; only get_only (chain end) sees the full sum,
// written to `acc`. `n` must be a multiple of 16.
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void cascade_reduce_put_only_f32(uint32_t n, const float *__restrict local) {
  for (uint32_t i = 0; i < n; i += 16)
    put_mcd(aie::load_v<16>(local + i).to_native());
}

void cascade_reduce_put_get_f32(uint32_t n, const float *__restrict local) {
  for (uint32_t i = 0; i < n; i += 16) {
    aie::vector<float, 16> pred(get_scd_v16float());
    aie::vector<float, 16> sum = aie::add(pred, aie::load_v<16>(local + i));
    put_mcd(sum.to_native());
  }
}

void cascade_reduce_get_only_f32(uint32_t n, const float *__restrict local,
                                 float *__restrict acc) {
  for (uint32_t i = 0; i < n; i += 16) {
    aie::vector<float, 16> pred(get_scd_v16float());
    aie::vector<float, 16> sum = aie::add(pred, aie::load_v<16>(local + i));
    aie::store_v(acc + i, sum);
  }
}

} // extern "C"
