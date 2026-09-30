// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Correctly rounded f32 sqrt and division on 16 int32 lanes, for kernels whose host model is
// numpy's IEEE result (K048). Callers run them under aie::rounding_mode::floor (K049).
#pragma once
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace rf {

inline uint32_t bits(float x) { return __builtin_bit_cast(uint32_t, x); }
inline float from_bits(uint32_t u) { return __builtin_bit_cast(float, u); }

using V = aie::vector<int32_t, 16>;
inline V bc(int32_t x) { return aie::broadcast<int32_t, 16>(x); }

inline V sqrt_rn(V u) {
  V e = aie::sub(aie::downshift(u, 23), bc(127));
  V m = aie::bit_or(aie::bit_and(u, bc(0x7FFFFF)), bc(0x800000));
  auto odd = aie::neq(aie::bit_and(e, bc(1)), bc(0));
  m = aie::select(m, aie::upshift(m, 1), odd);
  e = aie::select(e, aie::sub(e, bc(1)), odd);
  // digit-by-digit root of m << 25, two bits per step; the remainder stays under 2^27
  V res = bc(0), rem = bc(0);
  for (int k = 24; k >= 0; --k) {
    V two = 2 * k >= 25 ? aie::bit_and(aie::downshift(m, 2 * k - 25), bc(3))
            : 2 * k == 24 ? aie::upshift(aie::bit_and(m, bc(1)), 1) : bc(0);
    rem = aie::bit_or(aie::upshift(rem, 2), two);
    V trial = aie::bit_or(aie::upshift(res, 2), bc(1));
    auto ge = aie::ge(rem, trial);
    rem = aie::select(rem, aie::sub(rem, trial), ge);
    res = aie::select(aie::upshift(res, 1), aie::bit_or(aie::upshift(res, 1), bc(1)), ge);
  }
  V q = aie::downshift(res, 1);
  auto up = aie::eq(aie::bit_and(res, bc(1)), bc(1)) &
            (aie::neq(rem, bc(0)) | aie::neq(aie::bit_and(q, bc(1)), bc(0)));
  q = aie::select(q, aie::add(q, bc(1)), up);
  V E = aie::downshift(e, 1);
  auto ovf = aie::eq(q, bc(1 << 24));
  q = aie::select(q, aie::downshift(q, 1), ovf);
  E = aie::select(E, aie::add(E, bc(1)), ovf);
  return aie::bit_or(aie::upshift(aie::add(E, bc(127)), 23), aie::bit_and(q, bc(0x7FFFFF)));
}

inline V div_rn(V ua, V ub) {
  auto zero = aie::eq(aie::bit_and(ua, bc(0x7FFFFFFF)), bc(0));
  V ea = aie::downshift(ua, 23), eb = aie::downshift(ub, 23);
  V ma = aie::bit_or(aie::bit_and(ua, bc(0x7FFFFF)), bc(0x800000));
  V mb = aie::bit_or(aie::bit_and(ub, bc(0x7FFFFF)), bc(0x800000));
  V q = bc(0), r = bc(0);   // (ma << 26) / mb by shift-subtract; q < 2^27, r < 2^25
  for (int i = 49; i >= 0; --i) {
    V bit = i >= 26 ? aie::bit_and(aie::downshift(ma, i - 26), bc(1)) : bc(0);
    r = aie::bit_or(aie::upshift(r, 1), bit);
    q = aie::upshift(q, 1);
    auto ge = aie::ge(r, mb);
    r = aie::select(r, aie::sub(r, mb), ge);
    q = aie::select(q, aie::bit_or(q, bc(1)), ge);
  }
  auto big = aie::ge(q, bc(1 << 26));
  V m = aie::select(aie::downshift(q, 2), aie::downshift(q, 3), big);
  V rb = aie::select(aie::bit_and(aie::downshift(q, 1), bc(1)), aie::bit_and(aie::downshift(q, 2), bc(1)), big);
  V st = aie::select(aie::bit_and(q, bc(1)), aie::bit_and(q, bc(3)), big);
  auto up = aie::eq(rb, bc(1)) &
            (aie::neq(r, bc(0)) | aie::neq(st, bc(0)) | aie::neq(aie::bit_and(m, bc(1)), bc(0)));
  m = aie::select(m, aie::add(m, bc(1)), up);
  V E = aie::select(aie::sub(aie::sub(ea, eb), bc(1)), aie::sub(ea, eb), big);
  auto ovf = aie::eq(m, bc(1 << 24));
  m = aie::select(m, aie::downshift(m, 1), ovf);
  E = aie::select(E, aie::add(E, bc(1)), ovf);
  V o = aie::bit_or(aie::upshift(aie::add(E, bc(127)), 23), aie::bit_and(m, bc(0x7FFFFF)));
  return aie::select(o, bc(0), zero);
}


}  // namespace rf
