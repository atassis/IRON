// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Scales a bf16 tile in place by a compile-time constant (ATTN_SCALE). Written for
// iron/operators/attn_core/design.py to fold attention's `scores *= attn_scale` into softmax's
// own per-tile loop: attn_scale is trace-time-known (head_dim**-0.5, see llm_decode_spec.py), so
// this is one extra kernel call on a tile softmax already has resident, instead of a whole extra
// ElementwiseMul pipeline stage (its own ObjectFifos, its own aiex.configure).

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef ATTN_SCALE
#error "ATTN_SCALE must be defined at compile time, e.g. -DATTN_SCALE=0.08838834764831845f"
#endif

extern "C" {

void scale_tile_bf16(bfloat16 *restrict buf, const int32_t n)
{
    event0();
    ::aie::vector<bfloat16, 32> s = ::aie::broadcast<bfloat16, 32>((bfloat16)ATTN_SCALE);
    const int32_t F = n / 32;
    for (int32_t i = 0; i < F; i++) {
        ::aie::vector<bfloat16, 32> v = ::aie::load_v<32>(buf);
        ::aie::accum<accfloat, 32> acc = ::aie::mul(v, s);
        ::aie::store_v(buf, acc.to_vector<bfloat16>());
        buf += 32;
    }
    for (int32_t i = F * 32; i < n; i++) {
        buf[i - F * 32] = (bfloat16)((float)buf[i - F * 32] * (float)ATTN_SCALE);
    }
    event1();
}

}
