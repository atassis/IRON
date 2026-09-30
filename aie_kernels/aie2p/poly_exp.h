// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// f32 vector exp / log1p / softplus by polynomial, for activations whose error compounds or is read
// directly (gates, sigmoid, SiLU). The SFU tanh/exp2 LUT is ~5e-3..5e-2 relative; these are ~1e-4
// on aie2p, where f32 vector multiplies are emulated through bf16.

#pragma once

#include <aie_api/aie.hpp>

static constexpr unsigned V = 16;
using vf = aie::vector<float, V>;

static inline vf vmul(vf a, vf b) { return aie::mul(a, b).template to_vector<float>(); }
static inline vf vfma(vf a, vf b, float c) { return aie::add(vmul(a, b), aie::broadcast<float, V>(c)); }
// Peano cannot legalize G_FNEG on <16 x float>, so negation is a subtract from zero.
static inline vf negv(vf x) { return aie::sub(aie::broadcast<float, V>(0.0f), x); }

// exp2f_vec (mlir-aie aie_kernels/aie2p/exp2f_vec.cc).
static inline vf exp2v(vf x)
{
    x = aie::max(x, aie::broadcast<float, V>(-111.0f));
    x = aie::min(x, aie::broadcast<float, V>(127.999f));
    aie::vector<int32_t, V> ki = aie::to_fixed<int32_t>(x);
    ki = aie::sub(ki, aie::select(aie::broadcast<int32_t, V>(0), aie::broadcast<int32_t, V>(1),
                                  aie::lt(x, aie::to_float<float>(ki))));
    vf f = aie::sub(x, aie::to_float<float>(ki));
    vf p = aie::broadcast<float, V>(0.0013333558f);
    p = vfma(p, f, 0.0096181291f);
    p = vfma(p, f, 0.0555041087f);
    p = vfma(p, f, 0.2402265069f);
    p = vfma(p, f, 0.6931471805f);
    p = vfma(p, f, 1.0f);
    aie::vector<int32_t, V> e = aie::upshift(aie::add(ki, aie::broadcast<int32_t, V>(127)), 23);
    return vmul(p, e.template cast_to<float>());
}

static inline vf expv(vf x) { return exp2v(vmul(x, aie::broadcast<float, V>(1.4426950409f))); }

// log1p on [0, 1], degree-8 fit, max rel error 1.9e-7 in f32 Horner.
static inline vf log1pv(vf t)
{
    vf p = aie::broadcast<float, V>(0.005253457929939032f);
    p = vfma(p, t, -0.02958850748836994f);
    p = vfma(p, t, 0.07836166769266129f);
    p = vfma(p, t, -0.13674770295619965f);
    p = vfma(p, t, 0.19111430644989014f);
    p = vfma(p, t, -0.24844369292259216f);
    p = vfma(p, t, 0.33319270610809326f);
    p = vfma(p, t, -0.49999502301216125f);
    p = vfma(p, t, 1.0f);
    return vmul(t, p);
}

static inline vf softplusv(vf u)
{
    return aie::add(aie::max(u, aie::broadcast<float, V>(0.0f)), log1pv(expv(negv(aie::abs(u)))));
}
