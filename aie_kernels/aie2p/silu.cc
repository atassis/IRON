// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SiLU: out = x * sigmoid(x).
//
// The sigmoid is computed from a SOFTWARE f32 exp2 polynomial, not from the hardware SFU tanh.
// aie::tanh<bfloat16> on aie2p is the same coarse piecewise-linear SFU LUT as aie::exp2 -- measured
// rel-L2 5.370e-3 on [-8,8] rising to 4.932e-2 on [-0.5,0.5], and the error GROWS as the domain
// narrows, which is backwards for rounding and right for a fixed-knot interpolator. The degree-5
// poly below is 1467x-1726x better (3.2e-5 to 1.1e-4). Its LUT error is also BIASED rather than
// random, so in a deep residual stack it accumulates instead of cancelling: on the 28-layer Qwen3
// decode the tanh form left SiLU as the largest per-op error in the whole graph (2.3e-2 against
// ~4e-3 for every norm and residual add, and bit-exact GEMVs).
//
// Ported from route_b_kernels/ctx_ln/silu_brick.cc SILU_MODE=2 (standalone rel-L2 ~6e-6), narrowed
// to the bf16 in/out interface this operator exports.
//
// TWO CONTRACTS THIS KERNEL DEPENDS ON, both of which are silent when violated:
//   * STACK. This body spills a frame well past the IRON default 1024 B worker stack, and the
//     generated ld.script places that stack directly below the objectFIFO buffers with zero
//     clearance -- so an overflow corrupts a neighbouring buffer with no crash. The design must
//     size the worker stack past the frame (channeled_unary_design.py passes stack_size).
//   * ROUNDING. crRnd is one sticky per-core register and aie_api never initialises it; the
//     documented default is floor. SWAP and restore rather than set, so a kernel sharing this core
//     does not inherit whatever we wanted.

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// SOFTWARE f32 2^x for x <= 0. NOINLINE is LOAD-BEARING: inlining it makes Peano -O2 miscompile to
// NaN (a register-pressure codegen bug, recorded against the conv-module silu brick).
static __attribute__((noinline)) ::aie::vector<float, 16>
exp2f_neg16(::aie::vector<float, 16> x) {
  x = ::aie::max(x, ::aie::broadcast<float, 16>(-100.0f));
  ::aie::vector<int32_t, 16> ki = ::aie::to_fixed<int32_t>(x);
  ::aie::vector<float, 16> kf = ::aie::to_float<float>(ki);
  ::aie::vector<int32_t, 16> one = ::aie::broadcast<int32_t, 16>(1);
  ::aie::vector<int32_t, 16> zero = ::aie::broadcast<int32_t, 16>(0);
  ki = ::aie::sub(ki, ::aie::select(zero, one, ::aie::lt(x, kf)));
  ::aie::vector<float, 16> f = ::aie::sub(x, ::aie::to_float<float>(ki));
  ::aie::vector<float, 16> p = ::aie::broadcast<float, 16>(0.0013333558f);
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.0096181291f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.0555041087f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.2402265069f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.6931471805f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(1.0f));
  ::aie::vector<int32_t, 16> ebits =
      ::aie::upshift(::aie::add(ki, ::aie::broadcast<int32_t, 16>(127)), 23);
  ::aie::vector<float, 16> p2k = ebits.cast_to<float>();
  return ::aie::mul(p, p2k).to_vector<float>();
}

void silu_poly_bf16(bfloat16 *restrict input_vector, bfloat16 *restrict output_vector,
                    const int32_t vector_size)
{
    event0();
    // Hand the mode back on exit: crRnd is sticky per core and shared with whatever runs next.
    const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);

    constexpr int N = 16;
    const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
    const ::aie::vector<float, N> two = ::aie::broadcast<float, N>(2.0f);
    const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
    const ::aie::vector<float, N> neg_log2e = ::aie::broadcast<float, N>(-1.44269504089f);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < vector_size; i += N) {
        ::aie::vector<bfloat16, N> xb = ::aie::load_v<N>(input_vector + i);
        ::aie::accum<accfloat, N> xa;
        xa.from_vector(xb);
        ::aie::vector<float, N> xv = xa.template to_vector<float>();

        // sigmoid(x) computed on |x| so the exponential never overflows, then reassembled:
        //   x >= 0 -> 1 / (1 + 2^(-|x| log2 e))
        //   x <  0 -> s / (1 + s),  s = 2^(-|x| log2 e)
        ::aie::vector<float, N> absx = ::aie::max(xv, ::aie::sub(zero, xv));
        ::aie::vector<float, N> s =
            exp2f_neg16(::aie::mul(absx, neg_log2e).template to_vector<float>());
        ::aie::vector<float, N> denom = ::aie::add(one, s);
        // one Newton step on the reciprocal: aie::inv alone is not accurate enough to keep the
        // poly's advantage over the LUT it replaces.
        ::aie::vector<float, N> r0 = ::aie::inv(denom);
        ::aie::vector<float, N> dr0 = ::aie::mul(denom, r0).template to_vector<float>();
        ::aie::vector<float, N> r = ::aie::mul(r0, ::aie::sub(two, dr0)).template to_vector<float>();
        ::aie::vector<float, N> num = ::aie::select(one, s, ::aie::lt(xv, zero));
        ::aie::vector<float, N> sig = ::aie::mul(num, r).template to_vector<float>();
        // Narrow ONCE, from the accumulator, as the original kernel did: aie::mul yields an
        // accum and to_vector<bfloat16>() is the conversion that honours the rounding mode set
        // above. Going through an intermediate vector<float> would round twice.
        auto out_acc = ::aie::mul(xv, sig);
        ::aie::store_v(output_vector + i, out_acc.template to_vector<bfloat16>());
    }

    ::aie::set_rounding(saved_rounding);
    event1();
    return;
}

extern "C" {

void silu_bf16(bfloat16 *restrict input, bfloat16 *restrict output, int input_size)
{
    silu_poly_bf16(input, output, input_size);
}

} // extern "C"
