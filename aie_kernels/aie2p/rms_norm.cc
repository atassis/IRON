// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Scale-and-gain loop, shared by all three entry points below.
//
// RMS_BF16_SCALE=1 replaces the f32 multiplies with native bf16 ones: 22 bundles per 32 elements
// against 118, because aie2p has no native f32 vector multiply. It is a PRECISION CHANGE (rel-L2
// 2.317e-3 against the f32 form's 1.664e-3) and default-off; gate it on parity, not on rel-L2.
// The scale is a bf16 PAIR because a single bf16 inv_rms biases every element by up to 2^-9.
// Gain-first (multiply by gain before the inv_rms pair) was tried and dropped: same rounding
// COUNT, 25 vs 22 bundles/32 elem, rel-L2 unchanged (2.308e-3 vs 2.299e-3, real Gemma-4 gains).
#ifndef RMS_BF16_SCALE
#define RMS_BF16_SCALE 0
#endif

template <typename T, int N>
static inline void rms_scale_rows(const T *restrict input,
                                  const T *restrict input2,
                                  T *restrict output,
                                  float inv_rms,
                                  int vector_chunks)
{
#if RMS_BF16_SCALE
    const bfloat16 s_hi = (bfloat16)inv_rms;
    const bfloat16 s_lo = (bfloat16)(inv_rms - (float)s_hi);
    const ::aie::vector<bfloat16, N> hi = ::aie::broadcast<bfloat16, N>(s_hi);
    const ::aie::vector<bfloat16, N> lo = ::aie::broadcast<bfloat16, N>(s_lo);
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> x = ::aie::load_v<N>(input + i * N);
        ::aie::accum<accfloat, N> a = ::aie::mul(x, hi);
        a = ::aie::mac(a, x, lo);
        if (input2)
            a = ::aie::mul(a.template to_vector<bfloat16>(), ::aie::load_v<N>(input2 + i * N));
        ::aie::store_v(output + i * N, a.template to_vector<T>());
    }
#else
    ::aie::accum<accfloat, N> inv_rms_v;
    inv_rms_v.from_vector(::aie::broadcast<float, N>(inv_rms), 0);
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::accum<accfloat, N> reg_a;
        reg_a.from_vector(::aie::load_v<N>(input + i * N), 0);
        reg_a = ::aie::mul(reg_a.template to_vector<float>(), inv_rms_v.template to_vector<float>());
        if (input2) {
            ::aie::accum<accfloat, N> reg_b;
            reg_b.from_vector(::aie::load_v<N>(input2 + i * N), 0);
            reg_a = ::aie::mul(reg_a.template to_vector<float>(), reg_b.template to_vector<float>());
        }
        ::aie::store_v(output + i * N, reg_a.template to_vector<T>());
    }
#endif
}

template <typename T, int N>
void rms_norm_general(const T *restrict input,
                      const T *restrict input2,
                      T *restrict output,
                      int32_t cols,
                      float epsilon)
{
    event0();
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();

    int vector_chunks = cols / N;
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    float sum_sq = ::aie::reduce_add(add_res);

    int remaining = cols % N;
    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            float square = static_cast<float>(val) * static_cast<float>(val);
            sum_sq += square;
        }
    }

    float rms = sum_sq / cols + epsilon;
    float inv_rms = aie::invsqrt(rms);
    // Normalize in f32 and round once. Mirrors layer_norm.cc's accfloat path.
    rms_scale_rows<T, N>(input, input2, output, inv_rms, vector_chunks);

    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            T norm_val = static_cast<T>(static_cast<float>(val) * inv_rms);
            if (input2) {
                T mul_val = input2[start_idx + i];
                output[start_idx + i] = static_cast<T>(static_cast<float>(norm_val) * static_cast<float>(mul_val));
            } else {
                output[start_idx + i] = norm_val;
            }
        }
    }
    event1();
}

// x1 = x_in + gain_in*rms_in/rms(rms_in). `rms_in` and `output` are NOT `restrict` -- the
// residual prologue (gemv/design.py prologue="residual") calls this with output==rms_in, writing
// x1 over the just-consumed activation's own slot once its rms and value are no longer needed
// past this call. `gain_in`/`x_in` are separate buffers in every call site and stay `restrict`.
// PRECONDITION: N divides cols (no remainder loop -- every caller uses D=3840, N=32).
template <typename T, int N>
void rms_norm_residual_general(const T *rms_in,
                               const T *restrict gain_in,
                               const T *restrict x_in,
                               T *output,
                               int32_t cols,
                               float epsilon)
{
    event0();
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();
    int vector_chunks = cols / N;
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(rms_in + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    float inv_rms = aie::invsqrt(::aie::reduce_add(add_res) / cols + epsilon);

    ::aie::accum<accfloat, N> inv_rms_v;
    inv_rms_v.from_vector(::aie::broadcast<float, N>(inv_rms), 0);
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::accum<accfloat, N> a;
        a.from_vector(::aie::load_v<N>(rms_in + i * N), 0);
        a = ::aie::mul(a.template to_vector<float>(), inv_rms_v.template to_vector<float>());
        ::aie::accum<accfloat, N> g;
        g.from_vector(::aie::load_v<N>(gain_in + i * N), 0);
        a = ::aie::mul(a.template to_vector<float>(), g.template to_vector<float>());
        ::aie::accum<accfloat, N> x;
        x.from_vector(::aie::load_v<N>(x_in + i * N), 0);
        a = ::aie::add(a.template to_vector<float>(), x.template to_vector<float>());
        ::aie::store_v(output + i * N, a.template to_vector<T>());
    }
    event1();
}

#ifdef RMS_COLS
// Shape-specialized entry point. `cols` is a COMPILE-TIME constant here, which is what removes the
// two scalar-float constructs the runtime-cols form above cannot avoid:
//   * `sum_sq / cols` is a runtime float divide plus an int->float, and those are the ONLY reason
//     __divsf3 (1168 B) and __floatsisf link into a core. Measured on qkv_head_dp: dropping the
//     divide alone took .text 10416 -> 9248 B.
//   * the `cols % N` remainder loops are dead whenever N divides cols -- true for every shape this
//     decode uses (D=1024, HD=128, N=32) -- but they still link, and they carry scalar float math.
//     Removing them took .text 9248 -> 8448 B. Together: -1968 B, 18.9% of that core's program
//     memory, against a 16384 B region.
// Left OPT-IN behind RMS_COLS so the runtime-cols entry point above keeps its exact semantics for
// callers with a non-multiple-of-N size. Same idiom as mv.cc's -DDIM_K.
template <typename T, int N, int COLS>
void rms_norm_fixed(const T *restrict input,
                    const T *restrict input2,
                    T *restrict output,
                    float epsilon)
{
    static_assert(COLS % N == 0, "RMS_COLS must be a multiple of the vector width");
    event0();
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();
    for (int i = 0; i < COLS / N; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    // COLS is constexpr, so the reciprocal is folded at compile time -- and it is EXACT for the
    // powers of two this rail uses, so this is not a precision change against `sum_sq / cols`.
    constexpr float inv_cols = 1.0f / static_cast<float>(COLS);
    float inv_rms = aie::invsqrt(::aie::reduce_add(add_res) * inv_cols + epsilon);
    rms_scale_rows<T, N>(input, input2, output, inv_rms, COLS / N);
    event1();
}
#endif

// Length-parameterised form. `cols` and its reciprocal both arrive as arguments, so neither scalar-
// float construct the RMS_COLS form exists to avoid comes back: `sum_sq / cols` is what links
// __divsf3 (1168 B) and __floatsisf, and `cols % N` is what keeps the remainder loops alive.
// `cols / N` is an integer divide by a compile-time power of two, i.e. a shift.
// PRECONDITION: N divides cols. A remainder is not handled and not detectable here -- callers with
// a ragged length want `rms_norm_general` above.
template <typename T, int N>
void rms_norm_cols(const T *restrict input,
                   const T *restrict input2,
                   T *restrict output,
                   int32_t cols,
                   float inv_cols,
                   float epsilon)
{
    event0();
    const int vector_chunks = cols / N;
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    float inv_rms = aie::invsqrt(::aie::reduce_add(add_res) * inv_cols + epsilon);
    rms_scale_rows<T, N>(input, input2, output, inv_rms, vector_chunks);
    event1();
}

extern "C" {
void rms_norm_bf16_vector(bfloat16 *input, bfloat16 *output, int32_t size, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even); // round-to-nearest-even; do not inherit a floor rounding mode
                                                        // from a prior kernel
    rms_norm_general<bfloat16, 32>(input, nullptr, output, size, epsilon);
}

#ifdef RMS_COLS
// No `size` argument: the shape is RMS_COLS, so there is nothing for a caller to get wrong.
void weighted_rms_norm_fixed(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even); // round-to-nearest-even; do not inherit a floor rounding mode
    rms_norm_fixed<bfloat16, 32, RMS_COLS>(a_in, b_in, c_out, epsilon);
}
#endif
// ONE body, two MLIR-visible names. An external func.func is keyed by NAME and typed by memref
// SHAPE, so a caller at D and a caller at head_dim cannot share a declaration -- but under the
// bare-pointer calling convention both lower to the same ABI, so they can share an address. The
// alias is what keeps the second call site from costing a second 960 B copy of this loop on a core
// whose 16 KB program memory is the binding constraint.
void weighted_rms_norm_cols(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out,
                            int32_t cols, float inv_cols, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even); // round-to-nearest-even; do not inherit a floor rounding mode
    rms_norm_cols<bfloat16, 32>(a_in, b_in, c_out, cols, inv_cols, epsilon);
}
// Opt-in, like mv.cc's GEMV_ALIAS_SC and for the same reason: qkv_head_dp reaches the second name
// by compiling this source a second time under `prefix_symbols="hd_"`, so an unconditional alias
// puts two definitions of it in one archive -- `duplicate symbol: hd_weighted_rms_norm_cols`.
#ifdef RMS_ALIAS_HD
void hd_weighted_rms_norm_cols(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out,
                               int32_t cols, float inv_cols, float epsilon)
    __attribute__((alias("weighted_rms_norm_cols")));
#endif
void weighted_rms_norm(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int32_t size, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even); // round-to-nearest-even; do not inherit a floor rounding mode
                                                        // from a prior kernel
    rms_norm_general<bfloat16, 32>(a_in, b_in, c_out, size, epsilon);
}

// See rms_norm_residual_general's docstring for the formula and the aliasing contract. Called as
// gemv/design.py prologue="residual" step 1 (rms_in=o, gain_in=gain_a, x_in=x, output=o's own
// slot); step 2 is a plain weighted_rms_norm call, no aliasing.
void rms_norm_residual_add(bfloat16 *rms_in, bfloat16 *gain_in, bfloat16 *x_in, bfloat16 *output,
                           int32_t size, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    rms_norm_residual_general<bfloat16, 32>(rms_in, gain_in, x_in, output, size, epsilon);
}

// bf16 -> bf16, no conversion, so no rounding mode to set. gemv/design.py's prologue!="none" core
// body always dequeues >=2 B objects and always writes through one of several paths into b_norm so
// the shared matvec loop after it reads one buffer regardless of mode; this is the mode-off path
// (mode-on is weighted_rms_norm above, mode-residual is rms_norm_residual_add + weighted_rms_norm).
// See "the-norm-kernels-pay-an-emulated-f32-vector-multiply" for why a copy, not a same-buffer
// branch, is cheaper than duplicating the tile loop.
void bf16_copy_vector(bfloat16 *restrict input, bfloat16 *restrict output, int32_t size)
{
    constexpr int N = 32;
    int chunks = size / N;
    for (int i = 0; i < chunks; i++)
        ::aie::store_v(output + i * N, ::aie::load_v<N>(input + i * N));
    for (int i = chunks * N; i < size; i++)
        output[i] = input[i];
}
}
