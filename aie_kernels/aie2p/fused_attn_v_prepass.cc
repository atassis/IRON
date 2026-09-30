// SPDX-License-Identifier: Apache-2.0
// V pre-pass for Gemma-4's attention_k_eq_v (no stored V, no v_proj): recomputes V for a whole K
// block up front, one untiled row at a time -- see aie_kernels/generic/mv_taccum.cc's
// taccum_rows_kv_skip_v_bf16_f32, whose RoPE-inversion/gain-divide/RMSNorm math this ports
// (dropping its fused P*V accumulate for a direct store).

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef HEAD_DIM
#define HEAD_DIM 512
#endif

extern "C" {

// `kc`/`ang`/`v_out` are `rows` rows of HEAD_DIM each, row-major, contiguous -- no slicing, so one
// row's read/write is a single DMA-legal run. `ang` is laid out like rope.cc's `lut`: HEAD_DIM-wide,
// interleaved [cos, sin] pairs over HEAD_DIM/2 dims. `gain` is qk-norm's K weight, [HEAD_DIM],
// shared by every row.
void fa_v_prepass_rows(int32_t rows,
                       const bfloat16 *__restrict kc,
                       const bfloat16 *__restrict ang,
                       const bfloat16 *__restrict gain,
                       bfloat16 *__restrict v_out,
                       float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    constexpr uint32_t HALF = HEAD_DIM / 2;
    constexpr float inv_head_dim = 1.0f / HEAD_DIM;
    aie::vector<bfloat16, HALF> inv_g1 = aie::inv(aie::load_v<HALF>(gain));
    aie::vector<bfloat16, HALF> inv_g2 = aie::inv(aie::load_v<HALF>(gain + HALF));

    const bfloat16 *kp = kc;
    const bfloat16 *angp = ang;
    bfloat16 *vp = v_out;
    for (int32_t p = 0; p < rows; p++, kp += HEAD_DIM, angp += HEAD_DIM, vp += HEAD_DIM) {
        aie::vector<bfloat16, HEAD_DIM> ang_row = aie::load_v<HEAD_DIM>(angp);
        aie::vector<bfloat16, HALF> cos_val = aie::filter_even(ang_row, 1);
        aie::vector<bfloat16, HALF> sin_val = aie::filter_odd(ang_row, 1);
        aie::vector<bfloat16, HALF> y1 = aie::load_v<HALF>(kp);
        aie::vector<bfloat16, HALF> y2 = aie::load_v<HALF>(kp + HALF);
        // Inverse of the forward [y1,y2] = [x1*c-x2*s, x2*c+x1*s]: same cos/sin, cross terms
        // sign-flipped, then divide out the gain. mul() at HALF > 32 lanes returns an accum, not a
        // vector (aie_api's implicit accum->vector conversion is only defined up to 32 lanes).
        aie::vector<bfloat16, HALF> y1_cos = aie::mul(y1, cos_val).template to_vector<bfloat16>();
        aie::vector<bfloat16, HALF> y2_sin = aie::mul(y2, sin_val).template to_vector<bfloat16>();
        aie::vector<bfloat16, HALF> raw1 = aie::add(y1_cos, y2_sin);
        aie::vector<bfloat16, HALF> y2_cos = aie::mul(y2, cos_val).template to_vector<bfloat16>();
        aie::vector<bfloat16, HALF> y1_sin = aie::mul(y1, sin_val).template to_vector<bfloat16>();
        aie::vector<bfloat16, HALF> raw2 = aie::sub(y2_cos, y1_sin);
        aie::vector<bfloat16, HALF> raw1_g = aie::mul(raw1, inv_g1).template to_vector<bfloat16>();
        aie::vector<bfloat16, HALF> raw2_g = aie::mul(raw2, inv_g2).template to_vector<bfloat16>();
        aie::vector<bfloat16, HEAD_DIM> k_row = aie::concat(raw1_g, raw2_g);
        aie::vector<float, HEAD_DIM> sq = aie::mul_square(k_row).template to_vector<float>();
        float sum_sq = aie::reduce_add(sq);
        float inv_rms = aie::invsqrt(sum_sq * inv_head_dim + epsilon);
        aie::accum<accfloat, HEAD_DIM> scaled;
        scaled.from_vector(k_row, 0);
        scaled = aie::mul(scaled.template to_vector<float>(), aie::broadcast<float, HEAD_DIM>(inv_rms));
        aie::store_v(vp, scaled.template to_vector<bfloat16>());
    }
}

} // extern "C"
