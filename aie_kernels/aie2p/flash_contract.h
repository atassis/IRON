// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// What split-K flash attention in attn_block_dp depends on, asserted rather than assumed.
//
// Doctrine trigger [1], "on adoption": first use of a vendor type/API pins its byte size,
// alignment, units and AMBIENT-STATE requirements. The defects this file exists to prevent are the
// ones with no digits -- sizeof(bfp16ebs8) differing 1-vs-9 across backends, and a rounding mode
// living in a core register that nothing declares.
#pragma once

#include <stdint.h>
#include <aie_api/aie.hpp>

// ---- 1. Ambient core state -------------------------------------------------------------------
//
// aie_api NEVER sets the rounding register and the documented default is FLOOR. Every kernel here
// that converts to bf16 must call ::aie::set_rounding(conv_even) itself. Measured cost of getting
// this wrong on the fused decode: a softmax row summing to 0.995138 instead of 1, CONSTANT across
// widths 2..32 -- a per-element bias, which is what ruled out the reduction as the cause.
// There is no compile-time check for a register; this constant exists so the requirement has a
// name and one definition rather than being re-spelled per kernel.
#define FLASH_ROUNDING_MODE aie::rounding_mode::conv_even

// ---- 2. Vector-loop divisibility -------------------------------------------------------------
//
// The softmax loops step SM_VEC_LEN elements with no scalar tail, so a segment length that is not
// a whole number of vectors silently drops its remainder -- a wrong answer, not a fault.
// attn_block_dp enforces this in Python via lcm(stream-tile rows, kv block, FLASH_SM_VEC_LEN);
// this is the second copy, on the side that would actually corrupt.
#define FLASH_SM_VEC_LEN 64

// ---- 3. Type widths --------------------------------------------------------------------------
//
// The running state is f32 DELIBERATELY. mha.cc's scale_buffer keeps the running sum `l` in
// bfloat16, which is ~8 mantissa bits accumulating up to 32768 addends at a full window; that is
// fine at mha's block sizes and is not fine here. See tests/test_split_k_golden.py for the number.
static_assert(sizeof(bfloat16) == 2, "bf16 score/context buffers assume a 2-byte element");
static_assert(sizeof(float) == 4, "flash running state assumes a 4-byte f32 element");

// ---- 4. What we deliberately do NOT use ------------------------------------------------------
//
// mha.cc::rescale_O and mha.cc::matmul_PV are NOT reusable here and must not be wired in later by
// someone reading only their names. Both hardcode a 64x64 bf16 output tile -- `O + j*64 + k*8 +
// l*512` with all three loops fixed at 8 -- and carry their own "TODO: Make this generic for every
// tile size". attn_block_dp's context accumulator is [gqa][head_dim] in f32, so neither the layout
// nor the dtype matches. acc_rescale_f32 in softmax.cc is the replacement.
//
// mha.cc's 4-band scale_buffer protocol ([m_prev][m_i][l][segment-sum -> correction], each B_q
// wide) is likewise not what this design speaks. Our state is two f32 words per group.

// Running state, one per (core, gqa group): [0] running max, [1] running sum.
#define FLASH_STATE_WORDS 2
