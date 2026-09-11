# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-K online softmax must equal a full-row softmax, and the STATE dtype is what decides
whether it still does at a 32k window.

This is the oracle for aie_kernels/aie2p/softmax.cc::partial_softmax_f32state_bf16. It models what
the kernel actually does, in the kernel's own log2 domain: scores are scaled by log2e, the max is
taken over the SCALED values, and exp2 is used throughout -- not exp on unscaled scores.

  python -m pytest tests/test_split_k_golden.py -v -s
"""
import numpy as np
import pytest
from ml_dtypes import bfloat16

LOG2E = np.float32(1.4453125)   # softmax.cc's own constant, not 1.4426950408889634


def _bf(x):
    """Exact bfloat16 round trip (round-to-nearest-even), back to f32 for arithmetic."""
    return np.asarray(x, bfloat16).astype(np.float32)


def split_k_context(scores, V, L, state_bf16):
    """Online softmax over segments of L positions, context accumulated in f32.

    scores [S] f32 (pre-scaling), V [S, HD] f32 -> [HD] f32.

    `state_bf16` picks what mha.cc's scale_buffer does (True: running max AND running sum in
    bfloat16) versus what partial_softmax_f32state_bf16 does (False: both f32). The probability
    row `p` is bf16 in BOTH arms -- that is the sw buffer and it is bf16 either way, so this
    isolates the STATE dtype and nothing else.
    """
    st = _bf if state_bf16 else (lambda x: np.float32(x))
    S, HD = V.shape
    m = np.float32(-np.inf)
    l = np.float32(0.0)
    acc = np.zeros(HD, np.float32)

    for start in range(0, S, L):
        seg = (scores[start:start + L] * LOG2E).astype(np.float32)
        m_new = st(max(float(m), float(seg.max())))
        corr = np.float32(0.0) if not np.isfinite(m) else st(np.exp2(np.float32(m) - np.float32(m_new)))
        p = _bf(np.exp2(seg - np.float32(m_new)))          # sw is bf16 in the real kernel
        acc = acc * np.float32(corr) + p @ V[start:start + L]
        l = st(np.float32(l) * np.float32(corr) + np.float32(p.sum()))
        m = np.float32(m_new)

    return acc / np.float32(l)


def full_row_context(scores, V):
    s = (scores * LOG2E).astype(np.float32)
    p = _bf(np.exp2(s - s.max()))
    return (p @ V) / np.float32(p.sum())


def _case(S, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 4, S).astype(np.float32),
            rng.normal(0, 1, (S, 128)).astype(np.float32))


def _rel_l2(got, ref):
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


@pytest.mark.parametrize("S,L", [(4096, 2048), (8192, 2048), (32768, 2048), (32768, 4096)])
def test_f32_state_matches_full_row(S, L):
    """The split must be algebraically invisible. bf16 `p` sets the floor, so this compares
    against a full-row reference that ALSO rounds p to bf16 -- otherwise the test would be
    measuring bf16 rounding rather than the split."""
    scores, V = _case(S)
    rel = _rel_l2(split_k_context(scores, V, L, state_bf16=False), full_row_context(scores, V))
    print(f"  S={S:6d} L={L:5d} f32-state rel-L2 {rel:.3e}")
    assert rel < 1e-3, f"split-K with f32 state diverged: rel-L2 {rel:.3e}"


def test_bf16_state_cost_is_measured_not_assumed():
    """What mha.cc's bf16 scale_buffer costs, across the window sizes that matter. This is the
    number the f32-state decision rests on; it is printed, and the assert only guards the claim
    actually made in the contract header -- that bf16 state is worse at a 32k window."""
    rows = []
    for S in (4096, 8192, 32768):
        scores, V = _case(S)
        ref = full_row_context(scores, V)
        f32 = _rel_l2(split_k_context(scores, V, 2048, state_bf16=False), ref)
        b16 = _rel_l2(split_k_context(scores, V, 2048, state_bf16=True), ref)
        rows.append((S, f32, b16))
        print(f"  S={S:6d}  f32-state {f32:.3e}   bf16-state {b16:.3e}   ratio {b16/max(f32,1e-12):.1f}x")
    s32 = [r for r in rows if r[0] == 32768][0]
    assert s32[2] > s32[1], "bf16 state was not worse at 32k -- the contract header's claim is wrong"
