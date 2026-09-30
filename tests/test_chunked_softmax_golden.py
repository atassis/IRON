# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A chunked softmax must equal the unchunked one it replaces, at any segment count.

Oracle for `Softmax(segment=...)` -- aie_kernels/aie2p/softmax_chunked.cc against
softmax.cc::softmax_simple_bf16, which it must reproduce. Both are modelled in the kernels' own
log2 domain: scores scaled by softmax.cc's LOG2E, the max taken over the SCALED values, exp2
throughout, and the 64-lane f32 accumulator summed in the core's own order.

The bar here is EQUALITY, not a tolerance, and it is reachable because the chunked form runs the
same three passes over the same values -- it only changes what is resident in L1 while they run.
The distance to f64 is reported as a note, not a gate; it belongs to the unchunked op and this
change does not move it.

`aie::exp2` is modelled as exact-then-bf16. It is really a coarse SFU LUT, which is why the f64
column is a floor and not a prediction -- but both arms call it on identical arguments, so the
equality claim survives it.

  python -m pytest tests/test_chunked_softmax_golden.py -v -s
"""
import numpy as np
import pytest
from ml_dtypes import bfloat16

LOG2E = np.float32(1.4453125)   # softmax.cc's own constant, not 1.44269504089
VEC = 64                        # SM_VEC_LEN / FLASH_SM_VEC_LEN

LADDER = [8192, 16384, 32768, 65536, 131072, 262144]


def _bf(x):
    """Exact bfloat16 round trip (round-to-nearest-even), back to f32 for arithmetic."""
    return np.asarray(x, bfloat16).astype(np.float32)


def _scaled(x):
    """`aie::mul(input_bf16, log2e_vec)` -- an accfloat, so the bf16xbf16 product stays f32."""
    return _bf(x) * LOG2E


def _lane_accum(v, acc=None):
    """A 64-lane accfloat summed over `len(v)//64` iterations. Sequential in f32 per lane, which
    is what the core does; numpy's pairwise sum over the flat array would understate the
    accumulation error at the 4096-iteration end of the ladder -- and would not be the same
    number, which is the point."""
    acc = np.zeros(VEC, np.float32) if acc is None else acc
    for row in v.reshape(-1, VEC).astype(np.float32):
        acc = (acc + row).astype(np.float32)
    return acc


def unchunked(x):
    """softmax.cc::softmax_simple_bf16 -- one full-row tile, three walks over it."""
    s = _scaled(x)
    m = np.float32(0.0)                                      # max_val starts at 0, not -inf
    for i in range(0, len(s), VEC):
        m = max(m, np.float32(_bf(s[i:i + VEC]).max()))      # reduce_max over a bf16 vector
    p = _bf(np.exp2(s - _bf(m)))
    inv = _bf(np.float32(1.0) / np.float32(_lane_accum(p).sum(dtype=np.float32)))
    return _bf(p * inv)


def chunked(x, L, width=None):
    """softmax_chunked.cc -- the same three passes, one stream of the row each.

    `width` is the runtime unmasked width the design clamps per segment. A segment wholly past it
    is skipped by all three kernels (and pass 3 writes zeros); the straddling one is -inf-masked.
    """
    n = len(x)
    width = n if width is None else width

    def seg_unmasked(start):
        return max(0, min(width - start, L))

    def seg_in(start):
        """The L1 tile as the kernels see it: DMA'd bytes, then the tail mask if this is the
        straddling segment."""
        t = np.array(x[start:start + L], np.float32)
        u = seg_unmasked(start)
        t[u:] = -np.inf
        return t

    mx = np.float32(0.0)                                     # softmax_segment_init_f32
    lanes = np.zeros(VEC, np.float32)

    for start in range(0, n, L):                             # pass 1: max
        if seg_unmasked(start) <= 0:
            continue
        s = _scaled(seg_in(start))
        for i in range(0, L, VEC):
            mx = max(mx, np.float32(_bf(s[i:i + VEC]).max()))

    for start in range(0, n, L):                             # pass 2: sum
        if seg_unmasked(start) <= 0:
            continue
        lanes = _lane_accum(_bf(np.exp2(_scaled(seg_in(start)) - _bf(mx))), lanes)

    inv = _bf(np.float32(1.0) / np.float32(lanes.sum(dtype=np.float32)))
    out = np.zeros(n, np.float32)
    for start in range(0, n, L):                             # pass 3: normalise and write
        if seg_unmasked(start) <= 0:
            continue                                         # the kernel writes zeros here
        out[start:start + L] = _bf(_bf(np.exp2(_scaled(seg_in(start)) - _bf(mx))) * inv)
    return out


def online_two_pass(x, L):
    """The alternative that was NOT taken: softmax.cc::partial_softmax_f32state_bf16's running
    {max, sum} for pass A, then one apply pass. Two streams of the row instead of three -- but the
    segment sums compose through a bf16 correction factor, and that lands in the DIVISOR, so the
    error it introduces is a uniform scale on the whole row. Measured below."""
    m, l = np.float32(-np.inf), np.float32(0.0)
    for start in range(0, len(x), L):
        seg = _scaled(x[start:start + L])
        seg_max = np.float32(-np.inf)
        for i in range(0, len(seg), VEC):
            seg_max = max(seg_max, np.float32(_bf(seg[i:i + VEC]).max()))
        if seg_max == -np.inf:
            continue
        m_new = np.float32(max(seg_max, m))
        corr = np.float32(0.0) if m == -np.inf else _bf(np.exp2(m - m_new))
        seg_sum = np.float32(_lane_accum(_bf(np.exp2(seg - _bf(m_new)))).sum(dtype=np.float32))
        m, l = m_new, np.float32(l * corr + seg_sum)
    inv = _bf(np.float32(1.0) / l)
    out = np.empty(len(x), np.float32)
    for start in range(0, len(x), L):
        out[start:start + L] = _bf(_bf(np.exp2(_scaled(x[start:start + L]) - _bf(m))) * inv)
    return out


def golden(x):
    """f64 softmax of the bf16 input the device actually receives."""
    keep = np.isfinite(x)
    d = _bf(x[keep]).astype(np.float64)
    e = np.exp(d - d.max())
    out = np.zeros(len(x), np.float64)
    out[keep] = e / e.sum()
    return out


def _rel_l2(got, ref):
    ref = np.asarray(ref, np.float64)
    return float(np.linalg.norm(np.asarray(got, np.float64) - ref) / np.linalg.norm(ref))


def _row(S, seed=0, width=None):
    x = np.random.default_rng(seed).normal(0, 4, S).astype(np.float32)
    if width is not None:
        x[width:] = -np.inf
    return x


@pytest.mark.parametrize("S", LADDER)
@pytest.mark.parametrize("L", [512, 1024, 4096])
def test_chunking_is_bit_identical(S, L):
    x = _row(S)
    u, c = unchunked(x), chunked(x, L)
    ref = golden(x)
    print(f"  S={S:7d} L={L:5d} nseg={S//L:5d}  chunked-vs-unchunked {_rel_l2(c, u):.3e}  "
          f"(note: unchunked-vs-f64 {_rel_l2(u, ref):.3e})")
    assert np.array_equal(c, u), f"chunking moved the answer: rel-L2 {_rel_l2(c, u):.3e}"


@pytest.mark.parametrize("S,L,width", [(32768, 1024, 6912), (65536, 4096, 100),
                                       (262144, 1024, 262143), (8192, 512, 513),
                                       (16384, 1024, 1024)])
def test_masked_tail_and_wholly_masked_segments(S, L, width):
    """A build-constant segment count means segments past the runtime width are wholly -inf. All
    three kernels return early on those and pass 3 writes zeros; this is the arithmetic that has
    to survive it, against the unchunked op masking the same positions."""
    x = _row(S, width=width)
    u, c = unchunked(x), chunked(x, L, width=width)
    print(f"  S={S:7d} L={L:5d} width={width:6d} skipped-segs={S//L - -(-width//L):4d}  "
          f"chunked-vs-unchunked {_rel_l2(c, u):.3e}  chunked-vs-f64 {_rel_l2(c, golden(x)):.3e}")
    assert np.array_equal(c, u), "masked chunking moved the answer"
    assert np.abs(c[width:]).max() == 0.0, "masked tail is not exactly zero"


@pytest.mark.parametrize("seed", range(20))
def test_bit_identity_holds_across_seeds(seed):
    x = _row(65536, seed=seed)
    assert np.array_equal(chunked(x, 1024), unchunked(x))


def test_adversarial_shapes():
    """A monotone ramp raises the max in EVERY segment -- the worst case for any form that carries
    a running max. Bit-identity does not care, which is the argument for three passes."""
    S, L = 65536, 1024
    cases = {
        "ramp": np.linspace(-40, 40, S).astype(np.float32),
        "reverse-ramp": np.linspace(40, -40, S).astype(np.float32),
        "spike-at-end": np.concatenate(
            [np.random.default_rng(3).normal(0, 4, S - 1), [60.0]]).astype(np.float32),
        "all-equal": np.full(S, 3.5, np.float32),
    }
    for name, x in cases.items():
        u, c = unchunked(x), chunked(x, L)
        print(f"  {name:14s} chunked-vs-unchunked {_rel_l2(c, u):.3e}  "
              f"unchunked-vs-f64 {_rel_l2(u, golden(x)):.3e}")
        assert np.array_equal(c, u), name


def test_what_the_two_pass_alternative_would_have_cost():
    """Why the design streams the row three times rather than two.

    The online form saves one read of the scores. It pays for it in the divisor: `l` is composed
    across segments through a bf16 `exp2(m_prev - m_new)`, whose relative spacing is 2^-8, and a
    one-ULP miss there scales the entire row. Printed per seed, worst reported -- this is the
    number the choice rests on, and it is a MEASUREMENT of the alternative, not of what shipped.
    """
    worst_two, worst_three = 0.0, 0.0
    for seed in range(30):
        x = _row(65536, seed=seed)
        u = unchunked(x)
        worst_two = max(worst_two, _rel_l2(online_two_pass(x, 1024), u))
        worst_three = max(worst_three, _rel_l2(chunked(x, 1024), u))
    print(f"  S=65536 L=1024, 30 seeds: worst two-pass-vs-unchunked {worst_two:.3e}, "
          f"worst three-pass-vs-unchunked {worst_three:.3e}")
    assert worst_three == 0.0
    assert worst_two > 0.0, "the two-pass form was expected to drift; re-price the third read"
