# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Oracle for `Softmax(segment=L, vector_size_source="rows")` -- the composition op.py used to
refuse. Arithmetically this is `test_chunked_softmax_golden.py`'s per-row `chunked()`/`unchunked()`
applied to a BATCH of rows, each at its own causal width, because every state the design carries
(`mx`, `lanes`, and now the width itself) is reset/re-read per row -- row i's computation never
reads row j's state. See design.py's `_softmax_chunked` docstring for the wiring.

`rows=64` matches gen_llm_prefill.py's shipped batch M=64 (scenarios/generate-gemma4-12b.toml:283),
not the op's own `rows=Hq*M=1024` -- the real call tiles these same M widths across Hq=16 heads
(`causal_widths()`'s `np.tile(w, heads)`), and a repeated width is not a new arithmetic case.

  python -m pytest tests/test_chunked_causal_softmax_golden.py -v -s
"""
import numpy as np
import pytest

from tests.test_chunked_softmax_golden import chunked, unchunked, _rel_l2

from iron.operators.softmax.reference import generate_golden_widths

M_BATCH = 64  # scenarios/generate-gemma4-12b.toml:283


def _causal_batch(rows, cols, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 4, (rows, cols)).astype(np.float32)
    widths = generate_golden_widths(rows, cols).numpy()
    for i in range(rows):
        X[i, widths[i]:] = -np.inf
    return X, widths


@pytest.mark.parametrize("cols,L", [(262144, 1024), (262144, 4096), (65536, 1024), (8192, 512)])
def test_causal_batch_is_bit_identical(cols, L):
    """rows=M_BATCH, one chunk at base=0: widths 1..64. Every row's chunked composition must
    equal that row's unchunked masked softmax exactly."""
    rows = M_BATCH
    X, widths = _causal_batch(rows, cols)
    worst = 0.0
    for i in range(rows):
        u = unchunked(X[i])
        c = chunked(X[i], L, width=int(widths[i]))
        worst = max(worst, _rel_l2(c, u))
        assert np.array_equal(c, u), f"row {i} (width={widths[i]}) diverged"
    print(f"  cols={cols:7d} L={L:5d} rows={rows}: worst chunked-vs-unchunked rel-L2 {worst:.3e}")


def test_causal_batch_mid_chunk_base():
    """A chunk that does NOT start at position 0 -- base=6848, the tail of a 6912-token prompt at
    M_BATCH=64, so widths span [6849, 6912], all << cols. Exercises the wholly-past-mask segments
    the ramp at base=0 never reaches for a 262144-wide row."""
    rows, cols, L, base = M_BATCH, 262144, 1024, 6912 - M_BATCH
    rng = np.random.default_rng(1)
    X = rng.normal(0, 4, (rows, cols)).astype(np.float32)
    widths = generate_golden_widths(rows, cols, base=base).numpy()
    for i in range(rows):
        X[i, widths[i]:] = -np.inf
        u = unchunked(X[i])
        c = chunked(X[i], L, width=int(widths[i]))
        assert np.array_equal(c, u), f"row {i} (width={widths[i]}) diverged"


def test_causal_batch_adversarial_values():
    """The monotone-ramp/spike cases from the unmasked golden test, now under a real per-row
    causal width instead of the full row -- the case most likely to expose a width mixed up
    between rows, since every row's max differs."""
    rows, cols, L = M_BATCH, 262144, 1024
    widths = generate_golden_widths(rows, cols).numpy()
    ramp = np.linspace(-40, 40, cols).astype(np.float32)
    for i in range(rows):
        x = ramp.copy()
        x[int(widths[i]):] = -np.inf
        u = unchunked(x)
        c = chunked(x, L, width=int(widths[i]))
        assert np.array_equal(c, u), f"row {i} (width={widths[i]}) diverged"
