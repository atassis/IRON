# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract test for the quantizer's LEVEL RANGE (kernel-contract K025).

The packer's clamp range and the range the unpack path sign-extends are two declarations of one
contract. Each check stands for what their disagreement cost:

  decoder reach       the unpack must return every level the payload can hold. If it cannot, no
                      packer setting can reach the grid and `full_range` is a lie. The scale
                      search is on, because the default `amax/qmax` scale cannot land on a grid
                      whose extreme is the low level -- which is the whole defect.
  grid exactness      a matrix built ON the full 2**n grid must round-trip EXACTLY under
                      full_range. This is the property a QAT checkpoint has and the one whose
                      absence read for three sessions as "four bits is too coarse for this model"
                      -- 4.05-4.33% rel-L2 against 0.186%.
  restricted is worse the same matrix under the default range must be materially worse. Without
                      this the test passes when `full_range` silently stops doing anything.
  default unchanged   `full_range=False` must be byte-identical to omitting it, at every layout,
                      group, scale width and dtype -- the file's standing rule that a new knob
                      defaults to the old bytes.
"""
import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "quant", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common", "quant.py"))
quant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quant)

DTYPES = (("int4", -8, 7), ("int8", -128, 127))


def _roundtrip(W, group, dtype, **kw):
    layout = kw.pop("layout", "row_group_planar")
    sd = kw.pop("scale_dtype", "f32")
    rg = quant.derive_row_group([W.shape[1]], group, dtype, 8, scale_dtype=sd) \
        if layout == "row_group_planar" else 1
    packed = quant.quantize_weight(W, group, dtype, layout=layout, scale_dtype=sd,
                                   row_group=rg, **kw)
    return quant.dequantize_weight(packed, W.shape[0], W.shape[1], group, dtype,
                                   layout=layout, scale_dtype=sd, row_group=rg)


def test_decoder_reaches_every_level():
    for dtype, qlo, qhi in DTYPES:
        K, group = 256, 32
        levels = np.arange(qlo, qhi + 1)
        q = np.resize(levels, (4, K)).astype(np.float32)
        W = q * np.float32(0.0078125)
        R = _roundtrip(W, group, dtype, full_range=True, clip_search=True,
                       clip_range=(0.85, 1.05), n_clip_candidates=81) / np.float32(0.0078125)
        assert set(np.unique(R).astype(int)) >= {qlo, qhi}, \
            f"{dtype}: decoder never returned the endpoints {qlo}/{qhi}"


def test_on_grid_matrix_round_trips_exactly_only_under_full_range():
    rng = np.random.default_rng(0)
    for dtype, qlo, qhi in DTYPES:
        K, group = 512, 32
        q = rng.integers(qlo, qhi + 1, size=(32, K)).astype(np.float32)
        q[:, ::group] = qlo                            # pin the low level into every group
        W = (q * np.float32(0.0078125))                # power-of-two scale: exact in bf16 and f32
        full = _roundtrip(W, group, dtype, full_range=True, clip_search=True,
                          clip_range=(0.85, 1.05), n_clip_candidates=81)
        rest = _roundtrip(W, group, dtype, clip_search=True)
        err_full = np.linalg.norm(W - full) / np.linalg.norm(W)
        err_rest = np.linalg.norm(W - rest) / np.linalg.norm(W)
        assert err_full < 1e-6, f"{dtype}: on-grid matrix not exact under full_range ({err_full})"
        assert err_rest > 10 * max(err_full, 1e-9), \
            f"{dtype}: restricted range is not measurably worse ({err_rest}) -- knob is inert"


def test_full_range_false_is_the_default_bytes():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.02, (64, 1024)).astype(np.float32)
    n = 0
    for dtype, _, _ in DTYPES:
        for group in (32, 64, 128):
            for layout in ("header_first", "row_group_planar"):
                for clip in (False, True):
                    for sd in ("f32", "bf16"):
                        kw = dict(layout=layout, clip_search=clip, scale_dtype=sd)
                        try:
                            a = quant.quantize_weight(W, group, dtype, **kw)
                        except Exception:
                            continue
                        b = quant.quantize_weight(W, group, dtype, full_range=False, **kw)
                        assert np.array_equal(a, b), (dtype, group, layout, clip, sd)
                        n += 1
    assert n >= 24, f"only {n} configs exercised"


if __name__ == "__main__":
    test_decoder_reaches_every_level()
    test_on_grid_matrix_round_trips_exactly_only_under_full_range()
    test_full_range_false_is_the_default_bytes()
    print("quant_level_range: all checks pass")
