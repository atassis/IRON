#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Softmax's ring mask (rows_hole), sibling of softmax_row_widths.py's `rows` mode.

Batched prefill past a sliding window's wrap needs a HOLE in the middle, not a suffix: row i's
valid columns are [0, hole_lo) U [hole_hi, width) U [width, cols) excluded -- see
docs/superpowers/specs/2026-09-17-prefill-batch-past-the-sliding-window-design.md sec 1.3/2.1.
`vector_size_source="rows_hole"` streams a (hole_lo, hole_hi, width) int32 triple per row and the
core calls the new `mask_hole_bf16` kernel instead of `mask_bf16`.

Everything here is device-free: the designs are generated and inspected, never compiled to an
xclbin or dispatched (that gate is softmax_row_holes_build.py).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import aie.utils as aie_utils
from aie.iron.device import NPU2

aie_utils.set_current_device(NPU2())

from iron.common import AIEContext  # noqa: E402
from iron.operators.softmax.op import Softmax  # noqa: E402
from iron.operators.softmax.reference import masked_reference_hole, reference  # noqa: E402

ROWS, COLS = 8, 512


def build(**kwargs):
    op = Softmax(
        rows=ROWS,
        cols=COLS,
        context=AIEContext(build_dir="/tmp/softmax_row_holes"),
        **kwargs,
    )
    return op, str(op.get_mlir_artifact().generator())


# --- the new mode's contract --------------------------------------------------


def test_rows_hole_mode_is_a_different_artifact_from_rows_and_default():
    default, _ = build()
    rows, _ = build(vector_size_source="rows")
    hole, _ = build(vector_size_source="rows_hole")
    assert len({default.name, rows.name, hole.name}) == 3
    assert hole.name.endswith("_npu2") and "rows_hole" in hole.name


def test_rows_hole_arg_spec_widths_buffer_is_three_int32_per_row():
    op, _ = build(vector_size_source="rows_hole")
    specs = op.get_arg_spec()
    assert [s.direction for s in specs] == ["in", "in", "out"]
    assert specs[1].shape == (ROWS * 3,)
    assert specs[1].dtype == np.int32


def test_rows_hole_mode_declares_the_new_mask_kernel_not_the_old_one():
    _, mlir = build(vector_size_source="rows_hole")
    assert "func.func private @mask_hole_bf16(memref<512xbf16>, i32, i32, i32, i32)" in mlir
    assert "func.func private @mask_bf16" not in mlir


def test_rows_hole_mode_streams_three_ints_per_row():
    op, mlir = build(vector_size_source="rows_hole")
    assert f"aie.runtime_sequence(%arg0: memref<{ROWS * COLS}xbf16>, " in mlir
    assert f"%arg1: memref<{ROWS * 3}xi32>, " in mlir
    assert "!aie.objectfifo<memref<1xi32>>" in mlir
    core = mlir.split("aie.core(")[1].split("aie.end")[0]
    loop = core.split("scf.for")[2]
    # Three acquires of the shared scalar widths fifo per row, not one.
    assert loop.count("aie.objectfifo.acquire @width_0_0(Consume, 1)") == 3


def test_rows_hole_mode_rejects_a_competing_scalar_source():
    with pytest.raises(ValueError, match="vector_size_parameter must be None"):
        build(vector_size_source="rows_hole", vector_size_parameter="vs")


def test_rows_hole_and_segment_are_not_implemented_together():
    with pytest.raises(ValueError, match="not implemented together"):
        Softmax(rows=ROWS, cols=COLS, vector_size_source="rows_hole", segment=64,
                context=AIEContext(build_dir="/tmp/softmax_row_holes"))


def test_an_unknown_source_is_still_rejected():
    with pytest.raises(ValueError, match="must be None, 'rows' or 'rows_hole'"):
        build(vector_size_source="scalar")


# --- the CPU reference, checked against a hand-built [hole_lo,hole_hi) U [width,cols) mask -----


def _hand_masked_softmax(x, holes):
    """The definition `expand_mask` (docs/probes/prefill_ring_oracle.py) also computes, built
    independently here in torch rather than imported cross-repo: True = attend."""
    cols = x.shape[1]
    pos = torch.arange(cols).unsqueeze(0)
    lo = holes[:, 0].unsqueeze(1)
    hi = holes[:, 1].unsqueeze(1)
    width = holes[:, 2].unsqueeze(1)
    keep = ~(((pos >= lo) & (pos < hi)) | (pos >= width))
    return torch.softmax(x.masked_fill(~keep, float("-inf")), dim=-1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_masked_reference_hole_matches_a_hand_built_mask(seed):
    torch.manual_seed(seed)
    x = torch.rand(ROWS, COLS, dtype=torch.float32) * 4
    # A mix of "below the wrap" (hole_lo==hole_hi, no hole) and "past the wrap" (a real hole) rows.
    holes = torch.tensor([
        [0, 0, 400 + i] if i % 2 == 0 else [100 + i, 300, 500 + i]
        for i in range(ROWS)
    ], dtype=torch.int32)
    got = masked_reference_hole(x, holes)
    want = _hand_masked_softmax(x, holes)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_a_hole_of_zero_width_is_the_suffix_only_mask():
    """hole_lo == hole_hi (below the wrap, design sec 1.3) must reduce to masked_reference's plain
    suffix mask exactly."""
    from iron.operators.softmax.reference import masked_reference

    x = torch.rand(ROWS, COLS, dtype=torch.float32) * 4
    widths = torch.arange(ROWS, dtype=torch.int32) * 50 + 100
    holes = torch.stack([torch.zeros(ROWS, dtype=torch.int32), torch.zeros(ROWS, dtype=torch.int32),
                         widths], dim=1)
    got = masked_reference_hole(x, holes)
    want = masked_reference(x, widths)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_uniform_no_hole_full_width_is_the_unmasked_reference():
    x = torch.rand(ROWS, COLS, dtype=torch.float32) * 4
    holes = torch.tensor([[0, 0, COLS]] * ROWS, dtype=torch.int32)
    got = masked_reference_hole(x, holes)
    torch.testing.assert_close(got, reference(x), rtol=0, atol=0)


def test_the_reference_checks_it_was_given_one_triple_per_row():
    with pytest.raises(ValueError, match="expected 8 hole triples"):
        masked_reference_hole(torch.rand(ROWS, COLS), torch.zeros(3, 3, dtype=torch.int32))


def test_the_reference_rejects_a_width_the_device_would_read_as_garbage():
    x = torch.rand(ROWS, COLS)
    holes = torch.tensor([[0, 0, COLS]] * ROWS, dtype=torch.int32)
    for bad in (0, -1, COLS + 1):
        bad_holes = holes.clone()
        bad_holes[3, 2] = bad
        with pytest.raises(ValueError, match="width must lie in"):
            masked_reference_hole(x, bad_holes)


def test_the_reference_rejects_a_hole_outside_the_row():
    x = torch.rand(ROWS, COLS)
    holes = torch.tensor([[0, 0, COLS]] * ROWS, dtype=torch.int32)
    bad = holes.clone()
    bad[2, 1] = COLS + 1  # hole_hi past cols
    with pytest.raises(ValueError, match=r"hole \[hole_lo, hole_hi\)"):
        masked_reference_hole(x, bad)


def test_op_reference_dispatches_to_the_hole_mask():
    op, _ = build(vector_size_source="rows_hole")
    x = torch.rand(ROWS, COLS, dtype=torch.float32) * 4
    holes = torch.tensor([[0, 0, COLS]] * ROWS, dtype=torch.int32)
    got = op.reference(x.reshape(-1), widths=holes)
    torch.testing.assert_close(got, reference(x), rtol=0, atol=0)
