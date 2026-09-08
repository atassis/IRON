#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Softmax's per-row mask widths, and the guarantee that opting out changes nothing.

Batched prefill needs a different causal width per row: row ``i`` of a chunk at
absolute position ``base`` may only attend positions ``<= base + i``.
``vector_size_source="rows"`` streams one int32 per row alongside the input and
the core reads it inside the tile loop, instead of hoisting one scalar width
before it.

Everything here is device-free: the designs are generated and inspected, never
compiled to an xclbin or dispatched.
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
from iron.operators.softmax.reference import (  # noqa: E402
    generate_golden_widths,
    masked_reference,
    reference,
)

# The golden below was generated from this exact shape, so keep them together.
ROWS, COLS = 8, 512
GOLDEN = Path(__file__).parent / "softmax_default_r8_n512_c1_ch1_npu2.mlir"


def build(**kwargs):
    """A Softmax and the MLIR it generates, straight through op.py's wiring."""
    op = Softmax(
        rows=ROWS,
        cols=COLS,
        context=AIEContext(build_dir="/tmp/softmax_row_widths"),
        **kwargs,
    )
    return op, str(op.get_mlir_artifact().generator())


# --- opting out must change nothing ------------------------------------------


def test_default_mode_reproduces_the_pre_change_mlir():
    """The default path is byte-for-byte what it was before row widths existed.

    The golden was generated from a pristine ``git archive`` of the commit that
    introduced this test, so it genuinely predates the feature. A shipped 28-layer
    artifact rides this path. If a toolchain bump legitimately reformats the IR,
    regenerate the golden from the parent commit -- do not relax the comparison.
    """
    _, mlir = build()
    assert mlir == GOLDEN.read_text()


def test_default_mode_keeps_the_artifact_name():
    """``vector_size_source`` defaults to None, and ``MLIROperator.name`` filters
    fields that are None, so cached artifacts stay valid."""
    op, _ = build()
    assert op.name == f"Softmax_r{ROWS}_n{COLS}_c1_ch1_npu2"


def test_default_mode_keeps_two_arguments():
    op, _ = build()
    assert [(s.direction, s.shape) for s in op.get_arg_spec()] == [
        ("in", (ROWS * COLS,)),
        ("out", (ROWS * COLS,)),
    ]


def test_rows_mode_is_a_different_artifact():
    """A design-affecting flag absent from the artifact key is how two different
    designs come to share one xclbin."""
    default, _ = build()
    rows, _ = build(vector_size_source="rows")
    assert rows.name != default.name
    assert rows.name.endswith("_npu2") and "rows" in rows.name


# --- the new mode's contract --------------------------------------------------


def test_rows_mode_arg_spec_inserts_widths_before_the_output():
    """OperatorSequence splits a step as ``*in_specs, out_spec``, so a widths
    spec appended last would be mistaken for the operator's output."""
    op, _ = build(vector_size_source="rows")
    specs = op.get_arg_spec()
    assert [s.direction for s in specs] == ["in", "in", "out"]
    assert specs[1].shape == (ROWS,)
    assert specs[-1].direction == "out"


def test_rows_mode_widths_buffer_is_int32_sized():
    """AIERuntimeArgSpec defaults to bfloat16 and calculate_buffer_layout sizes
    the host buffer off spec.dtype, so an unset dtype under-allocates by 2x."""
    op, _ = build(vector_size_source="rows")
    widths = op.get_arg_spec()[1]
    assert widths.dtype == np.int32
    assert int(np.prod(widths.shape) * np.dtype(widths.dtype).itemsize) == ROWS * 4


def test_rows_mode_streams_one_int32_per_row():
    op, mlir = build(vector_size_source="rows")
    assert f"aie.runtime_sequence(%arg0: memref<{ROWS * COLS}xbf16>, " in mlir
    assert f"%arg1: memref<{ROWS}xi32>, " in mlir
    assert "!aie.objectfifo<memref<1xi32>>" in mlir
    # The width is loaded per tile-loop iteration, not hoisted above the loop.
    core = mlir.split("aie.core(")[1].split("aie.end")[0]
    loop = core.split("scf.for")[2]
    assert "aie.objectfifo.acquire @width_0_0(Consume, 1)" in loop
    assert "memref.load" in loop


def test_rows_mode_reuses_the_existing_mask_entry_point():
    """No kernel change: mask_bf16 already takes the width as an i32 argument."""
    _, mlir = build(vector_size_source="rows")
    assert "func.func private @mask_bf16(memref<512xbf16>, i32, i32)" in mlir
    assert "mask_rows" not in mlir


def test_rows_mode_drops_the_scalar_width_plumbing():
    """The width no longer comes from a write-RTP buffer, so none is emitted."""
    _, mlir = build(vector_size_source="rows")
    assert "rtp_0_0" not in mlir
    assert "aiex.npu.rtp_write" not in mlir


def test_widths_are_split_across_cores_the_same_way_the_rows_are():
    """Core k must receive the widths of exactly the rows core k computes."""
    op = Softmax(
        rows=8,
        cols=COLS,
        num_aie_columns=2,
        num_channels=2,
        vector_size_source="rows",
        context=AIEContext(build_dir="/tmp/softmax_row_widths"),
    )
    mlir = str(op.get_mlir_artifact().generator())
    widths_bds = [
        line
        for line in mlir.splitlines()
        if "xi32> offset" in line and "dma_bd" in line
    ]
    assert len(widths_bds) == 4
    offsets = sorted(int(line.split("offset = ")[1].split()[0]) for line in widths_bds)
    assert offsets == [0, 2, 4, 6]  # 8 rows / 4 cores = 2 rows each
    assert all("len = 2 " in line for line in widths_bds)


# --- validation ---------------------------------------------------------------


def test_an_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="must be None or 'rows'"):
        build(vector_size_source="scalar")


def test_rows_mode_rejects_a_competing_scalar_source():
    with pytest.raises(ValueError, match="vector_size_parameter must be None"):
        build(vector_size_source="rows", vector_size_parameter="vs")


# --- the CPU reference --------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_uniform_widths_are_the_scalar_mode(dtype):
    """A width vector of all-equal values must equal the unmasked reference."""
    x = (torch.rand(ROWS, COLS, dtype=torch.float32) * 4).to(dtype)
    full = masked_reference(x, torch.full((ROWS,), COLS, dtype=torch.int32))
    torch.testing.assert_close(full, reference(x), rtol=0, atol=0)


def test_masked_rows_match_a_hand_built_causal_softmax():
    x = torch.rand(ROWS, COLS, dtype=torch.float32) * 4
    widths = generate_golden_widths(ROWS, COLS, base=COLS - ROWS)
    got = masked_reference(x, widths)
    for i in range(ROWS):
        w = int(widths[i])
        torch.testing.assert_close(got[i, :w], torch.softmax(x[i, :w], dim=-1))
        assert torch.all(got[i, w:] == 0)


def test_the_reference_needs_the_widths_it_is_referencing():
    op, _ = build(vector_size_source="rows")
    with pytest.raises(ValueError, match="needs the per-row widths"):
        op.reference(torch.zeros(ROWS * COLS))


def test_the_reference_rejects_widths_the_device_would_read_as_garbage():
    """mask_bf16 loops ``for (i = width; i < total; i++)``: a negative width walks
    off the front of the tile and a zero width leaves the row all -inf. Neither is
    checkable on device -- the widths arrive as data -- so the reference refuses
    them rather than modelling them."""
    x = torch.rand(ROWS, COLS)
    for bad in (0, -1, COLS + 1):
        widths = torch.full((ROWS,), COLS, dtype=torch.int32)
        widths[3] = bad
        with pytest.raises(ValueError, match="widths must lie in"):
            masked_reference(x, widths)


def test_the_reference_checks_it_was_given_one_width_per_row():
    with pytest.raises(ValueError, match="expected 8 widths"):
        masked_reference(torch.rand(ROWS, COLS), torch.ones(3, dtype=torch.int32))
