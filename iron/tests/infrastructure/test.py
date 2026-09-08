#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`OperatorSequence`'s ``scratch_order`` layout parameter.

``calculate_buffer_layout()`` only needs each operator's ``get_arg_spec()``,
never a build or a device dispatch -- but constructing an ``ElementwiseAdd``
still resolves a target device for its ShimDMA-limit check, so
``aie.utils.set_current_device()`` is primed with an in-memory ``NPU2()``
descriptor up front, and ``pyxrt.device`` is patched to raise for the
duration of each test: a regression that reaches for the real NPU fails
loudly here instead of opening it.

Named ``test.py`` (not ``sequence.py``, unlike its neighbour) and run with
``--noconftest``, so the repo-root ``conftest.py`` -- whose
``pytest_collection_modifyitems`` queries the live device at collection --
never loads:

    pytest --noconftest iron/tests/infrastructure/test.py -v
"""

import pytest
import pyxrt

import aie.utils as aie_utils
from aie.iron.device import NPU2

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators.elementwise_add.op import ElementwiseAdd

# ElementwiseAdd.__post_init__ calls aie_utils.get_current_device(); binding an
# explicit device short-circuits that to a plain attribute read (see
# aie.utils.get_current_device), so it never falls through to the runtime
# probe that would open /dev/accel.
aie_utils.set_current_device(NPU2())

_SIZE = 1024
_TILE = 256
_LEN = _SIZE * 2  # bf16 bytes


@pytest.fixture(autouse=True)
def _forbid_npu_device_access(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("scratch_order tests must not open the NPU device")

    monkeypatch.setattr(pyxrt, "device", _boom)


def _make_sequence(name, scratch_names, ctx, scratch_order=None):
    """``out = (((a+b)+b)+b)+b``, threading through 3 scratch buffers named
    by ``scratch_names`` in runlist (first-appearance) order.
    """
    add = ElementwiseAdd(size=_SIZE, tile_size=_TILE, num_aie_columns=1, context=ctx)
    p, q, r = scratch_names
    return OperatorSequence(
        name=name,
        runlist=[
            (add, "a", "b", p),
            (add, p, "b", q),
            (add, q, "b", r),
            (add, r, "b", "out"),
        ],
        input_args=["a", "b"],
        output_args=["out"],
        context=ctx,
        scratch_order=scratch_order,
    )


def test_scratch_order_reconciles_layouts_across_runlist_orders(tmp_path):
    """Same 3 scratch names, introduced in different runlist orders: offsets
    disagree by default and agree once both sequences share a scratch_order.
    """
    ctx = AIEContext(build_dir=tmp_path)

    seq_pqr = _make_sequence("seq_pqr", ["p", "q", "r"], ctx)
    seq_rpq = _make_sequence("seq_rpq", ["r", "p", "q"], ctx)
    layout_pqr, _, _ = seq_pqr.calculate_buffer_layout()
    layout_rpq, _, _ = seq_rpq.calculate_buffer_layout()
    for name in ("p", "q", "r"):
        assert layout_pqr[name] != layout_rpq[name]

    order = ["p", "q", "r"]
    ordered_pqr = _make_sequence("seq_pqr_o", ["p", "q", "r"], ctx, scratch_order=order)
    ordered_rpq = _make_sequence("seq_rpq_o", ["r", "p", "q"], ctx, scratch_order=order)
    ol_pqr, _, _ = ordered_pqr.calculate_buffer_layout()
    ol_rpq, _, _ = ordered_rpq.calculate_buffer_layout()
    for name in ("p", "q", "r"):
        assert ol_pqr[name] == ol_rpq[name]


def test_scratch_order_none_matches_todays_layout(tmp_path):
    """``scratch_order=None`` reproduces the pre-existing first-appearance,
    zero-padding packing exactly, for a fixed example."""
    ctx = AIEContext(build_dir=tmp_path)
    seq = _make_sequence("seq_default", ["p", "q", "r"], ctx)

    layout, sizes, _ = seq.calculate_buffer_layout()

    assert layout["a"] == ("input", 0, _LEN)
    assert layout["b"] == ("input", _LEN, _LEN)
    assert layout["out"] == ("output", 0, _LEN)
    assert layout["p"] == ("scratch", 0, _LEN)
    assert layout["q"] == ("scratch", _LEN, _LEN)
    assert layout["r"] == ("scratch", 2 * _LEN, _LEN)
    assert sizes == (2 * _LEN, _LEN, 3 * _LEN)


def test_scratch_order_rejects_unknown_name(tmp_path):
    ctx = AIEContext(build_dir=tmp_path)
    seq = _make_sequence("seq_bad_unknown", ["p", "q", "r"], ctx, scratch_order=["nope"])
    with pytest.raises(ValueError) as excinfo:
        seq.calculate_buffer_layout()
    assert "'nope'" in str(excinfo.value)
    assert "not a scratch buffer" in str(excinfo.value)


def test_scratch_order_rejects_input_output_name(tmp_path):
    ctx = AIEContext(build_dir=tmp_path)
    seq = _make_sequence("seq_bad_io", ["p", "q", "r"], ctx, scratch_order=["a"])
    with pytest.raises(ValueError) as excinfo:
        seq.calculate_buffer_layout()
    assert "'a'" in str(excinfo.value)
    assert "input/output" in str(excinfo.value)


def test_scratch_order_rejects_duplicate_name(tmp_path):
    ctx = AIEContext(build_dir=tmp_path)
    seq = _make_sequence("seq_bad_dup", ["p", "q", "r"], ctx, scratch_order=["p", "p"])
    with pytest.raises(ValueError) as excinfo:
        seq.calculate_buffer_layout()
    assert "'p'" in str(excinfo.value)
    assert "duplicate" in str(excinfo.value)
