#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure tests for :class:`OperatorSequence`.

This is the first test module under ``iron/tests/`` and exercises the
sequencing infrastructure itself (dispatch-mode selection, fused-MLIR
generation, cross-mode output parity and the compare-mode self-check) rather
than any single operator.

The ``OperatorSequence`` dispatch modes covered here are:

* ``"auto"``     – picks ``"fused"`` on NPU2 (Strix) and ``"separate"`` on
                   NPU1 (Phoenix).
* ``"fused"``    – single-ELF dispatch (``aiex.configure`` / ``aiex.run``),
                   NPU2 only.
* ``"separate"`` – one xclbin per operator, chained (works on all platforms).
* ``"compare"``  – ``"separate"`` NPU path plus a per-step CPU-reference check.
* ``"reference"``– pure-CPU evaluation via each operator's ``reference()``.
"""

from pathlib import Path

import re

import pytest
import torch

import aie.utils as aie_utils
from aie.iron.device import NPU2

from iron.common.sequence import OperatorSequence
from iron.common.compilation.sequence import fuse_mlir
from iron.common.test_utils import verify_buffer
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.relu.op import ReLU


def _set_input(run, name, data):
    """Write a host tensor into an input buffer and push it to the device.

    Mirrors the caller contract for the fused single-ELF callable: after
    writing a get_buffer() sub-view via torch_view(), the caller is responsible
    for calling .to("npu") so the write reaches the NPU (a no-op sync for the
    separate/reference callables, whose __call__ syncs inputs themselves).
    """
    buf = run.get_buffer(name)
    buf.torch_view()[: data.numel()] = data.reshape(-1)
    buf.to("npu")


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_ADD_RELU_SIZE = 4096
_ADD_RELU_TILE = 1024
_ADD_RELU_COLS = 4


def _build_add_relu_sequence(context, dispatch, name):
    """out = relu(a + b), as a 2-step OperatorSequence."""
    add = ElementwiseAdd(
        size=_ADD_RELU_SIZE,
        tile_size=_ADD_RELU_TILE,
        num_aie_columns=_ADD_RELU_COLS,
        context=context,
    )
    relu = ReLU(
        size=_ADD_RELU_SIZE,
        num_aie_columns=_ADD_RELU_COLS,
        num_channels=1,
        tile_size=_ADD_RELU_TILE,
        context=context,
    )
    return OperatorSequence(
        name=name,
        runlist=[
            (add, "a", "b", "temp"),
            (relu, "temp", "out"),
        ],
        input_args=["a", "b"],
        output_args=["out"],
        dispatch=dispatch,
        context=context,
    )


# ---------------------------------------------------------------------------
# 1. Auto dispatch selects the platform default and runs correctly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [_ADD_RELU_SIZE])
def test_auto_dispatch_selects_platform_default(size, aie_context):
    """``dispatch="auto"`` must resolve to the full-ELF mode on Strix and to
    the separate-xclbin mode on Phoenix, and produce the correct result on
    whichever platform the test runs on."""
    torch.manual_seed(0)
    a = torch.rand(size, dtype=torch.bfloat16) * 4 - 2
    b = torch.rand(size, dtype=torch.bfloat16) * 4 - 2

    seq = _build_add_relu_sequence(aie_context, "auto", "infra_auto_add_relu")
    seq.compile()

    expected_mode = (
        "fused" if isinstance(aie_utils.get_current_device(), NPU2) else "separate"
    )
    assert seq._dispatch.name == expected_mode, (
        f"auto dispatch resolved to {seq._dispatch.name!r}, expected "
        f"{expected_mode!r} on this device"
    )

    run = seq.get_callable()
    _set_input(run, "a", a)
    _set_input(run, "b", b)
    run()
    out = run.get_buffer("out").torch_view()[:size].clone()

    expected = torch.nn.functional.relu(a + b)
    errors = verify_buffer(out, "out", expected, rel_tol=0.04, abs_tol=1e-6)
    assert not errors, f"auto-dispatch sequence produced {len(errors)} mismatches"


# ---------------------------------------------------------------------------
# 2. Compilation-only: the fused single-ELF MLIR is well formed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sequence", ["add_relu"])
def test_fused_mlir_contains_reconfiguration(sequence, aie_context, tmp_path):
    """The single-dispatch (fused) path emits one ``aie.device`` per operator
    plus a top-level device whose runtime sequence reconfigures the array
    between operators via ``aiex.configure`` / ``aiex.run``.

    Only the *generated MLIR* is inspected here (no ELF backend is invoked),
    so the check is device-agnostic and runs on all platforms even though the
    full fused dispatch itself requires NPU2.
    """
    seq = _build_add_relu_sequence(aie_context, "fused", "infra_fused_mlir")

    # Generate the fused MLIR directly, bypassing the ELF backend (which is
    # NPU2-only). This mirrors what set_up_artifacts() feeds to the compiler.
    seq.subbuffer_layout, seq.buffer_sizes, seq.slice_info = (
        seq.calculate_buffer_layout()
    )
    mlir_artifact = seq._dispatch.build_fused_mlir(seq)
    mlir_artifact.filename = str(tmp_path / mlir_artifact.filename)
    fuse_mlir(mlir_artifact)

    text = Path(mlir_artifact.filename).read_text()

    # Reconfiguration + dispatch ops between temporal steps.
    assert "aiex.configure" in text, "missing aiex.configure in fused MLIR"
    assert "aiex.run @sequence" in text, "missing aiex.run in fused MLIR"
    # Typed views of the byte arena handed to each operator's runtime sequence. memref.view,
    # not reinterpret_cast: the arena is i8 and the element type changes here, which
    # reinterpret_cast cannot express.
    assert "memref.view" in text, "missing buffer view in fused MLIR"
    # One inlined device per unique operator plus the top-level driver device.
    assert (
        "op0_ElementwiseAdd" in text and "op1_ReLU" in text
    ), "operator devices not inlined into fused module"
    assert (
        text.count("aie.device") >= 3
    ), "expected two operator devices plus a top-level device"


# ---------------------------------------------------------------------------
# 2b. extra_runlists emits one NAMED runtime sequence per variant, over one arena.
# ---------------------------------------------------------------------------


def _variant_sequence(context, name, with_variant):
    """Two runlists that differ only in WHICH design runs the second step.

    That is the shape a per-dispatch selection needs: same buffers, same arena, a different
    design per variant. `with_variant=False` is the negative control -- the same construction
    with the variant withheld, so the assertions below can fail.
    """
    add = ElementwiseAdd(
        size=_ADD_RELU_SIZE,
        tile_size=_ADD_RELU_TILE,
        num_aie_columns=_ADD_RELU_COLS,
        context=context,
    )
    relu_wide = ReLU(
        size=_ADD_RELU_SIZE,
        num_aie_columns=_ADD_RELU_COLS,
        num_channels=1,
        tile_size=_ADD_RELU_TILE,
        context=context,
    )
    relu_narrow = ReLU(
        size=_ADD_RELU_SIZE,
        num_aie_columns=_ADD_RELU_COLS,
        num_channels=1,
        tile_size=_ADD_RELU_TILE // 2,
        context=context,
    )
    base = [(add, "a", "b", "temp"), (relu_wide, "temp", "out")]
    extra = (
        {"sequence_narrow": [(add, "a", "b", "temp"), (relu_narrow, "temp", "out")]}
        if with_variant
        else None
    )
    return OperatorSequence(
        name=name,
        runlist=base,
        extra_runlists=extra,
        input_args=["a", "b"],
        output_args=["out"],
        dispatch="fused",
        context=context,
    )


def _fused_text(seq, tmp_path):
    seq.subbuffer_layout, seq.buffer_sizes, seq.slice_info = (
        seq.calculate_buffer_layout()
    )
    art = seq._dispatch.build_fused_mlir(seq)
    art.filename = str(tmp_path / art.filename)
    fuse_mlir(art)
    return Path(art.filename).read_text(), seq


@pytest.mark.parametrize("with_variant", [True, False])
def test_extra_runlists_emit_named_sequences(with_variant, aie_context, tmp_path):
    """A variant becomes its own named `aie.runtime_sequence` in the `main` device, which is
    what aiecc turns into a second control code and XRT resolves as `main:<name>`.

    The control (`with_variant=False`) is the same construction with the variant withheld: it
    must produce exactly one sequence and NOT mention the variant's name anywhere, so a change
    that silently ignored `extra_runlists` fails here rather than passing quietly.
    """
    seq = _variant_sequence(aie_context, f"infra_variants_{with_variant}", with_variant)
    text, seq = _fused_text(seq, tmp_path)

    # The MLIR printer ELIDES the symbol when it equals the assembly-format default, so the
    # `sequence` variant prints bare and only a non-default variant shows an `@name`. That is
    # also why every shipped artifact's `kernel_name: main:sequence` resolves against a fused
    # module whose top-level sequence carries no visible symbol -- do not read the bare form as
    # "unnamed". Top-level sequences are the ones taking the i8 arena.
    top_level = re.findall(r"aie\.runtime_sequence\s*(@\w+)?\(%arg0: memref<\d+xi8>", text)
    if not with_variant:
        assert "sequence_narrow" not in text, "control leaked a variant it was never given"
        assert len(top_level) == 1, f"control emitted {len(top_level)} top-level sequences"
        # The narrow design must not be built at all when no variant references it.
        assert len(seq.unique_designs()[0]) == 2
        return

    assert len(top_level) == 2, f"expected two top-level sequences, got {top_level}"
    assert top_level[0] is None or top_level[0] == "", "default variant must print bare"
    assert "@sequence_narrow" in top_level, f"variant sequence missing: {top_level}"
    # Three designs: the shared add, and one ReLU per variant.
    assert len(seq.unique_designs()[0]) == 3, "variant did not contribute its own design"
    # ONE arena for both. A buffer named by both variants keeps a single offset, which is why a
    # host may switch variants between dispatches without moving data.
    assert set(seq.subbuffer_layout) >= {"a", "b", "temp", "out"}
    # Each variant configures its OWN second-step device, so the two sequences are not copies.
    devices = re.findall(r"aiex\.configure @(\w+)", text)
    assert len(set(devices)) >= 3, f"variants share every device: {sorted(set(devices))}"


# ---------------------------------------------------------------------------
# 3. Every NPU dispatch mode produces bit-identical output.
# ---------------------------------------------------------------------------


def _run_add_relu(context, dispatch, a, b, name):
    """out = relu(a + b), returned as a host bf16 tensor."""
    seq = _build_add_relu_sequence(context, dispatch, name)
    seq.compile()
    run = seq.get_callable()
    _set_input(run, "a", a)
    _set_input(run, "b", b)
    run()
    return run.get_buffer("out").torch_view()[:_ADD_RELU_SIZE].clone()


@pytest.mark.parametrize("dispatch", ["separate", "fused", "compare"])
def test_dispatch_modes_bit_identical(dispatch, aie_context):
    """add -> relu must yield byte-for-byte identical output across every NPU
    dispatch mode: the compiled kernels are the same, so only the dispatch
    mechanism differs. The ``separate`` mode is the baseline (it runs on every
    platform)."""
    if dispatch == "fused" and not isinstance(aie_utils.get_current_device(), NPU2):
        pytest.skip("fused (single-ELF) dispatch requires NPU2")

    torch.manual_seed(0)
    a = torch.rand(_ADD_RELU_SIZE, dtype=torch.bfloat16) * 4 - 2
    b = torch.rand(_ADD_RELU_SIZE, dtype=torch.bfloat16) * 4 - 2

    baseline = _run_add_relu(
        aie_context, "separate", a, b, "infra_addrelu_parity_separate"
    )
    out = _run_add_relu(aie_context, dispatch, a, b, f"infra_addrelu_parity_{dispatch}")

    assert torch.equal(out, baseline), (
        f"dispatch={dispatch!r} output is not bit-identical to the separate "
        f"baseline"
    )


# ---------------------------------------------------------------------------
# 4. Compare mode flags (and by default raises on) a per-step reference/NPU
#    mismatch on its own.
#
#    Normally the reference is trusted and the NPU kernel is the suspect; here
#    we invert that (keep the NPU correct, vary the reference) because it is
#    easier to inject a known-wrong reference than a known-wrong kernel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reference_is_correct", [True, False])
def test_compare_mode_detects_wrong_reference(reference_is_correct, aie_context):
    """dispatch="compare" runs the NPU pipeline and, per step, re-runs the
    operator's ``reference()`` on the same NPU inputs. A correct reference must
    run cleanly (no flagged step); a wrong one must make compare mode raise on
    its own (``compare_raise_on_mismatch`` defaults to True)."""
    size = 256
    torch.manual_seed(0)
    a = torch.rand(size, dtype=torch.bfloat16)
    b = torch.rand(size, dtype=torch.bfloat16)

    op = ElementwiseAdd(
        size=size, tile_size=256, num_aie_columns=1, context=aie_context
    )
    if not reference_is_correct:
        # Override the reference on this instance to disagree with the NPU
        # kernel (which computes a + b). Keeping the real ElementwiseAdd class
        # leaves its name/compilation intact for the xclbin compare path.
        op.reference = lambda a, b: a + b + 1.0

    seq = OperatorSequence(
        name="infra_compare_add",
        runlist=[(op, "a", "b", "out")],
        input_args=["a", "b"],
        output_args=["out"],
        dispatch="compare",
        context=aie_context,
    )
    seq.compile()
    assert seq._dispatch.name == "compare"

    run = seq.get_callable()
    _set_input(run, "a", a)
    _set_input(run, "b", b)

    if reference_is_correct:
        run()  # must not raise
        flagged = any(step.get("mismatch") for step in run.last_step_stats)
        assert not flagged, "compare mode should not flag a matching reference"
    else:
        with pytest.raises(RuntimeError):
            run()  # compare mode reports the wrong reference by itself
