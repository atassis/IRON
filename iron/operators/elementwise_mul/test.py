#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.elementwise_mul.op import ElementwiseMul
from iron.operators.elementwise_mul.reference import generate_golden_reference, reference
from iron.common.test_utils import run_test, make_binary_elementwise_params


def get_params():
    return [
        pytest.param(il, nac, ts, marks=[] if not ext else [pytest.mark.extensive])
        for il, nac, ts, ext in make_binary_elementwise_params(
            [1024, 2048, 4096, 8192], 4096
        )
    ]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "input_length,num_aie_columns,tile_size",
    get_params(),
)
def test_elementwise_mul(input_length, num_aie_columns, tile_size, aie_context):
    golden_ref = generate_golden_reference(input_length=input_length)

    operator = ElementwiseMul(
        size=input_length,
        tile_size=tile_size,
        num_aie_columns=num_aie_columns,
        context=aie_context,
    )

    input_buffers = {"input1": golden_ref["A"], "input2": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


def test_broadcast_reference_matches_per_row_loop():
    """Host-only (no NPU): the broadcast reference's C == a per-row loop over `reference()`.

    This is the semantics the broadcast operator collapses into one dispatch: today's
    gen_llm_prefill.py calls the plain (non-broadcast) op once per row with the same [D] gain
    operand every time. Confirms torch's `[D] x [M, D]` broadcast reproduces that loop exactly,
    not merely approximately, before trusting it as the on-device gate in the parametrized test.
    """
    D, M = 3840, 64
    golden = generate_golden_reference(input_length=D, rows=M)
    a = golden["A"].reshape(M, D)
    b = golden["B"]
    c_broadcast = golden["C"].reshape(M, D)

    c_loop = torch.empty_like(c_broadcast)
    for r in range(M):
        c_loop[r] = reference(a[r], b)

    assert torch.equal(c_broadcast, c_loop), "broadcast reference diverged from per-row loop"


def get_broadcast_params():
    return [
        pytest.param(il, r, nac, ts, marks=[] if not ext else [pytest.mark.extensive])
        for il, r, nac, ts, ext in [
            # Small, CI-shaped correctness cases.
            (1024, 4, 4, 256, False),
            (2048, 8, 2, 1024, False),
            # The shipped Gemma-4-12B prefill arm's op_lscale shape: D=3840, M=64, cols=8,
            # tile_size=D//cols -- see gen_llm_prefill.py's op_lscale construction and
            # kb/prefill-is-dispatch-bound-and-one-op-is-41pct-of-the-runlist.md.
            (3840, 64, 8, 480, True),
        ]
    ]


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "input_length,rows,num_aie_columns,tile_size",
    get_broadcast_params(),
)
def test_elementwise_mul_broadcast(
    input_length, rows, num_aie_columns, tile_size, aie_context
):
    """`[D] x [M, D]` broadcast: one dispatch, gain read once per D-chunk instead of once per row.

    M=1 (decode) is deliberately NOT exercised here -- it takes the `rows=None` path, which is
    binary_elementwise_design unchanged (see test_elementwise_mul above).
    """
    golden_ref = generate_golden_reference(input_length=input_length, rows=rows)

    operator = ElementwiseMul(
        size=input_length,
        tile_size=tile_size,
        num_aie_columns=num_aie_columns,
        rows=rows,
        context=aie_context,
    )

    input_buffers = {"input1": golden_ref["A"], "input2": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
