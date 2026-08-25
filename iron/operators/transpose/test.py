#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import aie.utils as aie_utils

from iron.common.context import AIEContext
from iron.operators.transpose.op import Transpose
from iron.operators.transpose.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    max_aie_columns = aie_utils.get_current_device().cols
    input_lengths = [64, 2048]
    n_list = [64, 128, 256, 512]
    s_list = [8]
    m = 64
    n = 64

    params = []
    for M in input_lengths:
        for N in n_list:
            for s in s_list:
                for num_aie_columns in range(1, max_aie_columns + 1):
                    for num_channels in [1, 2]:
                        row_part = M // num_channels
                        col_part = N // num_aie_columns
                        if row_part % m != 0 or col_part % n != 0:
                            continue
                        check_length = (
                            row_part * col_part * num_channels * num_aie_columns
                        )
                        length = M * N
                        if check_length != length:
                            continue

                        is_regular = M == 2048 and N == 64
                        marks = [] if is_regular else [pytest.mark.extensive]

                        params.append(
                            pytest.param(
                                M,
                                N,
                                num_aie_columns,
                                num_channels,
                                m,
                                n,
                                s,
                                1,
                                marks=marks,
                            )
                        )

    # num_batches>1: batch B independent same-shape transposes into one dispatch
    # (regular shape, single column/channel). num_batches=2 runs in the default
    # suite; the larger batch is extensive.
    for nb in (2, 4):
        params.append(
            pytest.param(
                2048,
                64,
                1,
                1,
                m,
                n,
                8,
                nb,
                marks=[] if nb == 2 else [pytest.mark.extensive],
            )
        )

    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize("M,N,aie_columns,channels,m,n,s,num_batches", get_params())
def test_transpose(M, N, aie_columns, channels, m, n, s, num_batches, aie_context):
    golden_ref = generate_golden_reference(rows=M, cols=N, num_batches=num_batches)

    operator = Transpose(
        M=M,
        N=N,
        num_aie_columns=aie_columns,
        num_channels=channels,
        m=m,
        n=n,
        s=s,
        num_batches=num_batches,
        context=aie_context,
    )

    input_buffers = {"input": golden_ref["input"]}
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


# m, n and whether design.py's guard admits coalescing at M=N=64: n must equal N and
# the shape must be single-column/-channel, so n=32 is the fallback control.
_COALESCE_SHAPES = [(64, 64, True), (32, 64, True), (32, 32, False)]


@pytest.mark.parametrize("m,n,coalesces", _COALESCE_SHAPES)
def test_coalesce_batch_dma_reaches_the_design(m, n, coalesces, tmp_path):
    """coalesce_batch_dma is opt-in, so nothing at runtime notices if it stops being
    wired into the sequence body -- the design is silently the per-batch one. Gate on
    the generated MLIR: coalescing replaces the per-batch fill/drain unroll with one
    iterated fill BD plus the batch-chunked drain, so it must issue strictly fewer DMA
    tasks where it is admitted, and change nothing where it is not.
    """

    def dma_tasks(coalesce):
        mlir = (
            Transpose(
                M=64,
                N=64,
                num_aie_columns=1,
                num_channels=1,
                m=m,
                n=n,
                s=8,
                num_batches=4,
                coalesce_batch_dma=coalesce,
                context=AIEContext(build_dir=tmp_path),
            )
            .get_mlir_artifact()
            .generator()
        )
        return mlir.count("aiex.dma_configure_task_for")

    per_batch, coalesced = dma_tasks(False), dma_tasks(True)
    if coalesces:
        assert coalesced < per_batch
    else:
        assert coalesced == per_batch


# Device arms for the coalesced path, chosen to cover each branch design.py can take:
# the contiguous single-tile BD, the multi-tile chunked drain at one BD and at two
# (num_batches > _ITER_CAP), the fifodepth==2 case where the fill runs ahead of the core
# and the depth==1 case where it cannot (m*n > 4096), and an n != N shape the guard
# refuses so the fallback stays covered.
_COALESCE_DEVICE_ARMS = [
    # M,    N,   m,   n,  nb,  admitted
    (64, 64, 64, 64, 4, True),  # single-tile, depth 2
    (2048, 64, 64, 64, 4, True),  # multi-tile, one drain BD, depth 2
    (512, 64, 64, 64, 65, True),  # multi-tile, two drain BDs (nb > 64)
    (2048, 64, 128, 64, 4, True),  # multi-tile, depth 1 (m*n > 4096): fully serial
    (2048, 128, 64, 64, 4, False),  # n != N: guard refuses, per-batch fallback
]


@pytest.mark.parametrize("M,N,m,n,num_batches,admitted", _COALESCE_DEVICE_ARMS)
@pytest.mark.parametrize("coalesce", [False, True])
def test_coalesce_batch_dma_on_device(
    M, N, m, n, num_batches, admitted, coalesce, aie_context
):
    """Offline tap equivalence proves the index sequence, not the runtime.

    The coalesced fill is a single iterated BD that runs ahead of the core at
    fifodepth 2, and correctness there rests on ObjectFifo lock backpressure rather
    than on the taps. Transpose does no arithmetic, so the gate is exact equality
    with the golden rather than a tolerance -- both arms must reproduce it bitwise.
    """
    golden_ref = generate_golden_reference(rows=M, cols=N, num_batches=num_batches)

    operator = Transpose(
        M=M,
        N=N,
        num_aie_columns=1,
        num_channels=1,
        m=m,
        n=n,
        s=8,
        num_batches=num_batches,
        coalesce_batch_dma=coalesce,
        context=aie_context,
    )

    errors, _, _ = run_test(
        operator,
        {"input": golden_ref["input"]},
        {"output": golden_ref["output"]},
        rel_tol=0.0,
        abs_tol=0.0,
    )

    assert not errors, f"Test failed with errors: {errors}"
