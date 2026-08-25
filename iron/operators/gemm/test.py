#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pytest
import aie.utils as aie_utils
import torch
import ml_dtypes
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.operators.gemm.op import GEMM
from iron.operators.gemm.reference import generate_golden_reference
from iron.common.test_utils import run_test, verify_buffer


def get_params():
    dev = aie_utils.get_current_device()
    max_aie_columns = dev.cols
    device_type = dev.resolve().name
    # fmt: off
    #   M,     K,     N, num_aie_columns, b_col_maj, c_col_maj,   m,   k,   n, trace_size, partition_N
    regular_params = [
        (2048,  2048,  2048,               1,     False,     False,  64,  64,  64,          0, 1),
        (2048,  2048,  2048,               2,      True,     False,  64,  64,  64,          0, 1),
        (2048,  2048,  2048,               8,      True,      True,  64,  64,  64,          0, 1),
        ( 384,  1536,  1792,               4,      True,     False,  32,  48,  64,          0, 1),
        (1792,   896,  1152,               8,     False,      True,  64,  32,  48,          0, 1),
        ( 896,  1792,   640,               8,     False,      True,  32,  64,  80,          0, 1),
        ( 192,   384,    64,               4,     False,     False,  48,  96,  16,          0, 1),
        ( 192,   384,    64,               4,      True,      True,  48,  96,  16,          0, 1),
        (  64,   512,   256,               4,      True,     False,  16,  64,  64,          0, 4),
    ]
    extensive_params = [
        (2048,  2048,  2048,               8,     False,     False,  32,  32, 128,          0, 1),
        (2048,  2048,  8192,               2,     False,     False,  64,  64,  64,          0, 1),
        (2048,  8192,  2048,               2,     False,     False,  64,  64,  64,          0, 1),
        (2048,    64,  2048,               2,     False,     False,  64,  64,  64,          0, 1),
        (2048,    64,  8192,               2,     False,     False,  64,  64,  64,          0, 1),
        (2048,  2048,  2048,               8,      True,     False, 128,  32,  32,          0, 1),
        (2048,  2048,  8192,               2,      True,     False,  64,  64,  64,          0, 1),
        (2048,  8192,  2048,               2,      True,     False,  64,  64,  64,          0, 1),
        (2048,    64,  2048,               2,      True,     False,  64,  64,  64,          0, 1),
        (2048,    64,  8192,               2,      True,     False,  64,  64,  64,          0, 1),
        (2048,  2048,  2048,               2,     False,      True,   8,  16,  32,          0, 1),
        (2048,  2048,  8192,               2,     False,      True,  64,  64,  64,          0, 1),
        (2048,  8192,  2048,               2,     False,      True,  64,  64,  64,          0, 1),
        (2048,    64,  2048,               2,     False,      True,  64,  64,  64,          0, 1),
        (2048,    64,  8192,               2,     False,      True,  64,  64,  64,          0, 1),
    ]
    # fmt: on

    params = []

    # Helper to generate name and append param
    def add_params(param_list, is_extensive):
        for p in param_list:
            (
                M,
                K,
                N,
                num_aie_columns,
                b_col_maj,
                c_col_maj,
                m,
                k,
                n,
                trace_size,
                partition_N,
            ) = p

            # Skip tests that require more columns than available on the device
            if num_aie_columns > max_aie_columns:
                continue

            # Skip configurations with small tile sizes that don't meet AIE2 kernel constraints
            # AIE2 mm kernel requires m % (4 * r) == 0 where r=4 for bf16
            if device_type == "npu1" and m < 16:
                continue

            marks = [pytest.mark.extensive] if is_extensive else []
            params.append(pytest.param(*p, marks=marks))

    add_params(regular_params, is_extensive=False)
    add_params(extensive_params, is_extensive=True)

    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "M,K,N,num_aie_columns,b_col_maj,c_col_maj,m,k,n,trace_size,partition_N",
    get_params(),
)
def test_gemm(
    M,
    K,
    N,
    num_aie_columns,
    b_col_maj,
    c_col_maj,
    m,
    k,
    n,
    trace_size,
    partition_N,
    aie_context,
):
    total_N = N * partition_N

    golden_ref = generate_golden_reference(
        M=M,
        K=K,
        N=total_N,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
    )

    operator = GEMM(
        M=M,
        K=K,
        N=N,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        num_aie_columns=num_aie_columns,
        prio_accuracy=True,
        emulate_bf16_mmul_with_bfp16=False,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
        context=aie_context,
    )

    if partition_N == 1:
        input_buffers = {
            "A": golden_ref["input"].flatten(),
            "B": golden_ref["input_b"][0].flatten(),
        }
        output_buffers = {
            "C": golden_ref["output"][0].flatten(),
        }
        errors, latency_us, bandwidth_gbps = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
        )
    else:
        compilable = operator.compile()
        op_func = compilable.get_callable()

        # Convert B_full torch bfloat16 → numpy bfloat16 for partition_B
        B_full_np = (
            golden_ref["input_b"][0]
            .contiguous()
            .view(torch.uint16)
            .numpy()
            .view(ml_dtypes.bfloat16)
        )

        # Partition B using the operator method (handles slicing and padding)
        B_parts = compilable.partition_B(B_full_np, partition_N)

        # Create A XRTTensor (shared across all partitions)
        A_buf = XRTTensor.from_torch(golden_ref["input"].flatten())

        # Allocate per-partition B and C XRTTensors
        arg_spec = compilable.get_arg_spec()
        c_shape = arg_spec[2].shape
        c_dtype = arg_spec[2].dtype

        B_bufs = []
        C_bufs = []
        for i in range(partition_N):
            b_torch = (
                torch.from_numpy(B_parts[i].view(np.uint16))
                .view(torch.bfloat16)
                .flatten()
            )
            B_bufs.append(XRTTensor.from_torch(b_torch))
            C_bufs.append(XRTTensor(c_shape, dtype=c_dtype))

        # Run each partition
        start_time = time.perf_counter()
        for i in range(partition_N):
            op_func(A_buf, B_bufs[i], C_bufs[i])
        end_time = time.perf_counter()
        latency_us = (end_time - start_time) * 1e6

        # Read back and concatenate C partitions along the column dimension
        C_parts_torch = [buf.to_torch().reshape(c_shape) for buf in C_bufs]
        if c_col_maj:
            C_concat = torch.cat(C_parts_torch, dim=0)
        else:
            C_concat = torch.cat(C_parts_torch, dim=1)

        # Compare concatenated output to full reference
        C_expected = golden_ref["output"][0]
        buf_errors = verify_buffer(
            C_concat, "C", C_expected, rel_tol=0.005, abs_tol=0.005
        )
        errors = {"C": buf_errors} if buf_errors else {}

        # Calculate bandwidth
        a_bytes = golden_ref["input"].nelement() * 2  # bf16 = 2 bytes
        b_bytes = sum(p.nbytes for p in B_parts)
        c_bytes = C_concat.nelement() * 2
        total_bytes = a_bytes + b_bytes + c_bytes
        bandwidth_gbps = total_bytes / (latency_us * 1e-6) / 1e9

    gflops = (2.0 * M * K * total_N) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    assert not errors, "Test failed"


def get_m_stationary_params():
    """Shapes where BOTH dataflows are legal, so n-stationary can act as the reference."""
    dev = aie_utils.get_current_device()
    if dev.resolve().name != "npu2" or dev.cols < 8:
        return []
    # fmt: off
    #   M,     K,    N, cols, b_col_maj,  m,   k,   n
    return [
        (2048, 2048, 2048,  8,     False, 64,  64,  64),
        (2048, 2048, 2048,  8,      True, 64,  64,  64),
        (2048,  512, 1024,  8,     False, 64,  64,  64),
    ]
    # fmt: on


def get_m_stationary_skinny_params():
    """Shapes only the m-stationary dataflow can run: N < tile_n * cols."""
    dev = aie_utils.get_current_device()
    if dev.resolve().name != "npu2" or dev.cols < 8:
        return []
    # fmt: off
    #   M,     K,   N, cols, b_col_maj,  m,   k,   n
    return [
        (2048, 2048,  64,  8,     False, 64,  64,  64),
        (2048,  512, 128,  8,      True, 64,  64,  64),
    ]
    # fmt: on


def _run_gemm(M, K, N, num_aie_columns, b_col_maj, m, k, n, m_stationary, ctx):
    golden_ref = generate_golden_reference(
        M=M, K=K, N=N, b_col_maj=b_col_maj, c_col_maj=False
    )
    operator = GEMM(
        M=M,
        K=K,
        N=N,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        num_aie_columns=num_aie_columns,
        # m_stationary asserts against prio_accuracy (design.py), so the reference arm
        # has to give up f32 accumulation too or the comparison is not like-for-like.
        prio_accuracy=False,
        emulate_bf16_mmul_with_bfp16=False,
        b_col_maj=b_col_maj,
        c_col_maj=False,
        m_stationary=m_stationary,
        context=ctx,
    ).compile()

    arg_spec = operator.get_arg_spec()
    a_buf = XRTTensor.from_torch(golden_ref["input"].flatten())
    b_buf = XRTTensor.from_torch(golden_ref["input_b"][0].flatten())
    c_buf = XRTTensor(arg_spec[2].shape, dtype=arg_spec[2].dtype)
    op_func = operator.get_callable()
    op_func(a_buf, b_buf, c_buf)
    # to_torch() is a view on the XRT buffer, which does not survive loading the next
    # design; clone before the caller runs a second arm.
    return c_buf.to_torch().flatten().clone(), golden_ref["output"][0].flatten()


@pytest.mark.parametrize(
    "M,K,N,num_aie_columns,b_col_maj,m,k,n", get_m_stationary_params()
)
def test_gemm_m_stationary(M, K, N, num_aie_columns, b_col_maj, m, k, n, aie_context):
    """M-stationary must be bit-identical to the shipped N-stationary dataflow.

    Both accumulate a given output element over K in the same order -- the dataflow
    only moves WHERE that happens -- so this is an equality check, not a tolerance.
    Comparing against the f32 golden reference instead would drown the signal: without
    prio_accuracy, bf16 accumulation over K=2048 puts ~29% of elements past rel 0.005
    on BOTH dataflows.
    """
    args = (M, K, N, num_aie_columns, b_col_maj, m, k, n)
    got, _ = _run_gemm(*args, m_stationary=True, ctx=aie_context)
    aie_utils.DefaultNPURuntime.cleanup()
    ref, _ = _run_gemm(*args, m_stationary=False, ctx=aie_context)
    mismatches = int((got.view(torch.uint16) != ref.view(torch.uint16)).sum())
    assert (
        mismatches == 0
    ), f"{mismatches}/{got.numel()} elements differ from n-stationary"


@pytest.mark.parametrize(
    "M,K,N,num_aie_columns,b_col_maj,m,k,n", get_m_stationary_skinny_params()
)
def test_gemm_m_stationary_skinny_n(
    M, K, N, num_aie_columns, b_col_maj, m, k, n, aie_context
):
    """N below tile_n*cols, which the N-stationary path rejects -- the reason O6 exists.

    No equivalence arm is available here, so this only checks that the N-chunking is
    sane: a wrong tiling gives O(1) relative error, while correct tiling lands on the
    bf16 accumulation floor (measured 7.0e-3 at K=2048, 3.9e-3 at K=512, i.e. ~sqrt(K)).
    """
    got, expected = _run_gemm(
        M, K, N, num_aie_columns, b_col_maj, m, k, n, m_stationary=True, ctx=aie_context
    )
    got_f = got.float()
    exp_f = expected.float()
    rel_l2 = float(torch.linalg.norm(got_f - exp_f) / torch.linalg.norm(exp_f))
    print(f"\nrel_L2: {rel_l2:.3e}")
    assert (
        rel_l2 < 0.02
    ), f"rel_L2 {rel_l2:.3e} is far above the bf16 accumulation floor"
