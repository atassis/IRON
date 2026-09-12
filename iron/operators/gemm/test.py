#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
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


# --------------------------------------------------------------------------------------------
# weight_dtype: B streamed group-quantized, expanded on-core (see iron/common/quant.py's GEMM
# section and aie_kernels/generic/mm_quant.cc).
# --------------------------------------------------------------------------------------------


def _quant_operator(M, K, N, m, k, n, cols, group_size, weight_dtype, ctx=None, **kw):
    return GEMM(
        M=M, K=K, N=N, tile_m=m, tile_k=k, tile_n=n, num_aie_columns=cols,
        b_col_maj=True, emulate_bf16_mmul_with_bfp16=True, prio_accuracy=False,
        weight_dtype=weight_dtype, group_size=group_size, context=ctx, **kw,
    )


# (M, K, N, m, k, n, cols, group_size, weight_dtype). g=32 is what Gemma-4-12B's own int8 dump
# uses at every site but attn_o; g=64 is the other one it ships. K=3840/N=4096 is the `qkv` shape.
_QUANT_SHAPES = [
    (256, 3840, 4096, 64, 64, 64, 8, 32, "int8"),
    (256, 3840, 4096, 64, 64, 64, 8, 64, "int8"),
    (256, 3840, 2048, 64, 64, 64, 8, 32, "int8"),
    (256, 512, 512, 64, 64, 64, 8, 32, "int8"),
    (256, 512, 512, 64, 64, 64, 8, 64, "int4"),
]


def _run_on_device(operator, A_bf16, B_tensor):
    operator.compile()
    call = operator.get_callable()
    out_spec = operator.get_arg_spec()[2]
    c_buf = XRTTensor(out_spec.shape, dtype=out_spec.dtype)
    call(XRTTensor.from_torch(A_bf16.flatten()), XRTTensor.from_torch(B_tensor), c_buf)
    return c_buf.to_torch().reshape(operator.M, operator.N).to(torch.float32).numpy()


@pytest.mark.parametrize("M,K,N,m,k,n,cols,group_size,weight_dtype", _QUANT_SHAPES)
def test_gemm_quantized_weight(M, K, N, m, k, n, cols, group_size, weight_dtype, aie_context):
    """GATE: the quantized arm must be BIT-IDENTICAL to a plain bf16 GEMM fed the same weight.

    The control is the unquantized operator at the same shape with `unpack_B`'s output as a bf16 B
    -- so the only difference between the arms is where the expansion happens, DDR-side or
    on-core. Gating on 1:1 against it, rather than on a tolerance against an f32 golden, is what
    makes this a correctness test at all: at K=3840 the GEMM's own bf16/bfp16 arithmetic is ~1.3e-2
    rel-L2 with individual small outputs far worse, which swamps any layout bug a tolerance could
    see. rel-L2 against f32 is printed as a note, never asserted tightly.
    """
    if cols > aie_utils.get_current_device().cols:
        pytest.skip(f"needs {cols} columns")
    torch.manual_seed(13)
    A = torch.randn(M, K, dtype=torch.bfloat16) * 4
    W = (np.random.default_rng(13).standard_normal((N, K)) * 0.5).astype(np.float32)

    operator = _quant_operator(M, K, N, m, k, n, cols, group_size, weight_dtype, aie_context)
    packed = operator.pack_B(W)
    dequantized = operator.unpack_B(packed)

    c_quant = _run_on_device(operator, A, torch.from_numpy(packed))
    # Every arithmetic field copied off the operator: a control that differs in one of them
    # measures the mmul's accuracy settings instead of the expansion.
    control = dataclasses.replace(operator, weight_dtype="bf16", group_size=0)
    c_control = _run_on_device(control, A, torch.from_numpy(
        dequantized.astype(ml_dtypes.bfloat16).view(np.uint16)).view(torch.bfloat16).flatten())

    golden = A.to(torch.float32).numpy() @ dequantized.T
    rel = lambda x: float(np.linalg.norm(x - golden) / np.linalg.norm(golden))
    print(f"\nweight bytes: bf16={N * K * 2} packed({weight_dtype},g{group_size})={packed.nbytes} "
          f"ratio={N * K * 2 / packed.nbytes:.3f}x")
    print(f"rel-L2 vs f32 golden: quant {rel(c_quant):.5f}  bf16 control {rel(c_control):.5f}")

    assert np.array_equal(c_quant, c_control), (
        f"on-core expansion diverges from the DMA-delivered bf16 tile: "
        f"{np.count_nonzero(c_quant != c_control)} of {c_quant.size} outputs differ, "
        f"rel-L2 {float(np.linalg.norm(c_quant - c_control) / np.linalg.norm(c_control)):.3e}")
    assert rel(c_quant) < 0.05, "both arms agree but are far from the f32 golden"


@pytest.mark.parametrize("tile_k,tile_n,group_size,weight_dtype",
                         [(64, 64, 32, "int8"), (64, 64, 64, "int8"), (64, 32, 32, "int8"),
                          (64, 64, 64, "int4"), (128, 64, 32, "int8")])
def test_gemm_pack_is_the_layout_mm_quant_reads(tile_k, tile_n, group_size, weight_dtype):
    """The byte contract, checked on the host: a slab walked the way mm_quant.cc walks it must
    reproduce the bf16 tile gemm/design.py's `dims_to_stream` would have delivered.

    The kernel's loop nest is re-derived here rather than reusing `gemm_tile_permutation`, so this
    is a cross-check of the two sides and not a restatement of one of them. It is the check that
    would have caught a scale indexed by the wrong row -- the failure class this tree pays for on
    device (see the doctrine's seam rule).
    """
    from iron.common.quant import (gemm_tile_permutation, gemm_tile_scale_region_bytes,
                                   gemm_tile_slab_bytes, pack_gemm_weight, unpack_gemm_weight)

    s, t, cols = 8, 8, 3
    N, K = tile_n * cols, tile_k * 2
    rng = np.random.default_rng(7)
    W = (rng.standard_normal((N, K)) * 4).astype(np.float32)
    packed = pack_gemm_weight(W, tile_k, tile_n, group_size, weight_dtype, s, t, cols)
    deq = unpack_gemm_weight(packed, N, K, tile_k, tile_n, group_size, weight_dtype, s, t, cols)

    slab_bytes = gemm_tile_slab_bytes(tile_k, tile_n, group_size, weight_dtype, s, t)
    scale_region = gemm_tile_scale_region_bytes(tile_k, tile_n, group_size, weight_dtype, s, t)
    n_groups = tile_k // group_size
    k_blocks = K // tile_k
    perm = gemm_tile_permutation(tile_k, tile_n, s, t)
    slabs = packed.view(np.uint8).reshape(-1, slab_bytes)

    for kb in range(K // tile_k):
        for nt in range(N // tile_n):
            # Column-run order: column `nt % cols` owns a contiguous run, n-tile outer.
            slab = slabs[(nt % cols) * (N // tile_n // cols) * k_blocks
                         + (nt // cols) * k_blocks + kb]
            scales = slab[:4 * tile_n * n_groups].view(np.float32).reshape(tile_n, n_groups)
            body = slab[scale_region:]
            if weight_dtype == "int4":
                q = np.empty(tile_n * tile_k, dtype=np.int8)
                q[0::2] = np.where((body & 0xF) >= 8, (body & 0xF).astype(np.int8) - 16,
                                   (body & 0xF).astype(np.int8))
                q[1::2] = np.where((body >> 4) >= 8, (body >> 4).astype(np.int8) - 16,
                                   (body >> 4).astype(np.int8))
            else:
                q = body.view(np.int8)

            # mm_quant.cc's own nest: j over n-blocks, gi over groups, bi over the group's
            # t*s blocks; lane l of a block is row j*t + l//s, column (gi*g/s + bi)*s + l%s.
            got = np.empty(tile_n * tile_k, dtype=np.float32)
            for j in range(tile_n // t):
                for gi in range(n_groups):
                    sv = np.repeat(scales[j * t:(j + 1) * t, gi], s)
                    for bi in range(group_size // s):
                        p = (j * (tile_k // s) + gi * (group_size // s) + bi) * (t * s)
                        got[p:p + t * s] = q[p:p + t * s].astype(np.float32) * sv

            want = deq[nt * tile_n:(nt + 1) * tile_n,
                       kb * tile_k:(kb + 1) * tile_k].reshape(-1)[perm]
            assert np.array_equal(got, want), (
                f"slab (kb={kb}, nt={nt}) decodes to a different tile than the DMA layout")


@pytest.mark.parametrize("weight_dtype,group_size", [("int8", 32), ("int8", 64), ("int4", 64)])
def test_gemm_repack_of_a_row_dump_is_the_same_bytes(weight_dtype, group_size):
    """A weight already packed for the decode GEMV repacks into slabs with no requantization.

    This is what lets one dump serve both paths; if it ever stops holding, prefill and decode are
    multiplying different weights and nothing else would say so.
    """
    from iron.common.quant import pack_gemm_weight, quantize_weight, repack_gemm_weight

    N, K = 192, 256
    rng = np.random.default_rng(3)
    W = (rng.standard_normal((N, K)) * 4).astype(np.float32)
    row_form = quantize_weight(W, group_size, weight_dtype)
    assert np.array_equal(
        pack_gemm_weight(W, 64, 64, group_size, weight_dtype, 8, 8, 3),
        repack_gemm_weight(row_form, N, K, 64, 64, group_size, weight_dtype, 8, 8, 3))


def test_gemm_quant_group_must_not_straddle_a_k_tile():
    """The constraint gemv has no analogue of: gemv reads a whole row, GEMM tiles K."""
    with pytest.raises(ValueError, match="tile_k"):
        _quant_operator(256, 512, 512, 64, 64, 64, 8, 128, "int8")


def test_gemm_quant_needs_b_col_maj():
    with pytest.raises(ValueError, match="b_col_maj"):
        GEMM(M=256, K=512, N=512, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=8,
             b_col_maj=False, weight_dtype="int8", group_size=32)


def test_gemm_quant_rejects_the_affine_dtypes_gemv_takes():
    with pytest.raises(ValueError, match="weight_dtype"):
        GEMM(M=256, K=512, N=512, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=8,
             b_col_maj=True, weight_dtype="int8a", group_size=32)


def test_gemm_quant_does_not_share_an_artifact_with_the_plain_gemm():
    """A cached plain build must not be able to satisfy a quantized op (see GEMM.name)."""
    kw = dict(M=256, K=512, N=512, tile_m=64, tile_k=64, tile_n=64, num_aie_columns=8,
              b_col_maj=True)
    plain = GEMM(**kw).name
    quant = GEMM(weight_dtype="int8", group_size=32, **kw).name
    assert quant != plain and "wdtint8g32" in quant, quant
    assert GEMM(weight_dtype="int8", group_size=64, **kw).name != quant


def test_gemm_quant_b_operand_is_the_packed_byte_count():
    from iron.common.quant import gemm_packed_bytes

    op = _quant_operator(256, 3840, 4096, 64, 64, 64, 8, 32, "int8")
    spec = op.get_arg_spec()[1]
    assert spec.dtype is np.int8, spec.dtype
    assert spec.shape == (gemm_packed_bytes(4096, 3840, 64, 64, 32, "int8", 8, 8),), spec.shape
    # 4 B/group of scale over 32 elements is 1.125 B/element against bf16's 2.
    assert spec.shape[0] == 4096 * 3840 * 9 // 8
