#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import aie.utils as aie_utils

from iron.operators.gemv.op import GEMV
from iron.operators.gemv.quant import quantize_weight, dequantize_weight
from iron.operators.gemv.reference import (
    generate_golden_reference,
    generate_golden_reference_batched,
    generate_golden_reference_windowed,
    gelu_tanh_approx,
)
from iron.common.device_utils import get_kernel_dir
import numpy as np
import torch
from iron.common.test_utils import run_test


def get_params():
    max_aie_columns = aie_utils.get_current_device().cols

    params_list = [
        (128, 128, 1, 32, 128),
        (2048, 8192, 1, 1, 2048),
        (8192, 2048, 1, 4, 1024),
        (2048, 8192, 2, 1, 1024),
        (8192, 2048, 2, 4, 1024),
        (2048, 8192, 4, 1, 512),
        (8192, 2048, 4, 4, 1024),
        (2048, 8192, 8, 1, 256),
        (8192, 2048, 8, 4, 1024),
    ]

    params = []
    for p in params_list:
        M, K, num_aie_columns, tile_size_input, tile_size_output = p
        # Skip tests that require more columns than available on the device
        if num_aie_columns > max_aie_columns:
            continue
        params.append(pytest.param(*p))
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "M,K,num_aie_columns,tile_size_input,tile_size_output", get_params()
)
def test_gemv(M, K, num_aie_columns, tile_size_input, tile_size_output, aie_context):
    golden_ref = generate_golden_reference(M=M, K=K)

    operator = GEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size_input=tile_size_input,
        tile_size_output=tile_size_output,
        context=aie_context,
    )

    input_buffers = {"matrix": golden_ref["A"].flatten(), "vector": golden_ref["B"]}
    output_buffers = {"output": golden_ref["C"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-3
    )

    print(f"\nLatency: {latency_us:.1f} us")

    gflops = (2.0 * M * K) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


def get_quant_params():
    max_aie_columns = aie_utils.get_current_device().cols
    # (M, K, num_aie_columns, tile_size_input, tile_size_output, group_size, weight_dtype)
    params_list = [
        (256, 1024, 1, 4, 256, 128, "int4"),
        (256, 1024, 1, 4, 256, 128, "int8"),
        (256, 1024, 8, 4, 32, 64, "int4"),
    ]
    params = []
    for p in params_list:
        if p[2] > max_aie_columns:
            continue
        params.append(pytest.param(*p))
    return params


@pytest.mark.parametrize(
    "M,K,num_aie_columns,tile_size_input,tile_size_output,group_size,weight_dtype",
    get_quant_params(),
)
def test_gemv_quantized_weight(
    M, K, num_aie_columns, tile_size_input, tile_size_output, group_size, weight_dtype,
    aie_context,
):
    """GEMV with A (the weight matrix) streamed group-quantized int4/int8, dequantized on-core.

    Golden = (host dequant of the SAME packed bytes the device reads) @ B, so this isolates
    "did the kernel correctly decode the on-wire format and MAC it" from "how much does
    quantization hurt" -- the latter is a model-quality question this operator-level test does
    not make a claim about.
    """
    torch.manual_seed(11)
    A_f32 = (torch.randn(M, K, dtype=torch.float32) * 4).numpy()
    B = (torch.randn(K, dtype=torch.bfloat16) * 4)

    packed = quantize_weight(A_f32, group_size, weight_dtype)
    a_dequant = dequantize_weight(packed, M, K, group_size, weight_dtype)
    c_golden = torch.from_numpy(a_dequant).to(torch.bfloat16) @ B

    operator = GEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size_input=tile_size_input,
        tile_size_output=tile_size_output,
        weight_dtype=weight_dtype,
        group_size=group_size,
        context=aie_context,
    )

    input_buffers = {"matrix": torch.from_numpy(packed), "vector": B}
    output_buffers = {"output": c_golden}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-2
    )
    print(f"\nLatency: {latency_us:.1f} us")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    bf16_bytes = M * K * 2
    packed_bytes = packed.nbytes
    print(f"weight bytes: bf16={bf16_bytes} packed({weight_dtype},g{group_size})="
          f"{packed_bytes} ratio={bf16_bytes / packed_bytes:.3f}x")

    assert not errors, f"quantized GEMV ({weight_dtype}, g{group_size}) failed: {errors}"


def get_batched_params():
    max_cols = aie_utils.get_current_device().cols
    # (M, K, cols, tsi, tso, num_batches): exercise the coalesced path + fallback.
    plist = [
        (256, 128, 1, 1, 256, 4),  # tiny, coalesced
        (256, 128, 8, 1, 32, 100),  # large num_batches -> the size-uncapped dim
        (448, 64, 8, 1, 56, 192),  # multi-dim run split + large num_batches together
        (64, 1536, 1, 1, 64, 8),  # large K
        (1026, 64, 1, 1, 2, 2),  # run needs an even (granularity-aligned) split
        (1024, 1024, 1, 1, 64, 2),  # batch stride > 2**20 -> falls back to per-batch
        (512, 64, 8, 4, 64, 32),  # attn-style: tile_size_input>1, num_batches=heads
    ]
    out = []
    for p in plist:
        if p[2] > max_cols:
            continue
        out.append(pytest.param(*p))
    return out


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "M,K,num_aie_columns,tile_size_input,tile_size_output,num_batches",
    get_batched_params(),
)
def test_gemv_batched(
    M, K, num_aie_columns, tile_size_input, tile_size_output, num_batches, aie_context
):
    golden = generate_golden_reference_batched(M=M, K=K, num_batches=num_batches)
    operator = GEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size_input=tile_size_input,
        tile_size_output=tile_size_output,
        num_batches=num_batches,
        context=aie_context,
    )
    input_buffers = {
        "matrix": golden["A"].flatten(),
        "vector": golden["B"].flatten(),
    }
    output_buffers = {"output": golden["C"].flatten()}
    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-3
    )

    print(f"\nLatency: {latency_us:.1f} us")
    gflops = (2.0 * M * K * num_batches) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"batched GEMV failed: {errors}"


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "M,K,num_aie_columns,tile_size_input,tile_size_output",
    [
        pytest.param(128, 128, 1, 32, 128),
        pytest.param(2048, 8192, 1, 1, 2048),
        pytest.param(8192, 2048, 1, 4, 1024),
    ],
)
def test_gemv_gelu(
    M, K, num_aie_columns, tile_size_input, tile_size_output, aie_context
):
    """GEMV with the fused GELU epilogue (NPU2-only) vs a gelu(A @ B) golden."""
    if get_kernel_dir() != "aie2p":
        pytest.skip("gemv gelu epilogue is only available on NPU2 (aie2p)")

    golden_ref = generate_golden_reference(M=M, K=K)
    c_ref = golden_ref["C"].to(torch.float32).numpy()
    c_gelu = torch.from_numpy(gelu_tanh_approx(c_ref).astype(np.float32)).to(
        torch.bfloat16
    )

    operator = GEMV(
        M=M,
        K=K,
        num_aie_columns=num_aie_columns,
        tile_size_input=tile_size_input,
        tile_size_output=tile_size_output,
        epilogue="gelu",
        context=aie_context,
    )

    input_buffers = {"matrix": golden_ref["A"].flatten(), "vector": golden_ref["B"]}
    output_buffers = {"output": c_gelu}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.06, abs_tol=2e-2
    )

    print(f"\nLatency: {latency_us:.1f} us")
    gflops = (2.0 * M * K) / (latency_us * 1e-6) / 1e9
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


# GQA without a Repeat: batch_group consecutive batches read ONE matrix, so the operand holds
# num_batches//batch_group of them. The shape below is the decode's own (Hq=16 query heads over
# Hkv=8 kv heads), scaled down so it stays a fast test.
@pytest.mark.parametrize("M,K,num_batches,batch_group", [(256, 128, 16, 2), (128, 64, 8, 4)])
def test_batch_group_shrinks_the_matrix_operand(M, K, num_batches, batch_group):
    g = GEMV(M=M, K=K, num_batches=num_batches, batch_group=batch_group)
    spec = g.get_arg_spec()
    assert spec[0].shape[0] == num_batches // batch_group, (
        f"matrix operand must hold {num_batches // batch_group} matrices, got {spec[0].shape}"
    )
    assert spec[1].shape[0] == num_batches, "vector stays per-batch"
    assert spec[2].shape[0] == num_batches, "output stays per-batch"


def test_batch_group_must_divide_num_batches():
    with pytest.raises(ValueError, match="batch_group"):
        GEMV(M=256, K=128, num_batches=15, batch_group=2)


def test_batch_group_one_is_the_old_operand_shape():
    """The default must not move: batch_group=1 leaves every batch its own matrix."""
    a = GEMV(M=256, K=128, num_batches=4).get_arg_spec()[0].shape
    b = GEMV(M=256, K=128, num_batches=4, batch_group=1).get_arg_spec()[0].shape
    assert a == b == (4, 256, 128), a


# A NARROW WINDOW of a WIDE-STRIDED matrix. `M` is what gets computed, `alloc_M` is what is
# allocated per matrix -- decode attention computes n_past scores against a KV cache sized at
# max_seq. Only the buffer size and the per-matrix stride follow alloc_M; the run, the C tile and
# the core loop all follow M.
@pytest.mark.parametrize(
    "M,K,alloc_M,num_batches,batch_group", [(256, 128, 2048, 16, 2), (64, 128, 512, 8, 1)]
)
def test_alloc_M_sizes_the_operand_not_the_window(M, K, alloc_M, num_batches, batch_group):
    g = GEMV(M=M, K=K, alloc_M=alloc_M, num_batches=num_batches, batch_group=batch_group)
    spec = g.get_arg_spec()
    assert spec[0].shape == (num_batches // batch_group, alloc_M, K), (
        f"matrix operand must be sized by the ALLOCATION {alloc_M}, got {spec[0].shape}"
    )
    assert spec[2].shape[0] == num_batches, "output stays per-batch"
    assert spec[2].shape[1:] == (M,), f"output follows the WINDOW M={M}, got {spec[2].shape}"


def test_alloc_M_none_is_the_old_operand_shape():
    """The default must not move: alloc_M=None leaves the operand sized by M."""
    a = GEMV(M=256, K=128, num_batches=4).get_arg_spec()[0].shape
    b = GEMV(M=256, K=128, num_batches=4, alloc_M=None).get_arg_spec()[0].shape
    c = GEMV(M=256, K=128, num_batches=4, alloc_M=256).get_arg_spec()[0].shape
    assert a == b == c == (4, 256, 128), a


def test_alloc_M_windowed_does_not_share_a_name_with_the_plain_gemv():
    """A cached plain build must not be able to satisfy a windowed op (see GEMV.name)."""
    plain = GEMV(M=128, K=128, num_batches=4).name
    windowed = GEMV(M=128, K=128, num_batches=4, alloc_M=1024).name
    assert windowed != plain, f"windowed GEMV shares an artifact name with the plain one: {plain}"
    assert "am1024" in windowed, windowed
    # alloc_M == M is the same design as None, so it must NOT perturb the stable name.
    assert GEMV(M=128, K=128, num_batches=4, alloc_M=128).name == plain


def test_alloc_M_below_M_is_refused():
    with pytest.raises(ValueError, match="alloc_M"):
        GEMV(M=256, K=128, alloc_M=128)


@pytest.mark.parametrize(
    "M,K,alloc_M,num_batches,batch_group", [(256, 128, 2048, 16, 2), (128, 128, 1024, 4, 1)]
)
def test_gemv_narrow_window_reads_only_its_window(
    M, K, alloc_M, num_batches, batch_group, aie_context
):
    """Poisoned rows past the window catch a wrong PER-MATRIX STRIDE.

    Not a wrong window: a GEMV row is a dot product, so rows past M cannot reach the first M
    outputs at all. What alloc_M has to get right is that matrix m starts at m*alloc_M*K; at the
    old m*M*K it lands inside the previous matrix's poison. Both params keep n_matrices > 1 so
    that stride exists to be checked.
    """
    golden = generate_golden_reference_windowed(
        M=M, K=K, alloc_M=alloc_M, num_batches=num_batches, batch_group=batch_group
    )
    operator = GEMV(
        M=M,
        K=K,
        alloc_M=alloc_M,
        num_aie_columns=aie_utils.get_current_device().cols,
        tile_size_input=4,
        num_batches=num_batches,
        batch_group=batch_group,
        context=aie_context,
    )
    input_buffers = {"matrix": golden["A"].flatten(), "vector": golden["B"].flatten()}
    output_buffers = {"output": golden["C"].flatten()}
    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-3
    )
    assert not errors, f"windowed GEMV failed: {errors}"
