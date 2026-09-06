#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.operators.tmatvec.op import TMatVec
from iron.operators.tmatvec.reference import generate_golden_reference_tmatvec
from iron.common.test_utils import run_test


def test_one_matrix_per_column_is_required():
    with pytest.raises(ValueError, match="one matrix per column"):
        TMatVec(M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=4)


def test_rows_per_chunk_must_divide_K():
    with pytest.raises(ValueError, match="rows_per_chunk"):
        TMatVec(M=128, K=100, num_aie_columns=1, num_batches=1, rows_per_chunk=64)


def test_alloc_K_sizes_the_operand_not_the_reduction():
    op = TMatVec(M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2, alloc_K=2048)
    spec = op.get_arg_spec()
    assert spec[0].shape == (8, 2048, 128), spec[0].shape
    assert spec[1].shape == (16, 256), "W follows the REDUCED extent"
    assert spec[2].shape == (16, 128), "output is the row width"


def test_alloc_K_windowed_does_not_share_a_name_with_the_plain_op():
    plain = TMatVec(M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2).name
    win = TMatVec(
        M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2, alloc_K=2048
    ).name
    assert win != plain and "ak2048" in win, win


# The decode's own shape (Hq=16 query heads over Hkv=8 kv heads at head_dim 128), plus a small one.
@pytest.mark.parametrize(
    "M,K,num_batches,batch_group,rows_per_chunk",
    [(128, 512, 16, 2, 64), (128, 256, 16, 2, 32)],
)
def test_tmatvec_reduces_down_the_rows(
    M, K, num_batches, batch_group, rows_per_chunk, aie_context
):
    golden = generate_golden_reference_tmatvec(
        M=M, K=K, num_batches=num_batches, batch_group=batch_group
    )
    op = TMatVec(
        M=M,
        K=K,
        num_aie_columns=num_batches // batch_group,
        num_batches=num_batches,
        batch_group=batch_group,
        rows_per_chunk=rows_per_chunk,
        context=aie_context,
    )
    input_buffers = {"matrix": golden["A"].flatten(), "vector": golden["W"].flatten()}
    output_buffers = {"output": golden["C"].flatten()}
    errors, latency_us, bandwidth_gbps = run_test(
        op, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-2
    )
    print(f"\nLatency: {latency_us:.1f} us  Bandwidth: {bandwidth_gbps:.3f} GB/s\n")
    assert not errors, f"transposed matvec failed: {errors}"
