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


# The VECTOR-operand mirror of alloc_K: W is the reduction vector (`# vector (W)` in
# get_arg_spec), and its own allocation can differ from K exactly like A's.
def test_alloc_K_w_sizes_the_vector_operand_not_the_reduction():
    op = TMatVec(M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2, alloc_K_w=2048)
    spec = op.get_arg_spec()
    assert spec[0].shape == (8, 256, 128), "A stays sized by K (alloc_K unset)"
    assert spec[1].shape == (16, 2048), (
        f"W must be sized by the ALLOCATION 2048, got {spec[1].shape}"
    )
    assert spec[2].shape == (16, 128), "output is the row width"


def test_alloc_K_w_none_is_the_old_operand_shape():
    a = TMatVec(M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2).get_arg_spec()[1].shape
    b = TMatVec(
        M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2, alloc_K_w=None
    ).get_arg_spec()[1].shape
    c = TMatVec(
        M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2, alloc_K_w=256
    ).get_arg_spec()[1].shape
    assert a == b == c == (16, 256), a


def test_alloc_K_w_windowed_does_not_share_a_name_with_the_plain_op():
    plain = TMatVec(M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2).name
    win = TMatVec(
        M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2, alloc_K_w=2048
    ).name
    assert win != plain and "akw2048" in win, win
    assert TMatVec(
        M=128, K=256, num_aie_columns=8, num_batches=16, batch_group=2, alloc_K_w=256
    ).name == plain


def test_alloc_K_w_below_K_is_refused():
    with pytest.raises(ValueError, match="alloc_K_w"):
        TMatVec(M=128, K=256, num_aie_columns=1, num_batches=1, alloc_K_w=128)


def test_rows_per_chunk_must_fit_l1_not_just_divide():
    """K008: the default rows_per_chunk fits at head_dim 128 and does NOT at 256.

    Without this the only thing that notices is aiecc, which says "'aie.tile' op Basic sequential
    allocation also failed" -- a tile, not a size. Measured 2026-09-07 on Gemma-3-270M: the default
    fails to build and 32 succeeds.
    """
    # Qwen3-0.6B's shape, the one in the shipped decode: unchanged by this check.
    TMatVec(M=128, K=2048, num_aie_columns=8, num_batches=16, batch_group=2, rows_per_chunk=64)
    # Gemma-3-270M's: head_dim 256 doubles the A tile to 64 KB on its own.
    with pytest.raises(ValueError, match="does not fit L1"):
        TMatVec(M=256, K=2048, num_aie_columns=1, num_batches=4, batch_group=4, rows_per_chunk=64)
    TMatVec(M=256, K=2048, num_aie_columns=1, num_batches=4, batch_group=4, rows_per_chunk=32)


def test_the_fit_error_names_the_BLOCKING_TERM_when_no_chunk_fits():
    """At Gemma-4-12B's global geometry no rows_per_chunk fits, and A is not why.

    head_dim 512, one kv head, gqa_group 16: W alone is 16*2048*2 = 65536 B, the entire L1, before
    A/C/acc get a byte. Only A scales with rows_per_chunk, so the message must say the knob cannot
    help and name the term that can -- an earlier version said "needs a MemTile stage for A", which
    points at the wrong operand.
    """
    from iron.operators.tmatvec.design import check_l1_fits

    msg = check_l1_fits(M=512, K=2048, batch_group=16, rows_per_chunk=64)
    assert msg and "NO rows_per_chunk fits" in msg, msg
    assert "W (batch_group*K)" in msg and "65536" in msg, msg
    assert "cannot help" in msg, msg
    # The A-dominated case must still name the value that works, not the blocking term.
    tunable = check_l1_fits(M=256, K=2048, batch_group=4, rows_per_chunk=64)
    assert tunable and "largest rows_per_chunk that fits here is 32" in tunable, tunable


def test_the_fit_error_names_the_value_that_works():
    from iron.operators.tmatvec.design import largest_fitting_rows_per_chunk

    assert largest_fitting_rows_per_chunk(M=256, K=2048, batch_group=4) == 32
    assert largest_fitting_rows_per_chunk(M=128, K=2048, batch_group=2) == 64


@pytest.mark.parametrize(
    "M,K,alloc_K_w,num_batches,batch_group", [(128, 256, 2048, 16, 2), (128, 128, 1024, 8, 1)]
)
def test_tmatvec_narrow_window_reads_only_its_w_window(
    M, K, alloc_K_w, num_batches, batch_group, aie_context
):
    """Poisoned W columns past the window catch a wrong PER-BATCH STRIDE.

    Same technique as alloc_K's device test, on the other operand: W past K IS summed into every
    output if the reduction extent is wrong, so the poison catches both a wrong extent and a wrong
    per-batch stride.
    """
    golden = generate_golden_reference_tmatvec(
        M=M, K=K, num_batches=num_batches, batch_group=batch_group, alloc_K_w=alloc_K_w
    )
    op = TMatVec(
        M=M,
        K=K,
        alloc_K_w=alloc_K_w,
        num_aie_columns=num_batches // batch_group,
        num_batches=num_batches,
        batch_group=batch_group,
        context=aie_context,
    )
    input_buffers = {"matrix": golden["A"].flatten(), "vector": golden["W"].flatten()}
    output_buffers = {"output": golden["C"].flatten()}
    errors, latency_us, bandwidth_gbps = run_test(
        op, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-2
    )
    assert not errors, f"windowed TMatVec W failed: {errors}"


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
