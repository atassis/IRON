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


def test_m_chunk_must_divide_M():
    with pytest.raises(ValueError, match="m_chunk"):
        TMatVec(M=512, K=2048, num_aie_columns=1, num_batches=16, batch_group=16, m_chunk=100)


def test_m_chunk_unblocks_the_global_attention_geometry():
    """Gemma-4-12B's global layers: head_dim 512, one kv head, gqa_group 16.

    Unchunked this fits at NO rows_per_chunk and no context length -- `C+acc` is 65536 B, the whole
    L1, with no K term. Chunking the output moves that floor and lets W stream per K-chunk.
    """
    with pytest.raises(ValueError, match="does not fit L1"):
        TMatVec(M=512, K=6912, num_aie_columns=1, num_batches=16, batch_group=16, rows_per_chunk=16)
    TMatVec(
        M=512, K=6912, num_aie_columns=1, num_batches=16, batch_group=16,
        rows_per_chunk=16, m_chunk=256,
    )


def test_m_chunk_footprint_does_not_carry_K():
    """The property the long-context ceiling needs: L1 identical at 2048 and 262144.

    Unchunked, W is `batch_group*K` and grows without bound; the shipped S=6912 already needs
    221184 B for it alone. Chunked, no term carries K at all.
    """
    from iron.operators.tmatvec.design import l1_footprint_bytes

    at = [l1_footprint_bytes(512, K, 16, 16, 256) for K in (2048, 6912, 32768, 262144)]
    assert len(set(at)) == 1, at
    assert l1_footprint_bytes(512, 262144, 16, 16) > 65536 * 100, "unchunked must still blow up"


def test_the_fit_error_names_m_chunk_when_rows_per_chunk_cannot_help():
    from iron.operators.tmatvec.design import check_l1_fits, largest_fitting_m_chunk

    msg = check_l1_fits(M=512, K=6912, batch_group=16, rows_per_chunk=64)
    assert "cannot help" in msg and "m_chunk=" in msg, msg
    assert largest_fitting_m_chunk(M=512, K=6912, batch_group=16, rows_per_chunk=16) == 256


def test_m_chunk_does_not_share_a_cache_key_or_kernel_width():
    """It changes the core program, the fifo depths and all three access patterns."""
    base = dict(M=512, K=2048, num_aie_columns=1, num_batches=16, batch_group=16)
    chunked = TMatVec(**base, rows_per_chunk=16, m_chunk=128)
    assert "mc128" in chunked.name, chunked.name
    # DIM_N is the compiled row length, so the object is keyed on the chunk, and an m_chunk=128
    # slice of M=512 is legitimately the same object as an unchunked M=128 op.
    assert chunked._kernel_object == "tmv_128n.o", chunked._kernel_object


def test_m_chunk_and_block_size_do_not_compose():
    """Both want all four A access-pattern dims; composing them needs five."""
    from iron.operators.tmatvec.design import transposed_matvec

    with pytest.raises(AssertionError, match="four A dims"):
        transposed_matvec(
            dev=None, cols=1, M=512, K=2048, num_batches=16, batch_group=16,
            rows_per_chunk=16, m_chunk=128, block_size=512,
        )


def test_unchunked_footprint_is_byte_identical_to_the_validated_build():
    """Regression guard: 43008 B was read off a real aiecc ld.script at the shipped Qwen3 shape."""
    from iron.operators.tmatvec.design import l1_footprint_bytes

    assert l1_footprint_bytes(M=128, K=2048, batch_group=2, rows_per_chunk=64) == 43008


def test_runtime_extent_leaves_the_operands_alone():
    """The parameter bounds the REDUCTION, never the allocation: K still sizes W and A."""
    plain = TMatVec(M=128, K=256, num_aie_columns=1, num_batches=1)
    rtp = TMatVec(M=128, K=256, num_aie_columns=1, num_batches=1,
                  vector_size_parameter="sm_mask")
    assert [s.shape for s in rtp.get_arg_spec()] == [s.shape for s in plain.get_arg_spec()]


def test_runtime_extent_does_not_share_a_name_with_the_build_constant_op():
    """Different core program at the same shape, so a cached plain build must not satisfy it."""
    plain = TMatVec(M=128, K=256, num_aie_columns=1, num_batches=1)
    rtp = TMatVec(M=128, K=256, num_aie_columns=1, num_batches=1,
                  vector_size_parameter="sm_mask")
    assert plain.name != rtp.name
    assert rtp.name.endswith("_rtksm_mask")


def test_runtime_extent_reaches_the_generator():
    op = TMatVec(M=128, K=256, num_aie_columns=1, num_batches=1,
                 vector_size_parameter="sm_mask")
    assert op.get_mlir_artifact().generator.kwargs["vector_size_parameter"] == "sm_mask"
