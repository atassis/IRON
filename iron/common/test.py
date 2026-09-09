#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-function tests for iron.common.kv_layout -- no device, no toolchain."""

import pytest

from iron.common.kv_layout import KVLayout, derive_block_size, split_run, validate_block_size


# ---- KVLayout: T == S degenerates to the pre-blocking flat [Hkv, S, HD] formula ----

def test_degenerate_head_stride_is_the_old_S_times_HD():
    lay = KVLayout(Hkv=8, S=2048, HD=128, T=2048)
    assert lay.head_stride == 2048 * 128
    assert lay.num_blocks == 1


def test_degenerate_kv_off_is_the_old_pos_times_HD():
    lay = KVLayout(Hkv=8, S=2048, HD=128, T=2048)
    for pos in (0, 1, 511, 2047):
        assert lay.kv_off(pos) == pos * 128


def test_degenerate_offset_is_the_old_flat_HkvSHD_formula():
    lay = KVLayout(Hkv=8, S=2048, HD=128, T=2048)
    for head in range(8):
        for pos in (0, 1, 2047):
            assert lay.offset(head, pos) == head * 2048 * 128 + pos * 128


def test_total_elems_is_layout_independent():
    flat = KVLayout(Hkv=8, S=2048, HD=128, T=2048)
    blocked = KVLayout(Hkv=8, S=2048, HD=128, T=128)
    assert flat.total_elems == blocked.total_elems == 8 * 2048 * 128


# ---- Blocking changes strides, but every (head, pos) still lands on a UNIQUE offset that never
# collides with another (head, pos) pair, and reduces to the flat layout's offset when T == S. ----

def test_blocked_strides_do_not_scale_with_S():
    # The whole point: head_stride/block_stride at a fixed T must be identical regardless of S.
    for S in (2048, 4096, 8192, 32768):
        lay = KVLayout(Hkv=8, S=S, HD=128, T=128)
        assert lay.head_stride == 128 * 128 == 16384
        assert lay.block_stride == 8 * 128 * 128 == 131072


def test_blocked_offsets_are_bijective():
    Hkv, S, HD, T = 4, 512, 32, 64
    lay = KVLayout(Hkv=Hkv, S=S, HD=HD, T=T)
    seen = set()
    for head in range(Hkv):
        for pos in range(S):
            off = lay.offset(head, pos)
            assert off not in seen, f"collision at head={head} pos={pos} off={off}"
            seen.add(off)
    # Every offset must also stay inside the buffer's own declared extent.
    assert max(seen) < lay.total_elems


def test_blocked_offset_matches_the_S_over_T_Hkv_T_HD_layout_by_hand():
    # [S/T, Hkv, T, HD], block-major: offset(head,pos) = block*(Hkv*T*HD) + head*(T*HD) + within*HD
    Hkv, S, HD, T = 8, 512, 128, 128
    lay = KVLayout(Hkv=Hkv, S=S, HD=HD, T=T)
    for head in range(Hkv):
        for pos in (0, 1, 127, 128, 255, 256, 511):
            block, within = pos // T, pos % T
            expect = block * (Hkv * T * HD) + head * (T * HD) + within * HD
            assert lay.offset(head, pos) == expect


def test_kv_off_plus_head_base_reconstructs_offset():
    # The split every real call site relies on: head_base is a BUILD-time constant, kv_off is the
    # RUNTIME per-token scratchpad value, and their sum must equal the full offset.
    lay = KVLayout(Hkv=8, S=4096, HD=128, T=128)
    for head in (0, 3, 7):
        for pos in (0, 127, 128, 4095):
            assert lay.head_base(head) + lay.kv_off(pos) == lay.offset(head, pos)


def test_kv_off_is_head_independent():
    lay = KVLayout(Hkv=8, S=4096, HD=128, T=128)
    for pos in (0, 63, 128, 4095):
        vals = {lay.kv_off(pos) for _ in range(lay.Hkv)}
        assert len(vals) == 1


# ---- Validation ----

def test_S_must_be_a_whole_number_of_blocks():
    with pytest.raises(ValueError, match="whole number of blocks"):
        KVLayout(Hkv=8, S=100, HD=128, T=64)


def test_out_of_range_head_and_pos_refuse():
    lay = KVLayout(Hkv=8, S=128, HD=128, T=128)
    with pytest.raises(ValueError, match="out of range"):
        lay.head_base(8)
    with pytest.raises(ValueError, match="out of range"):
        lay.kv_off(128)


# ---- derive_block_size: the arithmetic the task asked to be verified independently ----

def test_derive_block_size_qwen3_shape_is_128():
    # Hkv=8, HD=128 (Qwen3-0.6B). At T=256 the block stride is 262144 elements = 131072 granules,
    # ONE over the mem-tile 17-bit field's 131071. At T=128 it is 65536 granules -- fits.
    assert derive_block_size(HD=128, Hkv=8) == 128


def test_derive_block_size_t256_would_overflow_the_memtile_field_by_one_granule():
    lay = KVLayout(Hkv=8, S=256, HD=128, T=256)
    granules = lay.block_stride // 2  # bf16: 2 elements/granule
    assert granules == (1 << 17)          # 131072 -- one over (1<<17)-1 = 131071
    with pytest.raises(ValueError, match="over the 17-bit"):
        validate_block_size(256, HD=128, Hkv=8)


def test_derive_block_size_t128_fits_both_shim_and_memtile_fields():
    validate_block_size(128, HD=128, Hkv=8)  # must not raise


def test_derive_block_size_scales_with_HD_and_Hkv():
    # Halving HD or Hkv should let T double and still fit the same field -- the product
    # Hkv*T*HD is what is bounded, not any one factor alone.
    base = derive_block_size(HD=128, Hkv=8)
    assert derive_block_size(HD=64, Hkv=8) == base * 2
    assert derive_block_size(HD=128, Hkv=4) == base * 2


# ---- split_run: the wrap-cap splitter tmatvec's blocked tap now shares with gemv's ----

def test_split_run_reconstructs_the_run():
    for run in (16384, 8192, 2048 * 128 // 8, 131072):
        got = split_run(run)
        assert got is not None, run
        hi, lo = got
        assert hi * lo == run
        assert hi <= 1023 and lo <= 1023
        assert lo % 2 == 0


def test_split_run_none_when_unsplittable():
    # An odd run has no even divisor at all, so no gran=2-aligned lo can exist.
    assert split_run(1031) is None  # 1031 is prime and odd
    assert split_run(2) == (1, 2)


# ---- DMA-tap address reconstruction: a small, aie-import-free simulator of the SHAPE
# gemv/design.py and tmatvec/design.py build for a blocked A operand, cross-checked against
# KVLayout. This is a MODEL of the addressing scheme those two files hand-transcribe (it does not
# execute their actual Python), so it catches a conceptual error in the scheme itself -- the actual
# transcription is checked by the real build's DDR-byte census (scripts/decode_ddr_bytes.py),
# which reads the compiled ELF's emitted shim BDs, not this simulation.

def _walk_4d(sizes, strides, offset):
    """Every element address a [sizes]/[strides] access pattern visits, in hardware iteration
    order (sizes[0] slowest .. sizes[3] fastest) -- mirrors what an aie.dma_bd literally walks."""
    addrs = []
    for i0 in range(sizes[0]):
        for i1 in range(sizes[1]):
            for i2 in range(sizes[2]):
                for i3 in range(sizes[3]):
                    addrs.append(offset + i0 * strides[0] + i1 * strides[1]
                                + i2 * strides[2] + i3 * strides[3])
    return addrs


def _gemv_column_tap(Hkv, S, HD, T, cols, col):
    """gemv/design.py's blocked A_taps_coalesced[col], reconstructed: sizes=[n_matrices,
    blocks_per_col, run_hi, run_lo], strides=[head_stride, block_stride, run_lo, 1]."""
    blocks_per_col = (S // cols) // T
    head_stride, block_stride = T * HD, Hkv * T * HD
    run_hi, run_lo = split_run(T * HD)
    sizes = [Hkv, blocks_per_col, run_hi, run_lo]
    strides = [head_stride, block_stride, run_lo, 1]
    offset = (col * blocks_per_col) * block_stride
    return sizes, strides, offset


def test_gemv_blocked_column_tap_matches_KVLayout_in_head_major_position_order():
    Hkv, S, HD, T, cols = 4, 1024, 32, 128, 8
    lay = KVLayout(Hkv=Hkv, S=S, HD=HD, T=T)
    per_col = S // cols
    for col in range(cols):
        sizes, strides, offset = _gemv_column_tap(Hkv, S, HD, T, cols, col)
        got = _walk_4d(sizes, strides, offset)
        # Expected: head-major, then this column's per_col CONSECUTIVE positions in order, each
        # head_dim elements -- exactly what the core's acquire loop consumes for group_reuse.
        expect = []
        for head in range(Hkv):
            for pos in range(col * per_col, (col + 1) * per_col):
                base = lay.offset(head, pos)
                expect.extend(range(base, base + HD))
        assert got == expect, f"col={col}"


def _tmatvec_column_tap(Hkv, S, HD, T, head):
    """tmatvec/design.py's blocked A_taps[head] (cols == n_matrices == Hkv, one head/column):
    sizes=[1, num_blocks, run_hi, run_lo], strides=[0, block_stride, run_lo, 1]."""
    num_blocks = S // T
    head_stride, block_stride = T * HD, Hkv * T * HD
    run_hi, run_lo = split_run(T * HD)
    sizes = [1, num_blocks, run_hi, run_lo]
    strides = [0, block_stride, run_lo, 1]
    offset = head * head_stride
    return sizes, strides, offset


def test_tmatvec_blocked_column_tap_matches_KVLayout_in_position_order():
    Hkv, S, HD, T = 8, 2048, 128, 128
    lay = KVLayout(Hkv=Hkv, S=S, HD=HD, T=T)
    for head in range(Hkv):
        sizes, strides, offset = _tmatvec_column_tap(Hkv, S, HD, T, head)
        got = _walk_4d(sizes, strides, offset)
        expect = []
        for pos in range(S):
            base = lay.offset(head, pos)
            expect.extend(range(base, base + HD))
        assert got == expect, f"head={head}"


def test_blocked_taps_agree_with_degenerate_T_equals_S_case():
    # At T == S (one block), both tap shapes must reconstruct the SAME addresses the historical
    # flat [Hkv, S, HD] layout uses -- i.e. blocking a design at T=S is a genuine no-op.
    Hkv, S, HD = 4, 512, 32
    lay_flat = KVLayout(Hkv=Hkv, S=S, HD=HD, T=S)
    for head in range(Hkv):
        sizes, strides, offset = _tmatvec_column_tap(Hkv, S, HD, S, head)
        got = _walk_4d(sizes, strides, offset)
        expect = [lay_flat.offset(head, pos) + d for pos in range(S) for d in range(HD)]
        assert got == expect
