#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-function tests for iron.common.kv_layout -- no device, no toolchain."""

import pytest

from iron.common.kv_layout import (KVLayout, blocked_access_pattern, derive_block_size,
                                   restride_rows, split_run, validate_block_size)


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


# ---- derive_block_size(S, n_cols=...): the second, column-share constraint the blocked GEMV
# enforces that the field bound above knows nothing about (gemma3-270m was unbuildable without
# it -- see the docstring for the Hkv=1 geometry that exposes the collision) ----

def test_derive_block_size_qwen3_shape_unchanged_by_column_constraint():
    # Regression guard: Qwen3-0.6B's default build (S=2048, COLS=8, S//COLS=256) never hits the
    # column bound -- the field bound alone already lands on T=128, which divides 256. Passing
    # S/n_cols must not move it.
    assert derive_block_size(HD=128, Hkv=8, S=2048, n_cols=8) == 128


def test_derive_block_size_gemma3_shape_divides_the_column_share():
    # Gemma3-270M (Hkv=1, HD=256): the field bound alone picks T=512 (see the bug this fixes --
    # blocked GEMV then needs S // COLS=256 to be a whole number of T=512 blocks, and isn't). The
    # column constraint caps T at S // n_cols=256, which the field bound alone does not know to do.
    T = derive_block_size(HD=256, Hkv=1, S=2048, n_cols=8)
    assert T == 256
    assert (2048 // 8) % T == 0


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


# ---- head_span: what a whole-window operand has to be allowed to address ----

def test_head_span_is_the_flat_slab_at_T_equals_S():
    assert KVLayout(Hkv=8, S=2048, HD=128, T=2048).head_span == 2048 * 128


def test_head_span_reaches_the_end_of_the_last_block():
    lay = KVLayout(Hkv=8, S=2048, HD=128, T=128)
    # The last element any head touches is the last of its slice of block num_blocks-1.
    assert lay.head_span == lay.offset(0, lay.S - 1) + lay.HD
    # And the last head's span must land exactly on the end of the buffer, not past it.
    assert lay.head_base(lay.Hkv - 1) + lay.head_span == lay.total_elems


# ---- blocked_access_pattern: the descriptor rewrite the prefill GEMMs read B through ----

def _walk(sizes, strides, offset):
    """Every address an n-D access pattern visits, sizes[0] slowest."""
    addrs, total = [], 1
    for s in sizes:
        total *= s
    for lin in range(total):
        addr, rem = offset, lin
        for d in range(len(sizes) - 1, -1, -1):
            rem, i = divmod(rem, sizes[d])
            addr += i * strides[d]
        addrs.append(addr)
    return addrs


# The two descriptors TensorTiler2D.step_tiler actually produces for the prefill attention GEMMs
# at M=256, S=2048, HD=128 -- scores (b_col_maj, B = [N=S, K=HD], 64x64 tiles over 8 columns) and
# ctx (plain, B = [K=S, N=HD], 64x16 tiles). Transcribed rather than regenerated so this file
# stays free of the aie import; if the tiling moves, the build's own descriptors move with it and
# these become a check of a shape nothing builds.
_SCORES = ([4, 2, 64, 64], [65536, 64, 128, 1], [c * 8192 for c in range(8)])
_CTX = ([1, 32, 64, 16], [0, 8192, 128, 1], [c * 16 for c in range(8)])


@pytest.mark.parametrize("sizes,strides,offsets", [_SCORES, _CTX])
def test_blocked_access_pattern_visits_the_same_elements_where_KVLayout_puts_them(
        sizes, strides, offsets):
    HD, S, Hkv, T = 128, 2048, 8, 128
    lay = KVLayout(Hkv=Hkv, S=S, HD=HD, T=T)
    for offset in offsets:
        b_off, b_sizes, b_strides = blocked_access_pattern(
            offset, sizes, strides, HD, T, lay.block_stride)
        got = _walk(b_sizes, b_strides, b_off)
        # The flat pattern's Nth address is (row, col) of head 0's logical [S, HD] matrix; the
        # blocked pattern's Nth must be where KVLayout says that (position, dim) lives.
        expect = [lay.offset(0, r) + c
                  for r, c in (divmod(a, HD) for a in _walk(sizes, strides, offset))]
        assert got == expect
        assert max(got) < lay.head_span


def test_blocked_access_pattern_is_the_identity_when_the_window_is_one_block():
    HD, S, Hkv = 128, 2048, 8
    flat = KVLayout(Hkv=Hkv, S=S, HD=HD, T=S)
    for sizes, strides, offsets in (_SCORES, _CTX):
        for offset in offsets:
            assert blocked_access_pattern(
                offset, sizes, strides, HD, flat.T, flat.block_stride
            ) == (offset, list(sizes), list(strides))


def test_blocked_access_pattern_splits_a_dim_that_crosses_a_block():
    # ctx walks 32 tiles of 64 positions. Under T=128 that is two tiles per block, so the one dim
    # becomes two: whole blocks outer, the pair inside one block inner.
    _, sizes, strides = blocked_access_pattern(0, *_CTX[:2], 128, 128, 131072)
    assert sizes == [1, 16, 2, 64, 16]
    assert strides == [0, 131072, 8192, 128, 1]


def test_blocked_access_pattern_rewrites_a_whole_block_step_without_splitting():
    # scores steps 512 positions at a time -- four whole blocks -- so the dim count is unchanged.
    _, sizes, strides = blocked_access_pattern(0, *_SCORES[:2], 128, 128, 131072)
    assert sizes == list(_SCORES[0])
    assert strides == [4 * 131072, 64, 128, 1]


def test_blocked_access_pattern_raises_on_a_step_it_cannot_express():
    # 3 positions per step against a 128-position block: neither whole blocks nor a divisor of one.
    with pytest.raises(ValueError, match="crosses a block boundary"):
        blocked_access_pattern(0, [100, 128], [3 * 128, 1], 128, 128, 131072)


def test_blocked_access_pattern_raises_on_a_step_that_is_neither_rows_nor_columns():
    with pytest.raises(ValueError, match="neither under one row"):
        blocked_access_pattern(0, [4, 8], [192, 1], 128, 128, 131072)


def test_blocked_access_pattern_rejects_overlapping_blocks():
    with pytest.raises(ValueError, match="under one block"):
        blocked_access_pattern(0, [4, 8], [128, 1], 128, 128, 128 * 128 - 2)


# ---- restride_rows: reading a per-head slice of a wider buffer in place ----

def test_restride_rows_is_the_identity_at_the_same_width():
    off, sizes, strides = restride_rows(8192, [4, 2, 64, 64], [0, 64, 128, 1], 128, 128)
    assert (off, sizes, strides) == (8192, [4, 2, 64, 64], [0, 64, 128, 1])


def test_restride_rows_visits_the_head_slice_of_the_wider_buffer():
    # The GEMM A tap for prefill's scores op: a dense [M=256, HD=128] matrix. Re-targeted at a
    # [M, QD=2048] buffer it must visit exactly head h's columns, row for row.
    HD, QD, M = 128, 2048, 256
    for h in (0, 5, 15):
        off, sizes, strides = restride_rows(0, [4, 2, 64, 64], [0, 64, 128, 1], HD, QD,
                                            col_base=h * HD)
        got = _walk(sizes, strides, off)
        # every address must decode to (row, column) inside head h's band of the wide buffer
        for a in got:
            r, c = divmod(a, QD)
            assert 0 <= r < M and h * HD <= c < (h + 1) * HD, (h, a)
        # and the SEQUENCE must match the dense pattern, element for element
        dense = _walk(sizes, [0, 64, 128, 1], 0)
        assert [divmod(a, QD)[0] * HD + divmod(a, QD)[1] - h * HD for a in got] == dense


def test_restride_rows_rescales_only_row_steps():
    # dim1's 64 is half a row and must NOT move; dim2's 128 is one row and must.
    _, _, strides = restride_rows(0, [4, 2, 64, 64], [0, 64, 128, 1], 128, 2048)
    assert strides == [0, 64, 2048, 1]


def test_restride_rows_takes_the_column_base_separately_from_the_offset():
    # Folding the column base into `offset` is the misuse this argument exists to prevent: it is
    # read as ROWS and lands the operand somewhere plausible and wrong.
    HD, QD = 128, 2048
    right, _, _ = restride_rows(0, [64, 64], [128, 1], HD, QD, col_base=5 * HD)
    wrong, _, _ = restride_rows(5 * HD, [64, 64], [128, 1], HD, QD)
    assert right == 5 * HD
    assert wrong == 5 * QD and wrong != right


def test_restride_rows_rejects_a_step_that_is_not_rows_or_columns():
    with pytest.raises(ValueError, match="neither under one row"):
        restride_rows(0, [4, 8], [192, 1], 128, 2048)
