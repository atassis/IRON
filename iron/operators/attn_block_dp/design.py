# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode attention block as ONE `aie.device`: norm, QKV, qk-norm, RoPE, KV-append, scores,
softmax and context, data-parallel with **one KV HEAD per core**.

    hn      = weighted_RMSNorm(cur, n_in)                     D-wide, replicated on every core
    q[g]    = RoPE(weighted_RMSNorm(Wq[head gqa*c+g] @ hn, n_qn), ang)     core c's gqa query heads
    k       = RoPE(weighted_RMSNorm(Wk[head c] @ hn, n_kn), ang)           core c's one kv head
    v       = Wv[head c] @ hn
    -- append k, v at kv_off --
    sc[g]   = kc[head c] @ q[g]                                S-wide row, core c's own kv head
    sw[g]   = softmax(mask(sc[g], n_past+1))
    cx[g]   = sum_p sw[g][p] * vc[head c][p][:]                transposed-A reduction

replacing FOUR consecutive designs in the decode runlist (QKVHeadDataParallel, the scores GEMV,
Softmax, TMatVec) with one, so the group costs one `aiex.configure` per layer instead of four.

WHY THE HEAD MAPPING IS THE WHOLE DESIGN. The four operators this replaces cannot be fused as
they stand -- not "with difficulty", not at all. Their per-column-direct-fill style spends one
shim DMA channel per column per operand, and the device has 16 of each direction TOTAL
(`iron/common/utils.py::get_shim_dma_limit`, 8 ShimNOCTiles x 2). Counted on this graph's own
shapes:

    operator                       shim in   shim out
    QKVHeadDataParallel            misc 1 + weight 8 = 9        out 8
    GEMV scores                    A 8 + B 8         = 16       C 8
    Softmax                        in 8              = 8        out 8
    TMatVec ctx                    A 8 + W 8         = 16       C 8
    naive union                                       49          32     against 16 / 16

`fuse/attn-core` (worktree wt-fuse-attn, commit 36153b0) built exactly that union for a
7-operator version and aiecc rejected it: "no ShimNOCTile on the device has 0 input/1 output DMA
channel(s) free: all 8 ShimNOCTile(s) are at 9/16 input, 16/16 output channels used" -- with
op_sck+op_scv+op_scores ALONE, 3 of its 7 operators, already over. That file also names the fix it
did not build, and this design is it: give every stage ONE consistent column meaning so the
intermediates never take a channel at all.

The meaning is **kv head**. TMatVec already places one kv head per column and Softmax already
partitions by head, so those two agree; the scores GEMV was the odd one out (it slices its OUTPUT
by SEQUENCE POSITION, every column holding a slice of every head, which is what made sc/softmax a
genuine 8-source/8-destination crossbar that `aie.objectfifo_link` refuses outright). Read as a
BATCH-per-column matvec instead -- column c reads only kv head c's cache and computes the FULL
S-wide row for its own gqa query heads -- and the crossbar disappears, because producer and
consumer are the same core. `q`, `sc` and `sw` then never leave L1, so they never reach DDR and
never take a shim channel.

The QKV head has to be re-mapped to match. `qkv_head_dp` assigns heads to cores by contiguous ROWS
of the concatenated Wqkv, which at Hq=16/Hkv=8 puts all sixteen q heads on cores 0-3 and nothing
else there -- a 4-source/8-destination exchange against the mapping above. Here core c owns query
heads [gqa*c, gqa*c+gqa), kv head c of K and kv head c of V: the same FOUR heads per core and the
same per-core weight BYTES, just a different four. They are not contiguous in Wqkv's stock
[Wq | Wk | Wv] row order, so the weight arrives as THREE fills on the one weight channel rather
than one -- the "same fifo, several fill() calls" idiom swiglu_mlp_dp uses for Wg/Wu/Wd. Three
BDs per core for 1 MB instead of one; the weight LAYOUT in DDR is untouched, which is what keeps
this a drop-in for the existing artifact.

Channels, after all that:

  MISC (1 input, BROADCAST to all N cores). HD-wide. Carries `cur` and `n_in` as D/HD chunks
  reassembled by an explicit-offset copy, then n_qn, n_kn and ang, acquired together and held.
  Verbatim qkv_head_dp's misc channel, including why it is HD-wide and not D-wide.

  STREAM (1 input, per core). ONE fifo carrying, in order: this core's three Wqkv runs, then its
  kv head's whole K cache, then its whole V cache. One tile shape serves all three because
  `tile_size_input * D == rows_per_chunk * HD` by construction -- the same shared-tile invariant
  swiglu_mlp_dp asserts for Wg/Wu (row width D) and Wd (row width FF).

  OUTPUT (1 of the 2 available). HD-wide, four rounds: k, v, then one per query head's context.

Two in, one out per compute tile against the hard 2-in/2-out of an AIE2P tile; **9 input and 8
output shim channels device-wide, of 16 each** -- identical to what QKVHeadDataParallel alone
spends today, with the other three operators absorbed for free.

WHAT THE DDR CENSUS DOES. Per layer at Qwen3-0.6B's shape this deletes `q`'s drain (4 KB), the
scores GEMV's re-read of `q` (8 columns x 4 KB = 32 KB), and both round trips of `sc` and `sw`
(4 x 64 KB) -- 299,008 B/layer, 8.372 MB/token over 28 layers. Nothing else moves: the K and V
cache reads are the same bytes in the same contiguous per-head runs, and Wqkv is the same bytes in
three runs instead of one. A fusion that moved MORE bytes would have lost whatever its configure
count.

NOT AN L2 RESIDENCY CLAIM. Nothing is staged through a MemTile here and nothing needs to be: the
intermediates are deleted, not relayed. Two decode-side attempts that merely relayed through L2
measured +0.150 ms and +4.54 ms ([[the-unit-that-costs-is-the-wait-not-the-issue]]) -- L2 pays for
REUSE, not for relay.

WHAT THIS TIES TO L1, AND IT IS NEW. `sc` and `sw` are now core-local, so the attention window S
costs `2 * gqa * S * 2` bytes of L1 that it did not cost before. At gqa=2 that is 16 KB at S=2048
and 32 KB at S=4096, and the budget below is what says which fits. S was previously bounded only
by DDR. State it as a constraint of this design, not of the hardware.

INHERITED, NOT INTRODUCED: the K/V cache fills are single BDs against a fifo whose object is one
chunk, which is the shape `tmatvec/design.py` records as a KNOWN DEFECT (correct standalone, racy
across repeated invocations in one runtime sequence). The shipped graph already runs TMatVec that
way once per layer; this does not make it worse and does not fix it.
"""

import aie.dialects.index as index
import aie.extras.dialects.arith as arith
from aie.dialects.aie import T
from ml_dtypes import bfloat16
import math
import numpy as np

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, Kernel, ObjectFifo, Program, Runtime, ScratchpadParameter,
                      TaskGroup, Worker, WorkerRuntimeBarrier, sync_parameters)

from iron.operators._trace import maybe_enable_trace

BF16 = bfloat16

# AIE2P core-tile local memory. Stated, not derived: the Python bindings expose no accessor
# (AIETargetModel::getLocalMemorySize() is C++ only).
FLASH_SM_VEC_LEN = 64   # mirrors aie_kernels/aie2p/flash_contract.h; a mismatch drops a tail
# The r of mv.cc's r-wide chunking. Passed to the build as -DVEC_SIZE and asserted against both
# reduction lengths below, so the value and the shapes it constrains cannot drift apart.
GEMV_VEC_SIZE = 64
L1_BYTES = 65536


def _flat_tap(total, size, offset=0):
    """A contiguous [offset:offset+size) window of an L3 buffer of `total` elements. `total` is the
    FULL declared size -- TensorAccessPattern validates offset+extent against it, so a bare
    (size,) is correct only at offset 0."""
    return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def l1_footprint_bytes(D, HD, gqa, L, tile_elems, weight_depth, stack_size):
    """Bytes this design places in one core's L1, by term. Computed rather than assumed -- the
    same check qkv_head_dp and swiglu_mlp_dp carry.

    `L` is the SPLIT length in positions, not the window. Since split-K landed, max_seq does not
    enter this budget at all: sc/sw are sized to one segment and the running max/sum carry the
    result across segments. Before that this argument was max_seq and capped the window at 4544.
    """
    misc = 3 * (HD * 2)                 # depth 3: n_qn, n_kn and ang are held together
    stream = weight_depth * (tile_elems * 2)
    out = 2 * (HD * 2)
    persistent = (
        3 * (D * 2)                     # cur, n_in, hn
        + 2 * (HD * 2)                  # raw, nrm
        + gqa * (HD * 2)                # the RoPE'd query heads, which never leave L1
        + 2 * gqa * (L * 2)             # sc and sw -- sized to one SPLIT, not to the window
        + gqa * (HD * 4)                # the f32 context accumulators
        + gqa * (3 * 4)                 # {running max, running sum, correction} f32, per group
    )
    return misc + stream + out + persistent + stack_size


def derive_attn_split(D, HD, gqa, S, tile_elems, weight_depth, stack_size, kv_block_size=None,
                      tile_size_input=4):
    """The largest legal segment length for window `S`, or `S` itself when the window fits L1.

    ONE owner for the rule, because `decode_layer_dp`'s construction check and this file's own
    build both need it and a second copy would drift. A window at or below the L1 bound returns
    `S`, which is the pre-split design byte for byte -- so a rung ladder spanning widths either
    side of the cap needs no per-rung configuration.
    """
    rpc = (tile_size_input * D) // HD
    gran = math.lcm(rpc, kv_block_size or S, FLASH_SM_VEC_LEN)
    if l1_footprint_bytes(D, HD, gqa, S, tile_elems, weight_depth, stack_size) <= L1_BYTES:
        return S
    fixed = l1_footprint_bytes(D, HD, gqa, 0, tile_elems, weight_depth, stack_size)
    per = l1_footprint_bytes(D, HD, gqa, 1, tile_elems, weight_depth, stack_size) - fixed
    cap = (L1_BYTES - fixed) // per
    return max((d for d in range(gran, min(cap, S) + 1, gran) if S % d == 0), default=0)


def attn_block_dp(
    dev,
    D,
    HD,
    Hq,
    Hkv,
    max_seq,
    attn_split=None,
    scores_rowbatch=1,
    epsilon=1e-6,
    tile_size_input=4,
    stack_size=0xD00,
    func_prefix="",
    n_aie_cols=8,
    kv_offset_parameter="kv_off",
    mask_parameter="sm_mask",
    window_parameter=None,   # None: the segment count stays a build constant. A name: read at runtime.
    trace_size=0,
    weight_depth=2,
    wqkv_head_major=False,
    kv_alloc=None,
    kv_block_size=None,
    fifo_prefix="",
    parts_only=False,
    norms_packed=False,
):
    """`func_prefix` is not optional once this design is placed in an OperatorSequence -- see
    gemv/design.py's identical parameter. N = n_aie_cols, one core per KV HEAD.

    `wqkv_head_major` picks the WEIGHT LAYOUT, and it is a real trade, not a tidy-up. False is the
    stock [Wq | Wk | Wv] blob, which costs THREE fills per core because this core's four heads are
    not adjacent in it -- 24 `dma_await_task` where the shipped op0 spends 8. True expects the blob
    pre-permuted to [core0's q,q,k,v | core1's ... ], one contiguous run per core, ONE fill. Same
    bytes either way; what moves is the await count, and the shipped graph's four designs spend 69
    awaits per layer against this design's 77 stock and 61 head-major. The permutation is a
    build-time numpy reorder of a blob this generator already writes, so it costs nothing at
    runtime -- but it makes the artifact incompatible with a loader expecting the stock order,
    which is why it is a parameter and not the only behaviour.

    `kv_alloc` (None default) separates the cache CAPACITY from `max_seq`, which stays the WINDOW:
    sc/sw, the KV-chunk loop and the mask are all unchanged, still S=max_seq wide. `kv_alloc` only
    sizes the KV_L3 buffer and the per-head stride, so a resident ladder of these designs at
    different windows can share one wide cache. `kv_block_size` (also None) blocks that cache the
    way gemv/tmatvec's `block_size` blocks theirs, one axis over. See the KV_ALLOC block below."""
    N = n_aie_cols
    tsi = tile_size_input
    S = max_seq
    QD, KVD = Hq * HD, Hkv * HD
    TOT = QD + 2 * KVD                      # rows of the concatenated Wqkv
    assert Hkv == N, (
        f"this design places one KV HEAD per core: Hkv ({Hkv}) must equal n_aie_cols ({N}). "
        f"It is TMatVec's own 'one matrix per column' rule, now binding on every stage."
    )
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})"
    gqa = Hq // Hkv
    assert D % HD == 0, f"this design carries `cur`/`n_in` as D/HD chunks; D={D} HD={HD}"
    assert HD % tsi == 0, f"HD ({HD}) must divide by tile_size_input ({tsi})"

    N_MISC_CHUNKS = D // HD
    N_W_TILES = HD // tsi                   # weight tiles per head row-block
    TILE_ELEMS = tsi * D

    # THE SHARED-TILE INVARIANT. One ObjectFifo carries Wqkv row-tiles (tsi rows of D) AND cache
    # row-chunks (rpc rows of HD); it works only because the two are the same number of elements.
    # Asserted rather than commented: it is what collapses two input channels into one, and a
    # shape where it fails needs a different tiling, not a partial fill.
    assert TILE_ELEMS % HD == 0, (
        f"the shared stream tile ({TILE_ELEMS} elements) must be a whole number of cache rows "
        f"(HD={HD})"
    )
    rpc = TILE_ELEMS // HD                  # cache rows per stream tile
    assert S % rpc == 0, (
        f"max_seq ({S}) must divide by the cache rows per stream tile ({rpc}), which "
        f"tile_size_input={tsi} and head_dim={HD} fix at tsi*D/HD"
    )

    # THE SPLIT. sc/sw are sized to L positions and the softmax carries a running max/sum across
    # segments, so L1 stops depending on the window entirely. attn_split=None keeps L == max_seq,
    # which is one segment and therefore byte-for-byte the pre-split design.
    # attn_split=None DERIVES the split instead of meaning "one segment", which is what lets a
    # window RUNG LADDER span widths that straddle the L1 cap without its caller knowing where the
    # cap is. A window that already fits keeps L == S and is byte-for-byte the pre-split design --
    # so every rung at or below 4542 positions is untouched, and only the wide ones segment.
    #
    # Derived rather than global for a concrete reason: gen_llm_decode builds each rung with the
    # SAME attn_split, and the shipped ladder carries rungs at 256 and 512. A global 1024 fails
    # `S % L` on both. The split is a property of the window, not of the build.
    if attn_split is not None:
        L = attn_split
    else:
        L = derive_attn_split(D, HD, gqa, S, TILE_ELEMS, weight_depth, stack_size,
                              kv_block_size, tsi)
        if not L:
            raise ValueError(
                f"no legal attn_split for max_seq={S}: need a divisor of {S} that is a multiple "
                f"of lcm(stream-tile rows={rpc}, kv block={kv_block_size or S}, softmax vector="
                f"{FLASH_SM_VEC_LEN}) and within the L1 bound. Pick a max_seq with one, or pass "
                f"attn_split explicitly."
            )
    assert S % L == 0, f"attn_split ({L}) must divide max_seq ({S})"
    assert L % rpc == 0, (
        f"attn_split ({L}) must be a whole number of stream tiles (rpc={rpc})"
    )
    assert L % FLASH_SM_VEC_LEN == 0, (
        f"attn_split ({L}) must be a whole number of softmax vectors ({FLASH_SM_VEC_LEN}): the "
        f"kernel's loops have no scalar tail, so a remainder is dropped silently"
    )
    NSPLIT = S // L
    SPLIT_CHUNKS = L // rpc

    used = l1_footprint_bytes(D, HD, gqa, L, TILE_ELEMS, weight_depth, stack_size)
    assert used <= L1_BYTES, (
        f"estimated L1 use {used} B exceeds {L1_BYTES} B at tsi={tsi} attn_split={L} gqa={gqa}. "
        f"sc+sw alone are {2 * gqa * L * 2} B and are the terms the SPLIT drives (max_seq={S} no "
        f"longer enters) -- lower attn_split, or shrink tile_size_input to free "
        f"{weight_depth * TILE_ELEMS} B per step."
    )

    # KV CAPACITY, separate from the WINDOW (S=max_seq, above -- unchanged, still what sizes sc/sw
    # and the segment count). `kv_alloc` is None by default, which keeps capacity == window, byte for
    # byte. `kv_block_size` blocks that capacity the way gemv/tmatvec's `block_size` blocks theirs.
    # iron.common.kv_layout.KVLayout is the single owner of the offset/stride arithmetic this needs
    # -- gemv/tmatvec's own block_size branches predate that module and restate the formula by
    # hand; this is the first caller that asks it instead.
    from iron.common.kv_layout import KVLayout, split_run, validate_block_size

    assert kv_alloc is None or kv_alloc >= S, (
        f"kv_alloc ({kv_alloc}) must be >= max_seq ({S}): it is the cache CAPACITY, not a second "
        f"window"
    )
    KV_ALLOC = S if kv_alloc is None else kv_alloc
    _KVT = KV_ALLOC if kv_block_size is None else kv_block_size
    kv_layout = KVLayout(Hkv=Hkv, S=KV_ALLOC, HD=HD, T=_KVT)  # validates KV_ALLOC % _KVT == 0
    kv_blocked = _KVT != KV_ALLOC
    if kv_blocked:
        # Stride-field bound: the same check gemv's MAX_STRIDE / tmatvec's hand-rolled assert make,
        # now checked once by the module both of those predate.
        validate_block_size(_KVT, HD, Hkv)
        # The WINDOW (not the allocation) is what tg3 streams, block by block starting at block 0
        # -- so it is the window that must be a whole number of blocks here.
        assert S % _KVT == 0, (
            f"blocked KV cache needs the attention window ({S}) to be a whole number of blocks "
            f"(kv_block_size={_KVT})"
        )
        _kv_num_blocks_window = S // _KVT
        _kv_blk_split = split_run(_KVT * HD)
        assert _kv_blk_split is not None, (
            f"blocked KV cache: no wrap-legal split for one block's run ({_KVT * HD} elements, "
            f"kv_block_size={_KVT})"
        )
        _kv_blk_hi, _kv_blk_lo = _kv_blk_split

    def _kv_read_tap(head_base):
        """This head's WINDOW (S positions) out of a cache allocated at KV_ALLOC. Unblocked: one
        contiguous S*HD run at head_base -- byte-identical to the pre-capacity flat tap when
        KV_ALLOC==S. Blocked: the window's S//kv_block_size blocks, block_stride apart, positions
        still delivered in order (block b holds positions [b*T,(b+1)*T) by KVLayout's own
        definition) so the core's chunked consumption is unaffected by where a block boundary
        falls relative to a stream tile."""
        if not kv_blocked:
            return _flat_tap(kv_layout.total_elems, S * HD, head_base)
        return TensorAccessPattern(
            tensor_dims=(kv_layout.total_elems,),
            offset=head_base,
            sizes=[1, _kv_num_blocks_window, _kv_blk_hi, _kv_blk_lo],
            strides=[0, kv_layout.block_stride, _kv_blk_lo, 1],
        )

    def _kv_split_tap(head_base, split):
        """Segment `split` of this head's window: the same walk _kv_read_tap does, over L positions
        instead of S. With attn_split=None there is one segment and this IS _kv_read_tap."""
        if not kv_blocked:
            return _flat_tap(kv_layout.total_elems, L * HD, head_base + split * L * HD)
        blocks_per_split = L // _KVT
        return TensorAccessPattern(
            tensor_dims=(kv_layout.total_elems,),
            offset=head_base + split * blocks_per_split * kv_layout.block_stride,
            sizes=[1, blocks_per_split, _kv_blk_hi, _kv_blk_lo],
            strides=[0, kv_layout.block_stride, _kv_blk_lo, 1],
        )

    kv_off_param = (ScratchpadParameter(kv_offset_parameter, np.int32)
                    if kv_offset_parameter is not None else None)
    mask_param = ScratchpadParameter(mask_parameter, np.int32)
    # Carries L, a WINDOW LENGTH in positions -- not a chunk count. A later task needs the length
    # itself for the DMA side, and one parameter meaning one thing is what keeps two consumers
    # (this core loop and that future fill) from disagreeing about what it means.
    win_param = (ScratchpadParameter(window_parameter, np.int32)
                 if window_parameter is not None else None)

    D_ty = np.ndarray[(D,), np.dtype[BF16]]
    HD_ty = np.ndarray[(HD,), np.dtype[BF16]]
    TILE_ty = np.ndarray[(TILE_ELEMS,), np.dtype[BF16]]
    SROW_ty = np.ndarray[(L,), np.dtype[BF16]]      # one SPLIT's scores, not the window
    ST_ty = np.ndarray[(3,), np.dtype[np.float32]]  # {running max, running sum, correction}
    ACC_ty = np.ndarray[(HD,), np.dtype[np.float32]]
    W_L3_ty = np.ndarray[(TOT * D,), np.dtype[BF16]]
    # PACKED NORM GAINS. n_in, n_qn and n_kn are three STATIC weight vectors; `ang` is not (the
    # host writes the RoPE angle row per token), so it stays its own argument. Packing the three
    # is a build-time concat of blobs the generator already emits, exactly as Wqkv is a concat of
    # Wq/Wk/Wv -- and it exists because aiecc caps a device at 16 host buffer arguments
    # (kMaxHostBOs, tools/aiecc/SidecarFiles.h), which a whole fused LAYER reaches at 17.
    NORMS = D + 2 * HD
    NORMS_L3_ty = np.ndarray[(NORMS,), np.dtype[BF16]]
    KV_L3_ty = np.ndarray[(kv_layout.total_elems,), np.dtype[BF16]]
    CX_L3_ty = np.ndarray[(QD,), np.dtype[BF16]]

    # ---- kernels: one archive, every core plays every role ----
    CORE_ARCHIVE = f"{func_prefix}attn_block_dp_core.a"
    copy_kernel = Kernel(
        f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE, [D_ty, HD_ty, np.int32, np.int32]
    )
    # Two declarations, ONE body: rms_norm.cc aliases the hd_ name onto the other, so the second
    # call site costs no program memory. They stay two declarations because an external func.func
    # is typed by memref shape and these callers are at D and at head_dim.
    wnorm_d_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_cols", CORE_ARCHIVE,
        [D_ty, D_ty, D_ty, np.int32, np.float32, np.float32]
    )
    wnorm_hd_kernel = Kernel(
        f"{func_prefix}hd_weighted_rms_norm_cols", CORE_ARCHIVE,
        [HD_ty, HD_ty, HD_ty, np.int32, np.float32, np.float32]
    )

    def rms_len(cols):
        """The (length, reciprocal) pair the norm takes, from ONE source so they cannot disagree.

        The kernel multiplies by the reciprocal rather than dividing, because a runtime float
        divide is the whole reason __divsf3 links into a core. Exact for the powers of two this
        rail uses, so it is not a precision change.
        """
        return cols, 1.0 / cols

    # K007: the shapes are picked here, so the divisibility the kernel's r-wide chunking needs is
    # asserted here. The runtime-K body cannot static_assert it the way the compile-time-K one does.
    assert D % GEMV_VEC_SIZE == 0 and HD % GEMV_VEC_SIZE == 0, (
        f"both reduction lengths must be a whole number of {GEMV_VEC_SIZE}-wide chunks, "
        f"got D={D}, head_dim={HD}"
    )
    # One runtime-K body under two names -- an external func.func is keyed by name and typed by
    # memref shape, and these callers are at D and at head_dim. The row-batched scores path takes K
    # as a template parameter, so when it is on the scores keep their own compile-time-K symbol.
    mv_kernel = Kernel(
        f"{func_prefix}matvec_rtk_bf16_bf16", CORE_ARCHIVE,
        [np.int32, np.int32, np.int32, TILE_ty, D_ty, HD_ty],
    )
    if scores_rowbatch == 1:
        sc_mv_kernel = Kernel(
            f"{func_prefix}sc_matvec_rtk_bf16_bf16", CORE_ARCHIVE,
            [np.int32, np.int32, np.int32, TILE_ty, HD_ty, SROW_ty],
        )
    else:
        sc_mv_kernel = Kernel(
            f"{func_prefix}sc_matvec_vectorized_bf16_bf16", CORE_ARCHIVE,
            [np.int32, np.int32, TILE_ty, HD_ty, SROW_ty],
        )
    rope_kernel = Kernel(f"{func_prefix}rope", CORE_ARCHIVE, [HD_ty, HD_ty, HD_ty, np.int32])
    mask_kernel = Kernel(
        f"{func_prefix}mask_bf16", CORE_ARCHIVE, [SROW_ty, np.int32, np.int32]
    )
    softmax_kernel = Kernel(
        f"{func_prefix}softmax_bf16", CORE_ARCHIVE, [SROW_ty, SROW_ty, np.int32]
    )
    # groups=1 on all three: this core runs the context reduction once PER QUERY HEAD against its
    # own accumulator, rather than once for the pair against a [gqa, HD] one. Same MACs, same A
    # tile read once from L1 for both -- and it drops the [gqa*S] softmax buffer and the [gqa*HD]
    # context buffer the grouped form would need, because each head's result is finished straight
    # into the drain tile.
    tz_kernel = Kernel(f"{func_prefix}taccum_zero_f32", CORE_ARCHIVE, [np.int32, ACC_ty])
    tr_kernel = Kernel(
        f"{func_prefix}taccum_rows_bf16_f32", CORE_ARCHIVE,
        [np.int32, np.int32, np.int32, np.int32, TILE_ty, SROW_ty, ACC_ty],
    )
    tf_kernel = Kernel(
        f"{func_prefix}taccum_finish_scaled_bf16", CORE_ARCHIVE, [np.int32, ST_ty, ACC_ty, HD_ty]
    )
    # Split-K's three. `softmax_kernel` above stays bound but unused on this path -- removing it is
    # a separate cleanup, and leaving it keeps one concern per change.
    psm_kernel = Kernel(
        f"{func_prefix}partial_softmax_f32state_bf16", CORE_ARCHIVE,
        [SROW_ty, SROW_ty, ST_ty, np.int32],
    )
    sinit_kernel = Kernel(f"{func_prefix}flash_state_init", CORE_ARCHIVE, [ST_ty])
    acc_rescale_kernel = Kernel(
        f"{func_prefix}acc_rescale_f32", CORE_ARCHIVE, [np.int32, ST_ty, ACC_ty]
    )

    misc_of = ObjectFifo(HD_ty, name=f"{fifo_prefix}misc", depth=3)
    stream_ofs = [ObjectFifo(TILE_ty, name=f"{fifo_prefix}stream_{c}", depth=weight_depth) for c in range(N)]
    out_ofs = [ObjectFifo(HD_ty, name=f"{fifo_prefix}out_{c}", depth=2) for c in range(N)]

    barriers = [WorkerRuntimeBarrier() for _ in range(N)]

    def core_fn(misc_c, stream_c, out_p, mask_src, win_src, barrier,
                cur_buf, nin_buf, hn_buf, raw_buf, nrm_buf, qh_bufs, sc_bufs, sw_bufs, acc_bufs,
                st_bufs,
                copy_k, wnorm_d_k, wnorm_hd_k, mv_k, rope_k,
                sc_mv_k, mask_k, softmax_k, tz_k, tr_k, tf_k,
                psm_k, sinit_k, acc_rescale_k):
        # Read AFTER wait_for_value(1), never before: the sequence calls sync_parameters() and only
        # then sets the barrier, so a read here sees THIS dispatch's value. Read earlier and it
        # samples the PREVIOUS dispatch's -- corruption with no clean recurrence.
        barrier.wait_for_value(1)
        mask_len = mask_src.read()
        win_len = win_src.read() if win_src is not None else None
        # nsplits is the SEGMENT trip count. CEIL, not floor: a window that is not a whole number
        # of segments still has to attend its tail, and floor would silently drop up to L-1
        # positions. Built with arith on the SSA value -- never Python `//`, which emits
        # arith.floordivsi instead of the intended op.
        nsplits = (arith.divsi(arith.addi(win_len, arith.constant(L - 1, T.i32())),
                               arith.constant(L, T.i32()))
                   if win_src is not None else NSPLIT)

        # The fill streams every built segment no matter what `nsplits` is -- the fill side has no
        # scratchpad parameter to shrink its own trip count by. A tile this core does not acquire
        # is not discarded, it is what the NEXT dispatch's first acquire returns: draining spends
        # the bytes the fill already moved, not the MACs. Guarded in Python, not emitted as an
        # `scf.if`, because the remainder is 0 on the constant-trip-count path and a zero-trip loop
        # is still a loop the unwindowed core program must not contain.
        def drain_remainder():
            # The fill always streams NSPLIT (K, V) segment pairs; a shorter runtime window
            # consumes fewer, and the unconsumed ones sit CONTIGUOUSLY at the tail because K and V
            # are interleaved per segment. So one drain at the end replaces the pre-split design's
            # two, and it counts tiles: SPLIT_CHUNKS per segment per cache, two caches.
            if win_src is not None:
                per_split = arith.constant(SPLIT_CHUNKS * 2, T.i32())
                total = arith.constant(NSPLIT * SPLIT_CHUNKS * 2, T.i32())
                for _ in range_(arith.subi(total, arith.muli(nsplits, per_split))):
                    stream_c.acquire(1)
                    stream_c.release(1)

        # step 1: rebuild cur and n_in from D/HD chunks, then hn = weighted_RMSNorm(cur, n_in).
        for i in range_(N_MISC_CHUNKS):
            ch = misc_c.acquire(1)
            copy_k(cur_buf, ch, HD, index.casts(T.i32(), i) * HD)
            misc_c.release(1)
        for i in range_(N_MISC_CHUNKS):
            ch = misc_c.acquire(1)
            copy_k(nin_buf, ch, HD, index.casts(T.i32(), i) * HD)
            misc_c.release(1)
        wnorm_d_k(cur_buf, nin_buf, hn_buf, *rms_len(D), epsilon)

        # step 2: n_qn, n_kn, ang -- read once per head, so acquired once and held.
        w3 = misc_c.acquire(3)
        nqn_t, nkn_t, ang_t = w3[0], w3[1], w3[2]

        # step 3: this core's gqa query heads. They stay in L1 -- this is the whole point; today
        # `q` is drained to DDR and read back by the scores GEMV eight times over.
        for g in range(gqa):
            for j in range_(N_W_TILES):
                row_off = index.casts(T.i32(), j) * tsi
                wt = stream_c.acquire(1)
                mv_k(tsi, row_off, D, wt, hn_buf, raw_buf)
                stream_c.release(1)
            wnorm_hd_k(raw_buf, nqn_t, nrm_buf, *rms_len(HD), epsilon)
            rope_k(nrm_buf, ang_t, qh_bufs[g], HD)

        # step 4: this core's k head, RoPE'd straight into the drain tile that appends it.
        kt = out_p.acquire(1)
        for j in range_(N_W_TILES):
            row_off = index.casts(T.i32(), j) * tsi
            wt = stream_c.acquire(1)
            mv_k(tsi, row_off, D, wt, hn_buf, raw_buf)
            stream_c.release(1)
        wnorm_hd_k(raw_buf, nkn_t, nrm_buf, *rms_len(HD), epsilon)
        rope_k(nrm_buf, ang_t, kt, HD)
        out_p.release(1)

        # step 5: this core's v head -- no norm, no RoPE, so the matvec writes the drain tile.
        vt = out_p.acquire(1)
        for j in range_(N_W_TILES):
            row_off = index.casts(T.i32(), j) * tsi
            wt = stream_c.acquire(1)
            mv_k(tsi, row_off, D, wt, hn_buf, vt)
            stream_c.release(1)
        out_p.release(1)
        misc_c.release(3)

        # steps 6-8, now ONE loop over window segments. sc/sw are L-wide, and what carries the
        # result across segments is the running {max, sum} in st_bufs plus a rescale of the f32
        # accumulator -- so nothing in this core's L1 scales with the window any more.
        for g in range(gqa):
            tz_k(1, acc_bufs[g])
            sinit_k(st_bufs[g])

        for sp in range_(nsplits):
            seg_lo = index.casts(T.i32(), sp) * L

            # scores for this segment. One A tile still serves BOTH query heads out of L1 -- the
            # group reuse gemv's batch_group buys with an access pattern is free here, because the
            # two heads sharing this kv head are on the same core.
            for i in range_(SPLIT_CHUNKS):
                row_off = index.casts(T.i32(), i) * rpc
                at = stream_c.acquire(1)
                for g in range(gqa):
                    if scores_rowbatch == 1:
                        sc_mv_k(rpc, row_off, HD, at, qh_bufs[g], sc_bufs[g])
                    else:
                        sc_mv_k(rpc, row_off, at, qh_bufs[g], sc_bufs[g])
                stream_c.release(1)

            # This segment's valid width, clamped into [0, L]. CLAMPED AT ZERO DELIBERATELY:
            # mask_bf16 loops `for i = unmasked_size; i < total_size` with no lower guard, so a
            # negative width -- which a segment wholly past n_past produces -- would write -inf
            # BEFORE the buffer. A wholly masked segment is a real case on the build-constant
            # window path, where nsplits is NSPLIT regardless of how few positions n_past has.
            seg_unmasked = arith.maxsi(
                arith.minsi(arith.subi(mask_len, seg_lo), arith.constant(L, T.i32())),
                arith.constant(0, T.i32()),
            )

            for g in range(gqa):
                mask_k(sc_bufs[g], seg_unmasked, L)
                # Unnormalised exp2 out; the running {max, sum} and this segment's correction all
                # land in st_bufs, because an IRON Kernel call discards its result and cannot hand
                # a scalar to the next kernel. The divide is deferred to tf_k -- the denominator is
                # not known until the last segment has been seen.
                psm_k(sc_bufs[g], sw_bufs[g], st_bufs[g], L)
                acc_rescale_k(HD, st_bufs[g], acc_bufs[g])

            # context for this segment, transposed-A over this core's V head.
            for i in range_(SPLIT_CHUNKS):
                w_off = index.casts(T.i32(), i) * rpc
                at = stream_c.acquire(1)
                for g in range(gqa):
                    # tr_k's 3rd arg is w_stride (mv_taccum.cc: w_in + g*w_stride + w_off), the
                    # spacing between GROUPS in a packed w buffer -- not a row length. groups=1
                    # makes it dead, but it is buffer geometry, so it follows sw's width to L.
                    tr_k(rpc, 1, L, w_off, at, sw_bufs[g], acc_bufs[g])
                stream_c.release(1)

        drain_remainder()
        for g in range(gqa):
            ct = out_p.acquire(1)
            tf_k(1, st_bufs[g], acc_bufs[g], ct)
            out_p.release(1)

    workers = []
    for c in range(N):
        workers.append(
            Worker(
                core_fn,
                [
                    misc_of.cons(), stream_ofs[c].cons(), out_ofs[c].prod(),
                    mask_param, win_param, barriers[c],
                    Buffer(D_ty, name=f"{fifo_prefix}cur_{c}"), Buffer(D_ty, name=f"{fifo_prefix}nin_{c}"),
                    Buffer(D_ty, name=f"{fifo_prefix}hn_{c}"),
                    Buffer(HD_ty, name=f"{fifo_prefix}raw_{c}"), Buffer(HD_ty, name=f"{fifo_prefix}nrm_{c}"),
                    [Buffer(HD_ty, name=f"{fifo_prefix}qh_{c}_{g}") for g in range(gqa)],
                    [Buffer(SROW_ty, name=f"{fifo_prefix}sc_{c}_{g}") for g in range(gqa)],
                    [Buffer(SROW_ty, name=f"{fifo_prefix}sw_{c}_{g}") for g in range(gqa)],
                    [Buffer(ACC_ty, name=f"{fifo_prefix}acc_{c}_{g}") for g in range(gqa)],
                    [Buffer(ST_ty, name=f"{fifo_prefix}st_{c}_{g}") for g in range(gqa)],
                    copy_kernel, wnorm_d_kernel, wnorm_hd_kernel, mv_kernel, rope_kernel,
                    sc_mv_kernel, mask_kernel, softmax_kernel, tz_kernel, tr_kernel, tf_kernel,
                    psm_kernel, sinit_kernel, acc_rescale_kernel,
                ],
                stack_size=stack_size,
            )
        )

    def sequence(*seq_args):
        if norms_packed:
            (cur, norms, wqkv, ang, kc, vc, cx, misc_p, stream_ps, out_cs) = seq_args
            nin_src, nqn_src, nkn_src = (norms, NORMS, 0), (norms, NORMS, D), (norms, NORMS, D + HD)
        else:
            (cur, nin, wqkv, nqn, nkn, ang, kc, vc, cx,
             misc_p, stream_ps, out_cs) = seq_args
            nin_src, nqn_src, nkn_src = (nin, D, 0), (nqn, HD, 0), (nkn, HD, 0)
        # THREE task groups, and the split is forced rather than chosen. A group boundary is a hard
        # barrier (Runtime.finish_task_group awaits at group CLOSE), and the invariant is
        # swiglu_mlp_dp's, stated one-directionally in time: every task in group k must be
        # reachable by the core using only groups <= k.
        #   tg1  the core's inputs up to the point it produces anything.
        #   tg2  the K/V append. It MUST close before tg3: the scores read this token's own row
        #        back out of the cache, so a merged group would read a stale one.
        #   tg3  the cache streams and the context drains. Fills precede drains inside ONE group,
        #        which is qkv_head_dp's own load-bearing shape -- splitting them deadlocks a core
        #        that interleaves input tiles with output rounds.
        # `wait=True` everywhere: a bare finish() with no waited task lowers to dma_free_task,
        # which recycles BD IDs at COMPILE time and emits no hardware wait. BD pools are per shim
        # TILE and shared across every objectFIFO mapped to it, so a later fill can reprogram a
        # descriptor whose transfer is still in flight and desync a lock count.
        sync_parameters()
        for c in range(N):
            barriers[c].set(1)

        tg1 = TaskGroup()
        misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg1)
        for buf, total, off, n in ((*nin_src, D), (*nqn_src, HD), (*nkn_src, HD)):
            misc_p.fill(buf, _flat_tap(total, n, off), wait=True, group=tg1)
        misc_p.fill(ang, _flat_tap(HD, HD), wait=True, group=tg1)
        for c in range(N):
            # The core consumes its heads in ONE order -- gqa query heads, then K, then V -- and
            # the layout decides whether that is one run or three. See `wqkv_head_major`.
            if wqkv_head_major:
                runs = [(c * (gqa + 2) * HD, (gqa + 2) * HD)]
            else:
                runs = [(gqa * c * HD, gqa * HD),                # q heads gqa*c .. gqa*c+gqa
                        ((Hq + c) * HD, HD),                     # k head c
                        ((Hq + Hkv + c) * HD, HD)]               # v head c
            for off, rows in runs:
                stream_ps[c].fill(
                    wqkv, _flat_tap(TOT * D, rows * D, off * D), wait=True, group=tg1
                )
        tg1.finish()

        tg2 = TaskGroup()
        for c in range(N):
            # [head][n_past][HD]: the head term is static, the position term is `kv_off` (element
            # units) patched into the BD base address per dispatch -- unchanged from the
            # StridedCopy this ultimately replaces, so the cache layout and the host's parameter
            # write are both untouched.
            for cache in (kc, vc):
                out_cs[c].drain(
                    cache, _flat_tap(kv_layout.total_elems, HD, kv_layout.head_base(c)),
                    wait=True, group=tg2, offset_parameter=kv_off_param,
                )
        tg2.finish()

        # ONE TASK GROUP PER SEGMENT, and the group count is a BD-budget constraint rather than a
        # style choice. A shim tile carries 16 simultaneously active buffer descriptors; this
        # design places its 8 cores in 2 COLUMNS, so one shim serves 4 of them, and a single group
        # holding every segment's fills would ask for 4 cores x NSPLIT x 2 caches -- 32 at
        # NSPLIT=4, which aiecc rejects with "Too many simultaneously active buffer descriptors on
        # tile (1,0)". Per segment it is 4 x 2 = 8, flat in NSPLIT, so a 16-segment window costs no
        # more descriptors than a 1-segment one.
        #
        # The (K, V) PAIRING inside a segment is what the core's loop requires: it reads segment
        # s's K, softmaxes, then reads segment s's V. A fill order that did not match is a wrong
        # answer rather than a deadlock, because both carry the same element count on one fifo.
        for sp in range(NSPLIT):
            tg = TaskGroup()
            for c in range(N):
                for cache in (kc, vc):
                    stream_ps[c].fill(
                        cache, _kv_split_tap(kv_layout.head_base(c), sp), wait=True, group=tg
                    )
            tg.finish()

        # The context drains get their own group, AFTER every segment. That satisfies the
        # one-directional invariant this block's header states -- the core produces cx only after
        # consuming the last segment, so every task here is reachable using only earlier groups --
        # and it keeps the final group off the 16-BD edge that folding it into the last segment's
        # group would sit exactly on (8 fills + 8 drains per shim).
        tg3 = TaskGroup()
        for c in range(N):
            for g in range(gqa):
                out_cs[c].drain(
                    cx, _flat_tap(QD, HD, (gqa * c + g) * HD), wait=True, group=tg3
                )
        tg3.finish()

    l3_types = ([D_ty, NORMS_L3_ty, W_L3_ty, HD_ty, KV_L3_ty, KV_L3_ty, CX_L3_ty] if norms_packed
                else [D_ty, D_ty, W_L3_ty, HD_ty, HD_ty, HD_ty, KV_L3_ty, KV_L3_ty, CX_L3_ty])
    handles = [misc_of.prod(),
               [of.prod() for of in stream_ofs], [of.cons() for of in out_ofs]]
    # `parts_only` hands the pieces to a caller that is building a LARGER aie.device out of this
    # half and another -- see decode_layer_dp. Nothing about the half changes; the caller supplies
    # `fifo_prefix` and `func_prefix` so the two halves' fifo names and kernel symbols stay
    # disjoint in the one device-wide symbol table, and concatenates the sequences.
    if parts_only:
        return dict(workers=workers, seq=sequence, l3_types=l3_types, handles=handles)

    rt = Runtime(sequence, l3_types + handles)

    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
