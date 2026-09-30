# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gemma4-weightless-attention-block variant A_g: positions split across N columns, one worker
core per column applying ALL Hq query heads to that column's own K/V slice -- there is exactly
one kv head globally, so there is nothing to split BY except position, unlike attn_block_dp's
one-kv-head-per-core mapping. Each core computes a PARTIAL flash-softmax state and a PARTIAL f32
context, per head, over its own slice; `attn_global_merge` folds the N partials into the true
result. `attn_global_flash` is the one-design variant: runtime-length K/V streams and a
cascade merge, see its docstring.

Reuses attn_block_dp's split-K flash algebra verbatim per SEGMENT (scores, mask, partial softmax,
rescale, context accumulate -- `sc_matvec_rtk_bf16_bf16`, `mask_bf16`,
`partial_softmax_f32state_bf16`, `acc_rescale_f32`, `taccum_rows_bf16_f32`). The only new algebra
is the CROSS-COLUMN fold (`flash_merge_column`, `aie_kernels/aie2p/softmax.cc`): the same
online-softmax combination applied across columns instead of positions, because merging N
independent partial (max, sum, context) triples is the same problem run-once-more.

Inputs and separate operations:
  - no per-head norm or RoPE: `q` arrives already normed and rotated -- the existing per-head
    chain stays exactly as it is, upstream of this op;
  - no K/V append: StridedCopy stays a separate op, unabsorbed;
  - `vc` is read in full, HD-wide.

THE UNEVEN COLUMN SPLIT (`column_splits`). S positions do not divide evenly into N columns as a
whole number of FLASH_SM_VEC_LEN-wide segments when S/N is not itself a multiple of
FLASH_SM_VEC_LEN (Gemma-4's S=6912, N=8: 6912/8=864, not a multiple of 64). Padding a column's
read past its own slice risks reading past the cache's true allocation into whatever the arena
places next -- unlike attn_block_dp's tiny qkv padding, which reads into KNOWN adjacent content of
the SAME tensor. Instead, S's own S//FLASH_SM_VEC_LEN segments distribute across N columns as
evenly as possible (divmod), so every column's range is exact and a whole number of segments --a
BUILD-TIME difference in loop trip count between cores, never a runtime branch or any padding.

HEAD GROUPS, THE L1 FALLBACK (`head_groups`). Every head's context accumulator (f32, HD wide) and
RoPE'd query vector must be resident for the WHOLE column sweep, because online softmax needs a
running state per head across every segment it has seen -- so holding all `Hq` heads at once costs
`Hq` times one head's worth, which is the dominant L1 term at Hq=16. `head_groups>1` trades that
away: it holds only `Hq/head_groups` heads' state at a time and re-reads this column's ENTIRE K/V
slice once per group -- correctness-preserving (each group's online softmax is independent and
complete on its own), a straight L1-for-bytes trade, not a new algorithm.
"""

import aie.dialects.index as index
import aie.extras.dialects.arith as arith
from aie.dialects.aie import T
from ml_dtypes import bfloat16
import numpy as np

from aie.helpers.dialects.scf import _for as range_, else_, if_
from aie.iron import (Buffer, CascadeFlow, Kernel, ObjectFifo, Program, Runtime,
                      ScratchpadParameter, TaskGroup, Worker, WorkerRuntimeBarrier, sync_parameters)

from iron.operators._trace import maybe_enable_trace
from iron.operators.attn_block_dp.design import FLASH_SM_VEC_LEN, L1_BYTES, _flat_tap

BF16 = bfloat16


def column_splits(S, N, seg=FLASH_SM_VEC_LEN):
    """N per-column (start, length) pairs summing to S, each a multiple of `seg`. The first
    `(S//seg) % N` columns get one extra segment -- plain divmod, not a special case. At
    Gemma-4's S=6912, N=8 this is four columns of 896 positions and four of 832."""
    if S % seg:
        raise ValueError(f"S ({S}) must be a whole number of {seg}-position segments")
    total_segs = S // seg
    if total_segs < N:
        raise ValueError(f"S ({S}) has only {total_segs} segments of {seg}, fewer than N ({N})")
    base, extra = divmod(total_segs, N)
    starts, pos = [], 0
    for c in range(N):
        segs_c = base + (1 if c < extra else 0)
        starts.append((pos, segs_c * seg))
        pos += segs_c * seg
    assert pos == S, f"column_splits internal error: {pos} != {S}"
    return starts


def worker_l1_footprint_bytes(HD, gqa, weight_depth, stack_size):
    """Bytes ONE worker core places in L1. No norm/rope/v_norm terms (see the module docstring);
    the query-vector and accumulator terms scale with `gqa` (heads resident PER GROUP PASS), the
    dominant cost, because online softmax needs every resident head's running state live for the
    whole column sweep. `L` is fixed at FLASH_SM_VEC_LEN -- see the task Worklog for why that is
    the only legal value at this shape, not a free choice."""
    L = FLASH_SM_VEC_LEN
    misc = 2 * (HD * 2)                   # Q broadcast, depth 2 -- no n_qn/n_kn/ang held here
    stream = weight_depth * (HD * 2)      # rpc=1: one K or V row per stream tile
    out = 2 * ((HD + 2) * 4)              # partial-export tile, f32, double-buffered
    persistent = (
        gqa * (HD * 2)                   # qh_bufs: the (already RoPE'd) query heads, held resident
        + 2 * gqa * (L * 2)               # sc + sw, one segment wide
        + gqa * (HD * 4)                  # f32 context accumulators
        + gqa * (3 * 4)                   # {max, sum, correction} f32, per head
    )
    return misc + stream + out + persistent + stack_size


def flash_worker_l1_bytes(HD, heads, block, stack_size, last=True, rows_per_element=1):
    """L1 of one attn_global_flash worker: K/V fifos, two ScratchpadParameter reads, and the
    per-row-call-overhead fix's combined-per-head buffers -- qh/sc/sw/acc are each ONE resident
    [heads, .]-shaped buffer (not `heads` separate Buffers) so the score and context kernels take
    one call per row instead of `heads` (kb: attn-global-flash-per-row-call-overhead). Byte total
    is identical to the per-head-Buffer layout it replaces; only the state triple stays one Buffer
    per head, since partial_softmax/rescale/cascade-merge stay per-head, per-BLOCK calls. Only the
    `last` worker of the cascade chain has an output fifo. `rows_per_element` is the flash-
    multirow design's Step 2 lever (docs/superpowers/specs/2026-09-25-flash-multirow-mmul-
    design.md section 2): the K/V fifo element widens to that many rows, scaling only the fifo
    term -- 1 reproduces today's byte totals exactly."""
    fifos = 2 * 2 * (rows_per_element * HD * 2)  # K and V object fifos, depth 2, rows_per_element bf16 rows each
    params = 2 * 8                        # sm_mask and gf_loop, one 8-byte scratchpad slot each
    out = 2 * (HD * 2) if last else 0     # cx tile, bf16, double-buffered
    combined = (
        heads * (HD * 2)                  # qh_all: resident RoPE'd query rows, bf16
        + heads * 2 * (block * 2)         # sc_all + sw_all, one block wide per head, bf16
        + heads * (HD * 4)                # acc_all: f32 context accumulators
    )
    state = heads * (3 * 4)               # per-head {max, sum, correction} f32, unbatched
    return fifos + params + out + combined + state + stack_size


def flash_tap(HD, block, columns, col):
    """Column `col`'s round-robin view of a flat [capacity, HD] cache: one `block`-row block at
    block `col`, then every `columns`-th block. The static pattern is one block; the runtime
    length decides how many the DMA walks."""
    return ([1, 1, block, HD], [0, columns * block * HD, HD, 1], col * block * HD)


def attn_global_worker(
    dev,
    HD,
    Hq,
    S,
    n_aie_columns=8,
    head_groups=1,
    weight_depth=1,
    stack_size=0xD00,
    mask_parameter="sm_mask",
    func_prefix="",
    fifo_prefix="",
    trace_size=0,
):
    """One core per column (`n_aie_columns`, default 8). `q` is Hq*HD elements, already normed
    and RoPE'd, broadcast identically to every column. `kc`/`vc` are READ-ONLY here (the append is
    a separate, unabsorbed StridedCopy) at a flat `[S, HD]` layout (Hkv=1, T=S -- the only layout
    Gemma-4's global geometry ships today; a blocked cache is not supported by this function).
    `partials` is `n_aie_columns*Hq*(HD+2)` f32 elements, one (context, max, sum) triple per
    (column, head), written [column][head]-major -- `attn_global_merge` reads it in that same
    order, so the two must not disagree about the layout independently of this file.
    """
    N = n_aie_columns
    if Hq % head_groups:
        raise ValueError(f"Hq ({Hq}) must be a multiple of head_groups ({head_groups})")
    gqa = Hq // head_groups
    L = FLASH_SM_VEC_LEN
    splits = column_splits(S, N, L)

    used = worker_l1_footprint_bytes(HD, gqa, weight_depth, stack_size)
    if used > L1_BYTES:
        raise ValueError(
            f"L1 use {used} B exceeds {L1_BYTES} B at head_groups={head_groups} (gqa={gqa} heads "
            f"resident at once): qh_bufs+acc+st alone are {gqa * (HD * 2 + HD * 4 + 12)} B. "
            f"Raise head_groups (Hq={Hq} must stay a multiple of it)."
        )

    QD = Hq * HD
    HD_ty = np.ndarray[(HD,), np.dtype[BF16]]
    SROW_ty = np.ndarray[(L,), np.dtype[BF16]]
    ST_ty = np.ndarray[(3,), np.dtype[np.float32]]
    ACC_ty = np.ndarray[(HD,), np.dtype[np.float32]]
    PARTIAL_ty = np.ndarray[(HD + 2,), np.dtype[np.float32]]
    Q_L3_ty = np.ndarray[(QD,), np.dtype[BF16]]
    KV_L3_ty = np.ndarray[(S * HD,), np.dtype[BF16]]
    PARTIAL_L3_ty = np.ndarray[(N * Hq * (HD + 2),), np.dtype[np.float32]]

    CORE_ARCHIVE = f"{func_prefix}attn_global_dp_core.a"
    copy_kernel = Kernel(f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE,
                        [HD_ty, HD_ty, np.int32, np.int32])
    sc_mv_kernel = Kernel(f"{func_prefix}sc_matvec_rtk_bf16_bf16", CORE_ARCHIVE,
                         [np.int32, np.int32, np.int32, HD_ty, HD_ty, SROW_ty])
    mask_kernel = Kernel(f"{func_prefix}mask_bf16", CORE_ARCHIVE, [SROW_ty, np.int32, np.int32])
    tz_kernel = Kernel(f"{func_prefix}taccum_zero_f32", CORE_ARCHIVE, [np.int32, ACC_ty])
    tr_kernel = Kernel(
        f"{func_prefix}taccum_rows_bf16_f32", CORE_ARCHIVE,
        [np.int32, np.int32, np.int32, np.int32, HD_ty, SROW_ty, ACC_ty],
    )
    psm_kernel = Kernel(
        f"{func_prefix}partial_softmax_f32state_bf16", CORE_ARCHIVE,
        [SROW_ty, SROW_ty, ST_ty, np.int32],
    )
    sinit_kernel = Kernel(f"{func_prefix}flash_state_init", CORE_ARCHIVE, [ST_ty])
    acc_rescale_kernel = Kernel(
        f"{func_prefix}acc_rescale_f32", CORE_ARCHIVE, [np.int32, ST_ty, ACC_ty]
    )
    export_kernel = Kernel(
        f"{func_prefix}taccum_export_partial_f32", CORE_ARCHIVE,
        [np.int32, ST_ty, ACC_ty, PARTIAL_ty],
    )

    misc_of = ObjectFifo(HD_ty, name=f"{fifo_prefix}gmisc", depth=2)
    stream_ofs = [ObjectFifo(HD_ty, name=f"{fifo_prefix}gstream_{c}", depth=weight_depth)
                  for c in range(N)]
    out_ofs = [ObjectFifo(PARTIAL_ty, name=f"{fifo_prefix}gout_{c}", depth=2) for c in range(N)]
    mask_param = ScratchpadParameter(mask_parameter, np.int32)
    barriers = [WorkerRuntimeBarrier() for _ in range(N)]

    def make_core_fn(nsplit_c):
        def core_fn(misc_c, stream_c, out_p, mask_src, barrier,
                    qh_bufs, sc_bufs, sw_bufs, acc_bufs, st_bufs,
                    copy_k, sc_mv_k, mask_k, tz_k, tr_k, psm_k, sinit_k, acc_rescale_k, export_k):
            # Read AFTER wait_for_value(1), then re-arm (K027): an unreleased Acquire leaves the
            # lock at 1, so a later dispatch's wait would pass immediately and read stale mask_len.
            barrier.wait_for_value(1)
            mask_len = mask_src.read()
            barrier.release_with_value(1)

            for hg in range(head_groups):
                for g in range(gqa):
                    qt = misc_c.acquire(1)
                    copy_k(qh_bufs[g], qt, HD, 0)
                    misc_c.release(1)
                for g in range(gqa):
                    tz_k(1, acc_bufs[g])
                    sinit_k(st_bufs[g])

                for sp in range_(nsplit_c):
                    seg_lo = index.casts(T.i32(), sp) * L

                    for i in range_(L):
                        row_off = index.casts(T.i32(), i)
                        kt = stream_c.acquire(1)
                        for g in range(gqa):
                            sc_mv_k(1, row_off, HD, kt, qh_bufs[g], sc_bufs[g])
                        stream_c.release(1)

                    # CLAMPED AT ZERO -- same reason attn_block_dp's does: mask_bf16 has no lower
                    # guard, and a segment wholly past n_past would write -inf before the buffer.
                    seg_unmasked = arith.maxsi(
                        arith.minsi(arith.subi(mask_len, seg_lo), arith.constant(L, T.i32())),
                        arith.constant(0, T.i32()),
                    )
                    for g in range(gqa):
                        mask_k(sc_bufs[g], seg_unmasked, L)
                        psm_k(sc_bufs[g], sw_bufs[g], st_bufs[g], L)
                        acc_rescale_k(HD, st_bufs[g], acc_bufs[g])

                    for i in range_(L):
                        w_off = index.casts(T.i32(), i)
                        vt = stream_c.acquire(1)
                        for g in range(gqa):
                            tr_k(1, 1, L, w_off, vt, sw_bufs[g], acc_bufs[g])
                        stream_c.release(1)

                for g in range(gqa):
                    pt = out_p.acquire(1)
                    export_k(HD, st_bufs[g], acc_bufs[g], pt)
                    out_p.release(1)
        return core_fn

    workers = []
    for c in range(N):
        start_c, len_c = splits[c]
        nsplit_c = len_c // L
        workers.append(
            Worker(
                make_core_fn(nsplit_c),
                [
                    misc_of.cons(), stream_ofs[c].cons(), out_ofs[c].prod(),
                    mask_param, barriers[c],
                    [Buffer(HD_ty, name=f"{fifo_prefix}gqh_{c}_{g}") for g in range(gqa)],
                    [Buffer(SROW_ty, name=f"{fifo_prefix}gsc_{c}_{g}") for g in range(gqa)],
                    [Buffer(SROW_ty, name=f"{fifo_prefix}gsw_{c}_{g}") for g in range(gqa)],
                    [Buffer(ACC_ty, name=f"{fifo_prefix}gacc_{c}_{g}") for g in range(gqa)],
                    [Buffer(ST_ty, name=f"{fifo_prefix}gst_{c}_{g}") for g in range(gqa)],
                    copy_kernel, sc_mv_kernel, mask_kernel, tz_kernel, tr_kernel,
                    psm_kernel, sinit_kernel, acc_rescale_kernel, export_kernel,
                ],
                stack_size=stack_size,
            )
        )

    def sequence(*seq_args):
        (q, kc, vc, partials, misc_p, stream_ps, out_cs) = seq_args
        sync_parameters()
        for c in range(N):
            barriers[c].set(1)

        tgq = TaskGroup()
        misc_p.fill(q, _flat_tap(QD, QD, 0), wait=True, group=tgq)
        tgq.finish()

        # ONE TASK GROUP PER (column, group, segment): 2 BDs (K, V) each, same BD-budget
        # discipline as attn_block_dp's own per-segment groups -- see that file's comment on why
        # a shim's 16-BD ceiling forces per-segment granularity rather than one group per column.
        # Batching several columns' same-index segment into one group is a real pipelining lever
        # (fewer hard barriers) left for a device-timed follow-up, not built here.
        for c in range(N):
            start_c, len_c = splits[c]
            nsplit_c = len_c // L
            for hg in range(head_groups):
                for sp in range(nsplit_c):
                    tg = TaskGroup()
                    seg_off = (start_c + sp * L) * HD
                    stream_ps[c].fill(kc, _flat_tap(S * HD, L * HD, seg_off), wait=True, group=tg)
                    stream_ps[c].fill(vc, _flat_tap(S * HD, L * HD, seg_off), wait=True, group=tg)
                    tg.finish()

        # Drains grouped per (column, group): gqa BDs each, well under one shim's 16.
        for c in range(N):
            for hg in range(head_groups):
                tg = TaskGroup()
                for g in range(gqa):
                    h = hg * gqa + g
                    out_cs[c].drain(
                        partials,
                        _flat_tap(N * Hq * (HD + 2), HD + 2, c * Hq * (HD + 2) + h * (HD + 2)),
                        wait=True, group=tg,
                    )
                tg.finish()

    l3_types = [Q_L3_ty, KV_L3_ty, KV_L3_ty, PARTIAL_L3_ty]
    handles = [misc_of.prod(), [of.prod() for of in stream_ofs], [of.cons() for of in out_ofs]]
    rt = Runtime(sequence, l3_types + handles)
    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()


def attn_global_merge(
    dev,
    HD,
    Hq,
    n_aie_columns=8,
    stack_size=0xD00,
    func_prefix="",
    fifo_prefix="",
    trace_size=0,
):
    """Folds `attn_global_worker`'s N per-column partials into the true softmax result: ONE core,
    N sequential folds per head via `flash_merge_column` (the worker's own online-softmax
    combination, applied across columns), then `taccum_finish_scaled_bf16` -- attn_block_dp's own
    finish kernel, unmodified -- to normalise and narrow to bf16.

    All Hq heads' running state is held AT ONCE (Hq*(HD*4+12) bytes) rather than one head at a
    time, because the partials buffer arrives [column][head]-major (each worker drains its own Hq
    heads contiguously): consuming it in that same order needs every head's fold live across all N
    columns, not finished one at a time. The alternative (finish one head, then the next) would
    need the partials transposed to [head][column]-major, a second design question left alone.
    """
    N = n_aie_columns
    QD = Hq * HD
    HD_ty = np.ndarray[(HD,), np.dtype[BF16]]
    ST_ty = np.ndarray[(3,), np.dtype[np.float32]]
    ACC_ty = np.ndarray[(HD,), np.dtype[np.float32]]
    PARTIAL_ty = np.ndarray[(HD + 2,), np.dtype[np.float32]]
    PARTIAL_L3_ty = np.ndarray[(N * Hq * (HD + 2),), np.dtype[np.float32]]
    CX_L3_ty = np.ndarray[(QD,), np.dtype[BF16]]

    CORE_ARCHIVE = f"{func_prefix}attn_global_merge_core.a"
    tz_kernel = Kernel(f"{func_prefix}taccum_zero_f32", CORE_ARCHIVE, [np.int32, ACC_ty])
    sinit_kernel = Kernel(f"{func_prefix}flash_state_init", CORE_ARCHIVE, [ST_ty])
    merge_kernel = Kernel(
        f"{func_prefix}flash_merge_column", CORE_ARCHIVE,
        [np.int32, ST_ty, ACC_ty, PARTIAL_ty],
    )
    finish_kernel = Kernel(
        f"{func_prefix}taccum_finish_scaled_bf16", CORE_ARCHIVE,
        [np.int32, ST_ty, ACC_ty, HD_ty],
    )

    in_of = ObjectFifo(PARTIAL_ty, name=f"{fifo_prefix}gmerge_in", depth=2)
    out_of = ObjectFifo(HD_ty, name=f"{fifo_prefix}gmerge_out", depth=2)

    # No ScratchpadParameter anywhere in this op (it reads no runtime mask/window/offset), so no
    # sync_parameters()/barrier -- gemv/design.py's sequence() gates both the same way
    # (`if runtime_m:`). An unconditional sync_parameters() with nothing to sync emits a
    # zero-sized aiex.npu.create_scratchpad, which aiecc rejects outright.
    def core_fn(in_c, out_p, acc_bufs, st_bufs, tz_k, sinit_k, merge_k, finish_k):
        for g in range(Hq):
            tz_k(1, acc_bufs[g])
            sinit_k(st_bufs[g])
        for c in range(N):
            for g in range(Hq):
                pt = in_c.acquire(1)
                merge_k(HD, st_bufs[g], acc_bufs[g], pt)
                in_c.release(1)
        for g in range(Hq):
            ct = out_p.acquire(1)
            finish_k(1, st_bufs[g], acc_bufs[g], ct)
            out_p.release(1)

    worker = Worker(
        core_fn,
        [
            in_of.cons(), out_of.prod(),
            [Buffer(ACC_ty, name=f"{fifo_prefix}macc_{g}") for g in range(Hq)],
            [Buffer(ST_ty, name=f"{fifo_prefix}mst_{g}") for g in range(Hq)],
            tz_kernel, sinit_kernel, merge_kernel, finish_kernel,
        ],
        stack_size=stack_size,
    )

    def sequence(*seq_args):
        (partials, cx, in_p, out_c) = seq_args

        tg1 = TaskGroup()
        in_p.fill(partials, _flat_tap(N * Hq * (HD + 2), N * Hq * (HD + 2), 0),
                 wait=True, group=tg1)
        tg1.finish()

        tg2 = TaskGroup()
        for g in range(Hq):
            out_c.drain(cx, _flat_tap(QD, HD, g * HD), wait=True, group=tg2)
        tg2.finish()

    l3_types = [PARTIAL_L3_ty, CX_L3_ty]
    handles = [in_of.prod(), out_of.cons()]
    rt = Runtime(sequence, l3_types + handles)
    prog = Program(dev, rt, workers=[worker])
    maybe_enable_trace(prog, trace_size, [worker])
    return prog.resolve_program()


def attn_global_flash(dev, HD, Hq, capacity, n_aie_columns=8, heads_per_core=16, block=64,
                      rows_per_element=1,
                      mask_parameter="sm_mask", len_parameter="gf_len", loop_parameter="gf_loop",
                      stack_size=0xD00, func_prefix="", fifo_prefix="", trace_size=0):
    """One design: N columns x R = Hq / heads_per_core workers; worker (c, r) holds head group r
    (heads r*hpc .. r*hpc+hpc-1) over column c's round-robin blocks. Column c's K and V fifos
    broadcast to its R workers. Each head group's N workers chain 0 -> N-1 on the cascade, which
    merges their per-head partials (aie_kernels/aie2p/flash_cascade_f32.cc); the chain's last
    worker writes its heads of cx. Inputs q [Hq*HD], kc and vc flat [capacity*HD] (read only; the
    append stays a separate op), output cx [Hq*HD] bf16.

    Per token the host writes `len_parameter` = nb - 1 (addr kind: the K and V BDs are one block
    long at build time and grow by nb - 1 blocks) and `loop_parameter` = nb (core kind), where
    nb = ceil(ceil(n_live / block) / N), both from one value; `mask_parameter` carries n_live.
    Nothing in the DMA pattern bounds the walk at `capacity`: the host must never write an nb
    past capacity / (block * N).
    """
    N = n_aie_columns
    if heads_per_core not in (Hq, 4) or Hq % heads_per_core:
        raise NotImplementedError(f"heads_per_core={heads_per_core}: only Hq and 4 are implemented")
    R = Hq // heads_per_core
    if R > 4:
        raise ValueError(f"{R} head groups need {R} core rows per column; the device has 4")
    if N < 2:
        raise ValueError(f"n_aie_columns ({N}) must be at least 2: the cascade chain has two ends")
    if capacity % (block * N):
        raise ValueError(f"capacity {capacity} is not a whole number of {N} x {block}-row blocks")
    # Step 2 of the flash-multirow design: rows_per_element > 1 needs a worker whose G rows fit
    # entirely within ONE fifo element (G <= rows_per_element), which only heads_per_core=4 has
    # room for in L1 -- see the design's section 2 L1 table (hpc16 already sits at 61 KB at r=1).
    if rows_per_element > 1 and heads_per_core != 4:
        raise NotImplementedError(
            f"rows_per_element={rows_per_element} needs heads_per_core=4 (L1); "
            f"heads_per_core={heads_per_core} stays at rows_per_element=1"
        )
    if block % rows_per_element:
        raise ValueError(f"block ({block}) must be a whole number of rows_per_element ({rows_per_element})")
    if rows_per_element > 1 and rows_per_element % heads_per_core:
        # The Q-prefix copy (core_fn, RPE>1 branch) assumes a worker's G=heads_per_core rows sit
        # entirely inside ONE fifo element, which needs RPE a whole multiple of G.
        raise ValueError(
            f"rows_per_element ({rows_per_element}) must be a whole number of heads_per_core "
            f"({heads_per_core}) so a worker's rows land inside one fifo element"
        )
    if Hq % rows_per_element:
        raise ValueError(f"Hq ({Hq}) must be a whole number of rows_per_element ({rows_per_element})")
    used = flash_worker_l1_bytes(HD, heads_per_core, block, stack_size, last=True,
                                 rows_per_element=rows_per_element)
    if used > L1_BYTES:
        raise ValueError(f"attn_global_flash worker needs {used} B of L1 (> {L1_BYTES})")

    QD, G, RPE = Hq * HD, heads_per_core, rows_per_element
    HD_ty = np.ndarray[(HD,), np.dtype[BF16]]
    RHD_ty = np.ndarray[(RPE * HD,), np.dtype[BF16]]   # one K/V fifo element: RPE rows, RPE=1 today
    ST_ty = np.ndarray[(3,), np.dtype[np.float32]]
    # G resident heads, ONE buffer each instead of G separate Buffers -- lets the score and
    # context kernels take one call per row instead of G (kb: attn-global-flash-per-row-call-
    # overhead). qh_all/acc_all are [G][HD]; sc_all/sw_all are [G][block].
    QALL_ty = np.ndarray[(G * HD,), np.dtype[BF16]]
    SALL_ty = np.ndarray[(G * block,), np.dtype[BF16]]
    AALL_ty = np.ndarray[(G * HD,), np.dtype[np.float32]]
    Q_L3_ty = np.ndarray[(QD,), np.dtype[BF16]]
    KV_L3_ty = np.ndarray[(capacity * HD,), np.dtype[BF16]]

    CORE = f"{func_prefix}attn_global_dp_core.a"
    copy_k = Kernel(f"{func_prefix}copy_offset_bf16_vector", CORE, [QALL_ty, HD_ty, np.int32, np.int32])
    # Step 2 Q prefix (core_fn, RPE>1 branch, below) needs a source offset too.
    copy_g_k = Kernel(f"{func_prefix}copy_offset_src_bf16_vector", CORE,
                      [QALL_ty, RHD_ty, np.int32, np.int32, np.int32])
    # One call per row per fifo: all G heads' dot products against the acquired K row.
    sc_mv = Kernel(f"{func_prefix}sc_matvec_g_rtk_bf16_bf16", CORE,
                   [np.int32, np.int32, np.int32, np.int32, QALL_ty, HD_ty, SALL_ty])
    # Step 1 of docs/superpowers/specs/2026-09-25-flash-multirow-mmul-design.md: G=4 heads'
    # chains interleaved (mv.cc's matvec_g4_rows_rtk_gstride) and the context accumulator chunks
    # held across rows (mv_taccum.cc's taccum_rows_g4). hpc4 only -- see `heads_per_core == 4`
    # below; hpc16 keeps sc_mv/tr_k. `rows` is always 1 until Step 2 changes the fifo element.
    sc_mv_rows = Kernel(f"{func_prefix}sc_matvec_g_rows_bf16_bf16", CORE,
                        [np.int32, np.int32, np.int32, np.int32, np.int32, QALL_ty, HD_ty, SALL_ty])
    # Step 2: same symbol, RPE-row-wide kblk (a whole fifo element instead of one row) -- kept as
    # a separate Kernel object so sc_mv_rows (Step 1, already device-gated at RPE=1) is untouched.
    sc_mv_rows_e = Kernel(f"{func_prefix}sc_matvec_g_rows_bf16_bf16", CORE,
                          [np.int32, np.int32, np.int32, np.int32, np.int32, QALL_ty, RHD_ty, SALL_ty])
    tz_k = Kernel(f"{func_prefix}taccum_zero_f32", CORE, [np.int32, AALL_ty])
    tr_k = Kernel(f"{func_prefix}taccum_rows_bf16_f32", CORE,
                  [np.int32, np.int32, np.int32, np.int32, HD_ty, SALL_ty, AALL_ty])
    tr_rows_k = Kernel(f"{func_prefix}taccum_rows_g4_bf16_f32", CORE,
                       [np.int32, np.int32, np.int32, np.int32, HD_ty, SALL_ty, AALL_ty])
    tr_rows_k_e = Kernel(f"{func_prefix}taccum_rows_g4_bf16_f32", CORE,
                         [np.int32, np.int32, np.int32, np.int32, RHD_ty, SALL_ty, AALL_ty])
    # Per-BLOCK, still one call per head (1/64 as frequent as the row calls above) -- the _off
    # variants take a head offset into the combined sc_all/sw_all/acc_all buffers.
    mask_k = Kernel(f"{func_prefix}mask_bf16_off", CORE, [SALL_ty, np.int32, np.int32, np.int32])
    psm_k = Kernel(f"{func_prefix}partial_softmax_f32state_bf16_off", CORE,
                   [SALL_ty, np.int32, SALL_ty, np.int32, ST_ty, np.int32])
    sinit_k = Kernel(f"{func_prefix}flash_state_init", CORE, [ST_ty])
    resc_k = Kernel(f"{func_prefix}acc_rescale_f32_off", CORE, [np.int32, ST_ty, AALL_ty, np.int32])
    put_k = Kernel(f"{func_prefix}flash_cascade_put_f32", CORE, [np.int32, ST_ty, AALL_ty, np.int32])
    fold_put_k = Kernel(f"{func_prefix}flash_cascade_fold_put_f32", CORE,
                        [np.int32, ST_ty, AALL_ty, np.int32])
    fold_finish_k = Kernel(f"{func_prefix}flash_cascade_fold_finish_bf16", CORE,
                           [np.int32, ST_ty, AALL_ty, np.int32, HD_ty])

    def sfx(r):                                # hpc == Hq keeps the single-group names
        return "" if R == 1 else f"_{r}"

    kq_ofs = [ObjectFifo(RHD_ty, name=f"{fifo_prefix}gfk_{c}", depth=2) for c in range(N)]
    v_ofs = [ObjectFifo(RHD_ty, name=f"{fifo_prefix}gfv_{c}", depth=2) for c in range(N)]
    cx_ofs = [ObjectFifo(HD_ty, name=f"{fifo_prefix}gfcx{sfx(r)}", depth=2) for r in range(R)]

    mask_p = ScratchpadParameter(mask_parameter, np.int32)
    loop_p = ScratchpadParameter(loop_parameter, np.int32)
    len_p = ScratchpadParameter(len_parameter, np.int32)
    barriers = [[WorkerRuntimeBarrier() for _ in range(R)] for _ in range(N)]

    def i32(v):
        return arith.constant(v, T.i32())

    def make_worker(col, grp):
        def core_fn(kq, vq, m_src, l_src, barrier, qh, sc, sw, acc, st,
                    copy_, scmv, mask_, tz, tr, psm, sinit, resc, merge_, out=None):
            barrier.wait_for_value(1)
            n_live = m_src.read()
            nb = l_src.read()
            barrier.release_with_value(1)      # K027: re-arm after the read
            if RPE > 1:
                # Step 2 Q prefix: Hq/RPE elements. A worker's G rows are contiguous and G
                # divides RPE (asserted above), so they sit entirely inside ONE element, at a
                # local row offset that repeats every RPE/G groups.
                elem_of_grp = (grp * G) // RPE
                local_off = (grp * G) % RPE
                for e in range(Hq // RPE):
                    qt = kq.acquire(1)
                    if e == elem_of_grp:
                        copy_(qh, qt, G * HD, local_off * HD, 0)
                    kq.release(1)
            else:
                for h in range(Hq):             # Q prefix: all Hq rows on the K fifo, keep G
                    qt = kq.acquire(1)
                    if grp * G <= h < (grp + 1) * G:
                        copy_(qh, qt, HD, (h - grp * G) * HD)
                    kq.release(1)
            tz(G, acc)
            for g in range(G):
                sinit(st[g])
            for j in range_(nb):
                lo = arith.muli(arith.addi(i32(col), arith.muli(index.casts(T.i32(), j), i32(N))),
                                i32(block))
                live = arith.maxsi(arith.minsi(arith.subi(n_live, lo), i32(block)), i32(0))
                with if_(arith.cmpi(arith.CmpIPredicate.sgt, live, i32(0))) as has_live:
                    if RPE > 1:
                        # Step 2: block/RPE elements, RPE rows each -- one scmv call per element.
                        for e in range_(block // RPE):
                            kt = kq.acquire(1)
                            e_row0 = arith.muli(index.casts(T.i32(), e), i32(RPE))
                            scmv(RPE, G, e_row0, HD, block, qh, kt, sc)
                            kq.release(1)
                    else:
                        # ONE call per row covering all G heads (was G calls) -- the per-row-
                        # call-overhead fix. hpc4 (G==4) binds sc_matvec_g_rows_bf16_bf16, one
                        # rows=1 wider in its arg list than sc_matvec_g_rtk_bf16_bf16.
                        for i in range_(block):
                            kt = kq.acquire(1)
                            if G == 4:
                                scmv(1, G, index.casts(T.i32(), i), HD, block, qh, kt, sc)
                            else:
                                scmv(G, index.casts(T.i32(), i), HD, block, qh, kt, sc)
                            kq.release(1)
                    for g in range(G):
                        mask_(sc, g * block, live, block)
                        psm(sc, g * block, sw, g * block, st[g], block)
                        resc(HD, st[g], acc, g * HD)
                    if RPE > 1:
                        # P*V: ceil(live/RPE) elements, each rows=min(RPE, live - e*RPE) -- no V
                        # row at or past `live` is ever read (poison isolation, per-row today).
                        n_full_elems = arith.ceildivsi(live, i32(RPE))
                        for e in range_(n_full_elems):
                            vt = vq.acquire(1)
                            e_row0 = arith.muli(index.casts(T.i32(), e), i32(RPE))
                            rows_this = arith.minsi(i32(RPE), arith.subi(live, e_row0))
                            tr(rows_this, G, block, e_row0, vt, sw, acc)
                            vq.release(1)
                        for _ in range_(arith.subi(i32(block // RPE), n_full_elems)):
                            vq.acquire(1)
                            vq.release(1)
                    else:
                        # P*V over the unmasked rows only: a never-written V row may hold NaN.
                        # ONE call per row covering all G heads (was G calls).
                        for i in range_(live):
                            vt = vq.acquire(1)
                            tr(1, G, block, index.casts(T.i32(), i), vt, sw, acc)
                            vq.release(1)
                        for _ in range_(arith.subi(i32(block), live)):
                            vq.acquire(1)
                            vq.release(1)
                with else_(has_live):          # wholly masked: consume, never softmax
                    n_elems = block // RPE if RPE > 1 else block
                    for _ in range_(n_elems):
                        kq.acquire(1)
                        kq.release(1)
                    for _ in range_(n_elems):
                        vq.acquire(1)
                        vq.release(1)
            # Every worker joins the chain for every head, live blocks or not, or the chain stalls.
            for g in range(G):
                if col < N - 1:
                    merge_(HD, st[g], acc, g * HD)
                else:
                    ct = out.acquire(1)
                    merge_(HD, st[g], acc, g * HD, ct)
                    out.release(1)
        return core_fn

    def merge_kernel(col):
        return put_k if col == 0 else fold_put_k if col < N - 1 else fold_finish_k

    def make(c, r):
        n = f"{c}{sfx(r)}"
        if RPE > 1:
            copy_bind, scmv_bind, tr_bind = copy_g_k, sc_mv_rows_e, tr_rows_k_e
        elif G == 4:
            copy_bind, scmv_bind, tr_bind = copy_k, sc_mv_rows, tr_rows_k
        else:
            copy_bind, scmv_bind, tr_bind = copy_k, sc_mv, tr_k
        return Worker(make_worker(c, r), [
            kq_ofs[c].cons(), v_ofs[c].cons(), mask_p, loop_p, barriers[c][r],
            Buffer(QALL_ty, name=f"{fifo_prefix}gfq_{n}"),
            Buffer(SALL_ty, name=f"{fifo_prefix}gfsc_{n}"),
            Buffer(SALL_ty, name=f"{fifo_prefix}gfsw_{n}"),
            Buffer(AALL_ty, name=f"{fifo_prefix}gfacc_{n}"),
            [Buffer(ST_ty, name=f"{fifo_prefix}gfst_{n}_{g}") for g in range(G)],
            copy_bind, scmv_bind, mask_k, tz_k, tr_bind, psm_k, sinit_k, resc_k, merge_kernel(c),
        ] + ([cx_ofs[r].prod()] if c == N - 1 else []), stack_size=stack_size)

    grid = [[make(c, r) for r in range(R)] for c in range(N)]
    for r in range(R):
        for c in range(N - 1):
            CascadeFlow(grid[c][r], grid[c + 1][r])
    workers = [w for col_workers in grid for w in col_workers]

    def sequence(q, kc, vc, cx, kq_ps, v_ps, cx_cs):
        sync_parameters()
        for b in (b for col_barriers in barriers for b in col_barriers):
            b.set(1)
        tg = TaskGroup()
        for c in range(N):
            sizes, strides, off = flash_tap(HD, block, N, c)
            # [heads, HD] rather than a flat QD, which would overflow the 10-bit wrap.
            kq_ps[c].fill(q, sizes=[1, 1, Hq, HD], strides=[0, 0, HD, 1], offset=0,
                          transfer_len=QD, group=tg)
            kq_ps[c].fill(kc, sizes=sizes, strides=strides, offset=off, transfer_len=block * HD,
                          length_parameter=len_p, length_granule=block * HD, group=tg)
            v_ps[c].fill(vc, sizes=sizes, strides=strides, offset=off, transfer_len=block * HD,
                         length_parameter=len_p, length_granule=block * HD, group=tg)
        for r in range(R):                     # head group r lands at heads r*G .. r*G+G-1
            cx_cs[r].drain(cx, sizes=[1, 1, G, HD], strides=[0, 0, HD, 1], offset=r * G * HD,
                           transfer_len=G * HD, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [Q_L3_ty, KV_L3_ty, KV_L3_ty, Q_L3_ty,
                            [of.prod() for of in kq_ofs], [of.prod() for of in v_ofs],
                            [of.cons() for of in cx_ofs]])
    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
