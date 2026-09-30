# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.extras.dialects.arith as arith
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
    sync_parameters,
)

"""
Transposed-A matvec: the reduction runs DOWN the rows of a row-major matrix.

    C[b][j] = sum over p of  W[b][p] * A[b // batch_group][p][j]

gemv's contraction is the other one -- one output per row, reducing ALONG a row -- and attention's
context step wants this one: out[d] = sum_p softmax[p] * V[p][d] with V stored [S][head_dim]. Doing
it as a dot product is what forces a physical transpose of the whole cache first.

THE MAPPING IS THE POINT, and it is why `n_matrices == cols` is required rather than convenient.
Core c owns matrix c and EVERY batch that reads it, so it streams that matrix ONCE and applies each
group member's own W vector out of L1. The alternative (splitting the output width across columns)
reads the same matrix batch_group times AND makes each row-run M/cols elements wide -- 32 B at
head_dim 128 over 8 columns, well under the ~128 B contiguity knee measured for this fabric, where
whole rows are 256 B and past it.

 - cols: AIE columns; must equal num_batches // batch_group (one matrix per column)
 - M: output width == the matrix's row width (head_dim)
 - K: reduction extent == the matrix's row COUNT (sequence length)
 - rows_per_chunk: rows of A streamed per kernel call; sets the L1 A tile (rows_per_chunk*M elems)
 - batch_group: batches sharing one matrix (GQA: query heads per kv head)
 - l1_bytes: core-tile local memory to size against; None = the AIE2P 64 KB
"""


# AIE2P core-tile local memory. Kept as a fallback default, NOT because it cannot be derived: the
# claim this comment used to make -- that the Python bindings expose no accessor and
# AIETargetModel::getLocalMemorySize() is C++ only -- is FALSE on this bindings build. Measured
# 2026-09-09: `get_target_model(int(dev.resolve())).get_local_memory_size()` returns 65536 for npu1
# and npu2 and every column variant, and the sibling gemv design now derives its budget that way.
# A callable that takes a `dev` should ask; this literal is for the paths that have none.
AIE2P_L1_BYTES = 65536
# The core's stack and locals. The stack alone defaults to 0x400, and this tree has twice paid for
# a frame that silently overwrote the objectFIFO buffers placed above it.
#
# THIS IS A POLICY CONSTANT AND IT UNDER-COUNTS -- measured, not suspected. Read off the generated
# ld.script at the Gemma-4 global geometry (M=512, K=6912, batch_group=16): real non-buffer usage
# is 1024 B of stack plus a DATA REGION that is NOT constant -- 7680 B at m_chunk=256/rpc=16 and
# 10240 B at m_chunk=128/rpc=64, so 8704 and 11264 B against this 4096. Both shapes still fit, but
# the second lands at 98.4% of L1 with 1024 B spare where this model reports 87.5%.
#
# It cannot be derived here: the data region is a property of the COMPILED program, which does not
# exist at construction time. So this check is necessary and not sufficient, and a shape the model
# passes near the ceiling must still be confirmed against a real ld.script before it is trusted.
L1_HEADROOM_BYTES = 4096


def l1_footprint_bytes(M, K, batch_group, rows_per_chunk, m_chunk=None, a_row_bytes=None):
    """Bytes this design places in one core's L1, by term.

    Without m_chunk only the A term scales with rows_per_chunk, which is why that is the knob the
    error names. With it, W is streamed per K-chunk instead of held whole, so NO term carries K and
    the footprint stops depending on the context length at all.

    `a_row_bytes`: bytes of ONE row of A when it is not plain bf16 -- a quantized row's own
    header+payload stride (iron.common.quant.row_stride_bytes). None (default) is Mc*2,
    byte-identical to before this parameter existed.
    """
    Mc = M if m_chunk is None else m_chunk
    a_row = Mc * 2 if a_row_bytes is None else a_row_bytes
    return (
        2 * rows_per_chunk * a_row      # A objectfifo, depth 2
        + (batch_group * K * 2 if m_chunk is None
           else 2 * batch_group * rows_per_chunk * 2)   # W: whole-K depth 1, or chunked depth 2
        + 2 * batch_group * Mc * 2      # C objectfifo, depth 2, bf16
        + batch_group * Mc * 4          # the f32 accumulator Buffer
    )


def largest_fitting_rows_per_chunk(M, K, batch_group, l1_bytes=None, m_chunk=None,
                                   a_row_bytes=None):
    """The largest legal rows_per_chunk that FITS, or 0 if no chunking makes this shape fit."""
    budget = AIE2P_L1_BYTES if l1_bytes is None else l1_bytes
    ok = [r for r in (1, 2, 4, 8, 16, 32, 64, 128, 256)
          if r <= K and K % r == 0
          and l1_footprint_bytes(M, K, batch_group, r, m_chunk, a_row_bytes) + L1_HEADROOM_BYTES
          <= budget]
    return max(ok) if ok else 0


def largest_fitting_m_chunk(M, K, batch_group, rows_per_chunk, l1_bytes=None):
    """The widest output chunk that fits, or 0 if none does.

    Widest rather than smallest because the tax m_chunk pays is re-reading W once per chunk, and
    because A's per-row contiguity is `m_chunk*2` bytes -- below the measured ~128 B knee a narrower
    chunk starts costing more in DMA efficiency than it saves in L1.
    """
    budget = AIE2P_L1_BYTES if l1_bytes is None else l1_bytes
    ok = [mc for mc in (16, 32, 64, 128, 256, 512, 1024)
          if mc <= M and M % mc == 0
          and l1_footprint_bytes(M, K, batch_group, rows_per_chunk, mc) + L1_HEADROOM_BYTES <= budget]
    return max(ok) if ok else 0


def _blocked_strides(K, blk, M, cols):
    """The blocked A layout's block count and strides, plus the field checks that bound them.

    ONE derivation, read by both the blocked tap and the blocked+m_chunk composition. The two
    used to be the same arithmetic written once and refused once; sharing it is what keeps a
    future edit from moving one and not the other.

    `K` (the REDUCED extent) rather than `alloc_K` sizes `num_blocks`: a windowed read still only
    walks the K rows actually consumed. `cols` stands in for iron.common.kv_layout.KVLayout's
    `Hkv` -- this design stays KV-agnostic, the same split of concerns as gemv/design.py.
    """
    from iron.common.kv_layout import split_run

    if K % blk:
        raise AssertionError(
            f"blocked TMatVec needs the reduced extent K ({K}) to be a whole number of blocks "
            f"(block_size={blk})"
        )
    num_blocks = K // blk
    head_stride, block_stride = blk * M, cols * blk * M
    # Fail loud in Python rather than let aiecc reject an out-of-range stride deep in MLIR
    # verification -- the same 20-bit shim / 4-byte-granule arithmetic gemv/design.py checks.
    max_granules, gran_elems = (1 << 20) - 1, 2
    if block_stride // gran_elems > max_granules:
        raise AssertionError(
            f"blocked TMatVec: block_stride ({block_stride} elements = "
            f"{block_stride // gran_elems} granules) exceeds the shim's 20-bit step field "
            f"({max_granules} granules) -- block_size={blk} is too large for cols={cols}"
        )
    return {"num_blocks": num_blocks, "head_stride": head_stride,
            "block_stride": block_stride, "split_run": split_run}


def check_l1_fits(M, K, batch_group, rows_per_chunk, l1_bytes=None, m_chunk=None,
                  a_row_bytes=None):
    """Raise if the tiling does not FIT. Returns the message, so callers pick the exception type.

    KERNEL-CONTRACT K008: the tiling must FIT, not merely divide -- and nothing downstream checks
    it. rows_per_chunk sets the A tile at rows_per_chunk*M, so the footprint scales with head_dim:
    the default 64 fits at M=128 (42.0 KB) and does not at M=256 (88.0 KB), where aiecc reports
    "'aie.tile' op Basic sequential allocation also failed" -- naming a TILE and not a SIZE, so it
    reads as a placement bug rather than "your chunk is too big". MEASURED 2026-09-07 bringing up
    Gemma-3-270M (M=256, batch_group=4): the default fails to build and 32 succeeds, which this
    arithmetic reproduces exactly.

    `a_row_bytes`: see l1_footprint_bytes -- a quantized A row's own byte stride.
    """
    budget = AIE2P_L1_BYTES if l1_bytes is None else l1_bytes
    used = l1_footprint_bytes(M, K, batch_group, rows_per_chunk, m_chunk, a_row_bytes)
    if used + L1_HEADROOM_BYTES <= budget:
        return None
    fits = largest_fitting_rows_per_chunk(M, K, batch_group, l1_bytes, m_chunk, a_row_bytes)
    Mc = M if m_chunk is None else m_chunk
    a = 2 * rows_per_chunk * (Mc * 2 if a_row_bytes is None else a_row_bytes)
    w = batch_group * K * 2 if m_chunk is None else 2 * batch_group * rows_per_chunk * 2
    c, acc = 2 * batch_group * Mc * 2, batch_group * Mc * 4
    # Report the BREAKDOWN, not just the total. Only A scales with rows_per_chunk, so when a shape
    # cannot fit at any chunk size the blocker is one of the other three and shrinking the chunk can
    # never help. MEASURED 2026-09-07 on Gemma-4-12B's global layers (head_dim 512, one kv head,
    # gqa_group 16): W alone is 16*2048*2 = 65536 B, the ENTIRE L1, before A, C or acc get a byte.
    # An earlier version of this message said "needs a MemTile stage for A" in that case, which
    # names the wrong operand and would have sent the reader to tune the one knob that does nothing.
    terms = f"A {a} + W {w} + C {c} + acc {acc}"
    if fits:
        advice = f"largest rows_per_chunk that fits here is {fits}."
    else:
        worst = max((w, "W (batch_group*K)"), (c, "C"), (acc, "acc"), key=lambda t: t[0])
        advice = (
            f"NO rows_per_chunk fits: A is the only term it scales, and the other three already "
            f"total {w + c + acc} B. The dominant one is {worst[1]} at {worst[0]} B -- shrinking "
            f"rows_per_chunk cannot help. "
        )
        # C+acc carries no K, so a shape blocked on it is blocked at every context length and every
        # chunk size; m_chunk is the only knob that moves it. Name the value rather than the knob --
        # two sessions have tuned rows_per_chunk against a term it does not scale.
        if m_chunk is None:
            mc = max((largest_fitting_m_chunk(M, K, batch_group, r, l1_bytes), r)
                     for r in (1, 2, 4, 8, 16, 32, 64) if r <= K and K % r == 0)
            advice += (
                f"m_chunk={mc[0]} at rows_per_chunk={mc[1]} fits -- it chunks C+acc AND streams W "
                f"per K-chunk, so the footprint stops carrying K."
                if mc[0] else
                "m_chunk does not rescue it either -- reduce batch_group."
            )
        else:
            advice += "reduce batch_group or widen m_chunk's divisor of M."
    return (
        f"TMatVec does not fit L1: {used} B ({terms}) + {L1_HEADROOM_BYTES} B headroom exceeds "
        f"{budget} B at M={M} K={K} batch_group={batch_group} rows_per_chunk={rows_per_chunk} "
        f"m_chunk={m_chunk}. " + advice
    )


def transposed_matvec(
    dev,
    cols,
    M,
    K,
    num_batches=1,
    batch_group=1,
    rows_per_chunk=64,
    kernel_object="mv_taccum.o",
    func_prefix="",
    verbose=False,
    alloc_K=None,
    l1_bytes=None,
    block_size=None,
    m_chunk=None,
    vector_size_parameter=None,
    weight_dtype="bf16",
    group_size=0,
    kernel_vector_size=64,
    layout="header_first",
    row_group=None,
    scale_dtype="f32",
):
    assert num_batches % batch_group == 0, (
        f"num_batches ({num_batches}) must be a multiple of batch_group ({batch_group})"
    )
    n_matrices = num_batches // batch_group
    assert n_matrices == cols, (
        f"this design places one matrix per column: num_batches//batch_group ({n_matrices}) "
        f"must equal cols ({cols})"
    )
    assert K % rows_per_chunk == 0, (
        f"rows_per_chunk ({rows_per_chunk}) must divide K ({K})"
    )
    # Rows ALLOCATED per matrix, when that differs from the rows REDUCED. Same idea as gemv's
    # alloc_M one axis over: the window is a row PREFIX here, so only the per-matrix stride moves.
    assert alloc_K is None or alloc_K >= K, (
        f"alloc_K ({alloc_K}) must be >= K ({K})"
    )
    _AK = K if alloc_K is None else alloc_K
    # BLOCKED A, one head per column so no cross-column head walk is needed here (unlike gemv's
    # M-split, which packs several matrices under one column) -- see gemv/design.py's block_size
    # docstring for the shared addressing shape. `block_size is None` is one block == `_AK`,
    # byte-identical to the pre-blocking layout.
    assert block_size is None or (block_size > 0 and _AK % block_size == 0), (
        f"block_size ({block_size}) must be a positive divisor of alloc_K ({_AK})"
    )
    _BLK = _AK if block_size is None else block_size
    blocked = _BLK != _AK

    n_chunks = K // rows_per_chunk

    # OUTPUT chunking. `C + acc = 8*batch_group*M` carries no K, so a shape blocked on it is blocked
    # at every context length -- that is the global attention geometry (M=512, batch_group=16),
    # 65536 B before A or W get a byte. Chunking M is the only knob that moves it, and because the
    # K-sweep is then re-run per chunk, W can stream per K-chunk instead of being held whole: the
    # footprint stops carrying K, so it is identical at S=2048 and S=262144. The tax is re-reading
    # W once per chunk. This is the INVERSE of mv_quant_taccum.cc's pattern -- there the REDUCTION
    # operand was chunked behind a small accumulator; here the accumulator is the too-wide thing.
    assert m_chunk is None or (0 < m_chunk <= M and M % m_chunk == 0), (
        f"m_chunk ({m_chunk}) must be a positive divisor of M ({M})"
    )
    _MC = M if m_chunk is None else m_chunk
    n_m_chunks = M // _MC
    chunked_m = m_chunk is not None
    # m_chunk COMPOSES with block_size in four dims, not the five this used to refuse for: each
    # mode's fourth dim goes on something the other makes unnecessary (blocked wrap-splits a whole
    # `_BLK*M` row-run and leaves d0 dead; an m-chunked read's run is just `_MC`). The composed
    # offset is `mc*_MC + b*block_stride + r*M + j` -- see the A tap below. Only when one block is
    # one K-chunk, because the L1 tile is `rows_per_chunk x _MC`.
    compose_blk_mc = chunked_m and blocked
    if compose_blk_mc and _BLK != rows_per_chunk:
        raise AssertionError(
            f"m_chunk ({m_chunk}) composes with block_size ({block_size}) only when one block is "
            f"one K-chunk -- block_size == rows_per_chunk ({rows_per_chunk}) -- because the L1 "
            f"tile is rows_per_chunk x m_chunk. Pick block_size={rows_per_chunk}, or a "
            f"rows_per_chunk that divides the block, which needs a 5th A dim this design does "
            f"not build."
        )

    # A's stream format. A row here is one CACHED POSITION (M wide), not an output feature, so
    # group_size divides M -- see mv_taccum_quant.cc's header and op.py's identical assert; this
    # generator's own asserts cover it being driven directly, same split as gemv/design.py.
    quantized = weight_dtype != "bf16"
    assert weight_dtype in ("bf16", "int4", "int8"), (
        f"unknown weight_dtype {weight_dtype!r} (expected 'bf16', 'int4' or 'int8')"
    )
    _ROW_BYTES = None
    if quantized:
        assert not chunked_m and not blocked, (
            "weight_dtype != 'bf16' does not compose with m_chunk or block_size yet"
        )
        assert vector_size_parameter is None, (
            "weight_dtype != 'bf16' does not support vector_size_parameter yet"
        )
        assert group_size > 0 and M % group_size == 0, (
            f"M={M} must be a whole number of groups (group_size={group_size})"
        )
        from iron.common.quant import row_stride_bytes

        _ROW_BYTES = row_stride_bytes(M, group_size, weight_dtype, scale_dtype)

    # K008: the tiling must FIT, not merely divide. op.py raises this at construction; the
    # assert here covers the generator being driven directly.
    _msg = check_l1_fits(M, K, batch_group, rows_per_chunk, l1_bytes, m_chunk, _ROW_BYTES)
    assert _msg is None, _msg

    A_ELEM_ty = np.int8 if quantized else bfloat16
    _A_ROW_UNITS = _ROW_BYTES if quantized else M   # bytes/row when quantized, else elements/row
    L1_A_ty = np.ndarray[(rows_per_chunk * (_ROW_BYTES if quantized else _MC),),
                         np.dtype[A_ELEM_ty]]
    L1_W_ty = np.ndarray[
        (batch_group * (rows_per_chunk if chunked_m else K),), np.dtype[bfloat16]
    ]
    L1_C_ty = np.ndarray[(batch_group * _MC,), np.dtype[bfloat16]]
    ACC_ty = np.ndarray[(batch_group * _MC,), np.dtype[np.float32]]

    L3_A_ty = np.ndarray[(n_matrices * _AK * _A_ROW_UNITS,), np.dtype[A_ELEM_ty]]
    L3_W_ty = np.ndarray[(num_batches * K,), np.dtype[bfloat16]]
    L3_C_ty = np.ndarray[(num_batches * M,), np.dtype[bfloat16]]

    # The fused dispatch prefixes BOTH the symbol and the object FILENAME with op{idx}_, so the
    # object reference has to carry func_prefix too -- prefixing only the symbol builds an object
    # nothing links against ("cannot open tmv_128n.o").
    obj = f"{func_prefix}{kernel_object}"
    k_zero = Kernel(f"{func_prefix}taccum_zero_f32", obj, [np.int32, ACC_ty])
    k_rows = Kernel(
        f"{func_prefix}taccum_rows_{'bf16' if not quantized else weight_dtype}_f32",
        obj,
        [np.int32, np.int32, np.int32, np.int32, L1_A_ty, L1_W_ty, ACC_ty],
    )
    k_finish = Kernel(
        f"{func_prefix}taccum_finish_bf16", obj, [np.int32, ACC_ty, L1_C_ty]
    )

    A_fifos = [ObjectFifo(L1_A_ty, name=f"A_L3L1_{c}", depth=2) for c in range(cols)]
    W_fifos = [
        ObjectFifo(L1_W_ty, name=f"W_L3L1_{c}", depth=2 if chunked_m else 1)
        for c in range(cols)
    ]
    C_fifos = [ObjectFifo(L1_C_ty, name=f"C_L1L3_{c}", depth=2) for c in range(cols)]

    # Runtime reduction extent -- see gemv/design.py's RUNTIME ROW EXTENT block for the mechanism
    # and its exactness condition. Here the axis is K, and `cols` is one column per kv head, so
    # there is no per-column floor: cost follows the runtime value all the way down.
    runtime_k = vector_size_parameter is not None
    ks_param = (
        ScratchpadParameter(vector_size_parameter, np.int32) if runtime_k else None
    )

    def active_chunks(width):
        """Chunks covering `width` rows, CEIL and clamped into [0, n_chunks].

        CEIL because a width that is not a whole number of chunks still has to reduce its tail;
        clamped above because the parameter is an upper-BOUNDED runtime value, never a way to
        exceed what was built.
        """
        up = arith.addi(width, arith.constant(rows_per_chunk - 1, T.i32()))
        return arith.maxsi(
            arith.minsi(
                arith.divsi(up, arith.constant(rows_per_chunk, T.i32())),
                arith.constant(n_chunks, T.i32()),
            ),
            arith.constant(0, T.i32()),
        )

    def core_body(A_cons, W_cons, C_prod, acc, zero, rows, finish, ks_src=None, barrier=None):
        for _ in range_(0xFFFFFFFF):
            if runtime_k:
                # Read AFTER the barrier, never before: the sequence syncs the scratchpad and only
                # then sets it, so a read here sees THIS dispatch's value.
                barrier.wait_for_value(1)
                n_active = active_chunks(ks_src.read())
            w = W_cons.acquire(1)
            zero(batch_group, acc)
            for i in range_(n_active if runtime_k else n_chunks):
                a = A_cons.acquire(1)
                # w_off advances by whole chunks; the group stride is this batch's own W length.
                rows(rows_per_chunk, batch_group, K, i * rows_per_chunk, a, w, acc)
                A_cons.release(1)
            if runtime_k:
                for _ in range_(arith.subi(arith.constant(n_chunks, T.i32()), n_active)):
                    A_cons.acquire(1)
                    A_cons.release(1)
            c = C_prod.acquire(1)
            finish(batch_group, acc, c)
            C_prod.release(1)
            W_cons.release(1)

    def core_body_m_chunked(A_cons, W_cons, C_prod, acc, zero, rows, finish,
                            ks_src=None, barrier=None):
        for _ in range_(0xFFFFFFFF):
            if runtime_k:
                barrier.wait_for_value(1)
                n_active = active_chunks(ks_src.read())
            for _mc in range_(n_m_chunks):
                zero(batch_group, acc)
                for _i in range_(n_active if runtime_k else n_chunks):
                    a = A_cons.acquire(1)
                    w = W_cons.acquire(1)
                    # Each W object is this chunk's [batch_group][rows_per_chunk], so the group
                    # stride is the chunk and the offset is 0 -- the whole-K form's w_off walk is
                    # what the tap now does in the shim.
                    rows(rows_per_chunk, batch_group, rows_per_chunk, 0, a, w, acc)
                    W_cons.release(1)
                    A_cons.release(1)
                if runtime_k:
                    # Per M-chunk, not once at the end: A and W interleave per chunk, so the
                    # surplus of one M-chunk sits before the next M-chunk's first active tile.
                    for _ in range_(arith.subi(arith.constant(n_chunks, T.i32()), n_active)):
                        A_cons.acquire(1)
                        W_cons.acquire(1)
                        W_cons.release(1)
                        A_cons.release(1)
                c = C_prod.acquire(1)
                finish(batch_group, acc, c)
                C_prod.release(1)

    # A: column c reads matrix c, whole contiguous rows, chunk by chunk. The per-matrix stride is
    # the ALLOCATION (_AK), not the reduced extent, so a windowed read still lands on the right one.
    # This single fill of K*M elements lowers to ONE BD against a fifo whose object is
    # rows_per_chunk*M, so 262144 elements stream through a depth-2 16384-element L1 buffer with no
    # per-object lock. That shape is real, and it is NOT a defect: probe_tmatvec_reinvoke.py ran
    # both arms on device 2026-09-15 and the operator is bit-identical (0/2048 differing, within and
    # across runs) when invoked twice in one runtime sequence with a separating design -- its gemv
    # control clean too. A 2026-09-07 comment here blamed the fused-decode nondeterminism on this
    # fill and prescribed routing A through a MemTile; both are withdrawn. op_ctx's own GEMV already
    # does the same thing harder -- same BD length into a SMALLER object, so twice the ring cycles,
    # shim->core with no MemTile -- and is clean on every layer.
    # The fused decode IS nondeterministic (22-51 of 336 buffers); it is in the graph around op_ctx,
    # not here. probe_decode_first_divergence.py is the instrument.
    if chunked_m:
        # A is [n_matrices][_AK][M]; this walks column c's own matrix as n_m_chunks column slices,
        # each swept in n_chunks row-blocks. Row stride is the FULL M, so the slice is strided and
        # its contiguous run is `_MC*2` bytes -- keep _MC >= 64 to stay above the ~128 B knee.
        #
        # Under `blocked` the ONLY differences are d1's stride and the base offset -- see the
        # compose_blk_mc note above for why four dims suffice and why W/C need no change.
        if compose_blk_mc:
            _blk_chk = _blocked_strides(K, _BLK, M, cols)
            _base, _d1_stride = _blk_chk["head_stride"], _blk_chk["block_stride"]
            _d1_size = _blk_chk["num_blocks"]
        else:
            _base, _d1_stride, _d1_size = _AK * M, rows_per_chunk * M, n_chunks
        A_taps = [
            TensorAccessPattern(
                tensor_dims=L3_A_ty.__args__[0],
                offset=c * _base,
                sizes=[n_m_chunks, _d1_size, rows_per_chunk, _MC],
                strides=[_MC, _d1_stride, M, 1],
            )
            for c in range(cols)
        ]
    elif blocked:
        # column c is head c (n_matrices == cols, asserted above): its `_AK` rows are
        # `num_blocks` blocks of `_BLK` rows, `cols` heads interleaved every block -- so column c's
        # OWN data starts at `c*head_stride` and its `num_blocks` blocks are `block_stride` apart
        # (`cols` standing in for iron.common.kv_layout.KVLayout's `Hkv`; this design stays
        # KV-agnostic, same split of concerns as gemv/design.py's block_size).
        #   sizes   = [1,            num_blocks,   run_hi, run_lo]
        #   strides = [0,            block_stride, run_lo, 1]
        # K (the REDUCED extent) rather than _AK sizes num_blocks: a windowed read (_AK > K) still
        # only walks the K rows actually consumed, same as the unblocked branch's `K * M` run does.
        _blk = _blocked_strides(K, _BLK, M, cols)
        num_blocks, head_stride = _blk["num_blocks"], _blk["head_stride"]
        block_stride, split_run = _blk["block_stride"], _blk["split_run"]
        blk_run_split = split_run(_BLK * M)
        assert blk_run_split is not None, (
            f"blocked TMatVec: no wrap-legal split for one block's run "
            f"({_BLK * M} elements, block_size={_BLK})"
        )
        blk_run_hi, blk_run_lo = blk_run_split
        A_taps = [
            TensorAccessPattern(
                tensor_dims=L3_A_ty.__args__[0],
                offset=c * head_stride,
                sizes=[1, num_blocks, blk_run_hi, blk_run_lo],
                strides=[0, block_stride, blk_run_lo, 1],
            )
            for c in range(cols)
        ]
    else:
        # Quantized: L3_A_ty is packed BYTES, one row _A_ROW_UNITS (= row_stride_bytes) wide
        # instead of M bf16 elements -- same substitution GEMV's own A tap makes, "elements" just
        # means "units of L3_A_ty" either way.
        A_taps = [
            TensorAccessPattern(
                tensor_dims=L3_A_ty.__args__[0],
                offset=c * _AK * _A_ROW_UNITS,
                sizes=[1, 1, 1, K * _A_ROW_UNITS],
                strides=[0, 0, 0, 1],
            )
            for c in range(cols)
        ]
    # W: column c takes its group's batches, which are contiguous because batches sharing a matrix
    # are consecutive by construction (batch b reads matrix b // batch_group).
    if chunked_m:
        # W is [num_batches][K]. Re-read whole per M-chunk (the tax), gathering each K-chunk's
        # slice across the group: batch_group runs of rows_per_chunk, K apart. The n_m_chunks
        # repeat is a zero stride, so it costs DDR reads but no extra descriptor.
        W_taps = [
            TensorAccessPattern(
                tensor_dims=L3_W_ty.__args__[0],
                offset=c * batch_group * K,
                sizes=[n_m_chunks, n_chunks, batch_group, rows_per_chunk],
                strides=[0, rows_per_chunk, K, 1],
            )
            for c in range(cols)
        ]
        C_taps = [
            TensorAccessPattern(
                tensor_dims=L3_C_ty.__args__[0],
                offset=c * batch_group * M,
                sizes=[1, n_m_chunks, batch_group, _MC],
                strides=[0, _MC, M, 1],
            )
            for c in range(cols)
        ]
    else:
        W_taps = [
            TensorAccessPattern(
                tensor_dims=L3_W_ty.__args__[0],
                offset=c * batch_group * K,
                sizes=[1, 1, 1, batch_group * K],
                strides=[0, 0, 0, 1],
            )
            for c in range(cols)
        ]
        C_taps = [
            TensorAccessPattern(
                tensor_dims=L3_C_ty.__args__[0],
                offset=c * batch_group * M,
                sizes=[1, 1, 1, batch_group * M],
                strides=[0, 0, 0, 1],
            )
            for c in range(cols)
        ]

    barriers = [WorkerRuntimeBarrier() for _ in range(cols)] if runtime_k else []

    workers = [
        Worker(
            core_body_m_chunked if chunked_m else core_body,
            [
                A_fifos[c].cons(),
                W_fifos[c].cons(),
                C_fifos[c].prod(),
                Buffer(type=ACC_ty, name=f"acc_{c}"),
                k_zero,
                k_rows,
                k_finish,
            ]
            + ([ks_param, barriers[c]] if runtime_k else []),
        )
        for c in range(cols)
    ]

    def sequence(A, W, C, A_prods, W_prods, C_conss):
        if runtime_k:
            sync_parameters()
            for c in range(cols):
                barriers[c].set(1)
        # Mirrors gemv's structure, and the shape matters. W is the LONG-LIVED operand -- the core
        # holds it across the whole chunk loop -- so it gets its OWN task group, finished LAST,
        # exactly as gemv does with B. Putting it in the same group as the per-chunk A fills and the
        # C drain, and interleaving the three per column, raced: two back-to-back invocations of
        # this design inside one runtime sequence (which is what 2+ decoder layers are) produced
        # different results run to run.
        tg_w = TaskGroup()
        for c in range(cols):
            W_prods[c].fill(W, W_taps[c], group=tg_w)
        # One group per chunk, finished before the next. A shim tile has 16 BDs, so issuing all
        # n_chunks fills for all columns at once ("Too many simultaneously active buffer
        # descriptors on tile (0,0)") is not an option -- gemv bounds the same way with its
        # per-wait groups.
        tg_ac = TaskGroup()
        for c in range(cols):
            A_prods[c].fill(A, A_taps[c], group=tg_ac)
        for c in range(cols):
            C_conss[c].drain(C, C_taps[c], group=tg_ac, wait=True)
        tg_ac.finish()
        tg_w.finish()

    rt = Runtime(
        sequence,
        [
            L3_A_ty,
            L3_W_ty,
            L3_C_ty,
            [f.prod() for f in A_fifos],
            [f.prod() for f in W_fifos],
            [f.cons() for f in C_fifos],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()
