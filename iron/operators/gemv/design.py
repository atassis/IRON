# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker

"""
Matrix-vector design

Calls into the mv.cc kernel code. That kernel computes `m_input` output rows per call.


 - cols: Number of AIE columns to split work across
 - M: number of rows in the matrix
 - K: number of columns in the matrix == number of rows in the vector
 - m_input: number of input rows stored on each AIE core == chunk size for data movement of input A
 - m_output: number of output rows stored on each AIE core == chunk size for data movement of output C
 - num_batches: number of iterations of this mat-vec to perform on contiguous matrices and vectors in memory (results concatenated)
"""


def my_matvec(
    dev,
    cols,
    M,
    K,
    m_input,
    m_output=None,
    num_batches=1,
    batch_group=1,
    kernel_object="mv.o",
    func_prefix="",
    verbose=False,
    epilogue="none",
    weight_dtype="bf16",
    group_size=0,
    alloc_M=None,
    alloc_M_out=None,
    barrier_chunk=1,
):
    if m_output is None:
        m_output = m_input

    if verbose:
        print(f"Device: {dev}")
        print(f"Matrix dimensions: M={M}, K={K}")
        print(f"Tiling: m_input={m_input}, m_output={m_output}")
        print(f"Columns: {cols}")

    # The reason for the following requirement is because we first acquire output rows from the C FIFO, then fill those acquiring rows of the A input.
    assert (
        m_output % m_input == 0 and m_output >= m_input
    ), "m_output must be a multiple of m_input"
    assert m_output <= M // cols, "m_output must be less than or equal to M/cols"
    assert (M // cols) % m_output == 0, "m_output must evenly divide M/cols"
    assert m_input <= M // cols, "m_input must be less than or equal to M/cols"
    assert (M // cols) % m_input == 0, "m_input must evenly divide M/cols"

    vectorized = True
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"
    # B (the vector) and C (the output) are always bf16 -- weight_dtype is an axis on A (the MxK
    # weight matrix) only, never on the activation/output path.
    dtype_b = np.dtype[bfloat16]

    assert M % cols == 0

    if weight_dtype == "bf16":
        dtype_in = np.dtype[bfloat16]
        dtype_in_str = "bf16"
        a_row_width = K  # elements/row, dtype_in-sized
    else:
        # Group-quantized A: see iron/operators/gemv/quant.py for the exact byte layout
        # (`[n_groups x f32 scale][payload]` per row) and aie_kernels/generic/mv_quant.cc for the
        # device-side dequant. Packing the scale into A's own buffer (rather than a 3rd FIFO) is
        # forced by the 2-input-DMA-channel budget: A and B already spend both.
        from iron.operators.gemv.quant import row_stride_bytes

        assert weight_dtype in ("int4", "int8"), f"unknown weight_dtype {weight_dtype!r}"
        assert num_batches == 1, (
            "GEMV weight_dtype != 'bf16' does not support num_batches>1 yet -- the per-column "
            "A layout assumes a single contiguous [row_stride-byte rows] slab per column "
            "(YAGNI: no quantized decode caller needs batching today)"
        )
        assert group_size > 0, "weight_dtype != 'bf16' needs an explicit group_size"
        dtype_in = np.dtype[np.int8]
        dtype_in_str = weight_dtype
        a_row_width = row_stride_bytes(K, group_size, weight_dtype)  # bytes/row, int8-sized

    L1_A_ty = np.ndarray[
        (
            m_input,
            a_row_width,
        ),
        dtype_in,
    ]
    L1_B_ty = np.ndarray[(K,), dtype_b]
    L1_C_ty = np.ndarray[(m_output,), dtype_out]
    # `batch_group` consecutive batches SHARE one matrix, so A holds num_batches//batch_group of
    # them, not num_batches. This is the GQA case: gqa_group query heads attend to one kv head, and
    # today that sharing is expressed by materialising a duplicate (Repeat) instead of by an access
    # pattern. batch_group=1 is the old behaviour exactly.
    assert num_batches % batch_group == 0, (
        f"num_batches ({num_batches}) must be a multiple of batch_group ({batch_group})"
    )
    n_matrices = num_batches // batch_group
    # A's rows ALLOCATED per matrix, which is not always the rows COMPUTED. The two differ whenever
    # a narrow window is read out of a buffer sized for a wider one -- decode attention reads
    # n_past rows of a KV cache allocated at max_seq. `M` stays the compute extent (it sizes the
    # run, the C tile and the core loop); `alloc_M` sizes the buffer and the per-matrix stride, so
    # a narrow read addresses the wide buffer correctly instead of walking off its own smaller one.
    assert alloc_M is None or alloc_M >= M, (
        f"alloc_M ({alloc_M}) must be >= M ({M}): it is the ALLOCATED row count, not a second window"
    )
    _AM = M if alloc_M is None else alloc_M
    # Rows ALLOCATED per output batch in C, when that differs from the rows COMPUTED (`M`) -- the
    # write-side mirror of alloc_M/_AM above. `M` stays the compute extent (it sizes the run and the
    # core loop); `alloc_M_out` sizes the C buffer and the per-batch stride, so a narrow write lands
    # inside a buffer allocated for a wider one instead of overlapping the next batch.
    assert alloc_M_out is None or alloc_M_out >= M, (
        f"alloc_M_out ({alloc_M_out}) must be >= M ({M}): it is the ALLOCATED row count per "
        f"output batch, not a second window"
    )
    _AMO = M if alloc_M_out is None else alloc_M_out
    L3_A_ty = np.ndarray[
        (n_matrices * _AM * a_row_width,),
        dtype_in,
    ]
    L3_B_ty = np.ndarray[(num_batches * K,), dtype_b]
    L3_C_ty = np.ndarray[(num_batches * _AMO,), dtype_out]

    func_type = "vectorized" if vectorized else "scalar"
    matvec = Kernel(
        f"{func_prefix}matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
        f"{func_prefix}{kernel_object}",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )
    # Optional fused activation over the full m_output C-tile, applied once per tile in core_body
    # (after the matvec inner-loop has filled all rows) rather than per matvec call, whose m_input
    # tile can be smaller than the 16-wide activation vector.
    assert epilogue in ("none", "gelu", "silu")
    gelu_kernel = None
    if epilogue != "none":
        # 32, not 16: both tile epilogues walk the C tile with a 32-lane iterator, so a tile that is
        # only 16-aligned makes the last iteration read and write past its end.
        assert (
            m_output % 32 == 0
        ), f"{epilogue} epilogue needs m_output % 32 == 0 (got {m_output})"
        gelu_kernel = Kernel(
            f"{func_prefix}{epilogue}_tile_bf16",
            f"{func_prefix}{kernel_object}",
            [np.int32, L1_C_ty],
        )

    MAX_WRAP = 1023
    # The shim NOC tile's BD step field is 20 bits -- not a convention, the hardware width:
    # `AIETargetModel.h::AIE2TargetModel::getDmaBdStepBits` returns 20 for ShimNOCTile (17 for a
    # MemTile, 13 for a core tile). But the field counts ADDRESS GRANULES, not elements:
    # `getAddressGenGranularity()` is 32 bits on AIE2/AIE2P, and `getHardwareStridesWraps` scales an
    # element stride by `elemWidth / addressGranularity` before `verifyStridesWraps` compares it.
    #
    # This bound was written in ELEMENTS and compared against the granule field width, which for
    # bf16 is exactly 2x too strict -- and its own FIXME below predicted it ("pull these shim BD
    # bounds from the MLIR-AIE target model rather than hard-coding them"). Cost, measured: the
    # decode's KV cache has a per-head stride of `alloc * head_dim` elements, so at head_dim=128 the
    # element bound caps the allocation at 8191 where the hardware allows 16383 -- one whole
    # doubling of the context a wide-allocation decode can address before its reads stop coalescing.
    MAX_STRIDE_GRANULES = (1 << 20) - 1
    GRAN_ELEMS = 2  # 4-byte shim granularity / 2-byte bf16 element
    # Elements per granule is a property of the ELEMENT TYPE, and this file hardcodes the bf16 ratio.
    # A quantized A is i8-typed (4 elements per granule), so the conversion would differ -- but
    # `num_batches` is asserted ==1 for a non-bf16 weight_dtype, which makes `coalesce` False and
    # this arithmetic unreachable on that path. Asserted rather than left to a comment.
    assert dtype_in_str == "bf16" or num_batches == 1, (
        f"coalescing arithmetic assumes {GRAN_ELEMS} elements/granule (bf16); "
        f"dtype_in={dtype_in_str} with num_batches={num_batches} would need its own ratio"
    )
    MAX_STRIDE = MAX_STRIDE_GRANULES * GRAN_ELEMS

    def split_run(run, lim=MAX_WRAP, gran=GRAN_ELEMS):
        """Factor a contiguous run into (hi, lo), both <= lim and lo a multiple of gran
        (the address-granularity-aligned inner size), lo maximal. None if no such
        split exists (caller then falls back to the per-batch path)."""
        lo_start = (lim // gran) * gran
        for lo in range(lo_start, 0, -gran):
            if run % lo == 0 and (run // lo) <= lim:
                return (run // lo, lo)
        return None

    A_run, A_bstride = (M // cols) * a_row_width, _AM * a_row_width
    C_run, C_bstride = (M // cols), _AMO
    A_split, C_split = split_run(A_run), split_run(C_run)
    coalesce = (
        num_batches > 1
        and num_batches % batch_group == 0
        and A_bstride <= MAX_STRIDE
        and C_bstride <= MAX_STRIDE
        and A_bstride % GRAN_ELEMS == 0
        and C_bstride % GRAN_ELEMS == 0
        and A_split is not None
        and C_split is not None
    )

    # GROUP REUSE: consume the shared matrix ONCE and run `batch_group` vectors over it, instead of
    # re-streaming it per group member.
    #
    # `batch_group` exists so GQA does not need a `Repeat` op materialising a duplicate KV cache in
    # DDR. It did save that copy -- and it did NOT save the read: the matrix operand carried an
    # outer BD dim of `batch_group` at STRIDE 0, and the shim DMA has no cache, so the cache was
    # physically streamed once per group member. Measured on device: at equal DDR bytes, one matrix
    # read sixteen times costs the same as sixteen distinct matrices read once (269.3 vs 268.0 us),
    # so the repeat was paid in full. On the Qwen3-0.6B decode that is 117.4 MB/token.
    #
    # Inverting the loop nesting -- A outer, the group inner -- removes it, and removes the
    # permutation with it: the [group, matrix] ordering was FORCED by the repeat (only the
    # outermost BD dim may carry a zero stride), and it is what made B need a 4-D wrap-capped tap.
    # With the matrix held, iteration order is plain matrix-major, output lands at
    # `q = batch_group*matrix + member` by construction, and A, B and C are all flat again.
    #
    # Gated on `coalesce` only to keep the change to one path: without it the per-batch fallback
    # would need its own restructuring for A and C to iterate different counts. That combination
    # keeps the old repeat, which is correct and slower, and shows up as unchanged DDR bytes in
    # `decode_ddr_bytes.py` rather than as silence.
    group_reuse = batch_group > 1 and coalesce
    n_vec = batch_group if group_reuse else 1

    # A's depth follows n_vec too, and it is NOT cosmetic. Reusing the tile means the core spends
    # n_vec matvec calls on each one, so a depth-2 fifo lets the DMA run only one tile ahead of a
    # core that now takes n_vec times as long to drain it -- and the stream stalls. Measured on
    # device at depth 2: the fix halved the DDR bytes (8.487 -> 4.293 MB on the scores arm) and
    # halved the achieved bandwidth with them (44.3 -> 23.1 GB/s), for a net ~zero. The prefetch
    # window has to grow with the work per tile. Costs 1 KB of L1 per extra slot on that arm.
    A_L3L1_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2 * n_vec) for i in range(cols)
    ]
    # B and C likewise: the core holds n_vec vectors and output tiles at once. At n_vec == 1 all
    # three are 2, 1 and 2 -- exactly as before.
    B_L3L1_fifos = [
        ObjectFifo(L1_B_ty, name=f"B_L3L1_{i}", depth=n_vec) for i in range(cols)
    ]
    C_L1L3_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2 * n_vec) for i in range(cols)
    ]

    def core_body(A_L3L1_fifo, B_L3L1_fifo, C_L1L3_fifo, matvec, gelu_kernel=None):
        one_idx = index.constant(1)
        for _ in range_(0xFFFFFFFF):  # batch dim handled as part of this loop
            b = B_L3L1_fifo.acquire(n_vec)
            # The kernel function computes m output rows; each core is responsible for (M/cols) output rows, so we need to call the kernel (M/cols)/m times.
            for i_idx in range_(M // m_output // cols):
                c = C_L1L3_fifo.acquire(n_vec)
                i_i32 = index.casts(T.i32(), i_idx)
                for j_idx in range_(m_output // m_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * m_input
                    a = A_L3L1_fifo.acquire(1)
                    # The A tile is acquired ONCE and every vector in the group runs over it. The
                    # group is a python-level unroll because batch_group is a build constant, and
                    # because `b`/`c` are indexable views only when more than one was acquired.
                    if n_vec == 1:
                        matvec(m_input, output_row_offset, a, b, c)
                    else:
                        for g in range(n_vec):
                            matvec(m_input, output_row_offset, a, b[g], c[g])
                    A_L3L1_fifo.release(1)
                if gelu_kernel is not None:
                    if n_vec == 1:
                        gelu_kernel(m_output, c)
                    else:
                        for g in range(n_vec):
                            gelu_kernel(m_output, c[g])
                C_L1L3_fifo.release(n_vec)
            B_L3L1_fifo.release(n_vec)

    workers = [
        Worker(
            core_body,
            [
                A_L3L1_fifos[i].cons(),
                B_L3L1_fifos[i].cons(),
                C_L1L3_fifos[i].prod(),
                matvec,
            ]
            + ([gelu_kernel] if epilogue != "none" else []),
        )
        for i in range(cols)
    ]

    # Distribution pattern for the input matrix A: each AIE core gets a contiguous chunk of rows.
    # The input matrix in DDR is MxK-sized (row-major); each core processes (M/cols)xK-sized matrices in chunks of mxK-sized tiles.
    # The chunking into mxK-sized tiles happens in the ObjectFIFO; the shim puts all data on the stream in sequence.
    # One tap per DELIVERY of the matrix. Reusing the group means that is once per MATRIX;
    # otherwise it stays once per batch, which for batch_group>1 is the same matrix twice.
    n_a_deliveries = n_matrices if group_reuse else num_batches
    A_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_A_ty.__args__[0],
                offset=col * (M // cols) * a_row_width
                + (d if group_reuse else d // batch_group) * _AM * a_row_width,
                sizes=[1, 1, 1, (M // cols) * a_row_width],
                strides=[0, 0, 0, 1],
            )
            for d in range(n_a_deliveries)
        ]
        for col in range(cols)
    ]

    # Every column gets the entirety of the vector B.
    # This design assumes that all of B fits on the cores.
    # B must follow the SAME permutation as C. The core consumes B from its FIFO in ITERATION
    # order, so once the A/C dims are [group, matrix] (forced: only the outermost dim may carry a
    # zero stride) a flat linear B hands step i the vector of head i while A/C are addressing head
    # group*matrix + member. That mismatch is silent -- every head simply gets the wrong query
    # vector -- and it reads as 0/8 parity, not as a near miss.
    #
    # At batch_group=1 this is the old flat read: sizes=[1, num_batches, 1, K] with offset m*K.
    # At batch_group=1 keep the ORIGINAL flat one-dimensional read byte for byte: leading 1s make
    # it a plain contiguous transfer whose length is not subject to the 10-bit wrap cap.
    #
    # When grouping, B must follow the SAME permutation as C -- the core consumes B from its FIFO in
    # ITERATION order, and A/C are forced into [group, matrix] because only the outermost dim may
    # carry a zero stride. A flat B then hands step i the vector of head i while A/C address head
    # group*matrix + member, which is silent and reads as 0/8 parity.
    #
    # That makes it genuinely 4-D, so its innermost size IS wrap-capped and K must be split exactly
    # like a run: op_ctx is gemv(M=head_dim, K=S), and at S=2048 an unsplit K trips
    # "Size 0 exceeds the [0:1023] range". op_scores (K=head_dim=128) never would, which is why the
    # k-only arm built and this one did not.
    # The permutation exists ONLY to satisfy the coalesced tap, whose [group, matrix] dim order is
    # forced (only the outermost dim may carry a zero stride). The per-batch FALLBACK has no such
    # constraint: its A taps address matrix `w // batch_group` in plain batch order, so a permuted B
    # hands delivery i the vector of head `member + batch_group*matrix` while A is on matrix
    # `i // batch_group` -- 14 of 16 deliveries pair the wrong operands.
    #
    # This combination was UNREACHABLE until a wide allocation made `coalesce` false at
    # batch_group>1: every shipped multi-batch GEMV coalesces. It is a defect this file's own
    # comment predicts one paragraph down ("that mismatch is silent -- every head simply gets the
    # wrong query vector -- and it reads as 0/8 parity, not as a near miss") and it went unexercised
    # because nothing could reach the path.
    if batch_group == 1 or group_reuse or not coalesce:
        # Flat, and for group_reuse that is the POINT: the core now consumes vectors in plain batch
        # order (matrix-major, member-inner), which is the order they already sit in, so the
        # permuted 4-D tap below -- and its wrap cap on K -- is not needed.
        B_tap = TensorAccessPattern(
            tensor_dims=L3_B_ty.__args__[0], offset=0,
            sizes=[1, 1, 1, num_batches * K], strides=[0, 0, 0, 1],
        )
    else:
        B_split = split_run(K)
        assert B_split is not None, (
            f"K ({K}) has no wrap-legal split; batch_group>1 needs a 4-D B tap"
        )
        k_hi, k_lo = B_split
        B_tap = TensorAccessPattern(
            tensor_dims=L3_B_ty.__args__[0], offset=0,
            sizes=[batch_group, n_matrices, k_hi, k_lo],
            strides=[K, batch_group * K, k_lo, 1],
        )

    # Collection pattern for the output vector C: each AIE core writes back its contiguous chunk of rows.
    C_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_C_ty.__args__[0],
                # Per-batch stride is the ALLOCATION (_AMO), not M -- see A_taps above.
                offset=col * (M // cols) + batch * _AMO,
                sizes=[1, 1, 1, (M // cols)],
                strides=[0, 0, 0, 1],
            )
            for batch in range(num_batches)
        ]
        for col in range(cols)
    ]

    # Batch coalescing replaces the per-batch unroll with a single iterated BD. The predicate and
    # the run splits are computed above, because `group_reuse` depends on them.
    #
    # Within one delivery the run is contiguous (A_run = (M//cols)*K elements). The stride between
    # deliveries is the full matrix (A_bstride = M*K), so for cols>1 each column gathers its own
    # slice out of every one with a gap in between. The contiguous run is split into two wrap dims
    # [run_hi, run_lo] ONLY to fit the AIE shim's 10-bit (1023) wrap-size cap.
    #
    # FIXME: pull these shim BD bounds from the MLIR-AIE target model rather than
    # hard-coding them; they live in verifyStridesWraps in
    # https://github.com/Xilinx/mlir-aie/blob/main/lib/Dialect/AIEX/IR/AIEXDialect.cpp

    # The outer dim used to be a dead placeholder (size 1, stride 0). It carries the GROUP now:
    # [group_member, matrix, run_hi, run_lo].
    #
    # The group must sit OUTERMOST because only the outer dim may have stride 0 -- aie.dma_bd
    # rejects a zero stride further in ("Stride 2 must be a positive integer"), which is what a
    # [matrix, group] ordering hits. So A repeats the whole matrix sweep per group member (outer
    # stride 0, inner A_bstride), and C compensates: its outer advances ONE batch and its inner
    # skips a whole group, so the output still lands at q = batch_group*matrix + member -- the
    # natural query-head numbering, needing no reordering downstream.
    #
    # At batch_group=1 this is byte-for-byte the original tap: sizes=[1, num_batches, ...],
    # strides=[0, bstride, ...]. A shim BD has exactly four dims and this uses all of them, so a
    # shape whose run needs a third dim cannot coalesce.
    def coalesced_tap(L3_ty, col_off, split, outer_stride, inner_stride, outer=None, inner=None):
        run_hi, run_lo = split
        return TensorAccessPattern(
            tensor_dims=L3_ty.__args__[0],
            offset=col_off,
            sizes=[batch_group if outer is None else outer,
                   n_matrices if inner is None else inner, run_hi, run_lo],
            strides=[outer_stride, inner_stride, run_lo, 1],
        )

    if coalesce:
        # Dropping the per-batch drain wait lets the single iterated fill BD run ahead of
        # the core. ObjectFifo lock backpressure keeps that safe: a producer that gets
        # ahead BLOCKS on the buffer lock (worst case a stall, never a corrupting
        # overrun). depth>=2 only buys OVERLAP of fill with compute, so it is a
        # performance guard here, not a correctness requirement (depth==1 is correct but
        # fully serial).
        assert all(f.depth >= 2 for f in A_L3L1_fifos) and all(
            f.depth >= 2 for f in C_L1L3_fifos
        ), "coalesced GEMV wants A/C ObjectFifo depth>=2 for fill/compute overlap"
        # With the group reused there is no repeat and no permutation: A walks its matrices and C
        # walks its batches, both plain and both with a dead outer dim.
        A_taps_coalesced = [
            coalesced_tap(L3_A_ty, col * (M // cols) * K, A_split, 0, A_bstride,
                          *((1, n_matrices) if group_reuse else (None, None)))
            for col in range(cols)
        ]
        C_taps_coalesced = [
            coalesced_tap(L3_C_ty, col * (M // cols), C_split,
                          0 if group_reuse else C_bstride,
                          C_bstride if group_reuse else batch_group * C_bstride,
                          *((1, num_batches) if group_reuse else (None, None)))
            for col in range(cols)
        ]

    def sequence(A, B, C, B_L3L1_fifos_prods, A_L3L1_fifos_prods, C_L1L3_fifos_conss):
        tg_b = TaskGroup()
        for col in range(cols):
            # Simple linear transfer of B, includes all batches in sequence
            B_L3L1_fifos_prods[col].fill(B, B_tap, group=tg_b)
        # Coalesced: one iterated BD per column covers all batches (num_waits==1, a
        # single drain wait for the whole column). Fallback (incl. num_batches==1): the
        # stock per-batch unroll (num_waits==num_batches, one wait per batch). The fills
        # and drains are otherwise identical; only the TAP and the wait count differ.
        num_waits = 1 if coalesce else num_batches
        # BARRIER CHUNK -- how many batches share one TaskGroup, i.e. one device-side drain wait.
        #
        # The fallback issues num_batches fills+drains per column AND num_batches barriers. Those
        # are two different costs and the wait count is the one nothing forced: the coalesced path
        # already argues, in this file, that dropping the per-batch wait is safe because ObjectFifo
        # lock backpressure blocks a runaway producer rather than corrupting it ("worst case a
        # stall, never a corrupting overrun"). The same fifos and the same locks are in play here.
        #
        # What DOES bound it is the shim's BD budget: batches in flight per column cannot exceed it,
        # and an earlier attempt to hold one fill per object across a whole design exceeded 16 BDs
        # and deadlocked (see tmatvec/design.py's KNOWN DEFECT note). So this is a chunk, not a
        # hoist -- 1 reproduces today's behaviour exactly, and the useful range is bounded above by
        # the BD budget rather than by taste.
        chunk = max(1, min(barrier_chunk, num_waits))
        for w0 in range(0, num_waits, chunk):
            tg_ac = TaskGroup()
            for w in range(w0, min(w0 + chunk, num_waits)):
                for col in range(cols):
                    a_tap = A_taps_coalesced[col] if coalesce else A_taps[col][w]
                    A_L3L1_fifos_prods[col].fill(A, a_tap, group=tg_ac)
                for col in range(cols):
                    c_tap = C_taps_coalesced[col] if coalesce else C_taps[col][w]
                    C_L1L3_fifos_conss[col].drain(
                        C,
                        c_tap,
                        group=tg_ac,
                        wait=True,
                    )
            tg_ac.finish()
        tg_b.finish()

    rt = Runtime(
        sequence,
        [
            L3_A_ty,
            L3_B_ty,
            L3_C_ty,
            [of.prod() for of in B_L3L1_fifos],
            [of.prod() for of in A_L3L1_fifos],
            [of.cons() for of in C_L1L3_fifos],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()
