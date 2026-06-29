# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker

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
    kernel_object="mv.o",
    func_prefix="",
    verbose=False,
    coalesce_batch_dma=False,
    dtype_a="bf16",
    epilogue="none",
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
    # B (vector) + C (output) are always bf16; A (matrix) may be int8 (a quantized resident K/V cache that
    # halves its LPDDR re-read) via dtype_a="int8" -> binds matvec_vectorized_i8_bf16 (widens int8 A to bf16,
    # MACs with bf16 B; per-tensor dequant scale folded host-side). bf16 path unchanged (default).
    dtype_in = np.dtype[bfloat16]  # B vector dtype
    dtype_in_str = "bf16"
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"
    # A (matrix) may be int8 (a quantized resident K/V cache that halves its LPDDR re-read). To avoid an
    # i8-typed MLIR buffer (the fusion arena is bf16 + reinterpret_cast can't change element type), the int8
    # bytes are STORED IN A bf16-TYPED buffer with HALF the inner dim (2 int8 per bf16 slot); the kernel
    # (matvec_vectorized_i8_bf16) reinterprets bf16->int8 at the C level. So A stays bf16 in MLIR (no fusion
    # change), DIM_K (the int8 contract length) stays K.
    if dtype_a == "int8":
        dtype_a_str = "i8"
        a_k = K // 2  # K int8 bytes per row = K//2 bf16 slots
    else:
        dtype_a_str = "bf16"
        a_k = K

    assert M % cols == 0

    L1_A_ty = np.ndarray[
        (
            m_input,
            a_k,
        ),
        dtype_in,
    ]
    L1_B_ty = np.ndarray[(K,), dtype_in]
    L1_C_ty = np.ndarray[(m_output,), dtype_out]
    L3_A_ty = np.ndarray[
        (num_batches * M * a_k,),
        dtype_in,
    ]
    L3_B_ty = np.ndarray[(num_batches * K,), dtype_in]
    L3_C_ty = np.ndarray[(num_batches * M,), dtype_out]

    func_type = "vectorized" if vectorized else "scalar"
    matvec = Kernel(
        f"{func_prefix}matvec_{func_type}_{dtype_a_str}_{dtype_out_str}",
        f"{func_prefix}{kernel_object}",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )
    # epilogue="gelu": fold a GELU(tanh) into the GEMV. Applied ONCE per C-tile over the full m_output rows
    # (in core_body, after the matvec inner-loop) — NOT per matvec call, whose m_input tile can be < the
    # gelu 16-wide vector. bf16-only.
    gelu_kernel = None
    if epilogue == "gelu":
        assert vectorized and dtype_a_str == "bf16", "gelu epilogue is bf16-vectorized only"
        assert m_output % 16 == 0, f"gelu epilogue needs m_output % 16 == 0 (got {m_output})"
        gelu_kernel = Kernel(
            f"{func_prefix}gelu_tile_bf16",
            f"{func_prefix}{kernel_object}",
            [np.int32, L1_C_ty],
        )

    A_L3L1_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2) for i in range(cols)
    ]
    B_L3L1_fifos = [
        ObjectFifo(L1_B_ty, name=f"B_L3L1_{i}", depth=1) for i in range(cols)
    ]
    C_L1L3_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2) for i in range(cols)
    ]

    def core_body(A_L3L1_fifo, B_L3L1_fifo, C_L1L3_fifo, matvec, gelu_kernel=None):
        one_idx = index.constant(1)
        for _ in range_(0xFFFFFFFF):  # batch dim handled as part of this loop
            b = B_L3L1_fifo.acquire(1)
            # The kernel function computes m output rows; each core is responsible for (M/cols) output rows, so we need to call the kernel (M/cols)/m times.
            for i_idx in range_(M // m_output // cols):
                c = C_L1L3_fifo.acquire(1)
                i_i32 = index.casts(T.i32(), i_idx)
                for j_idx in range_(m_output // m_input):
                    j_i32 = index.casts(T.i32(), j_idx)
                    output_row_offset = j_i32 * m_input
                    a = A_L3L1_fifo.acquire(1)
                    matvec(m_input, output_row_offset, a, b, c)
                    A_L3L1_fifo.release(1)
                # epilogue: gelu over the FULL m_output C-tile (the matvec inner-loop has filled all rows).
                if gelu_kernel is not None:
                    gelu_kernel(m_output, c)
                C_L1L3_fifo.release(1)
            B_L3L1_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                A_L3L1_fifos[i].cons(),
                B_L3L1_fifos[i].cons(),
                C_L1L3_fifos[i].prod(),
                matvec,
            ]
            + ([gelu_kernel] if epilogue == "gelu" else []),
        )
        for i in range(cols)
    ]

    # Distribution pattern for the input matrix A: each AIE core gets a contiguous chunk of rows.
    # The input matrix in DDR is MxK-sized (row-major); each core processes (M/cols)xK-sized matrices in chunks of mxK-sized tiles.
    # The chunking into mxK-sized tiles happens in the ObjectFIFO; the shim puts all data on the stream in sequence.
    A_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_A_ty.__args__[0],
                # A is laid out in a_k-wide rows (a_k = K for bf16, K//2 for the int8-bytes-in-bf16 buffer).
                offset=col * (M // cols) * a_k + batch * M * a_k,
                sizes=[1, 1, 1, (M // cols) * a_k],
                strides=[0, 0, 0, 1],
            )
            for batch in range(num_batches)
        ]
        for col in range(cols)
    ]

    # Every column gets the entirety of the vector B.
    # This design assumes that all of B fits on the cores.
    B_tap = TensorAccessPattern(
        tensor_dims=L3_B_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, num_batches * K],
        strides=[0, 0, 0, 1],
    )

    # Collection pattern for the output vector C: each AIE core writes back its contiguous chunk of rows.
    C_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_C_ty.__args__[0],
                offset=col * (M // cols) + batch * M,
                sizes=[1, 1, 1, (M // cols)],
                strides=[0, 0, 0, 1],
            )
            for batch in range(num_batches)
        ]
        for col in range(cols)
    ]

    # B-unroll -> BD-iteration: when coalesce_batch_dma, collapse the per-batch A
    # fill / C drain into ONE 4D-iterated transfer per column (batch dim folded into
    # the access pattern, stride = M*K for A / M for C). This replaces num_batches
    # DMA tasks per column with 1 (offline-verified to enumerate the identical access
    # sequence: scripts/tap_equivalence_check.py). Relies on the ObjectFifo depth for
    # flow control instead of the per-batch task-group `wait` (on-device validated).
    # Default off -> the per-batch path below is unchanged for other operators/designs.
    #
    # Pack the (num_batches x contiguous-run) access into ONE 4D BD per column:
    # batch dims OUTER (d0,d1), run dims INNER (d2,d3) so each batch's run stays
    # contiguous. Two AIE2p BD limits force splitting BOTH: every dim size <= 1023,
    # every stride <= 2**20. The run = (M//cols)*K (A) / (M//cols) (C) itself
    # overflows 1023 at production dims (S=448/T=1500 => run up to 12288), so it
    # needs its own dims too; with num_batches (up to B*H=1536) also split, this
    # still fits in 4 dims (B=16: 3 used; B=128: 4). Layout [bhi,blo,rhi,rlo],
    # strides [blo*bstride, bstride, rlo, 1] => flat index = batch*bstride + r,
    # IDENTICAL to the unrolled access (offline-verified scripts/tap_equivalence_check.py).
    # AIE2p dma_bd hardware dim limits (AIEXDialect.cpp verifier): logical sizes
    # [d0,d1,d2,d3] map to hw [wrap, UNCAPPED, wrap, iter] roughly as: d1 -> the
    # size-UNCAPPED hw dim (only stride<=2**20), d2,d3 -> wrap dims (size<=1023),
    # d0 -> iteration dim (size<=64). So: put the WHOLE batch in d1 (uncapped; its
    # stride=bstride must be <=2**20, true here: max M*K = 1536*64 = 98304), split
    # the contiguous run across d2,d3 (each <=1023), and leave d0=1. No batch split
    # needed. Layout [1, num_batches, run_hi, run_lo] strides [0, bstride, run_lo, 1]
    # => flat index = batch*bstride + r (IDENTICAL to unrolled; offline-verified).
    def _split_run(n, lim=1023):  # (hi, lo): lo=largest divisor<=lim (contiguous inner)
        for lo in range(min(lim, n), 0, -1):
            if n % lo == 0 and (n // lo) <= lim:
                return (n // lo, lo)
        raise ValueError(f"run={n} not splittable into two dims <= {lim}")

    def _coalesced_tap(L3_ty, col_off, run, bstride):
        assert bstride <= (1 << 20), f"batch stride {bstride} exceeds 2**20"
        rhi, rlo = _split_run(run)
        return TensorAccessPattern(
            tensor_dims=L3_ty.__args__[0],
            offset=col_off,
            sizes=[1, num_batches, rhi, rlo],
            strides=[0, bstride, rlo, 1],
        )

    # Constructed only when opting in, so the default path (and other GEMV callers)
    # never run _split_run/_split_batch (which could raise for an unsplittable run).
    A_taps_coalesced = (
        [
            _coalesced_tap(L3_A_ty, col * (M // cols) * K, (M // cols) * K, M * K)
            for col in range(cols)
        ]
        if coalesce_batch_dma
        else None
    )
    C_taps_coalesced = (
        [
            _coalesced_tap(L3_C_ty, col * (M // cols), (M // cols), M)
            for col in range(cols)
        ]
        if coalesce_batch_dma
        else None
    )

    rt = Runtime()
    with rt.sequence(L3_A_ty, L3_B_ty, L3_C_ty) as (A, B, C):
        # --- per-op NPU trace hook (opt-in via IRON_TRACE_SIZE env; no-op when unset
        # so production builds are unaffected). Route-(b) standalone per-op measurement. ---
        import os as _os

        if int(_os.environ.get("IRON_TRACE_SIZE", "0")) > 0:
            import aie.utils.trace as _tu

            _ev = _tu.events
            rt.enable_trace(
                int(_os.environ["IRON_TRACE_SIZE"]),
                workers=list(workers)[: int(_os.environ.get("IRON_TRACE_NTILES", "1"))],
                coretile_events=[
                    _ev.PortEvent(
                        _ev.CoreEvent.PORT_RUNNING_0, _ev.WireBundle.DMA, 0, True
                    ),
                    _ev.PortEvent(
                        _ev.CoreEvent.PORT_RUNNING_1, _ev.WireBundle.DMA, 1, True
                    ),
                    _ev.PortEvent(
                        _ev.CoreEvent.PORT_RUNNING_2, _ev.WireBundle.DMA, 0, False
                    ),
                    _ev.CoreEvent.INSTR_EVENT_0,
                    _ev.CoreEvent.INSTR_EVENT_1,
                    _ev.CoreEvent.MEMORY_STALL,
                    _ev.CoreEvent.LOCK_STALL,
                    _ev.CoreEvent.INSTR_VECTOR,
                ],
            )
        rt.start(*workers)
        tg_b = rt.task_group()
        for col in range(cols):
            # Simple linear transfer of B, includes all batches in sequence
            rt.fill(B_L3L1_fifos[col].prod(), B, B_tap, task_group=tg_b)
        if coalesce_batch_dma:
            tg_ac = rt.task_group()
            for col in range(cols):
                rt.fill(
                    A_L3L1_fifos[col].prod(),
                    A,
                    A_taps_coalesced[col],
                    task_group=tg_ac,
                )
            for col in range(cols):
                rt.drain(
                    C_L1L3_fifos[col].cons(),
                    C,
                    C_taps_coalesced[col],
                    task_group=tg_ac,
                    wait=True,
                )
            rt.finish_task_group(tg_ac)
        else:
            for batch in range(num_batches):
                tg_ac = rt.task_group()
                for col in range(cols):
                    rt.fill(
                        A_L3L1_fifos[col].prod(),
                        A,
                        A_taps[col][batch],
                        task_group=tg_ac,
                    )
                for col in range(cols):
                    rt.drain(
                        C_L1L3_fifos[col].cons(),
                        C,
                        C_taps[col][batch],
                        task_group=tg_ac,
                        wait=True,
                    )
                rt.finish_task_group(tg_ac)
        rt.finish_task_group(tg_b)

    return Program(dev, rt).resolve_program()
