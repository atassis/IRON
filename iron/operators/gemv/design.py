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
    dtype_in = np.dtype[bfloat16]
    dtype_in_str = "bf16"
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"

    assert M % cols == 0

    L1_A_ty = np.ndarray[
        (
            m_input,
            K,
        ),
        dtype_in,
    ]
    L1_B_ty = np.ndarray[(K,), dtype_in]
    L1_C_ty = np.ndarray[(m_output,), dtype_out]
    L3_A_ty = np.ndarray[
        (num_batches * M * K,),
        dtype_in,
    ]
    L3_B_ty = np.ndarray[(num_batches * K,), dtype_in]
    L3_C_ty = np.ndarray[(num_batches * M,), dtype_out]

    func_type = "vectorized" if vectorized else "scalar"
    matvec = Kernel(
        f"{func_prefix}matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
        f"{func_prefix}{kernel_object}",
        [np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
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

    def core_body(A_L3L1_fifo, B_L3L1_fifo, C_L1L3_fifo, matvec):
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
            ],
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
                offset=col * (M // cols) * K + batch * M * K,
                sizes=[1, 1, 1, (M // cols) * K],
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
        [_coalesced_tap(L3_A_ty, col * (M // cols) * K, (M // cols) * K, M * K)
         for col in range(cols)]
        if coalesce_batch_dma else None
    )
    C_taps_coalesced = (
        [_coalesced_tap(L3_C_ty, col * (M // cols), (M // cols), M)
         for col in range(cols)]
        if coalesce_batch_dma else None
    )

    rt = Runtime()
    with rt.sequence(L3_A_ty, L3_B_ty, L3_C_ty) as (A, B, C):
        rt.start(*workers)
        tg_b = rt.task_group()
        for col in range(cols):
            # Simple linear transfer of B, includes all batches in sequence
            rt.fill(B_L3L1_fifos[col].prod(), B, B_tap, task_group=tg_b)
        if coalesce_batch_dma:
            tg_ac = rt.task_group()
            for col in range(cols):
                rt.fill(
                    A_L3L1_fifos[col].prod(), A, A_taps_coalesced[col],
                    task_group=tg_ac,
                )
            for col in range(cols):
                rt.drain(
                    C_L1L3_fifos[col].cons(), C, C_taps_coalesced[col],
                    task_group=tg_ac, wait=True,
                )
            rt.finish_task_group(tg_ac)
        else:
            for batch in range(num_batches):
                tg_ac = rt.task_group()
                for col in range(cols):
                    rt.fill(
                        A_L3L1_fifos[col].prod(), A, A_taps[col][batch],
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
