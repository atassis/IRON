# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker

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
"""


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

    n_chunks = K // rows_per_chunk

    L1_A_ty = np.ndarray[(rows_per_chunk * M,), np.dtype[bfloat16]]
    L1_W_ty = np.ndarray[(batch_group * K,), np.dtype[bfloat16]]
    L1_C_ty = np.ndarray[(batch_group * M,), np.dtype[bfloat16]]
    ACC_ty = np.ndarray[(batch_group * M,), np.dtype[np.float32]]

    L3_A_ty = np.ndarray[(n_matrices * _AK * M,), np.dtype[bfloat16]]
    L3_W_ty = np.ndarray[(num_batches * K,), np.dtype[bfloat16]]
    L3_C_ty = np.ndarray[(num_batches * M,), np.dtype[bfloat16]]

    # The fused dispatch prefixes BOTH the symbol and the object FILENAME with op{idx}_, so the
    # object reference has to carry func_prefix too -- prefixing only the symbol builds an object
    # nothing links against ("cannot open tmv_128n.o").
    obj = f"{func_prefix}{kernel_object}"
    k_zero = Kernel(f"{func_prefix}taccum_zero_f32", obj, [np.int32, ACC_ty])
    k_rows = Kernel(
        f"{func_prefix}taccum_rows_bf16_f32",
        obj,
        [np.int32, np.int32, np.int32, np.int32, L1_A_ty, L1_W_ty, ACC_ty],
    )
    k_finish = Kernel(
        f"{func_prefix}taccum_finish_bf16", obj, [np.int32, ACC_ty, L1_C_ty]
    )

    A_fifos = [ObjectFifo(L1_A_ty, name=f"A_L3L1_{c}", depth=2) for c in range(cols)]
    W_fifos = [ObjectFifo(L1_W_ty, name=f"W_L3L1_{c}", depth=1) for c in range(cols)]
    C_fifos = [ObjectFifo(L1_C_ty, name=f"C_L1L3_{c}", depth=2) for c in range(cols)]

    def core_body(A_cons, W_cons, C_prod, acc, zero, rows, finish):
        for _ in range_(0xFFFFFFFF):
            w = W_cons.acquire(1)
            zero(batch_group, acc)
            for i in range_(n_chunks):
                a = A_cons.acquire(1)
                # w_off advances by whole chunks; the group stride is this batch's own W length.
                rows(rows_per_chunk, batch_group, K, i * rows_per_chunk, a, w, acc)
                A_cons.release(1)
            c = C_prod.acquire(1)
            finish(batch_group, acc, c)
            C_prod.release(1)
            W_cons.release(1)

    # A: column c reads matrix c, whole contiguous rows, chunk by chunk. The per-matrix stride is
    # the ALLOCATION (_AK), not the reduced extent, so a windowed read still lands on the right one.
    A_taps = [
        TensorAccessPattern(
            tensor_dims=L3_A_ty.__args__[0],
            offset=c * _AK * M,
            sizes=[1, 1, 1, K * M],
            strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]
    # W: column c takes its group's batches, which are contiguous because batches sharing a matrix
    # are consecutive by construction (batch b reads matrix b // batch_group).
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

    workers = [
        Worker(
            core_body,
            [
                A_fifos[c].cons(),
                W_fifos[c].cons(),
                C_fifos[c].prod(),
                Buffer(type=ACC_ty, name=f"acc_{c}"),
                k_zero,
                k_rows,
                k_finish,
            ],
        )
        for c in range(cols)
    ]

    def sequence(A, W, C, A_prods, W_prods, C_conss):
        tg = TaskGroup()
        for c in range(cols):
            W_prods[c].fill(W, W_taps[c], group=tg)
            A_prods[c].fill(A, A_taps[c], group=tg)
            C_conss[c].drain(C, C_taps[c], group=tg, wait=True)
        tg.finish()

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
