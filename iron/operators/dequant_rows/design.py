# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_


def dequant_rows_design(dev, rows, K, row_stride, tile_rows, num_columns, out_stride, out_col,
                        kernel_obj, func_prefix="", stack_size=None):
    """Each column expands a contiguous run of rows, `tile_rows` at a time, into row r of a
    [rows, out_stride] bf16 matrix at column `out_col` -- so the K-chunks GEMV stores separately
    can land side by side in one GEMM operand."""
    per_col = rows // num_columns
    n_tiles = per_col // tile_rows
    in_tile = tile_rows * row_stride
    in_ty = np.ndarray[(rows * row_stride,), np.dtype[np.int8]]
    out_ty = np.ndarray[(rows * out_stride,), np.dtype[bfloat16]]
    in_tile_ty = np.ndarray[(in_tile,), np.dtype[np.int8]]
    out_tile_ty = np.ndarray[(tile_rows * K,), np.dtype[bfloat16]]

    of_in = [ObjectFifo(in_tile_ty, name=f"dq_in_{c}") for c in range(num_columns)]
    of_out = [ObjectFifo(out_tile_ty, name=f"dq_out_{c}") for c in range(num_columns)]
    kernel = Kernel(f"{func_prefix}dequant_rows_int4", f"{func_prefix}{kernel_obj}",
                    [in_tile_ty, out_tile_ty])

    def core_fn(i, o, fn):
        for _ in range_(n_tiles):
            a = i.acquire(1)
            b = o.acquire(1)
            fn(a, b)
            i.release(1)
            o.release(1)

    workers = [Worker(core_fn, [of_in[c].cons(), of_out[c].prod(), kernel], stack_size=stack_size)
               for c in range(num_columns)]

    def in_tap(c):
        return TensorAccessPattern((1, rows * row_stride), c * per_col * row_stride,
                                   [1, 1, n_tiles, in_tile], [0, 0, in_tile, 1])

    # Rows in order, each cut into SEG-element runs: a [n_tiles, tile_rows, K] pattern over a
    # strided row compiled at 4608/9216 and wrote nothing, and aiecc refused it at 2560/5120.
    SEG = 256
    assert K % SEG == 0, f"K={K} must be a multiple of {SEG}"

    def out_tap(c):
        return TensorAccessPattern((1, rows * out_stride), c * per_col * out_stride + out_col,
                                   [1, per_col, K // SEG, SEG], [0, out_stride, SEG, 1])

    def sequence(A, C, in_ps, out_cs):
        tg = TaskGroup()
        for c in range(num_columns):
            in_ps[c].fill(A, in_tap(c), group=tg)
        for c in range(num_columns):
            out_cs[c].drain(C, out_tap(c), wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [in_ty, out_ty, [f.prod() for f in of_in], [f.cons() for f in of_out]])
    return Program(dev, rt, workers=workers).resolve_program()
