# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.helpers.dialects.scf import if_, else_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_

MODES = ("add", "mul", "gelu")


def pointwise_design(dev, size, num_columns, tile_size, mode, func_prefix=""):
    """Add, multiply or GELU over two input streams and one output, chosen by a mode RTP.

    The core program is the same for every mode: each sequence writes its own mode word before
    releasing the barrier, so operators that differ only in mode share one device body. GELU reads
    the first input and ignores the second, which its caller streams anyway.
    """
    if size % (num_columns * tile_size):
        raise ValueError(f"size ({size}) must be a multiple of num_columns * tile_size")
    n_tiles = size // num_columns // tile_size
    chunk = size // num_columns
    tensor_ty = np.ndarray[(size,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]

    of_in1s = [ObjectFifo(tile_ty, name=f"in1_{i}") for i in range(num_columns)]
    of_in2s = [ObjectFifo(tile_ty, name=f"in2_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(tile_ty, name=f"out_{i}") for i in range(num_columns)]
    modes = [Buffer(np.ndarray[(1,), np.dtype[np.int32]], name=f"mode_{i}", use_write_rtp=True)
             for i in range(num_columns)]
    barriers = [WorkerRuntimeBarrier() for _ in range(num_columns)]

    binary = [tile_ty, tile_ty, tile_ty, np.int32]
    add = Kernel(f"{func_prefix}eltwise_add_bf16_vector", f"{func_prefix}add.o", binary)
    mul = Kernel(f"{func_prefix}eltwise_mul_bf16_vector", f"{func_prefix}mul.o", binary)
    gelu = Kernel(f"{func_prefix}gelu_bf16", f"{func_prefix}gelu.o", [tile_ty, tile_ty, np.int32])

    def core_body(of_in1, of_in2, of_out, add_fn, mul_fn, gelu_fn, mode, barrier):
        barrier.wait_for_value(1)
        m = mode[0]
        barrier.release_with_value(1)  # re-arm before any output; see gemv/design.py tiles_rtp
        for _ in range_(n_tiles):
            a = of_in1.acquire(1)
            b = of_in2.acquire(1)
            c = of_out.acquire(1)
            with if_(m == 0) as is_add:
                add_fn(a, b, c, tile_size)
            with else_(is_add):
                with if_(m == 1) as is_mul:
                    mul_fn(a, b, c, tile_size)
                with else_(is_mul):
                    gelu_fn(a, c, tile_size)
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, [of_in1s[i].cons(), of_in2s[i].cons(), of_outs[i].prod(),
                           add, mul, gelu, modes[i], barriers[i]])
        for i in range(num_columns)
    ]
    taps = [TensorAccessPattern((1, size), chunk * i, [1, 1, 1, chunk], [0, 0, 0, 1])
            for i in range(num_columns)]

    def sequence(A, B, C, in1_prods, in2_prods, out_conses):
        for i in range(num_columns):
            modes[i][0] = MODES.index(mode)
            barriers[i].set(1)
        tg = TaskGroup()
        for i in range(num_columns):
            in1_prods[i].fill(A, taps[i], group=tg)
            in2_prods[i].fill(B, taps[i], group=tg)
        for i in range(num_columns):
            out_conses[i].drain(C, taps[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, tensor_ty,
         [of.prod() for of in of_in1s], [of.prod() for of in of_in2s],
         [of.cons() for of in of_outs]],
    )
    return Program(dev, rt, workers=workers).resolve_program()
