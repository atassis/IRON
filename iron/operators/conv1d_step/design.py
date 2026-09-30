# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def conv1d_step_design(
    dev,
    channels,
    taps,
    tile_channels,
    num_columns,
    kernel_fn_name,
    kernel_obj_file,
    func_prefix="",
    allocation_scheme=None,
    tokens=1,
    depth=2,
):
    """Each column owns a contiguous channel range and walks it in [rows, tile_channels] tiles,
    gathered from the [rows, channels] window (rows = taps - 1 + tokens) and the [taps, channels]
    weights with a row stride of `channels`, and drained back the same way."""
    per_col = channels // num_columns
    n_tiles = per_col // tile_channels
    rows = taps - 1 + tokens
    tensor_ty = np.ndarray[(rows * channels,), np.dtype[bfloat16]]
    w_ty = np.ndarray[(taps * channels,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(rows * tile_channels,), np.dtype[bfloat16]]
    w_tile_ty = np.ndarray[(taps * tile_channels,), np.dtype[bfloat16]]

    of_win = [ObjectFifo(tile_ty, name=f"win_{i}", depth=depth) for i in range(num_columns)]
    of_w = [ObjectFifo(w_tile_ty, name=f"w_{i}", depth=depth) for i in range(num_columns)]
    of_out = [ObjectFifo(tile_ty, name=f"out_{i}", depth=depth) for i in range(num_columns)]
    kernel = Kernel(f"{func_prefix}{kernel_fn_name}", f"{func_prefix}{kernel_obj_file}",
                    [tile_ty, w_tile_ty, tile_ty])

    def core_body(of_in, of_wt, of_o, fn):
        for _ in range_(n_tiles):
            a = of_in.acquire(1)
            b = of_wt.acquire(1)
            o = of_o.acquire(1)
            fn(a, b, o)
            of_in.release(1)
            of_wt.release(1)
            of_o.release(1)

    workers = [
        Worker(core_body, [of_win[i].cons(), of_w[i].cons(), of_out[i].prod(), kernel],
               allocation_scheme=allocation_scheme)
        for i in range(num_columns)
    ]
    def taps_for(n):
        return [TensorAccessPattern((n, channels), i * per_col, [1, n_tiles, n, tile_channels],
                                    [0, tile_channels, channels, 1])
                for i in range(num_columns)]

    win_taps, w_taps = taps_for(rows), taps_for(taps)

    def sequence(WIN, W, OUT, win_prods, w_prods, out_conses):
        tg = TaskGroup()
        for i in range(num_columns):
            win_prods[i].fill(WIN, win_taps[i], group=tg)
            w_prods[i].fill(W, w_taps[i], group=tg)
        for i in range(num_columns):
            out_conses[i].drain(OUT, win_taps[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [tensor_ty, w_ty, tensor_ty,
         [f.prod() for f in of_win], [f.prod() for f in of_w], [f.cons() for f in of_out]],
    )
    return Program(dev, rt, workers=workers).resolve_program()
