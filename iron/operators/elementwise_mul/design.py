# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
from iron.common.worker_compat import create_worker
from iron.operators._trace import maybe_enable_trace


def elementwise_mul_broadcast_design(
    dev,
    num_elements,
    num_columns,
    tile_size,
    m_rows,
    trace_size,
    kernel_fn_name,
    kernel_obj_file,
    func_prefix="",
    allocation_scheme=None,
):
    """`[m_rows, num_elements] x [num_elements] -> [m_rows, num_elements]`, one dispatch.

    Same per-column tiling as binary_elementwise_design, except "in2" (gain) carries no row
    dimension: each column's gain tile is fetched ONCE per D-chunk and held in L1 across the
    inner loop over `m_rows`, instead of being re-read once per row. That collapses the caller's
    per-row dispatch loop into one runtime sequence and cuts gain DMA traffic from
    `m_rows * chunk` elements to `chunk` (see gen_llm_prefill.py's op_lscale comment).

    Reached only via ElementwiseMul.rows is not None; the `rows=None` (m=1) path calls
    binary_elementwise_design directly and never imports this file -- see op.py.
    """
    per_tile_elements = 4096 if tile_size > 4096 else tile_size
    n = per_tile_elements * num_columns
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    chunk = num_elements // num_columns
    dtype = bfloat16

    # A/C: the row-broadcasted operand, m_rows*num_elements total, row-major [m_rows, num_elements].
    ac_tensor_ty = np.ndarray[(m_rows * num_elements,), np.dtype[dtype]]
    # B: the [num_elements] gain, unchanged shape from the non-broadcast design.
    b_tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    of_in1s = [ObjectFifo(tile_ty, name=f"in1_{i}") for i in range(num_columns)]
    of_in2s = [ObjectFifo(tile_ty, name=f"in2_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(tile_ty, name=f"out_{i}") for i in range(num_columns)]

    eltwise_kernel = Kernel(
        f"{func_prefix}{kernel_fn_name}",
        f"{func_prefix}{kernel_obj_file}",
        [tile_ty, tile_ty, tile_ty, np.int32],
    )

    def core_body(of_in1, of_in2, of_out, eltwise_fn):
        for _ in range_(N_div_n):
            # Gain chunk: fetched once per D-chunk, held across the row loop below.
            elem_in2 = of_in2.acquire(1)
            for _ in range_(m_rows):
                elem_in1 = of_in1.acquire(1)
                elem_out = of_out.acquire(1)
                eltwise_fn(elem_in1, elem_in2, elem_out, per_tile_elements)
                of_in1.release(1)
                of_out.release(1)
            of_in2.release(1)

    my_workers = [
        create_worker(
            core_body,
            [
                of_in1s[i].cons(),
                of_in2s[i].cons(),
                of_outs[i].prod(),
                eltwise_kernel,
            ],
            allocation_scheme=allocation_scheme,
        )
        for i in range(num_columns)
    ]

    # A/C access pattern per column: m_rows rows, row stride num_elements (the full row pitch),
    # chunk-of-chunks stride per_tile_elements within a row -- the row dimension does NOT collapse
    # to a flat run the way the non-broadcast design's does, because consecutive rows are
    # num_elements elements apart, not chunk apart (the other columns' slices sit in between).
    ac_taps = [
        TensorAccessPattern(
            (m_rows * num_elements,),
            chunk * i,
            [N_div_n, m_rows, per_tile_elements],
            [per_tile_elements, num_elements, 1],
        )
        for i in range(num_columns)
    ]
    # B access pattern per column: identical to the non-broadcast design's -- one flat
    # chunk-sized read, no row dimension.
    b_taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    def sequence(A, B, C, in1_prods, in2_prods, out_conses):
        tg = TaskGroup()
        for i in range(num_columns):
            in1_prods[i].fill(A, ac_taps[i], group=tg)
            in2_prods[i].fill(B, b_taps[i], group=tg)
        for i in range(num_columns):
            out_conses[i].drain(C, ac_taps[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [
            ac_tensor_ty,
            b_tensor_ty,
            ac_tensor_ty,
            [of_in1s[i].prod() for i in range(num_columns)],
            [of_in2s[i].prod() for i in range(num_columns)],
            [of_outs[i].cons() for i in range(num_columns)],
        ],
    )

    prog = Program(dev, rt, workers=my_workers)
    maybe_enable_trace(prog, trace_size, my_workers)
    return prog.resolve_program()
