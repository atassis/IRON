# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def channeled_unary_design(
    dev,
    size,
    num_columns,
    num_channels,
    tile_size,
    trace_size,
    kernel_fn_name,
    kernel_obj_file,
    tile_cap=4096,
    func_prefix="",
):
    xfr_dtype = bfloat16
    line_size = tile_cap if tile_size > tile_cap else tile_size
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]

    # When tile_cap > 4096 (e.g. 8192), tiles may exceed a single 8 KB bank,
    # so the FIFO depth must shrink to 1 to avoid exceeding local memory.
    fifo_kwargs = {}
    if tile_cap > 4096:
        fifodepth = 1 if line_size > 4096 else 2
        fifo_kwargs = {"depth": fifodepth}

    # Calculate number of iterations per core
    total_cores = num_columns * num_channels
    per_core_elements = size // total_cores
    N_div_n = per_core_elements // line_size

    # Chunk size sent per DMA channel
    chunk = size // num_columns // num_channels

    # Dataflow with ObjectFifos
    of_ins = [
        ObjectFifo(line_type, name=f"in{i}_{j}", **fifo_kwargs)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(line_type, name=f"out{i}_{j}", **fifo_kwargs)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # External, binary kernel definition
    kernel_fcn = Kernel(
        f"{func_prefix}{kernel_fn_name}",
        f"{func_prefix}{kernel_obj_file}",
        [line_type, line_type, np.int32],
    )

    # Task for the core to perform
    def core_fn(of_in, of_out, kernel_line):
        for _ in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel_line(elem_in, elem_out, line_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to perform the task
    # Large tile sizes (>4096) with LUT-based kernels need more stack space
    # than the default 1024 bytes due to spilled vector temporaries.
    worker_kwargs = {"stack_size": 0xD00} if line_size > 4096 else {}
    my_workers = [
        Worker(
            core_fn,
            [
                of_ins[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                kernel_fcn,
            ],
            **worker_kwargs,
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Create a TensorAccessPattern for each channel
    taps = [
        TensorAccessPattern(
            (1, size),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a_in, b_out):
        # Per-op on-NPU hardware trace (no-op when off -> production unaffected).
        # Honors either the trace_size param (LayerNorm plumbs it) OR the IRON_TRACE_SIZE
        # env (so GELU/SiLU/etc., which don't plumb trace_size through op.py, still trace).
        # Mirrors the dequant op + the matrix_multiplication trace example: configure the
        # compute tiles' trace units + route packets to DDR (ddr_id=4). Without this call the
        # IRON program never emits aie.trace/configure ops, so trace.txt comes back all-zeros.
        import os as _os

        _ts = (
            trace_size
            if (trace_size and trace_size > 0)
            else int(_os.environ.get("IRON_TRACE_SIZE", "0"))
        )
        if _ts > 0:
            import aie.utils.trace as _trace_utils

            _ev = _trace_utils.events
            rt.enable_trace(
                _ts,
                workers=list(my_workers)[
                    : int(_os.environ.get("IRON_TRACE_NTILES", "1"))
                ],
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
        rt.start(*my_workers)

        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a_in,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b_out,
                    taps[i * num_channels + j],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    # Place components and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())
