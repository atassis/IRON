# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    Buffer,
    WorkerRuntimeBarrier,
    ScratchpadParameter,
)
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from ml_dtypes import bfloat16


def softmax(
    dev,
    num_elements,
    num_aie_columns,
    num_channels,
    trace_size,
    tile_size,
    rtp_vector_size=None,
    mask_patch_value=0,
    mask_scratchpad=None,
    func_prefix="",
    kernel_obj_file="softmax.o",
):
    # Deep-C: when mask_scratchpad is a (name) string, the per-dispatch mask width becomes a runtime
    # `core`-kind scratchpad parameter (read on-tile via aiex.read_scratchpad_parameter, host-written
    # per dispatch) instead of the compile-time RTP constant / ELF-patch (mask_patch_value). This
    # keeps the ELF CONSTANT across tokens (registered once). The --aie-lower-scratchpad-parameters
    # pass inserts the lock + sync preamble. Mutually exclusive with mask_patch_value.
    use_sp = mask_scratchpad is not None
    if use_sp:
        assert (
            mask_patch_value == 0
        ), "mask_scratchpad and mask_patch_value are mutually exclusive"
    per_tile_elements = tile_size
    if rtp_vector_size is None:
        rtp_vector_size = per_tile_elements
    total_cores = num_aie_columns * num_channels
    per_core_elements = num_elements // total_cores
    if num_elements % total_cores != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {total_cores}."
        )
    N_div_n = per_core_elements // per_tile_elements
    chunk = num_elements // num_aie_columns // num_channels  # For offset calculation
    dtype = bfloat16

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}_{j}")
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}_{j}")
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    # AIE Core Function declaration
    softmax_kernel = Kernel(
        f"{func_prefix}softmax_bf16",
        f"{func_prefix}{kernel_obj_file}",
        [tile_ty, tile_ty, np.int32],
    )
    mask_kernel = Kernel(
        f"{func_prefix}mask_bf16",
        f"{func_prefix}{kernel_obj_file}",
        [tile_ty, np.int32, np.int32],
    )

    # Shared core-kind scratchpad parameter (one symbol, read by every core) for the runtime path.
    sm_param = ScratchpadParameter(mask_scratchpad, np.int32) if use_sp else None

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_out, softmax_kernel, mask_kernel, ctrl, barrier):
        if use_sp:
            vector_size = (
                ctrl.read()
            )  # aiex.read_scratchpad_parameter, fresh per dispatch
        else:
            barrier.wait_for_value(1)
            vector_size = ctrl[0]
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            mask_kernel(elem_in1, vector_size, per_tile_elements)
            softmax_kernel(elem_in1, elem_out, per_tile_elements)
            of_in1.release(1)
            of_out.release(1)

    rtps = (
        []
        if use_sp
        else [
            Buffer(
                np.ndarray[(1,), np.dtype[np.int32]],
                name=f"rtp_{i}_{j}",
                use_write_rtp=True,
            )
            for i in range(num_aie_columns)
            for j in range(num_channels)
        ]
    )

    barriers = (
        []
        if use_sp
        else [
            WorkerRuntimeBarrier()
            for i in range(num_aie_columns)
            for j in range(num_channels)
        ]
    )

    # Create a worker to run the task on a compute tile. In the scratchpad path the shared
    # sm_param replaces the per-core RTP buffer and the runtime barrier is unused.
    worker_args = lambda i, j: [
        of_in1s[i * num_channels + j].cons(),
        of_outs[i * num_channels + j].prod(),
        softmax_kernel,
        mask_kernel,
        sm_param if use_sp else rtps[i * num_channels + j],
        sm_param if use_sp else barriers[i * num_channels + j],
    ]
    my_workers = [
        Worker(core_body, worker_args(i, j))
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels.
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        # --- per-op NPU trace hook (opt-in via IRON_TRACE_SIZE env; no-op when unset
        # so production builds are unaffected). Route-(b) standalone per-op measurement. ---
        import os as _os

        if int(_os.environ.get("IRON_TRACE_SIZE", "0")) > 0:
            import aie.utils.trace as _tu

            _ev = _tu.events
            rt.enable_trace(
                int(_os.environ["IRON_TRACE_SIZE"]),
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
        if use_sp:
            # Runtime path: sync the host-written scratchpad word into the cores before they run.
            rt.sync_parameters()
            rt.start(*my_workers)
        else:
            rt.start(*my_workers)

            # Set run-time parameter controlling how many elements each core processes:
            # - Normal case (mask_patch_value == 0): set to rtp_vector_size (the actual active row
            #   width; elements beyond this are padding and ignored by the softmax computation).
            # - Masked case (mask_patch_value != 0): set to mask_patch_value, which the mask kernel
            #   uses as a threshold to zero out elements beyond the unmasked patch boundary.
            def set_rtps(*args):
                for rtp in args:
                    rtp[0] = mask_patch_value if mask_patch_value else rtp_vector_size

            rt.inline_ops(set_rtps, rtps)

            for i in range(num_aie_columns * num_channels):
                rt.set_barrier(barriers[i], 1)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_aie_columns):
            for j in range(num_channels):
                rt.fill(
                    of_in1s[i * num_channels + j].prod(),
                    A,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        # Drain the output objectFIFOs with data
        for i in range(num_aie_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    C,
                    taps[i * num_channels + j],
                    wait=True,  # wait for the transfer to complete and data to be available
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program()
