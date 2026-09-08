# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    ScratchpadParameter,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    Buffer,
    WorkerRuntimeBarrier,
    sync_parameters,
)
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from ml_dtypes import bfloat16
from iron.operators._trace import maybe_enable_trace


def softmax(
    dev,
    num_elements,
    num_aie_columns,
    num_channels,
    trace_size,
    tile_size,
    rtp_vector_size=None,
    vector_size_parameter=None,
    vector_size_source=None,
    func_prefix="",
    kernel_obj_file="softmax.o",
):
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

    # Per-row mask widths: instead of one scalar unmasked width shared by every
    # row, stream one int32 per row alongside the input. Batched prefill needs
    # this -- row i of a chunk at absolute position `base` may only attend
    # positions <= base + i, so each row in the batch has its own causal width.
    if vector_size_source not in (None, "rows"):
        raise ValueError(
            f"vector_size_source must be None or 'rows', got {vector_size_source!r}"
        )
    per_row_vector_size = vector_size_source == "rows"
    if per_row_vector_size and vector_size_parameter is not None:
        raise ValueError(
            "vector_size_source='rows' takes the width from the streamed buffer, so "
            "vector_size_parameter must be None (both would set the same argument)."
        )
    num_rows = num_elements // per_tile_elements

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    widths_ty = np.ndarray[(num_rows,), np.dtype[np.int32]]
    width_ty = np.ndarray[(1,), np.dtype[np.int32]]

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
    of_widths = (
        [
            ObjectFifo(width_ty, name=f"width_{i}_{j}")
            for i in range(num_aie_columns)
            for j in range(num_channels)
        ]
        if per_row_vector_size
        else []
    )

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

    # Vector size source: either a scratchpad Parameter (synced from host each
    # dispatch) or a write-RTP buffer set via rt.inline_ops at compile time.
    use_scratchpad = vector_size_parameter is not None
    vector_size_param = (
        ScratchpadParameter(vector_size_parameter, np.int32) if use_scratchpad else None
    )

    def core_body(
        of_in1, of_out, softmax_kernel, mask_kernel, vector_size_src, barrier
    ):
        barrier.wait_for_value(1)
        # `use_scratchpad` and `per_row_vector_size` are compile-time constants,
        # so exactly one width source is emitted into the core: a scratchpad
        # Parameter read, a write-RTP buffer load, or -- per row, inside the
        # loop below -- one element of the streamed widths ObjectFifo.
        if not per_row_vector_size:
            if use_scratchpad:
                vector_size = vector_size_src.read()
            else:
                vector_size = vector_size_src[0]
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            if per_row_vector_size:
                # The widths tap walks rows in the same order the input tap does,
                # so this element is this tile's own width.
                vector_size = vector_size_src.acquire(1)[0]
            mask_kernel(elem_in1, vector_size, per_tile_elements)
            softmax_kernel(elem_in1, elem_out, per_tile_elements)
            of_in1.release(1)
            of_out.release(1)
            if per_row_vector_size:
                vector_size_src.release(1)

    rtps = (
        []
        if use_scratchpad or per_row_vector_size
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

    barriers = [
        WorkerRuntimeBarrier()
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    # Create a worker to run the task on a compute tile
    def worker_args(i, j):
        idx = i * num_channels + j
        if per_row_vector_size:
            per_core_runtime = of_widths[idx].cons()
        elif use_scratchpad:
            per_core_runtime = vector_size_param
        else:
            per_core_runtime = rtps[idx]
        return [
            of_in1s[idx].cons(),
            of_outs[idx].prod(),
            softmax_kernel,
            mask_kernel,
            per_core_runtime,
            barriers[idx],
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

    # The width taps chop the same row range as `taps` does, one int32 per row
    # instead of `per_tile_elements` bf16, so core k gets the widths of exactly
    # the rows core k computes.
    width_taps = [
        TensorAccessPattern(
            (1, num_rows),
            N_div_n * (i * num_channels + j),
            [1, 1, 1, N_div_n],
            [0, 0, 0, 1],
        )
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    def sequence(A, *rest):
        if per_row_vector_size:
            W, C, in1_prods, out_conses, width_prods = rest
        else:
            C, in1_prods, out_conses = rest
        if use_scratchpad:
            # The host writes vector_size into the scratchpad via
            # ParameterScratchpad before each dispatch; sync delivers it to the
            # per-core parameter buffer.
            sync_parameters()
        else:
            # Set the static (compile-time) run-time parameter controlling how
            # many elements each core processes.
            for rtp in rtps:
                rtp[0] = rtp_vector_size

        for i in range(num_aie_columns * num_channels):
            barriers[i].set(1)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = TaskGroup()

        # Fill the width objectFIFOs first: a core blocks on its width acquire
        # before it can mask, so these must never trail the data they gate.
        if per_row_vector_size:
            for i in range(num_aie_columns):
                for j in range(num_channels):
                    width_prods[i * num_channels + j].fill(
                        W,
                        width_taps[i * num_channels + j],
                        group=tg,
                    )

        # Fill the input objectFIFOs with data
        for i in range(num_aie_columns):
            for j in range(num_channels):
                in1_prods[i * num_channels + j].fill(
                    A,
                    taps[i * num_channels + j],
                    group=tg,
                )
        # Drain the output objectFIFOs with data
        for i in range(num_aie_columns):
            for j in range(num_channels):
                out_conses[i * num_channels + j].drain(
                    C,
                    taps[i * num_channels + j],
                    wait=True,  # wait for the transfer to complete and data to be available
                    group=tg,
                )
        tg.finish()

    # Only the bare types become runtime_sequence arguments; the handle lists are
    # plumbing. The widths tensor goes between in and out, not after it:
    # OperatorSequence._iter_steps splits a runlist step as `*in_specs, out_spec`,
    # so the operator's output must stay the LAST argument.
    rt_args = [tensor_ty]
    if per_row_vector_size:
        rt_args.append(widths_ty)
    rt_args += [
        tensor_ty,
        [of.prod() for of in of_in1s],
        [of.cons() for of in of_outs],
    ]
    if per_row_vector_size:
        rt_args.append([of.prod() for of in of_widths])

    rt = Runtime(sequence, rt_args)

    # Place program components (assign them resources on the device) and generate an MLIR module
    prog = Program(dev, rt, workers=my_workers)
    maybe_enable_trace(prog, trace_size, my_workers)
    return prog.resolve_program()
