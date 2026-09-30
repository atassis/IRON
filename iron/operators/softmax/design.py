# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

import aie.dialects.index as index
import aie.extras.dialects.arith as arith
from aie.dialects.aie import T
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

SM_VEC_LEN = 64  # mirrors aie_kernels/aie2p/flash_contract.h; a mismatch drops a tail


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
    allocation_scheme=None,
    segment=None,
):
    if segment is not None:
        return _softmax_chunked(
            dev, num_elements, num_aie_columns, num_channels, trace_size, tile_size,
            segment, rtp_vector_size, vector_size_parameter, vector_size_source,
            func_prefix, kernel_obj_file, allocation_scheme,
        )
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
    #
    # "rows_hole" is the ring variant: past a sliding-window's wrap point the valid
    # columns are a HOLE in the middle, not a suffix, so each row streams a triple
    # (hole_lo, hole_hi, width) instead of one width. See
    # docs/superpowers/specs/2026-09-17-prefill-batch-past-the-sliding-window-design.md sec 1.3.
    if vector_size_source not in (None, "rows", "rows_hole"):
        raise ValueError(
            f"vector_size_source must be None, 'rows' or 'rows_hole', got {vector_size_source!r}"
        )
    hole_mode = vector_size_source == "rows_hole"
    per_row_vector_size = vector_size_source in ("rows", "rows_hole")
    if per_row_vector_size and vector_size_parameter is not None:
        raise ValueError(
            f"vector_size_source={vector_size_source!r} takes the width from the streamed "
            "buffer, so vector_size_parameter must be None (both would set the same argument)."
        )
    num_rows = num_elements // per_tile_elements
    row_ints = 3 if hole_mode else 1

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    widths_ty = np.ndarray[(num_rows * row_ints,), np.dtype[np.int32]]
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
    mask_kernel = (
        Kernel(
            f"{func_prefix}mask_hole_bf16",
            f"{func_prefix}{kernel_obj_file}",
            [tile_ty, np.int32, np.int32, np.int32, np.int32],
        )
        if hole_mode
        else Kernel(
            f"{func_prefix}mask_bf16",
            f"{func_prefix}{kernel_obj_file}",
            [tile_ty, np.int32, np.int32],
        )
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
            if hole_mode:
                # Three sequential scalar acquires off the same fifo -- the triple
                # a row streams, in (hole_lo, hole_hi, width) order (ring_mask_block).
                hole_lo = vector_size_src.acquire(1)[0]
                vector_size_src.release(1)
                hole_hi = vector_size_src.acquire(1)[0]
                vector_size_src.release(1)
                width = vector_size_src.acquire(1)[0]
                vector_size_src.release(1)
                mask_kernel(elem_in1, hole_lo, hole_hi, width, per_tile_elements)
            else:
                if per_row_vector_size:
                    # The widths tap walks rows in the same order the input tap does,
                    # so this element is this tile's own width.
                    vector_size = vector_size_src.acquire(1)[0]
                mask_kernel(elem_in1, vector_size, per_tile_elements)
            softmax_kernel(elem_in1, elem_out, per_tile_elements)
            of_in1.release(1)
            of_out.release(1)
            if per_row_vector_size and not hole_mode:
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
        Worker(core_body, worker_args(i, j), allocation_scheme=allocation_scheme)
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

    # The width taps chop the same row range as `taps` does, `row_ints` int32 per
    # row instead of `per_tile_elements` bf16, so core k gets the widths (or hole
    # triples) of exactly the rows core k computes.
    width_taps = [
        TensorAccessPattern(
            (1, num_rows * row_ints),
            row_ints * N_div_n * (i * num_channels + j),
            [1, 1, 1, row_ints * N_div_n],
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


def _softmax_chunked(
    dev,
    num_elements,
    num_aie_columns,
    num_channels,
    trace_size,
    tile_size,
    segment,
    rtp_vector_size,
    vector_size_parameter,
    vector_size_source,
    func_prefix,
    kernel_obj_file,
    allocation_scheme,
):
    """Softmax whose per-acquire L1 tile is a SEGMENT of a row, not the whole row.

    The unchunked design above sets the objectFIFO tile to `tile_size`, so L1 holds four
    row-length buffers and the largest context that places is 8063 positions -- and a second wall,
    the 16383-word core-tile `aie.dma_bd` length field, sits behind it at ~32768 on the same
    buffer. Both are properties of the TILE, so a per-segment tile clears both and L1 stops
    tracking the row length at all.

    The arithmetic is softmax.cc::softmax_simple_bf16's own three passes, one stream each: max,
    then sum, then normalise-and-write. Same passes in the same order over the same values, which
    is why it is bit-identical rather than approximately equal. The price is that the scores are
    streamed THREE times instead of once. The online (running max/sum) alternative streams them
    twice, but composes segment sums through a bf16 correction factor and was measured to move the
    answer by up to 7.5e-03 rel-L2 -- a uniform scale error on the row, since it lands in the
    divisor. tests/test_chunked_softmax_golden.py carries both numbers.

    ``vector_size_source="rows"`` composes with chunking: each row gets its own causal width off a
    streamed widths ObjectFifo instead of every row on a core sharing one RTP/scratchpad scalar.
    `k_max`/`k_sum`/`k_apply` already take that width as a plain per-call `unmasked` argument (see
    softmax_chunked.cc), so the kernels are unchanged -- only the source of that argument moves from
    a core-wide scalar read once to a per-row scalar read once per row per pass. The widths fifo is
    filled three times, matching the three fills of the data it gates, because the core makes three
    passes over each row and needs its own row's width fresh in each. ``"rows_hole"`` stays refused
    in op.py: it has no chunked-kernel form.
    """
    if tile_size % segment != 0:
        raise ValueError(f"tile_size ({tile_size}) must be a multiple of segment ({segment})")
    if segment % SM_VEC_LEN != 0:
        raise ValueError(
            f"segment ({segment}) must be a multiple of {SM_VEC_LEN}: the kernel loops step "
            f"{SM_VEC_LEN} elements with no scalar tail and would silently drop the remainder"
        )
    if vector_size_source not in (None, "rows"):
        raise ValueError(
            f"segment= supports vector_size_source None or 'rows', got {vector_size_source!r} "
            "(op.py should have refused this already)"
        )
    total_cores = num_aie_columns * num_channels
    if num_elements % total_cores != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {total_cores}."
        )
    if rtp_vector_size is None:
        rtp_vector_size = tile_size
    per_core_elements = num_elements // total_cores
    rows_per_core = per_core_elements // tile_size
    segs_per_row = tile_size // segment
    per_row_vector_size = vector_size_source == "rows"
    num_rows = num_elements // tile_size
    dtype = bfloat16

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(segment,), np.dtype[dtype]]
    lanes_ty = np.ndarray[(rows_per_core * SM_VEC_LEN,), np.dtype[np.float32]]
    mx_ty = np.ndarray[(rows_per_core,), np.dtype[np.float32]]
    widths_ty = np.ndarray[(num_rows,), np.dtype[np.int32]]
    width_ty = np.ndarray[(1,), np.dtype[np.int32]]

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

    k_init = Kernel(f"{func_prefix}softmax_segment_init_f32", f"{func_prefix}{kernel_obj_file}",
                    [lanes_ty, mx_ty, np.int32])
    k_max = Kernel(f"{func_prefix}softmax_segment_max_bf16", f"{func_prefix}{kernel_obj_file}",
                   [tile_ty, mx_ty, np.int32, np.int32, np.int32])
    k_sum = Kernel(f"{func_prefix}softmax_segment_sum_bf16", f"{func_prefix}{kernel_obj_file}",
                   [tile_ty, lanes_ty, mx_ty, np.int32, np.int32, np.int32])
    k_apply = Kernel(f"{func_prefix}softmax_segment_apply_bf16", f"{func_prefix}{kernel_obj_file}",
                     [tile_ty, tile_ty, lanes_ty, mx_ty, np.int32, np.int32, np.int32])

    use_scratchpad = vector_size_parameter is not None
    vector_size_param = (
        ScratchpadParameter(vector_size_parameter, np.int32) if use_scratchpad else None
    )

    def core_body(of_in1, of_out, k_init, k_max, k_sum, k_apply,
                  vector_size_src, barrier, lanes, mx):
        barrier.wait_for_value(1)
        if not per_row_vector_size:
            vector_size = vector_size_src.read() if use_scratchpad else vector_size_src[0]

        def seg_unmasked(vs, s):
            # This segment's valid width, clamped into [0, segment]. Clamped at zero because a
            # segment wholly past the mask boundary yields a negative width, and the kernels read
            # `<= 0` as "contributes nothing" -- see attn_block_dp/design.py for the same clamp.
            lo = index.casts(T.i32(), s) * segment
            return arith.maxsi(
                arith.minsi(arith.subi(vs, lo), arith.constant(segment, T.i32())),
                arith.constant(0, T.i32()),
            )

        for r in range_(rows_per_core):
            k_init(lanes, mx, index.casts(T.i32(), r))

        # Three passes, rows inside each. The fill issues the same tap three times, so the tile
        # arriving at each acquire is row-major within a pass -- passes outer is what that order
        # requires, and it is why the running state (and, in per-row mode, the width) is per row
        # rather than a single set. Per-row mode re-acquires its row's width once per pass, from a
        # fifo the host has filled three times to match.
        for r in range_(rows_per_core):
            ri = index.casts(T.i32(), r)
            if per_row_vector_size:
                vector_size = vector_size_src.acquire(1)[0]
            for s in range_(segs_per_row):
                elem = of_in1.acquire(1)
                k_max(elem, mx, ri, segment, seg_unmasked(vector_size, s))
                of_in1.release(1)
            if per_row_vector_size:
                vector_size_src.release(1)
        for r in range_(rows_per_core):
            ri = index.casts(T.i32(), r)
            if per_row_vector_size:
                vector_size = vector_size_src.acquire(1)[0]
            for s in range_(segs_per_row):
                elem = of_in1.acquire(1)
                k_sum(elem, lanes, mx, ri, segment, seg_unmasked(vector_size, s))
                of_in1.release(1)
            if per_row_vector_size:
                vector_size_src.release(1)
        for r in range_(rows_per_core):
            ri = index.casts(T.i32(), r)
            if per_row_vector_size:
                vector_size = vector_size_src.acquire(1)[0]
            for s in range_(segs_per_row):
                elem = of_in1.acquire(1)
                elem_out = of_out.acquire(1)
                k_apply(elem, elem_out, lanes, mx, ri, segment, seg_unmasked(vector_size, s))
                of_in1.release(1)
                of_out.release(1)
            if per_row_vector_size:
                vector_size_src.release(1)

    rtps = (
        []
        if use_scratchpad or per_row_vector_size
        else [
            Buffer(np.ndarray[(1,), np.dtype[np.int32]], name=f"rtp_{i}_{j}", use_write_rtp=True)
            for i in range(num_aie_columns)
            for j in range(num_channels)
        ]
    )
    barriers = [
        WorkerRuntimeBarrier()
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    def per_core_runtime(idx):
        if per_row_vector_size:
            return of_widths[idx].cons()
        if use_scratchpad:
            return vector_size_param
        return rtps[idx]

    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                k_init,
                k_max,
                k_sum,
                k_apply,
                per_core_runtime(i * num_channels + j),
                barriers[i * num_channels + j],
                Buffer(lanes_ty, name=f"{func_prefix}sm_lanes_{i}_{j}"),
                Buffer(mx_ty, name=f"{func_prefix}sm_max_{i}_{j}"),
            ],
            allocation_scheme=allocation_scheme,
        )
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    # The same contiguous per-core window the unchunked taps describe, spelled in its own
    # [row][segment] dims rather than one flat extent -- the flat form asks the shim for a single
    # descriptor whose length grows with the context, which is the class of field this whole
    # change exists to stop depending on.
    chunk = per_core_elements
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * (i * num_channels + j),
            [1, rows_per_core, segs_per_row, segment],
            [0, tile_size, segment, 1],
        )
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]
    # The width taps chop the same row range as `taps` does, one int32 per row instead of
    # `tile_size` bf16, so core k gets exactly the rows core k computes.
    width_taps = [
        TensorAccessPattern(
            (1, num_rows),
            rows_per_core * (i * num_channels + j),
            [1, 1, 1, rows_per_core],
            [0, 0, 0, 1],
        )
        for i in range(num_aie_columns)
        for j in range(num_channels)
    ]

    def sequence(A, *rest):
        if per_row_vector_size:
            W, C, in1_prods, out_conses, width_prods = rest
        else:
            C, in1_prods, out_conses = rest
        if use_scratchpad:
            sync_parameters()
        else:
            for rtp in rtps:
                rtp[0] = rtp_vector_size
        for i in range(total_cores):
            barriers[i].set(1)

        tg = TaskGroup()
        # Three fills of the same window, one per pass. Pass 3 cannot start until pass 2 has
        # finished the row it writes, so the drain is issued last and waited on. Widths fill
        # first each pass: a core blocks on its width acquire before its first data acquire.
        for _ in range(3):
            if per_row_vector_size:
                for k in range(total_cores):
                    width_prods[k].fill(W, width_taps[k], group=tg)
            for k in range(total_cores):
                in1_prods[k].fill(A, taps[k], group=tg)
        for k in range(total_cores):
            out_conses[k].drain(C, taps[k], wait=True, group=tg)
        tg.finish()

    rt_args = [tensor_ty]
    if per_row_vector_size:
        rt_args.append(widths_ty)
    rt_args += [tensor_ty, [of.prod() for of in of_in1s], [of.cons() for of in of_outs]]
    if per_row_vector_size:
        rt_args.append([of.prod() for of in of_widths])

    rt = Runtime(sequence, rt_args)
    prog = Program(dev, rt, workers=my_workers)
    maybe_enable_trace(prog, trace_size, my_workers)
    return prog.resolve_program()
