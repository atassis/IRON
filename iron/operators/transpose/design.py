# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def shuffle_transpose(
    dev, M, N, num_columns, num_channels, m, n, s, num_batches=1, func_prefix="",
    coalesce_batch_dma=False,
):
    num_elements = M * N
    per_tile_elements = m * n
    dtype = bfloat16

    if M % m != 0:
        raise ValueError(f"Matrix rows ({M}) must be a multiple of {m}.")
    if N % n != 0:
        raise ValueError(f"Matrix columns ({N}) must be a multiple of {n}.")
    if m % s != 0:
        raise ValueError(f"AIE tile rows ({m}) must be a multiple of {s}.")
    if n % s != 0:
        raise ValueError(f"AIE tile columns ({n}) must be a multiple of {s}.")
    if per_tile_elements > 8192:
        raise ValueError(
            f"Kernel tile size {per_tile_elements} needs to be below 8192 to fit within data memory."
        )

    # Minimum tile sizes required by the two kernels
    if s == 4 and (m <= 4 or n <= 4):
        raise ValueError(f"Kernel tile {s} needs AIE tile rows > 4 and columns > 4.")
    if s == 8 and (m <= 16 or n <= 16):
        raise ValueError(f"Kernel tile {s} needs AIE tile rows > 16 and columns > 16.")

    # Define tensor types. The runtime tensor spans all batches (contiguous matrices);
    # per-tile work on the cores is identical regardless of batch count.
    tensor_ty = np.ndarray[(num_batches * num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    fifodepth = 1 if per_tile_elements > 4096 else 2

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels. Partially transposes the input
    # data so that the kernel only needs to
    # transpose s*s-sized sub-tiles.
    # For num_batches>1 the L3 tensors hold that many contiguous (M,N) matrices, stacked along
    # the row dimension: in-dims (num_batches*M, N), out-dims (num_batches*N, M). At num_batches==1
    # these reduce to (M,N)/(N,M) — identical to the original single-transpose patterns. Each (i,j)
    # column/channel gets one TAP per batch (offset += batch*num_elements); the per-batch internal
    # sizes/strides are unchanged because each matrix is contiguous and row-major.
    in_dims = (num_batches * M, N)
    out_dims = (num_batches * N, M)
    taps_in_L3L2 = [
        [
            TensorAccessPattern(
                in_dims,
                batch * num_elements
                + (M // num_channels) * j * N
                + (N // num_columns) * i,
                [M // num_channels // m, N // num_columns // n, m, n],
                [m * N, n, N, 1],
            )
            for batch in range(num_batches)
        ]
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    taps_in_L2L1 = [
        TensorAccessPattern(
            (M, N),
            (M // num_channels) * j * N + (N // num_columns) * i,
            [m // s, s, n // s, s],
            [s, m, s * m, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    taps_out_L1L3 = [
        [
            TensorAccessPattern(
                out_dims,
                batch * num_elements
                + (N // num_columns) * i * M
                + (M // num_channels) * j,
                [M // num_channels // m, N // num_columns // n, n, m],
                [m, n * M, M, 1],
            )
            for batch in range(num_batches)
        ]
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # B-unroll -> BD-iteration (opt-in): collapse the per-batch L3 fill/drain into batched BD(s).
    # Two cases, both requiring num_columns==num_channels==1 and n==N (so the tile grid is only on M):
    #
    #  SINGLE-TILE (m==M): each per-batch L3 transfer is a CONTIGUOUS run of num_elements at offset
    #    batch*num_elements (the transpose is done by the kernel's batch-independent L2L1 TAP, not the
    #    L3 DMA). Both fill and drain coalesce to ONE batched 4D BD (batch in the size-uncapped dim,
    #    run split across two wrap dims <=1023) — identical to the GEMV lever.
    #
    #  MULTI-TILE (m<M): the FILL is still contiguous (grid_row stride m*N == inner block m*n since
    #    n==N -> telescopes), so one BD. The DRAIN is a transpose-SCATTER whose per-batch enumeration
    #    order [grid_row, n, m] must be preserved with batch OUTERMOST. The AIE2p iteration (outermost)
    #    BD dim is capped at <=64, so batch can't be a single >64 dim there -> the drain is BATCH-CHUNKED:
    #    ceil(num_batches/64) BDs, each [chunk<=64, grid_row, n, m] strides [num_elements, m, M, 1].
    #
    # Offline-verified to enumerate the identical DRAM access for both cases:
    #   scripts/tap_equivalence_transpose.py (single-tile), tap_equivalence_transpose_multitile.py (multi-tile).
    # Default off; any other shape (multi-column/-channel, or n!=N) falls back to the per-batch path.
    _grid_ok = num_columns == 1 and num_channels == 1 and n == N and M % m == 0
    _single_tile = _grid_ok and m == M
    _multi_tile = _grid_ok and m < M
    _do_coalesce = coalesce_batch_dma and (_single_tile or _multi_tile)
    _ITER_CAP = 64  # AIE2p iteration (outermost) BD dim cap (empirical, GEMV bring-up)

    def _split_run(n_, lim=1023):  # (hi, lo): lo = largest divisor <= lim (contiguous inner)
        for lo in range(min(lim, n_), 0, -1):
            if n_ % lo == 0 and (n_ // lo) <= lim:
                return (n_ // lo, lo)
        raise ValueError(f"transpose run={n_} not splittable into two dims <= {lim}")

    def _coalesced_contiguous(dims):  # ONE BD: contiguous num_elements run, batch in the uncapped dim
        assert num_elements <= (1 << 20), f"batch stride {num_elements} exceeds 2**20"
        rhi, rlo = _split_run(num_elements)
        return TensorAccessPattern(dims, 0, [1, num_batches, rhi, rlo], [0, num_elements, rlo, 1])

    def _chunked_drain(dims):  # batch-chunked transpose-scatter drain (multi-tile): one BD per <=64 batches
        grid = M // m
        # sizes [cb<=64, grid, n, m] must be <=1023 (cb<=_ITER_CAP by construction); strides
        # [num_elements, m, M, 1] must be <=2**20 (M is a STRIDE here, not a size).
        assert grid <= 1023 and n <= 1023 and m <= 1023
        assert num_elements <= (1 << 20) and M <= (1 << 20)
        taps = []
        for c0 in range(0, num_batches, _ITER_CAP):
            cb = min(_ITER_CAP, num_batches - c0)
            taps.append(TensorAccessPattern(
                dims, c0 * num_elements, [cb, grid, n, m], [num_elements, m, M, 1]))
        return taps

    if _single_tile:
        taps_in_L3L2_coalesced = [_coalesced_contiguous(in_dims)]
        taps_out_L1L3_coalesced = [_coalesced_contiguous(out_dims)]
    elif _multi_tile:
        taps_in_L3L2_coalesced = [_coalesced_contiguous(in_dims)]   # fill stays contiguous (1 BD)
        taps_out_L1L3_coalesced = _chunked_drain(out_dims)          # drain = ceil(nb/64) chunk BDs
    else:
        taps_in_L3L2_coalesced = None
        taps_out_L1L3_coalesced = None

    # AIE-array data movement with object fifos
    of_in1s_L3L2 = [
        ObjectFifo(tile_ty, name=f"of_in1s_L3L2_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_in1s_L2L1 = [
        of_in1s_L3L2[i * num_channels + j]
        .cons(dims_from_stream=taps_in_L2L1[i * num_channels + j].transformation_dims)
        .forward(obj_type=tile_ty, name=f"of_in1s_L2L1_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # AIE Core Function declaration
    transpose_kernel = Kernel(
        f"{func_prefix}transpose_{s}x{s}",
        f"{func_prefix}transpose_{m}x{n}.o",
        [tile_ty, tile_ty],
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_out, transpose_kernel):
        # Process num_batches contiguous matrices through the same FIFOs: num_batches x the per-matrix
        # tile iterations. The kernel only ever sees s*s sub-tiles, so it is batch-agnostic.
        for _ in range_(num_batches):
            # Number of sub-matrix "tile" iterations
            for _ in range_(N // n // num_columns):
                for _ in range_(M // m // num_channels):
                    elem_in1 = of_in1.acquire(1)
                    elem_out = of_out.acquire(1)
                    transpose_kernel(elem_in1, elem_out)
                    of_out.release(1)
                    of_in1.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = [
        Worker(
            core_body,
            [
                of_in1s_L2L1[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                transpose_kernel,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*my_workers)

        # Coalesced path: ONE task group; one contiguous fill BD + 1 (single-tile) or ceil(nb/64)
        # (multi-tile, batch-chunked) drain BD(s). Relies on ObjectFifo backpressure for flow control
        # instead of the per-batch `wait`.
        if _do_coalesce:
            tg = rt.task_group()
            rt.fill(of_in1s_L3L2[0].prod(), A, taps_in_L3L2_coalesced[0], task_group=tg)
            for _ot in taps_out_L1L3_coalesced:
                rt.drain(of_outs[0].cons(), C, _ot, wait=True, task_group=tg)
            rt.finish_task_group(tg)
        else:
            # One task group per batch (each a parallel fill+drain over all columns/channels), so the
            # num_batches contiguous matrices stream through the same FIFOs in sequence. At num_batches==1
            # this is a single pass — identical to the original single-transpose schedule.
            for batch in range(num_batches):
                # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
                tg = rt.task_group()

                # Fill the input objectFIFOs with data
                for i in range(num_columns):
                    for j in range(num_channels):
                        rt.fill(
                            of_in1s_L3L2[i * num_channels + j].prod(),
                            A,
                            taps_in_L3L2[i * num_channels + j][batch],
                            task_group=tg,
                        )
                # Drain the output objectFIFOs with data
                for i in range(num_columns):
                    for j in range(num_channels):
                        rt.drain(
                            of_outs[i * num_channels + j].cons(),
                            C,
                            taps_out_L1L3[i * num_channels + j][batch],
                            wait=True,  # wait for the transfer to complete and data to be available
                            task_group=tg,
                        )
                rt.finish_task_group(tg)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())
