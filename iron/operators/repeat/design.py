# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Repeat interleave
"""

import numpy as np

from aie.dialects.aiex import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup


def repeat(dev, dtype, rows, cols, repeat, transfer_size=None):
    elem_bytes = np.dtype(dtype).itemsize
    dtype = np.dtype[dtype]

    # Split cols into cols_split chunks of cols // cols_split. This is required to
    # satisfy hardware constraints on BD dimensions. We must choose a split that
    # does not exceed the hardware register sizes:
    #   - the chunk length is the innermost dim: <= 1023 (10-bit wrap) AND a whole number
    #     of 32-bit words, since the BD's innermost size is denominated in words
    #   - the chunk count is the next dim out: <= 1023, the same wrap field
    # An odd cols has only odd divisors, so no split of it is ever word-aligned at bf16;
    # that is reported here rather than left to the BD verifier.
    granule = max(1, 4 // elem_bytes)  # elements per 32-bit word

    def two_level_split(width):
        """Smallest divisor d of width with width//d and d both <= 1023 and width//d a
        multiple of granule, or None if width admits no such d."""
        for divisor in range(1, width + 1):
            if width % divisor:
                continue
            chunk = width // divisor
            if chunk <= 1023 and divisor <= 1023 and chunk % granule == 0:
                return divisor, chunk
        return None

    # A single BD tap gives only two split levels (count <= 1023, innermost chunk
    # <= 1023), so it tops out around 1023**2 elements -- e.g. gemma-4-12b's KV
    # repeat at max_seq=2048, head_dim=512 is exactly cols=2**20, just past that
    # ceiling. Past it (or when cols's divisors just don't leave a legal pair -- the
    # cols=2062 case below), split cols into n_chunks separate row-wide transfers,
    # each with its own two-level split, issued back to back through the same FIFO.
    # n_chunks is bounded by the same 1023-wrap field the two-level search already
    # uses: past that many sequential transfers the shape is unsupported, which is
    # what keeps this from "solving" cols=2062 by chunking it down to 1031 transfers
    # of 2 elements each.
    n_chunks = None
    split = None
    for n in range(1, min(1023, cols) + 1):
        if cols % n:
            continue
        found = two_level_split(cols // n)
        if found:
            n_chunks, split = n, found
            break
    if n_chunks is None:
        raise ValueError(
            f"Cannot split cols={cols} at {elem_bytes} bytes/element: no n_chunks <= 1023 "
            f"leaves cols//n_chunks with a divisor d such that (cols//n_chunks)//d <= 1023, "
            f"d <= 1023, and (cols//n_chunks)//d a multiple of {granule} "
            f"({granule} elements = one 32-bit word)."
        )
    cols_split, chunk = split
    chunk_cols = cols // n_chunks

    if transfer_size is None:
        transfer_size = cols

    inp_ty = np.ndarray[
        (rows, cols),
        dtype,
    ]
    out_ty = np.ndarray[
        (rows * repeat, cols),
        dtype,
    ]
    transfer_ty = np.ndarray[
        (transfer_size,),
        dtype,
    ]

    def input_tap(chunk_offset):
        return TensorAccessPattern(
            tensor_dims=(rows, cols),
            offset=chunk_offset,
            # The chunk LENGTH is innermost so the contiguous run is the innermost dim; the
            # chunk COUNT sits outside it. Swapping these two produces the same address
            # sequence, but putting the count innermost makes the unsplit case (cols_split
            # == 1) a 1-element innermost dim, which is not a whole 32-bit word for any
            # sub-word dtype and is rejected by the BD verifier.
            sizes=[repeat, rows, cols_split, chunk],
            strides=[0, cols, chunk, 1],
        )

    def output_tap(chunk_offset):
        return TensorAccessPattern(
            tensor_dims=(rows * repeat, cols),
            offset=chunk_offset,
            sizes=[repeat, rows, cols_split, chunk],
            strides=[cols, cols * repeat, chunk, 1],
        )

    # Use smaller FIFOs for the transfer amount
    fifo_in = ObjectFifo(transfer_ty, name="fifo_in", depth=2)
    fifo_out = fifo_in.cons().forward(name="fifo_out", depth=2)

    def sequence(inp, out, fifo_in_prod, fifo_out_cons):
        tg = TaskGroup()
        for c in range(n_chunks):
            fifo_in_prod.fill(inp, input_tap(c * chunk_cols), group=tg)
        for c in range(n_chunks):
            fifo_out_cons.drain(
                out, output_tap(c * chunk_cols), group=tg, wait=(c == n_chunks - 1)
            )
        tg.finish()

    rt = Runtime(
        sequence,
        [
            inp_ty,
            out_ty,
            fifo_in.prod(),
            fifo_out.cons(),
        ],
    )
    return Program(dev, rt).resolve_program()
