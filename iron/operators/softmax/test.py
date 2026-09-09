#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import aie.utils as aie_utils

from iron.operators.softmax.op import Softmax
from iron.operators.softmax.reference import (
    generate_golden_reference,
    generate_golden_reference_windowed,
)
from iron.common.test_utils import run_test


def get_optimal_columns_channels(input_length, tile_size, max_columns):
    """Helper function to determine optimal columns and channels for a given input length and tile size"""
    total_cores = input_length // tile_size

    if total_cores == 4:
        return 2, 2  # 4 cores: use 2x2 configuration
    elif total_cores == 8:
        return 2, 2  # 8 cores: use 2x2 configuration (N_div_n=2 iterations per core)
    elif total_cores == 2:
        return 1, 2  # 2 cores: use 1x2 configuration
    elif total_cores == 1:
        return 1, 1  # 1 core: use 1x1 configuration
    elif total_cores == 16:
        # For 16 cores, use 2x2 to avoid exceeding device capabilities
        # The 4x4 configuration causes placement issues on Phoenix
        return 2, 2  # Use 2x2, each core handles more iterations
    else:
        return 2, 2  # Default fallback


def get_params():
    max_aie_columns = aie_utils.get_current_device().cols
    input_lengths = [32768]
    tile_sizes = [1024, 512, 2048]

    params = []
    for input_length in input_lengths:
        for tile_size in tile_sizes:
            optimal_columns, optimal_channels = get_optimal_columns_channels(
                input_length, tile_size, max_aie_columns
            )
            # Skip if configuration exceeds device capabilities
            if optimal_columns > max_aie_columns:
                continue

            params.append(
                pytest.param(input_length, optimal_columns, optimal_channels, tile_size)
            )
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "input_length,num_aie_columns,num_channels,tile_size",
    get_params(),
)
def test_softmax(input_length, num_aie_columns, num_channels, tile_size, aie_context):

    rows = input_length // tile_size
    cols = tile_size

    golden_ref = generate_golden_reference(rows=rows, cols=cols)

    operator = Softmax(
        rows=rows,
        cols=cols,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        context=aie_context,
    )

    input_buffers = {"in": golden_ref["input"]}
    output_buffers = {"output": golden_ref["output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


# A WIDE-STRIDED row: `cols` is what gets computed, `alloc_cols` is what is allocated per row --
# decode's masked softmax runs over a scores row allocated at max_seq while only n_past columns
# are live. Only the buffer size and the per-row stride follow alloc_cols; the computed row length
# stays `cols`. Mirrors gemv's alloc_M tests.
@pytest.mark.parametrize(
    "rows,cols,alloc_cols", [(32, 128, 2048), (16, 64, 512)]
)
def test_alloc_cols_sizes_the_operand_not_the_window(rows, cols, alloc_cols):
    op = Softmax(rows=rows, cols=cols, alloc_cols=alloc_cols)
    spec = op.get_arg_spec()
    assert spec[0].shape == (rows, alloc_cols), (
        f"in operand must be sized by the ALLOCATION {alloc_cols}, got {spec[0].shape}"
    )
    assert spec[1].shape == (rows, alloc_cols), (
        f"out operand must be sized by the ALLOCATION {alloc_cols}, got {spec[1].shape}"
    )


def test_alloc_cols_none_is_the_old_operand_shape():
    """The default must not move: alloc_cols=None leaves the operand sized by cols, flat."""
    a = Softmax(rows=32, cols=128).get_arg_spec()[0].shape
    b = Softmax(rows=32, cols=128, alloc_cols=None).get_arg_spec()[0].shape
    c = Softmax(rows=32, cols=128, alloc_cols=128).get_arg_spec()[0].shape
    assert a == b == c == (32 * 128,), a


def test_alloc_cols_windowed_does_not_share_a_name_with_the_plain_op():
    """A cached plain build must not be able to satisfy a windowed op (see Softmax.name)."""
    plain = Softmax(rows=32, cols=128).name
    windowed = Softmax(rows=32, cols=128, alloc_cols=2048).name
    assert windowed != plain, f"windowed softmax shares an artifact name with the plain one: {plain}"
    assert "ac2048" in windowed, windowed
    # alloc_cols == cols is the same design as None, so it must NOT perturb the stable name.
    assert Softmax(rows=32, cols=128, alloc_cols=128).name == plain


def test_alloc_cols_below_cols_is_refused():
    with pytest.raises(ValueError, match="alloc_cols"):
        Softmax(rows=32, cols=128, alloc_cols=64)


@pytest.mark.parametrize(
    "rows,cols,alloc_cols,num_aie_columns,num_channels",
    [(32, 128, 2048, 2, 2), (16, 64, 512, 1, 2)],
)
def test_softmax_narrow_window_reads_only_its_window(
    rows, cols, alloc_cols, num_aie_columns, num_channels, aie_context
):
    """Poisoned columns past the window catch a wrong PER-ROW STRIDE on the READ side.

    Softmax mixes every element of a row into that row's max/sum, so reading past `cols` (a wrong
    stride) blows up the WHOLE row, not just the tail -- same technique as gemv's alloc_M poison
    test. The comparison itself uses max_error_rate for the padding fraction of the OUTPUT: unlike
    the input, this test cannot poison what the device backend leaves in the output buffer's
    unwritten columns (see generate_golden_reference_windowed's docstring and
    test_gemv_write_lands_at_the_allocated_stride for the same reasoning on gemv's alloc_M_out), so
    it tolerates only that fraction. A wrong stride corrupts every element of every row, which is
    far more than the padding fraction and still fails.
    """
    golden = generate_golden_reference_windowed(rows=rows, cols=cols, alloc_cols=alloc_cols)
    operator = Softmax(
        rows=rows,
        cols=cols,
        alloc_cols=alloc_cols,
        num_aie_columns=num_aie_columns,
        num_channels=num_channels,
        context=aie_context,
    )
    input_buffers = {"in": golden["input"].flatten()}
    output_buffers = {"output": golden["output"].flatten()}
    pad_fraction = (alloc_cols - cols) / alloc_cols
    errors, latency_us, bandwidth_gbps = run_test(
        operator,
        input_buffers,
        output_buffers,
        rel_tol=0.04,
        abs_tol=1e-6,
        max_error_rate=pad_fraction + 0.02,
    )
    assert not errors, f"windowed softmax failed: {errors}"
