# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden reference generator for softmax operator."""

import torch
from iron.common.test_utils import torch_dtype_map


def reference(x):
    """CPU reference: row-wise softmax over the last dim (ground truth)."""
    return torch.softmax(x, dim=-1)


def generate_golden_reference(rows: int, cols: int, dtype="bf16", seed=42):
    """
    Generate golden reference data for softmax.

    Returns:
        dict: Dictionary with tensors for inputs and outputs
    """
    torch.manual_seed(seed)
    val_range = 4
    input_tensor = torch.rand(rows, cols, dtype=torch_dtype_map[dtype]) * val_range
    output_tensor = reference(input_tensor)
    return {"input": input_tensor, "output": output_tensor}


def generate_golden_reference_windowed(
    rows: int, cols: int, alloc_cols: int, dtype="bf16", seed=42
):
    """Golden for a WIDE-STRIDED row: the in/out buffers are allocated with `alloc_cols` elements
    per row but only the first `cols` are computed, mirroring gemv's alloc_M golden.

    Input columns past `cols` are POISONED: a softmax over the wrong window (a wrong per-row
    stride) mixes poison into the row's own max/sum and blows up every element of that row, not
    just the tail -- so this catches the stride, not merely the window. The output's own padding
    is left at 0 here; the CALLER must not compare it exactly (the device backend does not
    guarantee the output buffer's initial content), only via a max_error_rate matched to the
    padding fraction. See test.py's device test for how that comparison is built.
    """
    torch.manual_seed(seed)
    val_range = 4
    input_tensor = torch.zeros(rows, alloc_cols, dtype=torch_dtype_map[dtype])
    input_tensor[:, :cols] = torch.rand(rows, cols, dtype=torch_dtype_map[dtype]) * val_range
    input_tensor[:, cols:] = 1000.0
    output_tensor = torch.zeros(rows, alloc_cols, dtype=torch_dtype_map[dtype])
    output_tensor[:, :cols] = reference(input_tensor[:, :cols])
    return {"input": input_tensor, "output": output_tensor}
