# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden reference generator for softmax operator."""

import torch
from iron.common.test_utils import torch_dtype_map


def reference(x):
    """CPU reference: row-wise softmax over the last dim (ground truth)."""
    return torch.softmax(x, dim=-1)


def masked_reference(x, widths):
    """CPU reference for per-row mask widths.

    Row ``i`` is softmaxed over ``x[i, :widths[i]]`` and the tail is zero, which
    is what the device produces: ``mask_bf16`` writes ``-inf`` past ``widths[i]``
    and ``softmax_bf16`` then exponentiates the whole row.

    A width of 0 leaves the row all-``-inf``; softmax of that is undefined, so
    this raises rather than returning the NaNs the device would.

    This models the MASK, not the arithmetic. It does not cover the device's
    bf16 rounding, and it assumes ``aie::exp2`` (a coarse SFU LUT) returns
    exactly 0 for the ``-inf`` the mask writes -- unverified on hardware.
    """
    widths = torch.as_tensor(widths, dtype=torch.long).reshape(-1)
    if widths.numel() != x.shape[0]:
        raise ValueError(f"expected {x.shape[0]} widths, got {widths.numel()}")
    if int(widths.min()) < 1 or int(widths.max()) > x.shape[1]:
        raise ValueError(
            f"widths must lie in [1, {x.shape[1]}]; got "
            f"[{int(widths.min())}, {int(widths.max())}]"
        )
    positions = torch.arange(x.shape[1]).unsqueeze(0)
    keep = positions < widths.unsqueeze(1)
    # Same dtype path as reference(), so all-`cols` widths reduce to it exactly.
    return torch.softmax(x.masked_fill(~keep, float("-inf")), dim=-1)


def generate_golden_widths(rows: int, cols: int, base: int = 0):
    """Causal widths for a chunk of ``rows`` tokens starting at ``base``.

    Row ``i`` attends positions ``<= base + i``, clamped to ``cols``.
    """
    return torch.clamp(torch.arange(rows, dtype=torch.int32) + base + 1, max=cols).to(
        torch.int32
    )


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
