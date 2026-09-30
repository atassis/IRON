# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.test_utils import torch_dtype_map


def reference(a, b):
    """CPU reference: element-wise multiplication (ground truth)."""
    return a * b


def generate_golden_reference(input_length: int, dtype="bf16", seed=42, rows: int | None = None):
    """rows=None: plain [input_length] x [input_length] (unchanged). rows=N: broadcast --
    A is [N, input_length], B stays [input_length], C = A * B via torch's own broadcasting (the
    same result a per-row loop `C[r] = A[r] * B` would produce). A/C come back flattened
    row-major, matching the operator's flat buffer arg spec.
    """
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]
    input_b = torch.rand(input_length, dtype=dtype_torch) * val_range
    if rows is None:
        input_a = torch.rand(input_length, dtype=dtype_torch) * val_range
        return {"A": input_a, "B": input_b, "C": reference(input_a, input_b)}
    input_a = torch.rand(rows, input_length, dtype=dtype_torch) * val_range
    output = reference(input_a, input_b)
    return {"A": input_a.reshape(-1), "B": input_b, "C": output.reshape(-1)}
