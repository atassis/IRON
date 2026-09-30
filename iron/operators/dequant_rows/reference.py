# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from iron.common.quant import dequantize_weight, quantize_weight


def generate_golden_reference(rows, K, group_size=32, out_stride=None, out_col=0, seed=0):
    """Packed int4 rows and the bf16 matrix the kernel should write: q * bf16(scale), rounded
    once to bf16, which is exact in f32 before the rounding."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((rows, K)).astype(np.float32)
    packed = quantize_weight(W, group_size, "int4")
    deq = dequantize_weight(packed, rows, K, group_size, "int4", emulate_kernel_scale_cast=True)
    out_stride = out_stride or K
    c = torch.zeros(rows, out_stride, dtype=torch.bfloat16)
    c[:, out_col:out_col + K] = torch.from_numpy(deq).to(torch.bfloat16)
    return {"a": torch.from_numpy(np.asarray(packed).view(np.int8).reshape(-1).copy()),
            "c": c.reshape(-1)}
