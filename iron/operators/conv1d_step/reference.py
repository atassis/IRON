# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def reference(window, w, taps, channels, tokens=1):
    """window: flat [taps - 1 + tokens, channels], w: flat [taps, channels], both bf16 -> the
    stepped window: the last taps - 1 inputs, then the tokens conv results."""
    rows = taps - 1 + tokens
    win = window.float().reshape(rows, channels)
    wt = w.float().reshape(taps, channels)
    y = torch.stack([(wt * win[t:t + taps]).sum(0) for t in range(tokens)])
    return torch.cat([win[tokens:], y], 0).reshape(-1).to(torch.bfloat16)


def generate_golden_reference(channels, taps=4, tokens=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    window = torch.randn((taps - 1 + tokens) * channels, generator=g).to(torch.bfloat16)
    w = (torch.randn(taps * channels, generator=g) * 0.5).to(torch.bfloat16)
    return {"window": window, "w": w, "out": reference(window, w, taps, channels, tokens)}
