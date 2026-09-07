# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16


def reference(cur, a, n_pf, Wg, Wu, Wd, D, FF, epsilon=1e-5):
    """CPU (f32) reference for the fused decode SwiGLU MLP block.

    Wg, Wu are flat [FF*D] row-major (FF rows, D cols); Wd is flat [D*FF] row-major (D rows,
    FF cols) -- the same on-wire layout gen_llm_decode.py loads weights into. Matches:
        x1  = cur + a
        hf  = RMSNorm_weighted(x1, n_pf, epsilon)
        nxt = x1 + Wd @ (SiLU(Wg @ hf) * (Wu @ hf))
    """
    cur = np.asarray(cur, np.float32)
    a = np.asarray(a, np.float32)
    n_pf = np.asarray(n_pf, np.float32)
    Wg = np.asarray(Wg, np.float32).reshape(FF, D)
    Wu = np.asarray(Wu, np.float32).reshape(FF, D)
    Wd = np.asarray(Wd, np.float32).reshape(D, FF)

    x1 = cur + a
    rms = np.sqrt(np.mean(x1 * x1) + epsilon)
    hf = (x1 / rms) * n_pf

    g = Wg @ hf
    # SiLU(x) = x * sigmoid(x). np.where(cond, exp(-x), exp(x)) evaluates BOTH branches eagerly,
    # so a plain one-liner still overflows on the discarded branch; index instead.
    sig = np.empty_like(g)
    pos = g >= 0
    sig[pos] = 1.0 / (1.0 + np.exp(-g[pos]))
    sig[~pos] = np.exp(g[~pos]) / (1.0 + np.exp(g[~pos]))
    silu_g = g * sig
    u = Wu @ hf
    gh = silu_g * u
    d = Wd @ gh
    nxt = x1 + d
    return nxt.astype(bfloat16)


def generate_golden_reference(D, FF, seed=42):
    rng = np.random.default_rng(seed)
    val_range = 1.0
    cur = (rng.standard_normal(D) * val_range).astype(bfloat16)
    a = (rng.standard_normal(D) * val_range).astype(bfloat16)
    n_pf = (rng.standard_normal(D) * val_range + 1.0).astype(bfloat16)
    Wg = (rng.standard_normal(FF * D) * val_range).astype(bfloat16)
    Wu = (rng.standard_normal(FF * D) * val_range).astype(bfloat16)
    Wd = (rng.standard_normal(D * FF) * val_range).astype(bfloat16)
    nxt = reference(cur, a, n_pf, Wg, Wu, Wd, D, FF)
    return {"cur": cur, "a": a, "n_pf": n_pf, "Wg": Wg, "Wu": Wu, "Wd": Wd, "nxt": nxt}
