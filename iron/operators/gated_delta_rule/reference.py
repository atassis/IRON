# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def reference(a, b, a_log, dt_bias, q, k, v, S):
    """HF Qwen3.5 torch_recurrent_gated_delta_rule for one token, in f64.
    q, k: [v_heads, dk] (already expanded per value head), v: [v_heads, dv], S: [v_heads, dk, dv]."""
    f = lambda x: torch.as_tensor(x).double()
    a, b, a_log, dt_bias, q, k, v, S = map(f, (a, b, a_log, dt_bias, q, k, v, S))
    alpha = torch.exp(-a_log.exp() * torch.nn.functional.softplus(a + dt_bias))
    beta = torch.sigmoid(b)
    S = S * alpha[:, None, None]
    err = (v - torch.einsum("hkv,hk->hv", S, k)) * beta[:, None]
    S = S + k[:, :, None] * err[:, None, :]
    return torch.einsum("hkv,hk->hv", S, q), S


def generate_golden_reference(v_heads=32, k_heads=16, dk=128, dv=128, tokens=1, l2_qk=False,
                              count=None, seed=0):
    """Inputs laid out the way Qwen3.5 holds them, one row per token, and the expected outputs.
    With `l2_qk`, q and k are passed raw and normalised in the reference, rounded to bf16 where
    the op writes them back. With `count`, only the first `count` tokens step the state and the
    rest have zero o rows."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)
    bf = lambda x: x.to(torch.bfloat16)
    a_log = torch.log(torch.rand(v_heads, generator=g) * 15.99 + 0.01)
    dt_bias = rn(v_heads) * 0.5
    S0 = S = rn(v_heads, dk, dv) * 0.1
    rep = v_heads // k_heads
    ab_len = 4 * dk
    abs_, mixeds, os = [], [], []
    for t in range(tokens):
        a, b = bf(rn(v_heads) * 2), bf(rn(v_heads) * 2)
        if l2_qk:
            q, k = bf(rn(k_heads, dk) * 3), bf(rn(k_heads, dk) * 3)
            l2 = lambda x: x.double() * torch.rsqrt((x.double() ** 2).sum(-1, keepdim=True) + 1e-6)
            qn, kn = bf(l2(q) * dk ** -0.5).float(), bf(l2(k)).float()
        else:
            q = bf(torch.nn.functional.normalize(rn(k_heads, dk), dim=-1) * dk ** -0.5)
            k = bf(torch.nn.functional.normalize(rn(k_heads, dk), dim=-1))
            qn, kn = q.float(), k.float()
        v = bf(rn(v_heads, dv))
        if count is None or t < count:
            o, S = reference(a.float(), b.float(), a_log, dt_bias, qn.repeat_interleave(rep, 0),
                             kn.repeat_interleave(rep, 0), v.float(), S)
        else:
            o = torch.zeros(v_heads, dv)
        ab = torch.zeros(ab_len, dtype=torch.bfloat16)
        ab[:v_heads], ab[v_heads:2 * v_heads] = a, b
        abs_.append(ab)
        mixeds.append(torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)]))
        os.append(bf(o.float()).reshape(-1))
    params = np.concatenate([-np.exp(a_log.numpy().astype(np.float64)).astype(np.float32),
                             dt_bias.numpy().astype(np.float32)])
    params = torch.from_numpy(params.view(np.uint16).copy()).view(torch.bfloat16)
    cnt = torch.zeros(dk, dtype=torch.bfloat16)
    if count is not None:
        cnt.view(torch.int16)[:2] = torch.tensor([count], dtype=torch.int32).view(torch.int16)
    return {"ab": torch.cat(abs_), "params": params, "count": cnt, "mixed": torch.cat(mixeds),
            "s_in": S0.float().reshape(-1), "s_out": S.float().reshape(-1), "o": torch.cat(os),
            "ab_len": ab_len, "q_off": 0, "k_off": k_heads * dk, "v_off": 2 * k_heads * dk,
            "mixed_len": mixeds[0].numel()}
