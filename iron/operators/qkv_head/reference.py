# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the fused QKV-head operator, composed from the operators it absorbs.

Not wired into any device-run test in this checkout (no NPU available here) -- provided so a
later on-device gate has a ready oracle, matching each absorbed op's own reference() exactly.
"""

import torch

from iron.operators.rms_norm.reference import reference as rms_norm_ref
from iron.operators.rope.reference import reference as rope_ref


def reference(cur, n_in, wq, wk, wv, n_qn, n_kn, ang, D, HD, Hq, Hkv, eps=1e-6):
    """Bit-for-bit the runlist group this operator fuses: norm -> {Wq,Wk,Wv} -> qk-norm -> RoPE.

    Args mirror the device buffers exactly (flat, bf16). `wq`/`wk`/`wv` are (QD*D,)/(KVD*D,)
    row-major flat; `ang` is (HD,) interleaved [cos,sin,...] for one position (angle_rows=1).
    Returns (q, k, v) flat bf16 tensors, shapes (Hq*HD,), (Hkv*HD,), (Hkv*HD,).
    """
    cur = torch.as_tensor(cur).reshape(1, D)
    n_in = torch.as_tensor(n_in).reshape(D)
    hn = rms_norm_ref(cur, n_in, weighted=True, eps=eps).reshape(D)  # (D,)

    wq_m = torch.as_tensor(wq).reshape(Hq * HD, D)
    wk_m = torch.as_tensor(wk).reshape(Hkv * HD, D)
    wv_m = torch.as_tensor(wv).reshape(Hkv * HD, D)

    q_raw = (wq_m.float() @ hn.float()).to(hn.dtype).reshape(Hq, HD)
    k_raw = (wk_m.float() @ hn.float()).to(hn.dtype).reshape(Hkv, HD)
    v = (wv_m.float() @ hn.float()).to(hn.dtype).reshape(Hkv, HD)

    n_qn = torch.as_tensor(n_qn).reshape(HD)
    n_kn = torch.as_tensor(n_kn).reshape(HD)
    ang = torch.as_tensor(ang).reshape(1, HD)

    q_normed = rms_norm_ref(q_raw, n_qn, weighted=True, eps=eps)  # (Hq, HD)
    k_normed = rms_norm_ref(k_raw, n_kn, weighted=True, eps=eps)  # (Hkv, HD)

    q_final = rope_ref(q_normed.reshape(Hq, 1, HD), ang, method_type=0, rows=1, cols=HD)
    k_final = rope_ref(k_normed.reshape(Hkv, 1, HD), ang, method_type=0, rows=1, cols=HD)

    return (
        q_final.reshape(Hq * HD),
        k_final.reshape(Hkv * HD),
        v.reshape(Hkv * HD),
    )
