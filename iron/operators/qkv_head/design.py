# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused QKV-head design: ONE aie.device for the decode group

    hn = weighted_RMSNorm(cur, n_in)                      # D-wide
    q  = Wq @ hn ; per-head weighted_RMSNorm(n_qn) ; RoPE(ang)   # Hq heads of HD
    k  = Wk @ hn ; per-head weighted_RMSNorm(n_kn) ; RoPE(ang)   # Hkv heads of HD
    v  = Wv @ hn                                          # raw, no norm/rope

Replaces 6 consecutive runlist entries (534/token -> 1 aiex.configure/token for this group) in
xdna-engine/designs/decode_fused/gen_llm_decode.py. `hn` never reaches DDR: each of `cols` workers
recomputes it locally from `cur`/`n_in` (redundant, but decode compute is ~free -- M=1) instead of
broadcasting one shared copy, trading a few KB of duplicated DMA for a much simpler dataflow (no
MemTile fan-out network). Each of the `cols` workers owns HEADS_Q_PER_COL query heads and
HEADS_KV_PER_COL key/value heads (contiguous slices of Wq/Wk/Wv), so Hq and Hkv must both divide
`cols` evenly -- Qwen3-0.6B (Hq=16, Hkv=8, cols=8) gives 2 and 1.

MEASURED CONSTRAINT (aiecc, this design): an AIE2P compute tile has exactly 2 input + 2 output DMA
channels, hard. A first version with 6 separate input ObjectFifos (cur, n_in, the matvec A-stream,
n_qn, n_kn, ang) failed to place with the tile diagnostic verbatim:
"tile (0, 3) requires 6 input/1 output DMA channels, but only 2 input/2 output available". Fixed by
collapsing to exactly 2 input channels, grouped by L1 tile shape:
  - `d_of` (D-wide): carries cur, then n_in (acquired together via `.acquire(2)` since the
    weighted_RMSNorm call needs both at once), then every row of this column's Wq/Wk/Wv slice, one
    D-wide row per matvec call. This FORCES tile_size_input (the matvec row-chunk) to 1 -- a real
    performance regression (128 matvec calls per head instead of 32 at tsi=4) traded for fitting
    the channel budget; restoring tsi>1 needs MemTile staging (aiecc's own suggested fix, "reduce
    the LTO's DMA fanin e.g. via memtile staging"), not attempted here.
  - `w_of` (HD-wide): carries n_qn, n_kn, ang, acquired together via `.acquire(3)` and held for the
    whole core body (they're each read multiple times: n_qn for every Q head, ang for every RoPE
    call).
One shared HD-wide output ObjectFifo (`c_of`) drains q-head-0, q-head-1, ..., k-head, v-head in
sequence -- 1 of the 2 available output channels, no consolidation needed there.
"""

import aie.dialects.index as index
from aie.dialects.aie import T
from ml_dtypes import bfloat16
import numpy as np

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker

BF16 = bfloat16


def qkv_head_fused(
    dev,
    D,
    HD,
    Hq,
    Hkv,
    QD,
    KVD,
    cols,
    epsilon=1e-6,
    mv_kernel_object="mv.o",
    rms_kernel_object="qkv_rms_norm.o",
    rope_kernel_object="rope_0.o",
    stack_size=0xD00,
    verbose=False,
):
    # tsi (matvec rows/call) is fixed at 1: the shared `d_of` channel below requires every row it
    # carries -- cur, n_in, and each Wq/Wk/Wv row -- to be exactly D-wide. See module docstring.
    tsi = 1
    assert QD == Hq * HD, f"QD ({QD}) must equal Hq*HD ({Hq * HD})"
    assert KVD == Hkv * HD, f"KVD ({KVD}) must equal Hkv*HD ({Hkv * HD})"
    assert Hq % cols == 0, f"Hq ({Hq}) must be divisible by cols ({cols})"
    assert Hkv % cols == 0, f"Hkv ({Hkv}) must be divisible by cols ({cols})"

    heads_q_per_col = Hq // cols
    heads_kv_per_col = Hkv // cols
    wq_rows_per_col = heads_q_per_col * HD
    wk_rows_per_col = heads_kv_per_col * HD
    wv_rows_per_col = heads_kv_per_col * HD

    if verbose:
        print(f"Device: {dev}, cols={cols}")
        print(f"D={D} HD={HD} Hq={Hq} Hkv={Hkv} QD={QD} KVD={KVD} tsi={tsi}")
        print(f"heads/col: q={heads_q_per_col} kv={heads_kv_per_col}")

    # ---- tensors living in DRAM (L3) ----
    cur_ty = np.ndarray[(D,), np.dtype[BF16]]
    nin_ty = np.ndarray[(D,), np.dtype[BF16]]
    wq_ty = np.ndarray[(QD * D,), np.dtype[BF16]]
    wk_ty = np.ndarray[(KVD * D,), np.dtype[BF16]]
    wv_ty = np.ndarray[(KVD * D,), np.dtype[BF16]]
    nqn_ty = np.ndarray[(HD,), np.dtype[BF16]]
    nkn_ty = np.ndarray[(HD,), np.dtype[BF16]]
    ang_ty = np.ndarray[(HD,), np.dtype[BF16]]
    q_ty = np.ndarray[(QD,), np.dtype[BF16]]
    k_ty = np.ndarray[(KVD,), np.dtype[BF16]]
    v_ty = np.ndarray[(KVD,), np.dtype[BF16]]

    # ---- tensors living in L1 (per column) ----
    hn_ty = np.ndarray[(D,), np.dtype[BF16]]
    d_tile_ty = np.ndarray[(D,), np.dtype[BF16]]  # shared: cur, n_in, then every Wq/Wk/Wv row
    small_tile_ty = np.ndarray[(HD,), np.dtype[BF16]]  # shared: n_qn, n_kn, ang
    c_tile_ty = np.ndarray[(HD,), np.dtype[BF16]]  # shared HD-wide output tile: q,q,...,k,v

    # ---- kernel declarations (module-scope symbols, reused by every column) ----
    matvec_kernel = Kernel(
        "matvec_vectorized_bf16_bf16",
        mv_kernel_object,
        [np.int32, np.int32, d_tile_ty, hn_ty, c_tile_ty],
    )
    # qkv_rms_norm_{d,hd}(a_in, b_in, c_out, size, epsilon): normalize-then-multiply-by-weight in
    # one call (rms_norm_qkv.cc, a local copy of aie_kernels/aie2p/rms_norm.cc's
    # weighted_rms_norm under two names) -- no separate elementwise-mul kernel needed. Two names
    # because this core calls it at two shapes (D-wide for hn, HD-wide for qk-norm); one IRON
    # Kernel() per symbol name emits one `func.func private` declaration each, so reusing a
    # single name at two shapes would emit two conflicting declarations in the same module.
    rms_norm_d_kernel = Kernel(
        "qkv_rms_norm_d",
        rms_kernel_object,
        [d_tile_ty, d_tile_ty, hn_ty, np.int32, np.float32],
    )
    rms_norm_hd_kernel = Kernel(
        "qkv_rms_norm_hd",
        rms_kernel_object,
        [c_tile_ty, small_tile_ty, c_tile_ty, np.int32, np.float32],
    )
    rope_kernel = Kernel(
        "rope",
        rope_kernel_object,
        [c_tile_ty, small_tile_ty, c_tile_ty, np.int32],
    )

    def core_body(
        d_of, w_of, c_of,
        hn_buf, qraw_buf, qnorm_buf, kraw_buf, knorm_buf,
        matvec_kernel, rms_norm_d_kernel, rms_norm_hd_kernel, rope_kernel,
    ):
        # hn = weighted_RMSNorm(cur, n_in), D-wide, kept in L1 only. cur/n_in share `d_of` with
        # the matvec row-stream below, so both must be live at once: acquire(2) up front.
        d2 = d_of.acquire(2)
        rms_norm_d_kernel(d2[0], d2[1], hn_buf, D, epsilon)
        d_of.release(2)

        def project_head(raw_buf, weight_t, norm_buf):
            for i_idx in range_(HD // tsi):
                i_i32 = index.casts(T.i32(), i_idx)
                row_off = i_i32 * tsi
                a_t = d_of.acquire(1)
                matvec_kernel(tsi, row_off, a_t, hn_buf, raw_buf)
                d_of.release(1)
            rms_norm_hd_kernel(raw_buf, weight_t, norm_buf, HD, epsilon)

        # n_qn, n_kn, ang share `w_of`: acquire once, hold for the whole body (each is read
        # multiple times -- n_qn per Q head, ang per RoPE call on either q or k).
        w3 = w_of.acquire(3)
        nqn_t, nkn_t, ang_t = w3[0], w3[1], w3[2]

        # Q heads: project, qk-norm, RoPE, drain -- one head at a time.
        for _h in range(heads_q_per_col):
            project_head(qraw_buf, nqn_t, qnorm_buf)
            c_t = c_of.acquire(1)
            rope_kernel(qnorm_buf, ang_t, c_t, HD)
            c_of.release(1)

        # K heads: same, weighted by n_kn.
        for _h in range(heads_kv_per_col):
            project_head(kraw_buf, nkn_t, knorm_buf)
            c_t = c_of.acquire(1)
            rope_kernel(knorm_buf, ang_t, c_t, HD)
            c_of.release(1)

        w_of.release(3)

        # V heads: raw matvec straight into the drain tile, no norm/rope.
        for _h in range(heads_kv_per_col):
            c_t = c_of.acquire(1)
            for i_idx in range_(HD // tsi):
                i_i32 = index.casts(T.i32(), i_idx)
                row_off = i_i32 * tsi
                a_t = d_of.acquire(1)
                matvec_kernel(tsi, row_off, a_t, hn_buf, c_t)
                d_of.release(1)
            c_of.release(1)

    # ---- per-column ObjectFifos + Buffers + Worker ----
    d_ofs, w_ofs, c_ofs = [], [], []
    workers = []
    for c in range(cols):
        d_of = ObjectFifo(d_tile_ty, name=f"d_of_{c}", depth=2)
        w_of = ObjectFifo(small_tile_ty, name=f"w_of_{c}", depth=3)
        c_of = ObjectFifo(c_tile_ty, name=f"c_of_{c}", depth=2)
        d_ofs.append(d_of)
        w_ofs.append(w_of)
        c_ofs.append(c_of)

        hn_buf = Buffer(hn_ty, name=f"hn_{c}")
        qraw_buf = Buffer(c_tile_ty, name=f"qraw_{c}")
        qnorm_buf = Buffer(c_tile_ty, name=f"qnorm_{c}")
        kraw_buf = Buffer(c_tile_ty, name=f"kraw_{c}")
        knorm_buf = Buffer(c_tile_ty, name=f"knorm_{c}")

        workers.append(
            Worker(
                core_body,
                [
                    d_of.cons(), w_of.cons(), c_of.prod(),
                    hn_buf, qraw_buf, qnorm_buf, kraw_buf, knorm_buf,
                    matvec_kernel, rms_norm_d_kernel, rms_norm_hd_kernel, rope_kernel,
                ],
                stack_size=stack_size,
            )
        )

    # ---- runtime sequence: DRAM <-> per-column L1 taps ----
    def full_tap(ty, n):
        return TensorAccessPattern(
            tensor_dims=ty.__args__[0], offset=0, sizes=[1, 1, 1, n], strides=[0, 0, 0, 1]
        )

    def slice_tap(ty, offset, n):
        return TensorAccessPattern(
            tensor_dims=ty.__args__[0], offset=offset, sizes=[1, 1, 1, n], strides=[0, 0, 0, 1]
        )

    def sequence(
        cur, nin, wq, wk, wv, nqn, nkn, ang, q, k, v,
        d_prods, w_prods, c_conss,
    ):
        # One TaskGroup PER COLUMN (12 tasks: 5 d_of fills + 3 w_of fills + 4 c_of drains),
        # finished before the next column starts. A single TaskGroup spanning all 8 columns (96
        # tasks) tripped aiecc: "Too many simultaneously active buffer descriptors on tile (0,0),
        # which supports up to 16" -- the runtime-sequence issuer's own active-BD budget, not a
        # per-column shim limit; 12 fits, a bigger interleave (e.g. across 2 columns) might not.
        for c in range(cols):
            tg = TaskGroup()
            # cur, n_in, then this column's Wq/Wk/Wv rows -- all D-wide, all through `d_of`.
            d_prods[c].fill(cur, full_tap(cur_ty, D), group=tg)
            d_prods[c].fill(nin, full_tap(nin_ty, D), group=tg)
            d_prods[c].fill(
                wq, slice_tap(wq_ty, c * wq_rows_per_col * D, wq_rows_per_col * D), group=tg
            )
            d_prods[c].fill(
                wk, slice_tap(wk_ty, c * wk_rows_per_col * D, wk_rows_per_col * D), group=tg
            )
            d_prods[c].fill(
                wv, slice_tap(wv_ty, c * wv_rows_per_col * D, wv_rows_per_col * D), group=tg
            )
            # n_qn, n_kn, ang -- all HD-wide, all through `w_of`.
            w_prods[c].fill(nqn, full_tap(nqn_ty, HD), group=tg)
            w_prods[c].fill(nkn, full_tap(nkn_ty, HD), group=tg)
            w_prods[c].fill(ang, full_tap(ang_ty, HD), group=tg)
            # This column's outputs: q heads, then k head(s), then v head(s), from the SAME
            # output channel, draining into three different DRAM destinations.
            for h in range(heads_q_per_col):
                off = c * wq_rows_per_col + h * HD
                c_conss[c].drain(q, slice_tap(q_ty, off, HD), wait=True, group=tg)
            for h in range(heads_kv_per_col):
                off = c * wk_rows_per_col + h * HD
                c_conss[c].drain(k, slice_tap(k_ty, off, HD), wait=True, group=tg)
            for h in range(heads_kv_per_col):
                off = c * wv_rows_per_col + h * HD
                c_conss[c].drain(v, slice_tap(v_ty, off, HD), wait=True, group=tg)
            tg.finish()

    rt = Runtime(
        sequence,
        [
            cur_ty, nin_ty, wq_ty, wk_ty, wv_ty, nqn_ty, nkn_ty, ang_ty, q_ty, k_ty, v_ty,
            [of.prod() for of in d_ofs],
            [of.prod() for of in w_ofs],
            [of.cons() for of in c_ofs],
        ],
    )

    return Program(dev, rt, workers=workers).resolve_program()
