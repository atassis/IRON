# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Designs around the chain GEMM kernel (aie_kernels/aie2p/chain_mm_bfp16.cc).

- `chain_row`: one core row, C columns, a W->E cascade; column c holds K slice c, converts its
  int4 sub-blocks on arrival and multiplies one RB x 8-row block; the chain end writes f32.
- `chain_solo`: one core running the same C chain positions in turn, the accumulator kept in
  memory between them: the same K order, so the oracle for `chain_row`.
- `one_core`: one core, one input and one output object, one kernel call (numerics probes).
- `dgemm`: all 8 x 4 cores at one projection's shape, for program-memory and L1 measurement.

Core (r, c) is Tile(c, 2 + r). Byte offsets use `bfp16_bytes`, never an element size.
"""

import numpy as np
from aie.iron import (Buffer, CascadeFlow, Kernel, ObjectFifo, Program, Runtime,
                      ScratchpadParameter, TaskGroup, Worker, WorkerRuntimeBarrier,
                      sync_parameters)
from aie.iron.device import NPU2, Tile
from aie.helpers.dialects.scf import _for as range_

U8 = np.uint8
BLK = 72          # one 8x8 bfp16ebs8 subtile: 64 mantissas + 8 exponents
GROUP = 32        # int4 quant group along K


def ty(n, dt=U8):
    return np.ndarray[(n,), np.dtype[dt]]


def bfp16_bytes(elems):
    assert elems % 64 == 0, elems
    return elems // 64 * BLK


def sub_bytes(sub_k, nb):
    """One int4 g32 sub-block: bf16 scales then nibbles (kernel header)."""
    return sub_k // GROUP * nb * 2 + sub_k * nb // 2


def shape_check(kc, nb, sub_k, rb):
    assert kc % 8 == 0 and nb % 16 == 0 and kc % sub_k == 0 and sub_k % GROUP == 0, (kc, nb, sub_k)
    assert rb in (1, 2), rb


def chain_row(C, kc, nb, sub_k, rb, obj, stack_size=0x800, epilogue="f32"):
    shape_check(kc, nb, sub_k, rb)
    n_sub = kc // sub_k
    a_b, w_b, sb = bfp16_bytes(8 * rb * kc), bfp16_bytes(kc * nb), sub_bytes(sub_k, nb)
    out_b = 8 * rb * nb * 4 if epilogue == "f32" else bfp16_bytes(8 * rb * nb // 2)
    A_ty, S_ty, W_ty, O_ty = ty(a_b), ty(sb), ty(w_b), ty(out_b)
    conv = Kernel("chain_convert_w", obj, [S_ty, W_ty, np.int32])
    first = Kernel(f"chain_mm_first_r{rb}", obj, [A_ty, W_ty])
    mid = Kernel(f"chain_mm_mid_r{rb}", obj, [A_ty, W_ty])
    scr_ty = ty(8 * rb * nb, np.uint16)
    last = (Kernel(f"chain_mm_last_f32_r{rb}", obj, [A_ty, W_ty, O_ty]) if epilogue == "f32" else
            Kernel(f"chain_mm_last_geluup_r{rb}", obj, [A_ty, W_ty, O_ty, scr_ty]))
    a_of = [ObjectFifo(A_ty, depth=1, name=f"a{c}") for c in range(C)]
    w_of = [ObjectFifo(S_ty, depth=2, name=f"w{c}") for c in range(C)]
    o_of = ObjectFifo(O_ty, depth=1, name="o")

    def make(c):
        def fn(a, w, wblk, cv, mm, *o):
            for s in range(n_sub):
                e = w.acquire(1)
                cv(e, wblk, s)
                w.release(1)
            x = a.acquire(1)
            if c == C - 1:
                y = o[0].acquire(1)
                if epilogue == "f32":
                    mm(x, wblk, y)
                else:
                    mm(x, wblk, y, o[1])
                o[0].release(1)
            else:
                mm(x, wblk)
            a.release(1)
        mm = first if c == 0 else last if c == C - 1 else mid
        args = [a_of[c].cons(), w_of[c].cons(), Buffer(W_ty, name=f"wblk{c}"), conv, mm]
        if c == C - 1:
            args.append(o_of.prod())
            if epilogue != "f32":
                args.append(Buffer(scr_ty, name="gscr"))
        return Worker(fn, args, tile=Tile(c, 2), stack_size=stack_size)

    ws = [make(c) for c in range(C)]
    for c in range(C - 1):
        CascadeFlow(ws[c], ws[c + 1])

    def seq(A, W, O, ap, wp, oc):
        tg = TaskGroup()
        for c in range(C):
            ap[c].fill(A, sizes=[1, 1, 1, a_b], strides=[0, 0, 0, 1], offset=c * a_b,
                       transfer_len=a_b, group=tg)
            wp[c].fill(W, sizes=[1, 1, n_sub, sb], strides=[0, 0, sb, 1], offset=c * n_sub * sb,
                       transfer_len=n_sub * sb, group=tg)
        oc.drain(O, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [ty(C * a_b), ty(C * n_sub * sb), O_ty,
                       [a_of[c].prod(tile=Tile(c, 0)) for c in range(C)],
                       [w_of[c].prod(tile=Tile(c, 0)) for c in range(C)],
                       o_of.cons(tile=Tile(C - 1, 0))])
    return Program(NPU2(), rt, workers=ws).resolve_program()


def chain_solo(C, kc, nb, sub_k, rb, obj, stack_size=0x800):
    shape_check(kc, nb, sub_k, rb)
    n_sub = kc // sub_k
    a_b, w_b, sb = bfp16_bytes(8 * rb * kc), bfp16_bytes(kc * nb), sub_bytes(sub_k, nb)
    out_b = 8 * rb * nb * 4
    A_ty, S_ty, W_ty, O_ty = ty(a_b), ty(sb), ty(w_b), ty(out_b)
    conv = Kernel("chain_convert_w", obj, [S_ty, W_ty, np.int32])
    solo = Kernel(f"chain_mm_solo_r{rb}", obj, [A_ty, W_ty, O_ty, np.int32])
    a_of = ObjectFifo(A_ty, depth=1, name="a")
    w_of = ObjectFifo(S_ty, depth=2, name="w")
    o_of = ObjectFifo(O_ty, depth=1, name="o")

    def fn(a, w, o, wblk, cv, so):
        y = o.acquire(1)
        for c in range(C):
            for s in range(n_sub):
                e = w.acquire(1)
                cv(e, wblk, s)
                w.release(1)
            x = a.acquire(1)
            so(x, wblk, y, 1 if c == 0 else 0)
            a.release(1)
        o.release(1)

    wk = Worker(fn, [a_of.cons(), w_of.cons(), o_of.prod(), Buffer(W_ty, name="wblk"), conv, solo],
                tile=Tile(0, 2), stack_size=stack_size)

    def seq(A, W, O, ap, wp, oc):
        tg = TaskGroup()
        ap.fill(A, group=tg)
        wp.fill(W, group=tg)
        oc.drain(O, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [ty(C * a_b), ty(C * n_sub * sb), O_ty,
                       a_of.prod(tile=Tile(0, 0)), w_of.prod(tile=Tile(0, 0)),
                       o_of.cons(tile=Tile(0, 0))])
    return Program(NPU2(), rt, workers=[wk]).resolve_program()


def one_core(fn_name, in_bytes, out_bytes, obj, stack_size=0x800):
    I_ty, O_ty = ty(in_bytes), ty(out_bytes)
    k = Kernel(fn_name, obj, [I_ty, O_ty])
    i_of = ObjectFifo(I_ty, depth=1, name="i")
    o_of = ObjectFifo(O_ty, depth=1, name="o")

    def fn(i, o, kk):
        x = i.acquire(1)
        y = o.acquire(1)
        kk(x, y)
        o.release(1)
        i.release(1)

    wk = Worker(fn, [i_of.cons(), o_of.prod(), k], tile=Tile(0, 2), stack_size=stack_size)

    def seq(I, O, ip, oc):
        tg = TaskGroup()
        ip.fill(I, group=tg)
        oc.drain(O, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [I_ty, O_ty, i_of.prod(tile=Tile(0, 0)), o_of.cons(tile=Tile(0, 0))])
    return Program(NPU2(), rt, workers=[wk]).resolve_program()


def dgemm(kc, nb, sub_k, rb, n_blocks, epilogue, obj, n_cols=8, n_rows=4, stack_size=0x1000,
          tok_parameter="rf_tok", acc_rows=0, stack_last=None):
    """All cores of one projection: per column a shim weight fifo split 4 ways in the MemTile,
    one activation fifo broadcast to the column; per row a W->E chain; each chain end drains
    through MemTile 2r / 2r+1's shim (static route). The token-block count is a core RTP read
    behind a barrier re-armed after the read (K027); the shim side is sized for one block."""
    shape_check(kc, nb, sub_k, rb)
    n_sub = kc // sub_k
    a_b, w_b, sb = bfp16_bytes(8 * rb * kc), bfp16_bytes(kc * nb), sub_bytes(sub_k, nb)
    out_b = {"geluup": bfp16_bytes(8 * rb * nb // 2), "bfp16": bfp16_bytes(8 * rb * nb),
             "bf16": 8 * rb * nb * 2, "acc": 0}[epilogue]
    A_ty, S_ty, W_ty = ty(a_b), ty(sb), ty(w_b)
    conv = Kernel("chain_convert_w", obj, [S_ty, W_ty, np.int32])
    first = Kernel(f"chain_mm_first_r{rb}", obj, [A_ty, W_ty])
    mid = Kernel(f"chain_mm_mid_r{rb}", obj, [A_ty, W_ty])
    scr_ty = ty(8 * rb * nb, np.uint16)          # gate and up, bf16, one token block
    acc_ty = ty((acc_rows or 8 * rb) * nb, np.float32)   # down: the chain end's [P_cap x NB] f32
    if epilogue == "geluup":
        O_ty = ty(out_b)
        last = Kernel(f"chain_mm_last_geluup_r{rb}", obj, [A_ty, W_ty, O_ty, scr_ty])
    elif epilogue == "acc":
        O_ty = None
        last = Kernel(f"chain_mm_last_acc_r{rb}", obj, [A_ty, W_ty, acc_ty])
    else:
        O_ty = ty(out_b)
        last = Kernel(f"chain_mm_last_{epilogue}_r{rb}", obj, [A_ty, W_ty, O_ty])

    w_l3 = [ObjectFifo(ty(n_rows * sb), depth=2, name=f"wl3_{c}") for c in range(n_cols)]
    w_sub = [w_l3[c].cons().split([r * sb for r in range(n_rows)], tile=Tile(c, 1),
                                   obj_types=[S_ty] * n_rows, depths=[2] * n_rows,
                                   names=[f"w_{c}_{r}" for r in range(n_rows)])
             for c in range(n_cols)]
    a_l3 = [ObjectFifo(A_ty, depth=2, name=f"al3_{c}") for c in range(n_cols)]
    a_b_of = [a_l3[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"a_{c}") for c in range(n_cols)]
    o_l1 = [ObjectFifo(O_ty, depth=2, name=f"ol1_{r}") for r in range(n_rows)] if O_ty else []
    o_l3 = [o_l1[r].cons().forward(tile=Tile(min(2 * r, n_cols - 1), 1), depth=2, name=f"ol3_{r}")
            for r in range(len(o_l1))]
    tok = ScratchpadParameter(tok_parameter, np.int32)
    bars = [[WorkerRuntimeBarrier() for _ in range(n_rows)] for _ in range(n_cols)]

    def make(r, c):
        is_last = c == n_cols - 1

        def fn(bar, tp, w, a, wblk, cv, mm, *extra):
            bar.wait_for_value(1)
            nt = tp.read()
            bar.release_with_value(1)          # K027: re-arm after the read
            for _ in range_(n_blocks):
                for s in range(n_sub):
                    e = w.acquire(1)
                    cv(e, wblk, s)
                    w.release(1)
                for _ in range_(nt):
                    x = a.acquire(1)
                    if is_last and epilogue == "acc":
                        mm(x, wblk, extra[0])
                    elif is_last:
                        y = extra[0].acquire(1)
                        if epilogue == "geluup":
                            mm(x, wblk, y, extra[1])
                        else:
                            mm(x, wblk, y)
                        extra[0].release(1)
                    else:
                        mm(x, wblk)
                    a.release(1)

        mm = first if c == 0 else last if is_last else mid
        args = [bars[c][r], tok, w_sub[c][r].cons(), a_b_of[c].cons(),
                Buffer(W_ty, name=f"wblk_{c}_{r}"), conv, mm]
        if is_last:
            if epilogue == "acc":
                args.append(Buffer(acc_ty, name=f"dacc_{r}"))
            else:
                args.append(o_l1[r].prod())
                if epilogue == "geluup":
                    args.append(Buffer(scr_ty, name=f"gscr_{r}"))
        ss = stack_last if (is_last and stack_last) else stack_size
        return Worker(fn, args, tile=Tile(c, 2 + r), stack_size=ss)

    grid = [[make(r, c) for r in range(n_rows)] for c in range(n_cols)]
    for r in range(n_rows):
        for c in range(n_cols - 1):
            CascadeFlow(grid[c][r], grid[c + 1][r])

    def seq(A, W, O, wp, ap, oc):
        sync_parameters()
        for b in (b for col in bars for b in col):
            b.set(1)
        tg = TaskGroup()
        for c in range(n_cols):
            wp[c].fill(W, group=tg)
            ap[c].fill(A, group=tg)
        for r, h in enumerate(oc):
            h.drain(O, wait=True, group=tg)
        tg.finish()

    O_rt = O_ty if O_ty else ty(64)
    rt = Runtime(seq, [A_ty, ty(n_rows * sb), O_rt,
                       [w_l3[c].prod(tile=Tile(c, 0)) for c in range(n_cols)],
                       [a_l3[c].prod(tile=Tile(c, 0)) for c in range(n_cols)],
                       [o_l3[r].cons(tile=Tile(min(2 * r, n_cols - 1), 0)) for r in range(len(o_l3))]])
    return Program(NPU2(), rt, workers=[w for col in grid for w in col]).resolve_program()
