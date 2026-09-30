# SPDX-License-Identifier: Apache-2.0
"""Fused prefill attention, one pipeline at 32 rows: QK, softmax and two x V halves in one column.

Each K byte feeds 32 rows, so QK's work per key meets its input channel's rate. x V splits the
head: each worker keeps a [32 x head_dim/2] f32 accumulator and gets its own V half, read strided
from DDR. The softmax sits between the two and hands each its own P and cl (K031). QK is not
adjacent to the softmax, so S^T goes by DMA, single-buffered: QK's L1 has no room for a second.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

from iron.operators.fused_attn.feed_design import half_tap, two_hop

ROWS, KEYS = 32, 64


def fused_attn_x2(dev, n_blocks, head_dim, slice_w, mask):
    assert n_blocks >= 2, "the x V workers peel the last block"
    n_sl, hd2 = head_dim // slice_w, head_dim // 2
    n_half = n_sl // 2
    keys = n_blocks * KEYS
    bf, f32, i32 = np.dtype[bfloat16], np.dtype[np.float32], np.dtype[np.int32]
    q_sl = np.ndarray[(ROWS * slice_w,), bf]
    kv_sl = np.ndarray[(KEYS * slice_w,), bf]
    qT_ty = np.ndarray[(head_dim * ROWS,), bf]
    sT_ty = np.ndarray[(KEYS * ROWS,), f32]
    p_ty = np.ndarray[(ROWS * KEYS,), bf]
    cl_ty = np.ndarray[(2 * ROWS,), f32]
    st_ty = np.ndarray[(2 * ROWS + 4,), f32]
    w_ty = np.ndarray[(ROWS,), i32]
    o_ty = np.ndarray[(ROWS * hd2,), f32]
    inv_ty = np.ndarray[(ROWS,), f32]
    ch_ty = np.ndarray[(ROWS * slice_w,), bf]

    qk_o, pv_o = "fa_qk32.o", "fa_pv32.o"
    qT_k = Kernel("fa_qT_slice", qk_o, [q_sl, qT_ty, np.int32])
    zs_k = Kernel("fa_zero_sT", qk_o, [sT_ty])
    qk_k = Kernel("fa_qkT_slice", qk_o, [kv_sl, qT_ty, sT_ty, np.int32])
    init_k = Kernel("fa_sm_round_init", qk_o, [st_ty, w_ty])
    sm_k = Kernel("fa_smT_block_x2", qk_o, [sT_ty, p_ty, p_ty, cl_ty, cl_ty, st_ty, w_ty, np.int32])
    zo_k = Kernel("fa_zero_o", pv_o, [o_ty, np.int32])
    pv_k = Kernel("fa_pv_slice", pv_o, [p_ty, kv_sl, o_ty, np.int32])
    resc_k = Kernel("fa_o_rescale", pv_o, [o_ty, cl_ty, np.int32])
    inv_k = Kernel("fa_inv_l", pv_o, [cl_ty, inv_ty])
    fin_k = Kernel("fa_o_finish_slice", pv_o, [o_ty, inv_ty, ch_ty, np.int32])

    in_q, q_t = two_hop("q", ROWS, head_dim, slice_w, 1)
    in_k, k_t = two_hop("k", KEYS, head_dim, slice_w, 2)
    in_v0, v0_t = two_hop("v0", KEYS, hd2, slice_w, 2)
    in_v1, v1_t = two_hop("v1", KEYS, hd2, slice_w, 2)
    wid = ObjectFifo(w_ty, name="widths", depth=1)
    s_f = ObjectFifo(sT_ty, name="sT", depth=1)
    p_f = [ObjectFifo(p_ty, name=f"p{i}", depth=2) for i in range(2)]
    cl_f = [ObjectFifo(cl_ty, name=f"cl{i}", depth=2) for i in range(2)]
    o_f = [ObjectFifo(ch_ty, name=f"o{i}_out", depth=2) for i in range(2)]

    def qk_body(q_in, k_in, s_out, qT, qTk, zsk, qkk):
        for sl in range(n_sl):
            q = q_in.acquire(1)
            qTk(q, qT, sl)
            q_in.release(1)
        for _ in range_(n_blocks):
            s = s_out.acquire(1)
            zsk(s)
            for sl in range(n_sl):
                k = k_in.acquire(1)
                qkk(k, qT, s, sl)
                k_in.release(1)
            s_out.release(1)

    def sm_body(s_in, w_in, p0_out, p1_out, c0_out, c1_out, st, initk, smk):
        w = w_in.acquire(1)
        initk(st, w)
        for _ in range_(n_blocks):
            s = s_in.acquire(1)
            p0, p1 = p0_out.acquire(1), p1_out.acquire(1)
            c0, c1 = c0_out.acquire(1), c1_out.acquire(1)
            smk(s, p0, p1, c0, c1, st, w, mask)
            s_in.release(1)
            p0_out.release(1)
            p1_out.release(1)
            c0_out.release(1)
            c1_out.release(1)
        w_in.release(1)

    def pv_block(p_in, cl_in, v_in, o, pvk, resck):
        p, c = p_in.acquire(1), cl_in.acquire(1)
        resck(o, c, n_half)
        for sl in range(n_half):
            v = v_in.acquire(1)
            pvk(p, v, o, sl)
            v_in.release(1)
        return c

    def pv_body(p_in, cl_in, v_in, o_out, o, inv, zok, pvk, resck, invk, fink):
        zok(o, ROWS * hd2)
        for _ in range_(n_blocks - 1):
            pv_block(p_in, cl_in, v_in, o, pvk, resck)
            p_in.release(1)
            cl_in.release(1)
        c = pv_block(p_in, cl_in, v_in, o, pvk, resck)
        invk(c, inv)
        p_in.release(1)
        cl_in.release(1)
        for sl in range(n_half):
            ch = o_out.acquire(1)
            fink(o, inv, ch, sl)
            o_out.release(1)

    def pv_worker(i, v_t, row):
        return Worker(pv_body, fn_args=[p_f[i].cons(), cl_f[i].cons(), v_t.cons(), o_f[i].prod(),
                                        Buffer(o_ty, name=f"o_acc{i}"),
                                        Buffer(inv_ty, name=f"inv_l{i}"),
                                        zo_k, pv_k, resc_k, inv_k, fin_k],
                      tile=Tile(col=0, row=row), stack_size=0x1000)

    workers = [
        pv_worker(0, v0_t, 2),
        Worker(sm_body, fn_args=[s_f.cons(), wid.cons(), p_f[0].prod(), p_f[1].prod(),
                                 cl_f[0].prod(), cl_f[1].prod(), Buffer(st_ty, name="sm_state"),
                                 init_k, sm_k],
               tile=Tile(col=0, row=3), stack_size=0x1000),
        pv_worker(1, v1_t, 4),
        Worker(qk_body, fn_args=[q_t.cons(), k_t.cons(), s_f.prod(), Buffer(qT_ty, name="qT"),
                                 qT_k, zs_k, qk_k],
               tile=Tile(col=0, row=5), stack_size=0x800),
    ]

    q_ty = np.ndarray[(ROWS * head_dim,), bf]
    kv_ty = np.ndarray[(keys * head_dim,), bf]

    def o_tap(half):
        return TensorAccessPattern((ROWS, head_dim), half * hd2, [1, n_half, ROWS, slice_w],
                                   [0, slice_w, head_dim, 1])

    def sequence(Q, Wd, K, V, O, q_h, w_h, k_h, v0_h, v1_h, o0_h, o1_h):
        tg = TaskGroup()
        q_h.fill(Q, group=tg)
        w_h.fill(Wd, group=tg)
        k_h.fill(K, group=tg)
        v0_h.fill(V, tap=half_tap(keys, head_dim, 0), group=tg)
        v1_h.fill(V, tap=half_tap(keys, head_dim, 1), group=tg)
        o0_h.drain(O, tap=o_tap(0), wait=True, group=tg)
        o1_h.drain(O, tap=o_tap(1), wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [q_ty, w_ty, kv_ty, kv_ty, q_ty, in_q.prod(), wid.prod(), in_k.prod(),
                            in_v0.prod(), in_v1.prod(), o_f[0].cons(), o_f[1].cons()])
    return Program(dev, rt, workers=workers).resolve_program()
