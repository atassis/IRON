# SPDX-License-Identifier: Apache-2.0
"""Fused prefill attention, one pipeline: 16 query rows over n_blocks blocks of 64 keys.

QK, softmax and x V on three adjacent cores of one column. Q, K and V reach their core through
two memtile hops (feed_design.two_hop). S^T, P and the per-block {correction, running sum} pass
through neighbour memory; each of those FIFOs has one consumer, the only case objectFIFO shares.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

from iron.operators.fused_attn.feed_design import two_hop

ROWS, KEYS = 16, 64


def fused_attn_one_pipe(dev, n_blocks, head_dim, slice_w, mask):
    assert n_blocks >= 2, "the x V worker peels the last block"
    n_sl = head_dim // slice_w
    bf, f32, i32 = np.dtype[bfloat16], np.dtype[np.float32], np.dtype[np.int32]
    q_sl = np.ndarray[(ROWS * slice_w,), bf]
    kv_sl = np.ndarray[(KEYS * slice_w,), bf]
    qT_ty = np.ndarray[(head_dim * ROWS,), bf]
    sT_ty = np.ndarray[(KEYS * ROWS,), f32]
    p_ty = np.ndarray[(ROWS * KEYS,), bf]
    cl_ty = np.ndarray[(2 * ROWS,), f32]
    st_ty = np.ndarray[(2 * ROWS + 4,), f32]
    w_ty = np.ndarray[(ROWS,), i32]
    o_ty = np.ndarray[(ROWS * head_dim,), f32]
    inv_ty = np.ndarray[(ROWS,), f32]
    ch_ty = np.ndarray[(ROWS * slice_w,), bf]

    qT_k = Kernel("fa_qT_slice", "fa_qk.o", [q_sl, qT_ty, np.int32])
    zs_k = Kernel("fa_zero_sT", "fa_qk.o", [sT_ty])
    qk_k = Kernel("fa_qkT_slice", "fa_qk.o", [kv_sl, qT_ty, sT_ty, np.int32])
    init_k = Kernel("fa_sm_round_init", "fa_qk.o", [st_ty, w_ty])
    sm_k = Kernel("fa_smT_block", "fa_qk.o", [sT_ty, p_ty, cl_ty, st_ty, w_ty, np.int32])
    zo_k = Kernel("fa_zero_o", "fa_pv.o", [o_ty, np.int32])
    pv_k = Kernel("fa_pv_slice", "fa_pv.o", [p_ty, kv_sl, o_ty, np.int32])
    resc_k = Kernel("fa_o_rescale", "fa_pv.o", [o_ty, cl_ty, np.int32])
    inv_k = Kernel("fa_inv_l", "fa_pv.o", [cl_ty, inv_ty])
    fin_k = Kernel("fa_o_finish_slice", "fa_pv.o", [o_ty, inv_ty, ch_ty, np.int32])

    in_q, q_t = two_hop("q", ROWS, head_dim, slice_w, 1)
    in_k, k_t = two_hop("k", KEYS, head_dim, slice_w, 2)
    in_v, v_t = two_hop("v", KEYS, head_dim, slice_w, 2)
    wid = ObjectFifo(w_ty, name="widths", depth=1)
    s_f = ObjectFifo(sT_ty, name="sT", depth=2)
    p_f = ObjectFifo(p_ty, name="p", depth=2)
    cl_f = ObjectFifo(cl_ty, name="cl", depth=2)
    o_f = ObjectFifo(ch_ty, name="o_out", depth=2)

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

    def sm_body(s_in, w_in, p_out, cl_out, st, initk, smk):
        w = w_in.acquire(1)
        initk(st, w)
        for _ in range_(n_blocks):
            s, p, c = s_in.acquire(1), p_out.acquire(1), cl_out.acquire(1)
            smk(s, p, c, st, w, mask)
            s_in.release(1)
            p_out.release(1)
            cl_out.release(1)
        w_in.release(1)

    def pv_block(p_in, cl_in, v_in, o, pvk, resck):
        p, c = p_in.acquire(1), cl_in.acquire(1)
        resck(o, c, n_sl)
        for sl in range(n_sl):
            v = v_in.acquire(1)
            pvk(p, v, o, sl)
            v_in.release(1)
        return c

    def pv_body(p_in, cl_in, v_in, o_out, o, inv, zok, pvk, resck, invk, fink):
        zok(o, ROWS * head_dim)
        for _ in range_(n_blocks - 1):
            pv_block(p_in, cl_in, v_in, o, pvk, resck)
            p_in.release(1)
            cl_in.release(1)
        c = pv_block(p_in, cl_in, v_in, o, pvk, resck)
        invk(c, inv)
        p_in.release(1)
        cl_in.release(1)
        for sl in range(n_sl):
            ch = o_out.acquire(1)
            fink(o, inv, ch, sl)
            o_out.release(1)

    workers = [
        Worker(qk_body, fn_args=[q_t.cons(), k_t.cons(), s_f.prod(), Buffer(qT_ty, name="qT"),
                                 qT_k, zs_k, qk_k],
               tile=Tile(col=0, row=2), stack_size=0x1000),
        Worker(sm_body, fn_args=[s_f.cons(), wid.cons(), p_f.prod(), cl_f.prod(),
                                 Buffer(st_ty, name="sm_state"), init_k, sm_k],
               tile=Tile(col=0, row=3), stack_size=0x1000),
        Worker(pv_body, fn_args=[p_f.cons(), cl_f.cons(), v_t.cons(), o_f.prod(),
                                 Buffer(o_ty, name="o_acc"), Buffer(inv_ty, name="inv_l"),
                                 zo_k, pv_k, resc_k, inv_k, fin_k],
               tile=Tile(col=0, row=4), stack_size=0x1000),
    ]

    kv_ty = np.ndarray[(n_blocks * KEYS * head_dim,), bf]
    q_ty = np.ndarray[(ROWS * head_dim,), bf]
    o_tap = TensorAccessPattern((ROWS, head_dim), 0, [1, n_sl, ROWS, slice_w],
                                [0, slice_w, head_dim, 1])

    def sequence(Q, Wd, K, V, O, q_h, w_h, k_h, v_h, o_h):
        tg = TaskGroup()
        q_h.fill(Q, group=tg)
        w_h.fill(Wd, group=tg)
        k_h.fill(K, group=tg)
        v_h.fill(V, group=tg)
        o_h.drain(O, tap=o_tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [q_ty, w_ty, kv_ty, kv_ty, q_ty, in_q.prod(), wid.prod(), in_k.prod(),
                            in_v.prod(), o_f.cons()])
    return Program(dev, rt, workers=workers).resolve_program()
