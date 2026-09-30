# SPDX-License-Identifier: Apache-2.0
"""Single-core checks for the new kernels, inputs resident, one or two outputs drained."""
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.helpers.taplib.tap import TensorAccessPattern

BF, F32, I32 = np.dtype[bfloat16], np.dtype[np.float32], np.dtype[np.int32]


def _program(dev, core, args, outs):
    """One or two outputs, each drained whole."""
    workers = [Worker(core, fn_args=[o.prod() for o in outs] + args, stack_size=0x1000)]
    tys = [o.obj_type for o in outs]
    if len(outs) == 1:
        def sequence(A, a_h):
            tg = TaskGroup()
            a_h.drain(A, wait=True, group=tg)
            tg.finish()
    else:
        def sequence(A, B, a_h, b_h):
            tg = TaskGroup()
            a_h.drain(A, wait=True, group=tg)
            b_h.drain(B, wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, tys + [o.cons() for o in outs])
    return Program(dev, rt, workers=workers).resolve_program()


def _obj(rows, kind):
    return f"fa_{kind}{'' if rows == 16 else rows}.o"


def qt_check(dev, rows, q_slices):
    """q_slices [n, rows*64] resident; slice i of Q^T is built from q_slices[i % n] (n = 2 at 32
    rows: eight would not fit beside the 32 KB Q^T); out: the whole Q^T."""
    n = len(q_slices)
    q_sl = np.ndarray[(rows * 64,), BF]
    qT_ty = np.ndarray[(512 * rows,), BF]
    k = Kernel("fa_qT_slice", _obj(rows, "qk"), [q_sl, qT_ty, np.int32])
    bufs = [Buffer(q_sl, initial_value=q_slices[i].astype(bfloat16), name=f"q{i}") for i in range(n)]
    out = ObjectFifo(qT_ty, name="qT_out", depth=1)

    def core(o_out, *a):
        qs, kern = a[:n], a[n]
        o = o_out.acquire(1)
        for i in range(8):
            kern(qs[i % n], o, i)
        o_out.release(1)

    return _program(dev, core, bufs + [k], [out])


def smt_check(dev, rows, s_blocks, widths, mask):
    """s_blocks: two S^T tile buffers [64*rows] f32; out: the second block's P and cl."""
    s_ty, p_ty = np.ndarray[(64 * rows,), F32], np.ndarray[(rows * 64,), BF]
    cl_ty = np.ndarray[(2 * rows,), F32]
    st_ty, w_ty = np.ndarray[(2 * rows + 4,), F32], np.ndarray[(rows,), I32]
    init = Kernel("fa_sm_round_init", _obj(rows, "qk"), [st_ty, w_ty])
    blk = Kernel("fa_smT_block", _obj(rows, "qk"), [s_ty, p_ty, cl_ty, st_ty, w_ty, np.int32])
    s0 = Buffer(s_ty, initial_value=s_blocks[0], name="s0")
    s1 = Buffer(s_ty, initial_value=s_blocks[1], name="s1")
    w = Buffer(w_ty, initial_value=widths.astype(np.int32), name="widths")
    st = Buffer(st_ty, name="sm_state")
    p_out = ObjectFifo(p_ty, name="p_out", depth=1)
    cl_out = ObjectFifo(cl_ty, name="cl_out", depth=1)

    def core(p_o, c_o, a0, a1, wb, stb, init_k, blk_k):
        p, c = p_o.acquire(1), c_o.acquire(1)
        init_k(stb, wb)
        blk_k(a0, p, c, stb, wb, mask)
        blk_k(a1, p, c, stb, wb, mask)
        p_o.release(1)
        c_o.release(1)

    return _program(dev, core, [s0, s1, w, st, init, blk], [p_out, cl_out])


def finish_check(dev, rows, o_tiles, cl):
    """o_tiles: one x V worker's f32 accumulator, [n_sl][rows][64] tiled (n_sl = 4 at 32 rows,
    the worker's half of the head); cl [2*rows]; out: [rows, n_sl*64] bf16 row-major."""
    n_sl = len(o_tiles) // (rows * 64)
    hd = n_sl * 64
    o_ty, cl_ty = np.ndarray[(rows * hd,), F32], np.ndarray[(2 * rows,), F32]
    inv_ty, ch_ty = np.ndarray[(rows,), F32], np.ndarray[(rows * 64,), BF]
    resc = Kernel("fa_o_rescale", _obj(rows, "pv"), [o_ty, cl_ty, np.int32])
    inv_k = Kernel("fa_inv_l", _obj(rows, "pv"), [cl_ty, inv_ty])
    fin = Kernel("fa_o_finish_slice", _obj(rows, "pv"), [o_ty, inv_ty, ch_ty, np.int32])
    o = Buffer(o_ty, initial_value=o_tiles, name="o_acc")
    c = Buffer(cl_ty, initial_value=cl, name="cl")
    inv = Buffer(inv_ty, name="inv_l")
    out = ObjectFifo(ch_ty, name="o_out", depth=2)

    def core(o_out, ob, cb, ib, resc_k, inv_kk, fin_k):
        resc_k(ob, cb, n_sl)
        inv_kk(cb, ib)
        for sl in range(n_sl):
            ch = o_out.acquire(1)
            fin_k(ob, ib, ch, sl)
            o_out.release(1)

    workers = [Worker(core, fn_args=[out.prod(), o, c, inv, resc, inv_k, fin], stack_size=0x1000)]
    tap = TensorAccessPattern((rows, hd), 0, [1, n_sl, rows, 64], [0, 64, hd, 1])

    def sequence(O, o_h):
        tg = TaskGroup()
        o_h.drain(O, tap=tap, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(rows * hd,), BF], out.cons()])
    return Program(dev, rt, workers=workers).resolve_program()
