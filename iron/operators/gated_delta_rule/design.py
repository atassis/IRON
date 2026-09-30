# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import aie.dialects.index as index
from aie.dialects.aie import T
from ml_dtypes import bfloat16
import numpy as np
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker


def _flat_tap(total, size, offset=0):
    return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def gated_delta_rule_design(
    dev,
    v_heads,
    k_heads,
    dk,
    dv,
    dvc,
    num_columns,
    ab_len,
    ab_off,
    mixed_len,
    q_off,
    k_off,
    v_off,
    kernel_obj,
    func_prefix="",
    stack_size=None,
):
    """One decode step of the gated delta rule over every value head.

    Each column owns v_heads/num_columns consecutive heads. Its `misc` stream carries, in order:
    the raw [a | b] gate logits, the [neg_a | dt_bias] params (f32 bytes), then q and k for the
    k-heads its value heads read (k-head j serves value heads rep*j .. rep*j+rep-1, so each is sent
    once and reused), then v, one dk-wide element each, copied into core-local buffers as they
    arrive so the stream stays two deep (a compute tile has 16 BDs and each fifo slot takes one).
    The state streams through in [dk, dvc] chunks with one 4-D pattern per direction, in place in
    L3, so a column needs 7 shim BDs.
    """
    hpc = v_heads // num_columns
    rep = v_heads // k_heads
    n_chunks = dv // dvc
    assert dv == dk, "misc elements are dk-wide and carry v heads too"
    assert 2 * v_heads * 4 <= dk * 2, "the f32 params must fit one misc element"
    assert 2 * v_heads <= dk and ab_off + dk <= ab_len, "the [a|b] element must be in bounds"
    assert hpc % rep == 0, "a column's heads must cover whole k-heads"

    el_ty = np.ndarray[(dk,), np.dtype[bfloat16]]
    s_ty = np.ndarray[(dk * dvc,), np.dtype[np.float32]]
    gates_ty = np.ndarray[(2 * v_heads,), np.dtype[np.float32]]

    kpc = hpc // rep
    n_loc = 2 * kpc + hpc
    loc_ty = np.ndarray[(n_loc * dk,), np.dtype[bfloat16]]
    misc_ofs = [ObjectFifo(el_ty, name=f"misc_{c}", depth=2) for c in range(num_columns)]
    sin_ofs = [ObjectFifo(s_ty, name=f"sin_{c}", depth=2) for c in range(num_columns)]
    sout_ofs = [ObjectFifo(s_ty, name=f"sout_{c}", depth=2) for c in range(num_columns)]
    o_ofs = [ObjectFifo(el_ty, name=f"o_{c}", depth=2) for c in range(num_columns)]

    obj = f"{func_prefix}{kernel_obj}"
    gates_k = Kernel(f"{func_prefix}gdn_gates_all", obj, [el_ty, el_ty, gates_ty])
    copy_k = Kernel(f"{func_prefix}gdn_copy", obj, [el_ty, loc_ty, np.int32])
    unit_k = Kernel(f"{func_prefix}gdn_unit", obj,
                    [loc_ty, gates_ty, np.int32, np.int32, np.int32, np.int32, np.int32,
                     s_ty, s_ty, el_ty])

    def core_fn(misc, sin, sout, o_p, gates_buf, loc_buf, gates_kern, cp, unit_kern, head0):
        pair = misc.acquire(2)
        gates_kern(pair[0], pair[1], gates_buf)
        misc.release(2)
        for i in range(n_loc):  # [q | k | v] in stream order
            el = misc.acquire(1)
            cp(el, loc_buf, i * dk)
            misc.release(1)
        for m in range(hpc):
            o_el = o_p.acquire(1)
            for c in range_(n_chunks):
                s_i = sin.acquire(1)
                s_o = sout.acquire(1)
                unit_kern(loc_buf, gates_buf, head0 + m, (m // rep) * dk, (kpc + m // rep) * dk,
                          (2 * kpc + m) * dk, index.casts(T.i32(), c), s_i, s_o, o_el)
                sin.release(1)
                sout.release(1)
            o_p.release(1)

    workers = [
        Worker(core_fn,
               [misc_ofs[c].cons(), sin_ofs[c].cons(), sout_ofs[c].prod(), o_ofs[c].prod(),
                Buffer(gates_ty, name=f"gates_{c}"), Buffer(loc_ty, name=f"loc_{c}"),
                gates_k, copy_k, unit_k, c * hpc],
               stack_size=stack_size)
        for c in range(num_columns)
    ]

    s_total = v_heads * dk * dv

    def s_tap(c):
        return TensorAccessPattern((1, s_total), c * hpc * dk * dv, [hpc, n_chunks, dk, dvc],
                                   [dk * dv, dvc, dv, 1])

    def qk_tap(base, c):
        return _flat_tap(mixed_len, kpc * dk, base + c * kpc * dk)

    def sequence(ab, params, mixed, s_in, s_out, o, misc_ps, sin_ps, sout_cs, o_cs):
        tg = TaskGroup()
        for c in range(num_columns):
            misc_ps[c].fill(ab, _flat_tap(ab_len, dk, ab_off), wait=True, group=tg)
            misc_ps[c].fill(params, _flat_tap(dk, dk), wait=True, group=tg)
            misc_ps[c].fill(mixed, qk_tap(q_off, c), wait=True, group=tg)
            misc_ps[c].fill(mixed, qk_tap(k_off, c), wait=True, group=tg)
            misc_ps[c].fill(mixed, _flat_tap(mixed_len, hpc * dv, v_off + c * hpc * dv),
                            wait=True, group=tg)
            sin_ps[c].fill(s_in, s_tap(c), wait=True, group=tg)
            sout_cs[c].drain(s_out, s_tap(c), wait=True, group=tg)
            o_cs[c].drain(o, _flat_tap(v_heads * dv, hpc * dv, c * hpc * dv), wait=True, group=tg)
        tg.finish()

    ab_ty = np.ndarray[(ab_len,), np.dtype[bfloat16]]
    params_ty = np.ndarray[(dk,), np.dtype[bfloat16]]
    mixed_ty = np.ndarray[(mixed_len,), np.dtype[bfloat16]]
    s_l3_ty = np.ndarray[(s_total,), np.dtype[np.float32]]
    o_ty = np.ndarray[(v_heads * dv,), np.dtype[bfloat16]]
    rt = Runtime(
        sequence,
        [ab_ty, params_ty, mixed_ty, s_l3_ty, s_l3_ty, o_ty,
         [f.prod() for f in misc_ofs], [f.prod() for f in sin_ofs],
         [f.cons() for f in sout_ofs], [f.cons() for f in o_ofs]],
    )
    return Program(dev, rt, workers=workers).resolve_program()


def gated_delta_rule_seq_design(
    dev,
    v_heads,
    k_heads,
    dk,
    dv,
    dvc,
    num_columns,
    tokens,
    ab_len,
    ab_off,
    mixed_len,
    q_off,
    k_off,
    v_off,
    kernel_obj,
    func_prefix="",
    stack_size=None,
    l2_qk=False,
    counted=False,
):
    """`tokens` consecutive steps of the gated delta rule, each [dk, dvc] state chunk held in L1
    across all of them, so the state crosses L3 once per call instead of once per token.

    ab and mixed are [tokens, ab_len] and [tokens, mixed_len] row-major, o is [tokens, v_heads*dv].
    A column's `misc` stream carries the params, one [a | b] row per token, then per k-head group
    the group's q, k and v rows for every token, copied into core-local buffers as they arrive.
    With `l2_qk` the q and k rows arrive raw and are L2-normalised in L1 (see gdn_l2qk). With
    `counted` a `count` argument (int32 in its first four bytes) limits the steps to its value.
    """
    hpc = v_heads // num_columns
    rep = v_heads // k_heads
    kpc = hpc // rep
    n_chunks = dv // dvc
    assert dv == dk, "misc elements are dk-wide and carry v heads too"
    assert 2 * v_heads * 4 <= dk * 2, "the f32 params must fit one misc element"
    assert 2 * v_heads <= dk and ab_off + dk <= ab_len, "the [a|b] element must be in bounds"
    assert hpc % rep == 0, "a column's heads must cover whole k-heads"

    el_ty = np.ndarray[(dk,), np.dtype[bfloat16]]
    s_ty = np.ndarray[(dk * dvc,), np.dtype[np.float32]]
    gates_ty = np.ndarray[(tokens * 2 * v_heads,), np.dtype[np.float32]]
    loc_ty = np.ndarray[((2 + rep) * tokens * dk,), np.dtype[bfloat16]]
    o_el_ty = np.ndarray[(tokens * dv,), np.dtype[bfloat16]]
    l1 = (2 * dk * 2 + dk * 2 + tokens * 2 * v_heads * 4 + (2 + rep) * tokens * dk * 2
          + 2 * dk * dvc * 4 + 2 * tokens * dv * 2 + (stack_size or 0))
    assert l1 <= 60 * 1024, f"{tokens} tokens need {l1} B of L1"

    misc_ofs = [ObjectFifo(el_ty, name=f"misc_{c}", depth=2) for c in range(num_columns)]
    sin_ofs = [ObjectFifo(s_ty, name=f"sin_{c}", depth=1) for c in range(num_columns)]
    sout_ofs = [ObjectFifo(s_ty, name=f"sout_{c}", depth=1) for c in range(num_columns)]
    o_ofs = [ObjectFifo(o_el_ty, name=f"o_{c}", depth=2) for c in range(num_columns)]

    obj = f"{func_prefix}{kernel_obj}"
    copy_k = Kernel(f"{func_prefix}gdn_copy", obj, [el_ty, el_ty, np.int32])
    copy_at_k = Kernel(f"{func_prefix}gdn_copy_at", obj, [el_ty, loc_ty, np.int32, np.int32])
    gates_k = Kernel(f"{func_prefix}gdn_gates_t", obj, [el_ty, el_ty, gates_ty, np.int32])
    l2_k = Kernel(f"{func_prefix}gdn_l2qk", obj, [loc_ty])
    seq_k = (Kernel(f"{func_prefix}gdn_seq_n", obj,
                    [loc_ty, gates_ty, np.int32, np.int32, np.int32, np.int32, el_ty, s_ty, s_ty,
                     o_el_ty])
             if counted else
             Kernel(f"{func_prefix}gdn_seq", obj,
                    [loc_ty, gates_ty, np.int32, np.int32, np.int32, np.int32, s_ty, s_ty,
                     o_el_ty]))

    def core_fn(misc, sin, sout, o_p, params_buf, cnt_buf, gates_buf, loc_buf, cp, cp_at,
                gates_kern, l2_kern, seq_kern, head0):
        el = misc.acquire(1)
        cp(el, params_buf, 0)
        misc.release(1)
        if counted:
            el = misc.acquire(1)
            cp(el, cnt_buf, 0)
            misc.release(1)
        for t in range_(tokens):
            el = misc.acquire(1)
            gates_kern(el, params_buf, gates_buf, index.casts(T.i32(), t))
            misc.release(1)
        for g in range_(kpc):
            for r in range_(2 + rep):  # q, k, then this group's v heads, `tokens` rows each
                for t in range_(tokens):
                    el = misc.acquire(1)
                    cp_at(el, loc_buf, index.casts(T.i32(), r), index.casts(T.i32(), t))
                    misc.release(1)
            if l2_qk:
                l2_kern(loc_buf)
            for j in range_(rep):
                o_el = o_p.acquire(1)
                for c in range_(n_chunks):
                    s_i = sin.acquire(1)
                    s_o = sout.acquire(1)
                    ids = (head0, index.casts(T.i32(), g), index.casts(T.i32(), j),
                           index.casts(T.i32(), c))
                    if counted:
                        seq_kern(loc_buf, gates_buf, *ids, cnt_buf, s_i, s_o, o_el)
                    else:
                        seq_kern(loc_buf, gates_buf, *ids, s_i, s_o, o_el)
                    sin.release(1)
                    sout.release(1)
                o_p.release(1)

    workers = [
        Worker(core_fn,
               [misc_ofs[c].cons(), sin_ofs[c].cons(), sout_ofs[c].prod(), o_ofs[c].prod(),
                Buffer(el_ty, name=f"params_{c}"), Buffer(el_ty, name=f"cnt_{c}"),
                Buffer(gates_ty, name=f"gates_{c}"),
                Buffer(loc_ty, name=f"loc_{c}"), copy_k, copy_at_k, gates_k, l2_k, seq_k,
                c * hpc],
               stack_size=stack_size)
        for c in range(num_columns)
    ]

    s_total = v_heads * dk * dv
    o_len = tokens * v_heads * dv

    def s_tap(c):
        return TensorAccessPattern((1, s_total), c * hpc * dk * dv, [hpc, n_chunks, dk, dvc],
                                   [dk * dv, dvc, dv, 1])

    def rows_tap(total, stride, offset, n=1, n_stride=0):
        return TensorAccessPattern((1, total), offset, [1, n, tokens, dk], [0, n_stride, stride, 1])

    def sequence(ab, params, *rest):
        count = rest[0] if counted else None
        mixed, s_in, s_out, o, misc_ps, sin_ps, sout_cs, o_cs = rest[1:] if counted else rest
        # The state and o transfers stay live for the whole call; each misc phase is awaited
        # before the next is issued, so a shim tile holds at most one phase's BDs.
        tg_s = TaskGroup()
        for c in range(num_columns):
            sin_ps[c].fill(s_in, s_tap(c), wait=True, group=tg_s)
            sout_cs[c].drain(s_out, s_tap(c), wait=True, group=tg_s)
            o_cs[c].drain(o, TensorAccessPattern((1, o_len), c * hpc * dv, [1, hpc, tokens, dv],
                                                 [0, dv, v_heads * dv, 1]),
                          wait=True, group=tg_s)
        tg = TaskGroup()
        for c in range(num_columns):
            misc_ps[c].fill(params, _flat_tap(dk, dk), wait=True, group=tg)
            if counted:
                misc_ps[c].fill(count, _flat_tap(dk, dk), wait=True, group=tg)
            misc_ps[c].fill(ab, rows_tap(tokens * ab_len, ab_len, ab_off), wait=True, group=tg)
        tg.finish()
        for g in range(kpc):  # two BDs a group: its q and k rows, then its v heads' rows
            tg = TaskGroup()
            for c in range(num_columns):
                kh = c * kpc + g
                misc_ps[c].fill(mixed, rows_tap(tokens * mixed_len, mixed_len, q_off + kh * dk,
                                                2, k_off - q_off), wait=True, group=tg)
                misc_ps[c].fill(mixed, rows_tap(tokens * mixed_len, mixed_len,
                                                v_off + kh * rep * dv, rep, dv),
                                wait=True, group=tg)
            tg.finish()
        tg_s.finish()

    ab_ty = np.ndarray[(tokens * ab_len,), np.dtype[bfloat16]]
    params_ty = np.ndarray[(dk,), np.dtype[bfloat16]]
    mixed_ty = np.ndarray[(tokens * mixed_len,), np.dtype[bfloat16]]
    s_l3_ty = np.ndarray[(s_total,), np.dtype[np.float32]]
    o_ty = np.ndarray[(o_len,), np.dtype[bfloat16]]
    rt = Runtime(
        sequence,
        [ab_ty, params_ty, *([params_ty] if counted else []), mixed_ty, s_l3_ty, s_l3_ty, o_ty,
         [f.prod() for f in misc_ofs], [f.prod() for f in sin_ofs],
         [f.cons() for f in sout_ofs], [f.cons() for f in o_ofs]],
    )
    return Program(dev, rt, workers=workers).resolve_program()
