# SPDX-License-Identifier: Apache-2.0
"""Check: Q (16 rows) and K (n_blocks x 64 keys) through both memtile hops into one core, which
copies every slice it receives to an output, so the host sees exactly the delivered stream."""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.dataflow.objectfifo import ObjectFifoLink
from aie.iron.device import AnyMemTile

from iron.operators.fused_attn.layouts import slice_major_dims, slice_tile_dims


def two_hop(name, rows, head_dim, slice_w, depth):
    """DDR -> memtile (laid down slice-major) -> one 8x8-tiled slice per consumer object."""
    bf = np.dtype[bfloat16]
    obj = np.ndarray[(rows * head_dim,), bf]
    feed = ObjectFifo(obj, name=f"in_{name}", depth=depth)
    tiles = ObjectFifo(obj, name=f"{name}_tiles", depth=depth,
                       dims_to_stream=slice_tile_dims(rows, head_dim, slice_w),
                       consumer_obj_type=np.ndarray[(rows * slice_w,), bf])
    ObjectFifoLink(feed.cons(dims_from_stream=slice_major_dims(rows, head_dim, slice_w)),
                   [tiles.prod()], AnyMemTile, [], [0])
    return feed, tiles


def tiled_feed(dev, n_blocks, head_dim, slice_w):
    n_sl = head_dim // slice_w
    bf = np.dtype[bfloat16]
    q_sl = np.ndarray[(16 * slice_w,), bf]
    k_sl = np.ndarray[(64 * slice_w,), bf]
    in_q, q_t = two_hop("q", 16, head_dim, slice_w, 1)
    in_k, k_t = two_hop("k", 64, head_dim, slice_w, 2)
    out_q = ObjectFifo(q_sl, name="out_q", depth=2)
    out_k = ObjectFifo(k_sl, name="out_k", depth=2)
    copy_q = Kernel("passThroughTile", "fa_feed_passThrough.o", [q_sl, q_sl, np.int32, np.int32])
    copy_k = Kernel("passThroughLine", "fa_feed_passThrough.o", [k_sl, k_sl, np.int32])

    def core(q_in, k_in, q_out, k_out, cq, ck):
        for _ in range_(n_sl):
            a, b = q_in.acquire(1), q_out.acquire(1)
            cq(a, b, 16, slice_w)
            q_in.release(1)
            q_out.release(1)
        for _ in range_(n_blocks * n_sl):
            a, b = k_in.acquire(1), k_out.acquire(1)
            ck(a, b, 64 * slice_w)
            k_in.release(1)
            k_out.release(1)

    worker = Worker(core, fn_args=[q_t.cons(), k_t.cons(), out_q.prod(), out_k.prod(), copy_q, copy_k])
    q_ty = np.ndarray[(16 * head_dim,), bf]
    k_ty = np.ndarray[(n_blocks * 64 * head_dim,), bf]

    def sequence(Q, K, OQ, OK, q_h, k_h, oq_h, ok_h):
        tg = TaskGroup()
        q_h.fill(Q, group=tg)
        k_h.fill(K, group=tg)
        oq_h.drain(OQ, wait=True, group=tg)
        ok_h.drain(OK, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [q_ty, k_ty, q_ty, k_ty, in_q.prod(), in_k.prod(), out_q.cons(),
                            out_k.cons()])
    return Program(dev, rt, workers=[worker]).resolve_program()


def half_tap(keys, head_dim, half):
    """One head-dim half of a row-major [keys, head_dim] tensor, keys in runs of up to 512 so no
    shim dimension exceeds 1023."""
    hd2 = head_dim // 2
    run = min(keys, 512)
    return TensorAccessPattern((keys, head_dim), half * hd2, [1, keys // run, run, hd2],
                               [0, run * head_dim, head_dim, 1])


def half_feed(dev, n_blocks, head_dim, slice_w, half):
    """Check: one head-dim half of K through both memtile hops into a copying core."""
    hd2, keys = head_dim // 2, n_blocks * 64
    n_sl = hd2 // slice_w
    bf = np.dtype[bfloat16]
    k_sl = np.ndarray[(64 * slice_w,), bf]
    in_h, h_t = two_hop("h", 64, hd2, slice_w, 2)
    out = ObjectFifo(k_sl, name="out_h", depth=2)
    copy = Kernel("passThroughLine", "fa_feed_passThrough.o", [k_sl, k_sl, np.int32])

    def core(h_in, h_out, ck):
        for _ in range_(n_blocks * n_sl):
            a, b = h_in.acquire(1), h_out.acquire(1)
            ck(a, b, 64 * slice_w)
            h_in.release(1)
            h_out.release(1)

    worker = Worker(core, fn_args=[h_t.cons(), out.prod(), copy])
    tap = half_tap(keys, head_dim, half)

    def sequence(K, O, h_h, o_h):
        tg = TaskGroup()
        h_h.fill(K, tap=tap, group=tg)
        o_h.drain(O, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [np.ndarray[(keys * head_dim,), bf], np.ndarray[(keys * hd2,), bf],
                            in_h.prod(), out.cons()])
    return Program(dev, rt, workers=[worker]).resolve_program()
