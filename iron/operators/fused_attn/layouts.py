# SPDX-License-Identifier: Apache-2.0
"""Host model of the memtile's two hops and of mm.cc's 8x8 tile layout.

Dims are (size, stride) pairs in elements, outermost first -- the form objectFIFO dims take. The
receive side lays a row-major [rows, head_dim] object down slice-major; the send side re-emits it
one slice at a time as [row tile][col tile][8][8] tiles. Split this way each hop fits its DMA:
the one-hop form needs five dimensions.
"""
import numpy as np

T = 8


def stream_order(dims):
    """Element offsets a (size, stride) pattern visits, in order."""
    offs = np.zeros(1, dtype=np.int64)
    for size, stride in dims:
        offs = (offs[:, None] + np.arange(size, dtype=np.int64)[None, :] * stride).reshape(-1)
    return offs


def slice_major_dims(rows, head_dim, slice_w):
    """Memtile receive side: row-major in, [slice][rows][slice_w] in memory."""
    return [(rows, slice_w), (head_dim // slice_w, rows * slice_w), (slice_w, 1)]


def slice_tile_dims(rows, head_dim, slice_w):
    """Memtile send side: [slice][row tile][col tile][8][8]; slice and row tile merge into one
    dimension because a slice is exactly rows/8 row tiles further on."""
    return [(head_dim // slice_w * rows // T, T * slice_w), (slice_w // T, T), (T, slice_w), (T, 1)]


def slice_tiles(x, slice_w):
    """What a consumer should receive for row-major x [rows, head_dim], flattened."""
    rows, head_dim = x.shape
    return (x.reshape(rows // T, T, head_dim // slice_w, slice_w // T, T)
             .transpose(2, 0, 3, 1, 4).reshape(-1))


def relay(x, slice_w):
    """What the two hops deliver for row-major x, computed from the dims alone."""
    rows, head_dim = x.shape
    mem = np.empty(x.size, dtype=x.dtype)
    mem[stream_order(slice_major_dims(rows, head_dim, slice_w))] = x.reshape(-1)
    return mem[stream_order(slice_tile_dims(rows, head_dim, slice_w))]


def st_tiles(s):
    """S^T tiles from s [keys, rows]: [key tile][row tile][8 keys][8 rows]."""
    k, r = s.shape
    return s.reshape(k // T, T, r // T, T).transpose(0, 2, 1, 3).reshape(-1)


def p_tiles(p):
    """P tiles from p [keys, rows]: [row tile][key tile][8 rows][8 keys]."""
    k, r = p.shape
    return p.T.reshape(r // T, T, k // T, T).transpose(0, 2, 1, 3).reshape(-1)


def o_tiles(o):
    """O accumulator from o [rows, head_dim]: [slice][row tile][hd tile][8 rows][8 hd]."""
    r, hd = o.shape
    return o.reshape(r // T, T, hd // 64, 64 // T, T).transpose(2, 0, 3, 1, 4).reshape(-1)
