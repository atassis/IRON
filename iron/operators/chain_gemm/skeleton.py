# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream skeleton of the resident forward's D_gemm: every stream of the design with no compute,
for a CPU-only place-and-route check. Core (r, c) is Tile(c, 2 + r), MemTile c is Tile(c, 1).

Streams per column c: a shim weight fifo split in MemTile c to the column's 4 cores; an activation
fifo MemTile c -> the column's 4 cores; `n_wb` elementwise write-backs from rows `wb_rows` into
MemTile c. Per row r: a W->E cascade over columns 0..N-1; core (r, N-1) sends to MemTiles 2r and
2r+1. The rstd broadcast goes from core (0, N-1) to core (rstd_row, c) of every other column.
`wb_last=False` drops column N-1's write-back fifos (it reuses core (3, N-1)'s chain output).
"""

import argparse

import numpy as np
from aie.iron import CascadeFlow, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import NPU2, Tile

U8 = np.uint8
N_COLS, N_ROWS = 8, 4
W_SUB = 4608           # one 128 K-row x 64-column int4 g32 sub-block (spec 1.3)
ACT = 8 * 480 * 9 // 8  # one 8-row bfp16 activation block, Kc = 480
OUT = 8 * 64 * 2       # one 8 x 64 bf16 output tile
RSTD = 64


def ty(n):
    return np.ndarray[(n,), np.dtype[U8]]


def dgemm_skeleton(n_cols=N_COLS, wb_rows=(2, 3), rstd_row=0, rstd=True, wb_last=True):
    N, R = n_cols, N_ROWS
    last = N - 1
    w_in = [ObjectFifo(ty(R * W_SUB), depth=2, name=f"w_l3_{c}") for c in range(N)]
    w_sub = [w_in[c].cons().split([i * W_SUB for i in range(R)], tile=Tile(c, 1),
                                   obj_types=[ty(W_SUB)] * R, depths=[2] * R,
                                   names=[f"w_{c}_{r}" for r in range(R)]) for c in range(N)]
    a_in = [ObjectFifo(ty(ACT), depth=2, name=f"a_l3_{c}") for c in range(N)]
    a_b = [a_in[c].cons().forward(tile=Tile(c, 1), depth=2, name=f"a_{c}") for c in range(N)]

    # MemTile c's inbound join: its chain output(s) plus the write-backs, drained at shim c.
    chain_src = {}   # memtile col -> chain-end row
    for r in range(R):
        for m in (2 * r, 2 * r + 1):
            if m < N:
                chain_src[m] = r
    joins, join_parts = [], []
    for m in range(N):
        n_in = (1 if m in chain_src else 0) + (len(wb_rows) if (wb_last or m != N - 1) else 0)
        j = ObjectFifo(ty(n_in * OUT), depth=2, name=f"o_l3_{m}")
        parts = j.prod().join([i * OUT for i in range(n_in)], tile=Tile(m, 1),
                              obj_types=[ty(OUT)] * n_in, depths=[2] * n_in,
                              names=[f"o_{m}_{i}" for i in range(n_in)])
        joins.append(j)
        join_parts.append(parts)
    # chain-end outputs
    out_of = {r: [] for r in range(R)}
    for m, r in chain_src.items():
        out_of[r].append(join_parts[m][0])
    wb_of = {m: join_parts[m][(1 if m in chain_src else 0):] for m in range(N)}

    rstd_of = ObjectFifo(ty(RSTD), depth=2, name="rstd") if rstd else None

    def body(r, c):
        outs = out_of[r] if c == last else []
        wbs = [wb_of[c][wb_rows.index(r)]] if r in wb_rows and (wb_last or c != last) else []
        rstd_prod = rstd and r == 0 and c == last
        rstd_cons = rstd and r == rstd_row and c != last

        def fn(w, a, *rest):
            rest = list(rest)
            rs = rest.pop(0) if (rstd_prod or rstd_cons) else None
            w.acquire(1)
            a.acquire(1)
            a.release(1)
            w.release(1)
            if rs is not None:
                rs.acquire(1)
                rs.release(1)
            for o in rest:
                o.acquire(1)
                o.release(1)

        args = [w_sub[c][r].cons(), a_b[c].cons()]
        if rstd_prod:
            args.append(rstd_of.prod())
        elif rstd_cons:
            args.append(rstd_of.cons())
        args += [o.prod() for o in outs] + [o.prod() for o in wbs]
        return Worker(fn, args, tile=Tile(c, 2 + r))

    grid = [[body(r, c) for r in range(R)] for c in range(N)]
    for r in range(R):
        for c in range(N - 1):
            CascadeFlow(grid[c][r], grid[c + 1][r])

    def seq(*a):
        wp, ap, oc = a[-3], a[-2], a[-1]
        tg = TaskGroup()
        for c in range(N):
            wp[c].fill(a[0], group=tg)
            ap[c].fill(a[1], group=tg)
        for m in range(N):
            oc[m].drain(a[2], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq, [ty(R * W_SUB), ty(ACT), ty(3 * OUT),
                       [w_in[c].prod(tile=Tile(c, 0)) for c in range(N)],
                       [a_in[c].prod(tile=Tile(c, 0)) for c in range(N)],
                       [joins[m].cons(tile=Tile(m, 0)) for m in range(N)]])
    return Program(NPU2(), rt, workers=[w for col in grid for w in col]).resolve_program()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cols", type=int, default=N_COLS)
    p.add_argument("--wb-rows", default="2,3")
    p.add_argument("--rstd-row", type=int, default=0)
    p.add_argument("--no-rstd", action="store_true")
    p.add_argument("--no-wb-last", action="store_true")
    a = p.parse_args()
    wb = tuple(int(x) for x in a.wb_rows.split(",") if x != "")
    print(dgemm_skeleton(a.cols, wb_rows=wb, rstd_row=a.rstd_row, rstd=not a.no_rstd,
                         wb_last=not a.no_wb_last))
