# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One `aie.device` for the decode SwiGLU MLP block (GROUP C of the three-way decode fusion).

Replaces 6 consecutive per-token designs (residual-add, weighted RMSNorm, gate GEMV, up GEMV,
SiLU, elementwise-mul, down GEMV, residual-add -- op_gate/op_up already share one design, so 6
distinct designs sit in the runlist today) with ONE device, cutting 6 `aiex.configure` down to 1
for this block:

    x1  = cur + a
    hf  = RMSNorm_weighted(x1, n_pf)
    nxt = x1 + Wd @ (SiLU(Wg @ hf) * (Wu @ hf))

FIRST CUT put all of this on ONE core and it does not place: aiecc's placer rejected it with
"tile (0, 3) requires 5 input/1 output DMA channels, but only 2 input/2 output available" --
every AIE2P COMPUTE TILE has its own 2-input/2-output DMA-channel budget, independent of (and much
tighter than) the device-wide 16-channel ShimDMA budget (iron.common.utils.get_shim_dma_limit)
that gemv/rms_norm/binary_elementwise already check for `num_aie_columns`. A single core touching
6 external buffers plus cross-stage data was never going to fit 2 inputs, regardless of column
count. See the fuse/mlp-block report for the exact error and the single-core MLIR it came from.

Architecture (5 cores, none with more than 2 inputs / 1 output), `cols=1` throughout so this
stays a placement baseline, not a throughput one:

    P1 (cur, a)         -> x1                    add
    P2 (x1, n_pf)       -> hf                    weighted_rms_norm
    A  (hf, Wg-then-Wu) -> gh = silu(g)*u         matvec x2 (shared A-tile stream) + silu + mul
    B  (gh, Wd)         -> d                      matvec
    P3 (x1, d)          -> nxt                    add

x1 is produced once by P1 and read by BOTH P2 and P3 (an ObjectFifo broadcast: two `.cons()`
handles on one producer, still ONE output port at P1 -- fan-out happens in the stream-switch
fabric, not at the source tile's own DMA). Every arrow above is a plain point-to-point ObjectFifo
between two Workers; NONE of x1/hf/g/u/gh/d ever crosses a Runtime `.fill()`/`.drain()`, so none of
them reaches L3/DDR -- only cur, a, n_pf, Wg, Wu, Wd (in) and nxt (out) cross the shim, exactly the
6-in/1-out interface the fusion is required to expose.

gate and up share ONE ObjectFifo for their A-tile stream (same K=D=1024 shape), filled twice (Wg's
range, then Wu's) by two separate Runtime.fill() calls on the same producer handle -- the same
idiom GEMV itself uses for num_batches>1 on one column. Down's weight (K=FF=3072) instantiates the
SAME mv.cc extern "C" name (`matvec_vectorized_bf16_bf16`, DIM_K is baked in at compile time, not
part of the name) on a DIFFERENT physical core -- but symbol uniqueness is checked over the WHOLE
MLIR MODULE (one `aie.device` == one symbol table), not per core's linked ELF, so two `Kernel(...)`
declarations with the same name still verify-fail as "redefinition of symbol" even on different
cores. op.py symbol-prefixes down's compiled object ("down_") so its Kernel() can use a distinct
MLIR-level name.
"""

from ml_dtypes import bfloat16
import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern


def _l1_single_core_tile(M, K, l1_bytes=65536, reserve=8192, tsi_candidates=(4, 2, 1), granule=8):
    """Largest (tile_size_input, tile_size_output) for a SINGLE core (cols=1) doing the WHOLE
    M-row matvec, i.e. tile_size_output == M (one C tile, no output-loop tiling).

    Same three constraints as designs/decode_fused/gen_llm_decode.py's gemv_tile_output (tso a
    multiple of tile_size_input; tso a multiple of `granule` bf16 elements -- see that function's
    docstring for the tso%8 device bug it guards against; A+B+C double-buffered must fit L1), just
    evaluated at cols=1 instead of searching per-column tiling. Raises if nothing fits.
    """
    for tsi in tsi_candidates:
        if M % tsi:
            continue
        budget = l1_bytes - reserve - 2 * (tsi * K * 2) - 2 * (K * 2)
        if budget <= 0:
            continue
        cap = budget // 4  # C tile, double-buffered, 2 bytes/element
        if M <= cap and M % granule == 0:
            return tsi, M
    raise ValueError(f"single-core matvec M={M} K={K}: no tile_size_input fits L1 ({l1_bytes} B)")


def my_swiglu_mlp_fused(dev, D, FF, epsilon=1e-5, stack_size=0x800):
    D_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    FF_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    Wg_L3_ty = np.ndarray[(FF * D,), np.dtype[bfloat16]]
    Wd_L3_ty = np.ndarray[(D * FF,), np.dtype[bfloat16]]

    tsi_gu, tso_gu = _l1_single_core_tile(FF, D)   # gate/up: M=FF, K=D
    tsi_d, tso_d = _l1_single_core_tile(D, FF)     # down:    M=D,  K=FF
    assert tso_gu == FF and tso_d == D             # single core -> one C tile covers the whole op
    n_gu_tiles = tso_gu // tsi_gu                  # 768 at D=1024, FF=3072
    n_d_tiles = tso_d // tsi_d                     # 512 at the same shapes

    Agu_tile_ty = np.ndarray[(tsi_gu, D), np.dtype[bfloat16]]
    Wd_tile_ty = np.ndarray[(tsi_d, FF), np.dtype[bfloat16]]

    # ---- kernels ----
    add_kernel = Kernel("eltwise_add_bf16_vector", "add.o", [D_ty, D_ty, D_ty, np.int32])
    wnorm_kernel = Kernel(
        "weighted_rms_norm", "rms_norm.o", [D_ty, D_ty, D_ty, np.int32, np.float32]
    )
    # matvec + silu + mul all run on core A -> bundled into one archive (op.py).
    CORE_A_ARCHIVE = "swiglu_mlp_fused_core_a.a"
    mv_gu_kernel = Kernel(
        "matvec_vectorized_bf16_bf16", CORE_A_ARCHIVE,
        [np.int32, np.int32, Agu_tile_ty, D_ty, FF_ty],
    )
    silu_kernel = Kernel("silu_tile_bf16", CORE_A_ARCHIVE, [np.int32, FF_ty])
    mul_kernel = Kernel("eltwise_mul_bf16_vector", CORE_A_ARCHIVE, [FF_ty, FF_ty, FF_ty, np.int32])
    # Down's own matvec: different K, same extern "C" name as mv_gu_kernel. Symbol uniqueness is
    # checked over the WHOLE MLIR MODULE (one `aie.device` == one symbol table), not per core's
    # linked ELF -- two `Kernel(...)` Python objects with the same `name` but different arg types
    # verify-fail as "redefinition of symbol" even though they end up on different physical cores.
    # op.py's KernelObjectArtifact(prefix_symbols="down_") renames the compiled object's export so
    # the MLIR-level name below actually resolves.
    mv_d_kernel = Kernel(
        "down_matvec_vectorized_bf16_bf16", f"down_gemv_{FF}k_64vs.o",
        [np.int32, np.int32, Wd_tile_ty, FF_ty, D_ty],
    )

    # ---- ObjectFifos: 6 external (shim) + 4 on-core (Worker-to-Worker, never touch L3) ----
    cur_of = ObjectFifo(D_ty, name="cur_in", depth=2)
    a_of = ObjectFifo(D_ty, name="a_in", depth=2)
    npf_of = ObjectFifo(D_ty, name="npf_in", depth=2)
    agu_of = ObjectFifo(Agu_tile_ty, name="Agu_in", depth=2)   # Wg, then Wu
    wd_of = ObjectFifo(Wd_tile_ty, name="Wd_in", depth=2)
    nxt_of = ObjectFifo(D_ty, name="nxt_out", depth=2)

    x1_of = ObjectFifo(D_ty, name="x1", depth=2)     # P1 -> {P2, P3} (broadcast, 1 output port)
    hf_of = ObjectFifo(D_ty, name="hf", depth=2)     # P2 -> A
    gh_of = ObjectFifo(FF_ty, name="gh", depth=2)    # A  -> B
    d_of = ObjectFifo(D_ty, name="d", depth=2)       # B  -> P3

    # ---- core A's purely-local scratch (never crosses a tile boundary) ----
    g_buf = Buffer(FF_ty, name="g_buf")
    u_buf = Buffer(FF_ty, name="u_buf")

    def core_p1(cur_c, a_c, x1_p, add_k):
        cur = cur_c.acquire(1)
        av = a_c.acquire(1)
        x1 = x1_p.acquire(1)
        add_k(cur, av, x1, D)
        cur_c.release(1)
        a_c.release(1)
        x1_p.release(1)

    def core_p2(x1_c, npf_c, hf_p, wnorm_k):
        x1 = x1_c.acquire(1)
        npf = npf_c.acquire(1)
        hf = hf_p.acquire(1)
        wnorm_k(x1, npf, hf, D, epsilon)
        x1_c.release(1)
        npf_c.release(1)
        hf_p.release(1)

    def core_a(hf_c, agu_c, gh_p, g, u, mv_gu_k, silu_k, mul_k):
        hf = hf_c.acquire(1)
        for j in range_(n_gu_tiles):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * tsi_gu
            wg_tile = agu_c.acquire(1)
            mv_gu_k(tsi_gu, row_off, wg_tile, hf, g)
            agu_c.release(1)
        silu_k(FF, g)
        for j in range_(n_gu_tiles):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * tsi_gu
            wu_tile = agu_c.acquire(1)
            mv_gu_k(tsi_gu, row_off, wu_tile, hf, u)
            agu_c.release(1)
        hf_c.release(1)
        gh = gh_p.acquire(1)
        mul_k(g, u, gh, FF)   # gh = silu(gate) * up
        gh_p.release(1)

    def core_b(gh_c, wd_c, d_p, mv_d_k):
        gh = gh_c.acquire(1)
        d = d_p.acquire(1)
        for j in range_(n_d_tiles):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * tsi_d
            wd_tile = wd_c.acquire(1)
            mv_d_k(tsi_d, row_off, wd_tile, gh, d)
            wd_c.release(1)
        gh_c.release(1)
        d_p.release(1)

    def core_p3(x1_c, d_c, nxt_p, add_k):
        x1 = x1_c.acquire(1)
        d = d_c.acquire(1)
        nxt = nxt_p.acquire(1)
        add_k(x1, d, nxt, D)
        x1_c.release(1)
        d_c.release(1)
        nxt_p.release(1)

    workers = [
        Worker(core_p1, [cur_of.cons(), a_of.cons(), x1_of.prod(), add_kernel], stack_size=stack_size),
        Worker(core_p2, [x1_of.cons(), npf_of.cons(), hf_of.prod(), wnorm_kernel], stack_size=stack_size),
        Worker(
            core_a,
            [hf_of.cons(), agu_of.cons(), gh_of.prod(), g_buf, u_buf,
             mv_gu_kernel, silu_kernel, mul_kernel],
            stack_size=stack_size,
        ),
        Worker(core_b, [gh_of.cons(), wd_of.cons(), d_of.prod(), mv_d_kernel], stack_size=stack_size),
        Worker(core_p3, [x1_of.cons(), d_of.cons(), nxt_of.prod(), add_kernel], stack_size=stack_size),
    ]

    def flat_tap(n):
        return TensorAccessPattern((1, n), 0, [1, 1, 1, n], [0, 0, 0, 1])

    def sequence(cur, a, npf, Wg, Wu, Wd, nxt, cur_p, a_p, npf_p, agu_p, wd_p, nxt_c):
        tg = TaskGroup()
        cur_p.fill(cur, flat_tap(D), group=tg)
        a_p.fill(a, flat_tap(D), group=tg)
        npf_p.fill(npf, flat_tap(D), group=tg)
        agu_p.fill(Wg, flat_tap(FF * D), group=tg)
        agu_p.fill(Wu, flat_tap(FF * D), group=tg)
        wd_p.fill(Wd, flat_tap(D * FF), group=tg)
        nxt_c.drain(nxt, flat_tap(D), wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [
            D_ty, D_ty, D_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, D_ty,
            cur_of.prod(), a_of.prod(), npf_of.prod(), agu_of.prod(), wd_of.prod(), nxt_of.cons(),
        ],
    )

    return Program(dev, rt, workers=workers).resolve_program()
