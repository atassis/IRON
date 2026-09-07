# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel decode SwiGLU MLP: every core runs every stage on its own 1/N slice, unlike
fuse/mlp-block's spatial 5-core PIPELINE (measured +45% slower -- one core streamed all 12.58 MB
of gate+up weights while 27 of 32 cores sat idle). Fusion stays TEMPORAL (one aie.device, one
aiex.configure); parallelism is SPATIAL, across N cores, at every stage:

    x1  = cur + a                             every core, full D (replicated -- cheap, 2 KB)
    hf  = RMSNorm_weighted(x1, n_pf)          every core, full D (replicated -- avoids a reduction)
    g   = Wg[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    u   = Wu[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    gh  = silu(g) * u                         core c's own FF/N slice
    -- ALL-GATHER gh (the one unavoidable exchange -- down's K is the full FF) --
    d   = Wd[c*D/N:(c+1)*D/N] @ gh            core c's own D/N output rows
    nxt[c*D/N:(c+1)*D/N] = x1[same] + d       core c's own D/N output rows

N compute tiles means N cores each with the SAME 2-input/2-output DMA-channel budget the
single-core fuse/mlp-block attempt tripped over. Three consolidations make N cores fit it:

  MISC (1 in): cur, a, n_pf are D-shaped, needed once each. The all-gathered gh is FF-shaped, but
  FF is a whole multiple of D (ratio R = FF/D), so it comes back as R separate D-sized reads on
  the SAME channel and is reassembled by an explicit-offset copy kernel. depth=2 lets the core
  hold cur+a simultaneously (`.acquire(2)`) for the first add; every other use is one at a time.
  Broadcast to all N cores via N `.cons()` handles on one producer -- fan-out is in the
  stream-switch fabric, not the source's own DMA (the mechanism fuse/mlp-block's P1 already uses
  to feed both P2 and P3 from one output port).

  WEIGHT (1 in): Wg, Wu (row width D) and Wd (row width FF) are three different L3 buffers, but
  the shared ObjectFifo tile is sized so ONE flat shape serves all three: TSI_GU rows of D
  elements == TSI_D rows of FF elements (TSI_D = TSI_GU // R). Reused sequentially for Wg's tiles,
  then Wu's, then Wd's -- the same "same fifo, several fill() calls" idiom fuse/mlp-block's
  gate/up A-tile stream already uses.

  OUTPUT (1 of the 2 available, one spare): gh's own FF/N slice is emitted in R D/N-sized chunks
  (an offset-mul kernel reading straight out of the g/u buffers, no separate gh-sized scratch) onto
  the SAME small ObjectFifo the final residual reuses for nxt -- R+1 sequential produce/drain
  rounds share one channel.

The gather has no native all-to-all primitive: ObjectFifoLink is many-to-one XOR one-to-many,
never both (confirmed against attn_core's identical wall in this same codebase). It round-trips
through an internal DRAM scratch buffer instead: each core drains its own gh slice in R chunks
(N*R small, disjoint, offset-addressed drains -- no join, no memtile fan-in, since at n_aie_rows=1
each core already owns a whole column's slice), then every core re-reads the FULL scratch buffer
back over the MISC channel. Cost is trivial (2*FF elements moved, ~0.03% of the layer's traffic);
correctness rests only on a TaskGroup barrier separating the drains from the refill.

n_aie_rows is fixed at 1 (N == n_aie_cols, one core per column, no within-column MemTile
split/join): that is the only topology this file builds and gates. Extending to n_aie_rows>1
(N=16/32, spreading a column's slice across its 4 physical rows) needs an additional per-column
MemTile join (gh) / split (weight) stage and is NOT implemented here -- see the operator report.
"""

from ml_dtypes import bfloat16
import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

# Shared weight-tile row counts (see module docstring, WEIGHT channel). Fixed, not searched: this
# design is gated at Qwen3-0.6B's D=1024/FF=3072 (R=3) shape only, and 6/2 is verified below to
# fit L1 at every N in {8, 16, 32} this file is built against.
TSI_GU = 6
TSI_D = 2


def _flat_tap(n, offset=0):
    return TensorAccessPattern((1, n), offset, [1, 1, 1, n], [0, 0, 0, 1])


def my_swiglu_mlp_dp(dev, D, FF, epsilon=1e-5, stack_size=0x800, func_prefix="", n_aie_cols=8):
    """`func_prefix` is required (not optional) by iron.common.sequence.FusedDispatch the moment
    this design is placed in an OperatorSequence -- see gemv/design.py's identical parameter for
    the same reason. `n_aie_cols` is N here (n_aie_rows fixed at 1, see module docstring)."""
    N = n_aie_cols
    assert FF % D == 0, f"this design assumes FF ({FF}) is a whole multiple of D ({D})"
    R = FF // D  # =3 at Qwen3-0.6B's shape; also N_GH_CHUNKS (misc) and N_GH_ROUNDS (output)
    assert D % N == 0 and FF % N == 0, f"D={D}, FF={FF} must both be divisible by N={N}"
    D_PER_CORE = D // N
    FF_PER_CORE = FF // N
    assert FF_PER_CORE % 32 == 0, (
        f"silu_tile_bf16 walks its buffer 32 lanes at a time with no tail handling; "
        f"FF/N ({FF_PER_CORE}) must be a multiple of 32"
    )
    assert TSI_GU * D == TSI_D * FF, "shared weight tile must be byte-identical for gate/up and down"
    assert FF_PER_CORE % TSI_GU == 0 and D_PER_CORE % TSI_D == 0, (
        f"N={N}: FF/N ({FF_PER_CORE}) must divide by TSI_GU ({TSI_GU}) and "
        f"D/N ({D_PER_CORE}) must divide by TSI_D ({TSI_D})"
    )
    N_GU_TILES = FF_PER_CORE // TSI_GU
    N_D_TILES = D_PER_CORE // TSI_D
    WTILE_ELEMS = TSI_GU * D

    # L1 budget check (64 KB/core) -- see module docstring's channel accounting for what each
    # buffer is. Computed, not guessed: this is exactly the "hanging numbers are bugs" rule.
    L1_BYTES = 65536
    misc_bytes = 2 * (D * 2)  # depth=2
    weight_bytes = 2 * (WTILE_ELEMS * 2)  # depth=2
    out_bytes = 2 * (D_PER_CORE * 2)  # depth=2
    persistent_bytes = 2 * (D * 2) + (FF * 2) + 2 * (FF_PER_CORE * 2) + (D_PER_CORE * 2)
    # x1_buf + hf_buf         gh_buf      g_buf + u_buf          d_buf
    total = misc_bytes + weight_bytes + out_bytes + persistent_bytes + stack_size
    assert total <= L1_BYTES, (
        f"N={N}: estimated L1 use {total} B exceeds {L1_BYTES} B "
        f"(misc={misc_bytes} weight={weight_bytes} out={out_bytes} "
        f"persistent={persistent_bytes} stack={stack_size})"
    )

    D_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    FF_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    DPC_ty = np.ndarray[(D_PER_CORE,), np.dtype[bfloat16]]
    FFPC_ty = np.ndarray[(FF_PER_CORE,), np.dtype[bfloat16]]
    WTILE_ty = np.ndarray[(WTILE_ELEMS,), np.dtype[bfloat16]]
    Wg_L3_ty = np.ndarray[(FF * D,), np.dtype[bfloat16]]
    Wd_L3_ty = np.ndarray[(D * FF,), np.dtype[bfloat16]]
    GH_SCRATCH_ty = np.ndarray[(FF,), np.dtype[bfloat16]]

    # ---- kernels (one archive per core -- every core plays every role) ----
    CORE_ARCHIVE = f"{func_prefix}swiglu_mlp_dp_core.a"
    # Two DIFFERENT bindings, not one reused: a Kernel() fixes ONE func.func signature for its
    # symbol, and the two call sites acquire differently-sized buffers (x1=cur+a is full-D; the
    # final residual is D/N-sized). The plain add costs nothing extra -- eltwise_add_bf16_vector
    # is already linked into every other design that touches add.cc.
    add_kernel = Kernel(
        f"{func_prefix}eltwise_add_bf16_vector", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.int32]
    )
    add_off_kernel = Kernel(
        f"{func_prefix}eltwise_add_offset_a_bf16_vector", CORE_ARCHIVE,
        [D_ty, DPC_ty, DPC_ty, np.int32, np.int32],
    )
    wnorm_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.int32, np.float32]
    )
    mv_gu_kernel = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, D_ty, FFPC_ty],
    )
    # Down's own matvec: different DIM_K, same extern "C" name as mv_gu_kernel -- symbol
    # uniqueness is device-wide (one aie.device, one symbol table), so op.py compiles this one
    # from a prefixed object (see fuse/mlp-block's identical mv.cc reuse for the same reason).
    mv_d_kernel = Kernel(
        f"{func_prefix}down_matvec_vectorized_bf16_bf16", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, FF_ty, DPC_ty],
    )
    silu_kernel = Kernel(f"{func_prefix}silu_tile_bf16", CORE_ARCHIVE, [np.int32, FFPC_ty])
    mul_off_kernel = Kernel(
        f"{func_prefix}eltwise_mul_offset_ab_bf16_vector", CORE_ARCHIVE,
        [FFPC_ty, FFPC_ty, DPC_ty, np.int32, np.int32],
    )
    copy_off_kernel = Kernel(
        f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE, [FF_ty, D_ty, np.int32, np.int32]
    )

    # ---- ObjectFifos: misc(1) + weight(N) + output(N), all direct L3<->L1 (n_aie_rows=1 --
    # no MemTile split/join needed; the automatic placer inserts whatever staging one column
    # needs, exactly as the existing 8-column GEMV/RMSNorm/SiLU designs already rely on). ----
    misc_of = ObjectFifo(D_ty, name="misc", depth=2)
    weight_ofs = [ObjectFifo(WTILE_ty, name=f"weight_{c}", depth=2) for c in range(N)]
    out_ofs = [ObjectFifo(DPC_ty, name=f"out_{c}", depth=2) for c in range(N)]

    def core_fn(misc_c, weight_c, out_p,
                x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
                add_k, add_off_k, wnorm_k, mv_gu_k, mv_d_k, silu_k, mul_off_k, copy_off_k,
                core_id):
        # step 1: x1 = cur + a, full D, replicated on every core.
        pair = misc_c.acquire(2)
        add_k(pair[0], pair[1], x1_buf, D)
        misc_c.release(2)

        # step 2: hf = weighted_rms_norm(x1, n_pf), full D, replicated.
        npf = misc_c.acquire(1)
        wnorm_k(x1_buf, npf, hf_buf, D, epsilon)
        misc_c.release(1)

        # step 3: g = Wg[my rows] @ hf, then u = Wu[my rows] @ hf -- same shared weight channel,
        # continued (Wg's N_GU_TILES tiles, then Wu's).
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, g_buf)
            weight_c.release(1)
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, u_buf)
            weight_c.release(1)

        # step 4: g = silu(g), in place over the whole FF/N slice.
        silu_k(FF_PER_CORE, g_buf)

        # step 5: emit gh = silu(g)*u in R chunks of D/N, straight onto the shared output fifo
        # (no separate gh-slice buffer -- the offset read comes out of g_buf/u_buf directly).
        for r in range(R):
            ot = out_p.acquire(1)
            mul_off_k(g_buf, u_buf, ot, D_PER_CORE, r * D_PER_CORE)
            out_p.release(1)

        # step 5b: reassemble the all-gathered gh from R D-sized misc reads (the Runtime's
        # sequence issues these only AFTER every core's R output drains land in gh_scratch --
        # see the TaskGroup barrier in sequence()).
        for i in range(R):
            chunk = misc_c.acquire(1)
            copy_off_k(gh_buf, chunk, D, i * D)
            misc_c.release(1)

        # step 6: d = Wd[my rows] @ gh -- same shared weight channel, continued (Wd's N_D_TILES).
        for j in range_(N_D_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_D
            wt = weight_c.acquire(1)
            mv_d_k(TSI_D, row_off, wt, gh_buf, d_buf)
            weight_c.release(1)

        # step 7: nxt[my rows] = x1[my rows] + d, onto the shared output fifo's (R+1)-th round.
        ot = out_p.acquire(1)
        add_off_k(x1_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
        out_p.release(1)

    workers = []
    for c in range(N):
        x1_buf = Buffer(D_ty, name=f"x1_{c}")
        hf_buf = Buffer(D_ty, name=f"hf_{c}")
        gh_buf = Buffer(FF_ty, name=f"gh_{c}")
        g_buf = Buffer(FFPC_ty, name=f"g_{c}")
        u_buf = Buffer(FFPC_ty, name=f"u_{c}")
        d_buf = Buffer(DPC_ty, name=f"d_{c}")
        workers.append(
            Worker(
                core_fn,
                [
                    misc_of.cons(), weight_ofs[c].cons(), out_ofs[c].prod(),
                    x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
                    add_kernel, add_off_kernel, wnorm_kernel, mv_gu_kernel, mv_d_kernel,
                    silu_kernel, mul_off_kernel, copy_off_kernel,
                    c,
                ],
                stack_size=stack_size,
            )
        )

    def sequence(cur, a, npf, Wg, Wu, Wd, gh_scratch, nxt,
                 misc_p, weight_ps, out_cs):
        tg1 = TaskGroup()
        misc_p.fill(cur, _flat_tap(D), group=tg1)
        misc_p.fill(a, _flat_tap(D), group=tg1)
        misc_p.fill(npf, _flat_tap(D), group=tg1)
        for c in range(N):
            weight_ps[c].fill(Wg, _flat_tap(FF_PER_CORE * D, c * FF_PER_CORE * D), group=tg1)
        for c in range(N):
            weight_ps[c].fill(Wu, _flat_tap(FF_PER_CORE * D, c * FF_PER_CORE * D), group=tg1)
        for c in range(N):
            weight_ps[c].fill(Wd, _flat_tap(D_PER_CORE * FF, c * D_PER_CORE * FF), group=tg1)
        tg1.finish()

        # Barrier: gh_scratch must be fully written before any core reads it back. Every core's R
        # output-fifo drains for gh land at disjoint, contiguous offsets that together cover all
        # of gh_scratch exactly once, in the natural FF order Wd's rows expect.
        tg2 = TaskGroup()
        for c in range(N):
            for r in range(R):
                out_cs[c].drain(
                    gh_scratch, _flat_tap(D_PER_CORE, c * FF_PER_CORE + r * D_PER_CORE),
                    wait=True, group=tg2,
                )
        tg2.finish()

        tg3 = TaskGroup()
        for i in range(R):
            misc_p.fill(gh_scratch, _flat_tap(D, i * D), group=tg3)
        tg3.finish()

        tg4 = TaskGroup()
        for c in range(N):
            out_cs[c].drain(nxt, _flat_tap(D_PER_CORE, c * D_PER_CORE), wait=True, group=tg4)
        tg4.finish()

    rt = Runtime(
        sequence,
        [
            D_ty, D_ty, D_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, GH_SCRATCH_ty, D_ty,
            misc_of.prod(), [of.prod() for of in weight_ofs], [of.cons() for of in out_ofs],
        ],
    )

    return Program(dev, rt, workers=workers).resolve_program()
