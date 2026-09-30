# SPDX-License-Identifier: Apache-2.0
"""V pre-pass for Gemma-4's attention_k_eq_v (no stored V, no v_proj): recomputes V for a whole K
block up front, one untiled row at a time, from the already-cached rotated/qk-normed K (`kc`) --
so fused_attn's existing V-tap reads a real, stored buffer, byte-identical in shape/dtype to what
it always expected, and needs zero changes. See fa_v_prepass_rows (fused_attn_v_prepass.cc) for
the math: ported from mv_taccum.cc's taccum_rows_kv_skip_v_bf16_f32 (decode's proven kernel),
minus its fused P*V accumulate. One row (no slicing) is why this sidesteps the two toolchain walls
fused_attn's own tiled architecture hit trying to do this in-kernel (task history in
fused_attn/design.py's git log, commits 7f0509e and its revert).
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile


def v_prepass(dev, n_blocks, head_dim, epsilon=1e-6):
    keys = n_blocks * 64
    bf = np.dtype[bfloat16]
    row_ty = np.ndarray[(head_dim,), bf]

    kc_f = ObjectFifo(row_ty, name="vp_kc", depth=2)
    # Gain rides one prefix ROW ahead of the angle stream on the same fifo -- the trick
    # attn_global_dp's ang_ofs uses (sibling worktree wt-kv-skip-v-iron), so this design needs no
    # separate one-shot fifo just for a [head_dim] vector.
    ang_f = ObjectFifo(row_ty, name="vp_ang", depth=2)
    v_f = ObjectFifo(row_ty, name="vp_v", depth=2)

    rows_k = Kernel("fa_v_prepass_rows", "fa_v_prepass.o",
                    [np.int32, row_ty, row_ty, row_ty, row_ty, np.float32])
    copy_gain_k = Kernel("passThroughLine", "fa_feed_passThrough.o", [row_ty, row_ty, np.int32])

    def core(kc_in, ang_in, v_out, gain_buf, copy_gain, k):
        gt = ang_in.acquire(1)
        copy_gain(gt, gain_buf, head_dim)
        ang_in.release(1)
        for _ in range_(keys):
            a, b, o = kc_in.acquire(1), ang_in.acquire(1), v_out.acquire(1)
            k(1, a, b, gain_buf, o, epsilon)
            kc_in.release(1)
            ang_in.release(1)
            v_out.release(1)

    # aiecc's measured-stack-size check (K007-adjacent hanging-number check, kernel-contract.md):
    # this core's actual requirement is 5504 B (__muldi3/__mulsf3 called from the epsilon divide
    # and the runtime-loop index arithmetic); 0x1700 gives headroom without guessing again.
    worker = Worker(core, fn_args=[kc_f.cons(), ang_f.cons(), v_f.prod(),
                                   Buffer(row_ty, name="vp_gain"), copy_gain_k, rows_k],
                    tile=Tile(col=0, row=2), stack_size=0x1700)

    kv_ty = np.ndarray[(keys * head_dim,), bf]
    gain_ty = np.ndarray[(head_dim,), bf]

    def sequence(KC, GAIN, ANG, V, kc_h, ang_h, v_h):
        tg = TaskGroup()
        kc_h.fill(KC, group=tg)
        ang_h.fill(GAIN, group=tg)
        ang_h.fill(ANG, group=tg)
        v_h.drain(V, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [kv_ty, gain_ty, kv_ty, kv_ty, kc_f.prod(), ang_f.prod(), v_f.cons()])
    return Program(dev, rt, workers=[worker]).resolve_program()
