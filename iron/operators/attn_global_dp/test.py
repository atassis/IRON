#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only gate for the position-split global attention block (gemma4-weightless-attention-
block variant A_g): does aiecc PLACE AttnGlobalWorker's N cores and AttnGlobalMerge's one core,
and does each fit its 16 KB program-memory / 64 KB L1 budget?

Placement is NOT correctness -- same caveat attn_block_dp/test.py states. This gate cannot see a
wrong answer; it can only see whether the design places and how much room it has left.
"""
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_DEFAULT_BUILD_ROOT = Path.cwd() / "build" / "attn_global_dp"

from iron.common import AIEContext
from iron.operators.attn_global_dp.design import (
    L1_BYTES, column_splits, flash_tap, flash_worker_l1_bytes, worker_l1_footprint_bytes,
)
from iron.operators.attn_global_dp.op import AttnGlobalFlash, AttnGlobalMerge, AttnGlobalWorker

PROGRAM_MEM_BYTES = 0x4000

# Gemma-4-12B's own global geometry.
HD, HQ, S, N = 512, 16, 6912, 8

# attn_global_flash's round-robin block size: a distinct op parameter, not FLASH_SM_VEC_LEN
# -- it must be a MULTIPLE of that softmax vector width, and 64 is only its default.
BLOCK = 64


def test_column_splits_are_exact_and_aligned():
    """No padding, no remainder: every column's length is a multiple of FLASH_SM_VEC_LEN (64) and
    the N lengths sum to S exactly. At Gemma-4's shape (6912/8=864, not a multiple of 64) this is
    the whole reason for `column_splits` to exist rather than an even split."""
    splits = column_splits(S, N)
    assert len(splits) == N
    assert sum(length for _, length in splits) == S
    assert all(length % 64 == 0 for _, length in splits)
    # Contiguous, no gap or overlap.
    pos = 0
    for start, length in splits:
        assert start == pos
        pos += length


def test_worker_l1_fits_at_head_groups_one():
    """head_groups=1 (every head resident at once) is the shipped default -- verify it actually
    fits, at both weight_depths this task's Worklog measured."""
    used_wd1 = worker_l1_footprint_bytes(HD, HQ, 1, 0xD00)
    used_wd2 = worker_l1_footprint_bytes(HD, HQ, 2, 0xD00)
    assert used_wd1 < used_wd2 <= L1_BYTES, (used_wd1, used_wd2, L1_BYTES)


def test_worker_places_head_groups_one(tmp_path):
    op = AttnGlobalWorker(HD=HD, Hq=HQ, S=S, num_aie_columns=N, head_groups=1, weight_depth=2,
                          context=AIEContext(build_dir=tmp_path / "worker_hg1"))
    op.compile()
    text = Path(op.xclbin_artifact.mlir_input.filename).read_text()
    assert len(re.findall(r"\baie\.device\b", text)) == 1
    assert len(re.findall(r"\baie\.core\(", text)) == N


def test_merge_places_one_core(tmp_path):
    op = AttnGlobalMerge(HD=HD, Hq=HQ, num_aie_columns=N,
                         context=AIEContext(build_dir=tmp_path / "merge"))
    op.compile()
    text = Path(op.xclbin_artifact.mlir_input.filename).read_text()
    assert len(re.findall(r"\baie\.device\b", text)) == 1
    assert len(re.findall(r"\baie\.core\(", text)) == 1


def test_flash_worker_l1_fits_at_16_and_4_heads():
    """Gemma-4's Hq=16 (head_groups=1) and a 4-head fallback both fit the 64 KB L1 budget, and
    only the chain's last worker pays for the double-buffered bf16 cx tile."""
    for heads in (HQ, 4):
        last = flash_worker_l1_bytes(HD=HD, heads=heads, block=BLOCK, stack_size=0xD00)
        mid = flash_worker_l1_bytes(HD=HD, heads=heads, block=BLOCK, stack_size=0xD00, last=False)
        assert last <= L1_BYTES
        assert last - mid == 2 * HD * 2


def test_flash_tap_is_one_block_with_the_round_robin_jump_outside():
    """The static pattern is ONE block; the round-robin jump to the column's next block lives in
    the outer (d2) stride, and both step fields stay inside the shim BD's 20-bit/10-bit widths."""
    sizes, strides, offset = flash_tap(HD=HD, block=BLOCK, columns=N, col=N - 1)
    assert sizes == [1, 1, BLOCK, HD] and strides == [0, N * BLOCK * HD, HD, 1]
    assert offset == (N - 1) * BLOCK * HD
    words = [s // 2 for s in strides[1:3]]           # bf16 -> 32-bit words
    assert all(w < (1 << 20) for w in words)          # 20-bit step field
    assert all(s <= 1023 for s in (sizes[2], HD // 2))   # 10-bit wrap field (d1 rows, d0 words)


def test_attn_global_flash_constructs_at_capacity():
    mlir = AttnGlobalFlash(HD=HD, Hq=HQ, capacity=262144).get_mlir_artifact().generator()
    assert "length_parameter" in mlir
    assert mlir.count("aie.core") == 8
    assert mlir.count("aie.cascade_flow") == 7


def test_attn_global_flash_hpc4_is_four_chains_of_eight():
    """hpc4: 4 workers per column, one 8-long cascade chain per head group, one cx drain per
    chain end."""
    mlir = AttnGlobalFlash(HD=HD, Hq=HQ, capacity=262144, heads_per_core=4
                           ).get_mlir_artifact().generator()
    assert mlir.count("aie.core") == 4 * N
    assert mlir.count("aie.cascade_flow") == 4 * (N - 1)
    assert sum(f"@gfcx_{r}(" in mlir for r in range(4)) == 4


# ##########################################################################
# Step 2 (flash-multirow design, 2026-09-25): 8-row fifo elements, hpc4 only
# ##########################################################################


def test_flash_worker_l1_bytes_rpe8_matches_the_design_table():
    """docs/superpowers/specs/2026-09-25-flash-multirow-mmul-design.md section 2's table:
    20800/22848 B at rpe=1 (today), 49472/51520 B at rpe=8 -- only the K/V fifo term scales."""
    mid1 = flash_worker_l1_bytes(HD=HD, heads=4, block=BLOCK, stack_size=0xD00, last=False,
                                 rows_per_element=1)
    last1 = flash_worker_l1_bytes(HD=HD, heads=4, block=BLOCK, stack_size=0xD00, last=True,
                                  rows_per_element=1)
    mid8 = flash_worker_l1_bytes(HD=HD, heads=4, block=BLOCK, stack_size=0xD00, last=False,
                                 rows_per_element=8)
    last8 = flash_worker_l1_bytes(HD=HD, heads=4, block=BLOCK, stack_size=0xD00, last=True,
                                  rows_per_element=8)
    assert (mid1, last1) == (20800, 22848)
    assert (mid8, last8) == (49472, 51520)
    assert last8 <= L1_BYTES


def test_attn_global_flash_hpc4_rpe8_places_the_same_topology_as_rpe1():
    """The fifo/loop restructuring changes L1 and instruction count, not core/cascade topology."""
    mlir = AttnGlobalFlash(HD=HD, Hq=HQ, capacity=262144, heads_per_core=4, rows_per_element=8
                           ).get_mlir_artifact().generator()
    assert mlir.count("aie.core") == 4 * N
    assert mlir.count("aie.cascade_flow") == 4 * (N - 1)


def test_rpe8_rejects_hpc16_and_bad_divisibility():
    """K007: heads_per_core=4 is required (L1), and rows_per_element must divide block and Hq
    and be a multiple of heads_per_core -- see attn_global_flash's own asserts."""
    import pytest
    with pytest.raises(NotImplementedError):
        AttnGlobalFlash(HD=HD, Hq=HQ, capacity=262144, heads_per_core=HQ, rows_per_element=8
                        ).get_mlir_artifact().generator()
    with pytest.raises(ValueError):
        AttnGlobalFlash(HD=HD, Hq=HQ, capacity=262144, heads_per_core=4, rows_per_element=3
                        ).get_mlir_artifact().generator()


def test_rpe8_loop_arithmetic_covers_every_row_exactly_once():
    """Pure-Python walk of the Step 2 context-loop arithmetic (design.py's `core_fn`, RPE>1
    branch): for every live in 1..64 and RPE=8, every V row < live is accumulated exactly once
    and no row >= live is ever read."""
    RPE, block = 8, 64
    for live in range(1, block + 1):
        covered = []
        n_full_elems = -(-live // RPE)             # ceildivsi(live, RPE)
        for e in range(n_full_elems):
            e_row0 = e * RPE
            rows_this = min(RPE, live - e_row0)
            covered.extend(range(e_row0, e_row0 + rows_this))
        assert covered == list(range(live)), (live, covered)
        n_elems_total = block // RPE
        drain = n_elems_total - n_full_elems
        assert drain >= 0
        # Every element (full-block RPE-row chunk) is accounted for: processed + drained.
        assert n_full_elems + drain == n_elems_total


# ##########################################################################
# Task 9: device gate for attn_global_flash vs decode_flash_ref.py
# ##########################################################################

DEV_HD, DEV_HQ = 512, 16


def _blocks_per_column(n_live, block=BLOCK, columns=N):
    return -(-(-(-n_live // block)) // columns)


def _decode_flash_ref():
    """CPU oracle lives in the sibling wt-global-flash worktree, not this repo."""
    ref_dir = Path(__file__).resolve().parents[4] / "wt-global-flash" / "designs" / "decode_fused"
    if str(ref_dir) not in sys.path:
        sys.path.insert(0, str(ref_dir))
    from decode_flash_ref import decode_flash_attention
    return decode_flash_attention


def _build_flash_sequence(capacity, build_dir, heads_per_core=DEV_HQ):
    """One-op fused-ELF build. AttnGlobalFlash's own op.compile() (Task 8's build-only gate
    above) emits xclbin+insts.bin, which AieccXclbinInstsCompilationRule does NOT pass
    `--get-scratchpad-parameters` to -- only AieccFullElfCompilationRule does, and only
    OperatorSequence(dispatch="fused") requests a FullElfArtifact. Without it there is no
    ctrl-scratchpad BO to bind a ParameterScratchpad to."""
    from iron.common.sequence import OperatorSequence

    op = AttnGlobalFlash(HD=DEV_HD, Hq=DEV_HQ, capacity=capacity, heads_per_core=heads_per_core)
    seq = OperatorSequence(
        name=f"gf_flash_dev_cap{capacity}" + ("" if heads_per_core == DEV_HQ else
                                              f"_hpc{heads_per_core}"),
        runlist=[(op, "q", "kc", "vc", "cx")],
        input_args=["q", "kc", "vc"],
        output_args=["cx"],
        dispatch="fused",
        context=AIEContext(build_dir=build_dir),
    )
    seq.compile()
    return seq


def _dispatch(call, timeout_ms=5000):
    """SequenceCallable.__call__ waits on run_handle.wait() with NO timeout -- a device hang
    would hang this process forever. Drive the run handle directly with pyxrt's own timed
    wait(timeout_ms), the same primitive the bdlen-probe (xrt::run::wait(3000ms)) used."""
    import pyxrt
    call._sync_inputs()
    call.run_handle.start()
    t0 = time.perf_counter()
    ret = call.run_handle.wait(timeout_ms)
    dt = time.perf_counter() - t0
    if ret != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        raise TimeoutError(f"dispatch returned {ret} (not COMPLETED) after {dt * 1e3:.0f} ms")
    call._sync_outputs()
    return dt


def _write_params(call, n_live):
    """Scratchpad convention (confirmed against runtime_lib/test_lib/parameter_scratchpad.h and
    AIELowerScratchpadParameters.cpp): ParameterScratchpad.write() takes the RAW semantic value
    for BOTH core- and addr-kind parameters. The <<2 for core-kind (sm_mask, gf_loop) is applied
    INSIDE writeBits() (params.txt records each name's kind); addr-kind (gf_len) passes through
    unshifted. The plan's "write sm_mask = n_live << 2" is the convention for a caller that
    writes raw scratchpad bytes directly (npu_decode.rs's own write_scratchpad, which bypasses
    this library and must replicate the shift by hand) -- pre-shifting HERE, through
    ParameterScratchpad, would double the shift and corrupt the value."""
    import numpy as np
    nb = _blocks_per_column(n_live)
    p = call.params
    p.write("sm_mask", np.int32(n_live))
    p.write("gf_loop", np.int32(nb))
    p.write("gf_len", np.int32(nb - 1))
    p.sync()
    return nb


def _rand_inputs(capacity, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(DEV_HQ, DEV_HD, dtype=torch.float32, generator=g) / (DEV_HD ** 0.5))
    kc = torch.randn(capacity, DEV_HD, dtype=torch.float32, generator=g)
    vc = torch.randn(capacity, DEV_HD, dtype=torch.float32, generator=g)
    return q.to(torch.bfloat16), kc.to(torch.bfloat16), vc.to(torch.bfloat16)


def _write_qkv(call, q, kc, vc):
    call.get_buffer("q").torch_view()[:] = q.flatten()
    call.get_buffer("kc").torch_view()[:] = kc.flatten()
    call.get_buffer("vc").torch_view()[:] = vc.flatten()


def run_device_correctness(capacity=4096, n_live_values=None, timeout_ms=5000,
                           heads_per_core=DEV_HQ, build_root=None):
    """Returns one row per n_live; each row's `out` is the clean run's cx, so two head splits
    can be compared bit for bit."""
    build_root = build_root or os.environ.get("IRON_TEST_BUILD_ROOT", _DEFAULT_BUILD_ROOT)
    import numpy as np
    import torch

    decode_flash_attention = _decode_flash_ref()
    n_live_values = n_live_values or [n for n in (1, 63, 64, 447, 448, 1000, 4096) if n <= capacity]

    build_dir = Path(build_root) / f"cap{capacity}_hpc{heads_per_core}_fused"
    seq = _build_flash_sequence(capacity, build_dir, heads_per_core)
    call = seq.get_callable()

    q, kc, vc = _rand_inputs(capacity)
    q_np, kc_np, vc_np = (t.float().numpy() for t in (q, kc, vc))

    rows = []
    print(f"{'n_live':>7} {'nb':>4} {'rel_l2':>10} {'poison':>7} {'repeat':>7}  wall_ms(a,b,c)")
    for n_live in n_live_values:
        golden = decode_flash_attention(q_np, kc_np, vc_np, n_live)

        _write_qkv(call, q, kc, vc)
        nb = _write_params(call, n_live)
        t0 = _dispatch(call, timeout_ms)
        out_a = call.get_buffer("cx").torch_view().clone()
        out_np = out_a.float().numpy().reshape(DEV_HQ, DEV_HD)
        rel_l2 = float(np.linalg.norm(out_np - golden) / np.linalg.norm(golden))

        kc_p, vc_p = kc.clone(), vc.clone()
        if n_live < capacity:
            kc_p[n_live:] = float("nan")
            vc_p[n_live:] = float("nan")
        _write_qkv(call, q, kc_p, vc_p)
        _write_params(call, n_live)
        t1 = _dispatch(call, timeout_ms)
        out_b = call.get_buffer("cx").torch_view().clone()
        poison_ok = torch.equal(out_a.view(torch.int16), out_b.view(torch.int16))

        _write_qkv(call, q, kc, vc)
        _write_params(call, n_live)
        t2 = _dispatch(call, timeout_ms)
        out_c = call.get_buffer("cx").torch_view().clone()
        repeat_ok = torch.equal(out_a.view(torch.int16), out_c.view(torch.int16))

        rows.append(dict(n_live=n_live, nb=nb, rel_l2=rel_l2, poison_ok=poison_ok,
                         repeat_ok=repeat_ok, out=out_a, wall_ms=(t0 * 1e3, t1 * 1e3, t2 * 1e3)))
        print(f"{n_live:7d} {nb:4d} {rel_l2:10.4e} {str(poison_ok):>7} {str(repeat_ok):>7}  "
              f"({t0 * 1e3:.2f},{t1 * 1e3:.2f},{t2 * 1e3:.2f})")
    return rows


def run_device_slope(capacity=32768, n_live_points=(512, 4096, 16384, 32768),
                     timed_reps=30, warmup_reps=5, timeout_ms=5000, heads_per_core=(DEV_HQ,),
                     build_root=None):
    """One linear fit per head split. With several splits each rep runs every split in turn
    (order alternated per rep, n_live order alternated per rep) after one untimed dispatch that
    absorbs the hardware-context switch."""
    build_root = build_root or os.environ.get("IRON_TEST_BUILD_ROOT", _DEFAULT_BUILD_ROOT)
    import numpy as np

    q, kc, vc = _rand_inputs(capacity)
    calls = {}
    for hpc in heads_per_core:
        seq = _build_flash_sequence(capacity, Path(build_root) / f"cap{capacity}_hpc{hpc}_fused", hpc)
        calls[hpc] = seq.get_callable()
        _write_qkv(calls[hpc], q, kc, vc)
        for n_live in n_live_points:
            _write_params(calls[hpc], n_live)
            for _ in range(warmup_reps):
                _dispatch(calls[hpc], timeout_ms)

    timings = {(h, n): [] for h in heads_per_core for n in n_live_points}
    for rep in range(timed_reps):
        order = n_live_points if rep % 2 == 0 else tuple(reversed(n_live_points))
        hpcs = heads_per_core if rep % 2 == 0 else tuple(reversed(heads_per_core))
        for hpc in hpcs:
            call = calls[hpc]
            if len(heads_per_core) > 1:
                _write_params(call, order[0])
                _dispatch(call, timeout_ms)
            for n_live in order:
                _write_params(call, n_live)
                timings[(hpc, n_live)].append(_dispatch(call, timeout_ms))

    xs = np.array(n_live_points, dtype=np.float64)
    results = {}
    for hpc in heads_per_core:
        meds = np.array([np.median(timings[(hpc, n)]) * 1e3 for n in n_live_points])
        mins = np.array([np.min(timings[(hpc, n)]) * 1e3 for n in n_live_points])
        slope, base = np.polyfit(xs, meds, 1)          # ms per position, over the medians
        fit = base + slope * xs
        resid = meds - fit
        print(f"--- hpc{hpc}")
        print(f"{'n_live':>8} {'median_ms':>10} {'min_ms':>10} {'fit_ms':>10} {'resid_ms':>10}")
        for n, m, mn, f, r in zip(n_live_points, meds, mins, fit, resid):
            print(f"{n:8d} {m:10.4f} {mn:10.4f} {f:10.4f} {r:10.4f}")
        print(f"fit: ms = {base:.4f} + {slope:.6f} * n_live")
        print(f"slope = {slope * 1e3:.4f} us/position/layer (bar: <= 0.075 us/position/layer)")
        results[hpc] = dict(n_live=list(n_live_points), median_ms=meds.tolist(),
                            min_ms=mins.tolist(), base_ms=float(base),
                            slope_us_per_pos=float(slope * 1e3), residual_ms=resid.tolist())
    return results if len(heads_per_core) > 1 else results[heads_per_core[0]]


def main():
    head_groups = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    weight_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    used = worker_l1_footprint_bytes(HD, HQ // head_groups, weight_depth, 0xD00)
    print(f"predicted worker L1 (head_groups={head_groups}, weight_depth={weight_depth}): "
          f"{used} B / {L1_BYTES} B ({100.0 * used / L1_BYTES:.1f}%)")
    print(f"column splits: {column_splits(S, N)}")

    def build_and_report(op, name):
        print(f"\n=== {name}: {op.name} ===")
        op.compile()
        mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
        text = mlir_path.read_text()
        n_cores = len(re.findall(r"\baie\.core\(", text))
        build_subdir = mlir_path.parent / f"{mlir_path.stem}.mlir.d"
        tiles = sorted(tuple(int(x) for x in d.name.removeprefix("elfs_main_core_").split("_"))
                       for d in build_subdir.glob("elfs_main_core_*") if d.is_dir())
        print(f"  {n_cores} cores over {len(tiles)} tiles: {tiles}")
        if shutil.which("llvm-size"):
            for col, row in tiles:
                elf = build_subdir / f"elfs_main_core_{col}_{row}" / f"elfs_main_core_{col}_{row}.elf"
                out = subprocess.run(["llvm-size", "-A", str(elf)], capture_output=True, text=True)
                tb = next((int(ln.split()[1]) for ln in out.stdout.splitlines()
                           if ln.startswith(".text")), None)
                if tb is not None:
                    print(f"    ({col},{row}): .text={tb} B ({100.0 * tb / PROGRAM_MEM_BYTES:.1f}%)")

    worker = AttnGlobalWorker(
        HD=HD, Hq=HQ, S=S, num_aie_columns=N, head_groups=head_groups, weight_depth=weight_depth,
        context=AIEContext(build_dir=Path(__file__).resolve().parents[4] / "build" /
                           f"attn_global_worker_hg{head_groups}_wd{weight_depth}"),
    )
    build_and_report(worker, "worker")

    merge = AttnGlobalMerge(
        HD=HD, Hq=HQ, num_aie_columns=N,
        context=AIEContext(build_dir=Path(__file__).resolve().parents[4] / "build" /
                           "attn_global_merge"),
    )
    build_and_report(merge, "merge")

    print("\nPASS")


if __name__ == "__main__":
    if "--device-correctness" in sys.argv:
        _cap = 4096
        if "--cap" in sys.argv:
            _cap = int(sys.argv[sys.argv.index("--cap") + 1])
        run_device_correctness(capacity=_cap)
    elif "--device-slope" in sys.argv:
        run_device_slope()
    else:
        sys.exit(main())
