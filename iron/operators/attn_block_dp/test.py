#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only gate for the fused attention block: does aiecc PLACE one `aie.device` that runs
norm, QKV, qk-norm, RoPE, KV-append, scores, softmax and context on N cores' own KV head, and does
each core's program fit the 16 KB region?

PLACEMENT IS THE QUESTION. The four operators this replaces spend 49 shim input and 32 shim output
DMA channels between them, against a device-wide 16 of each; `fuse/attn-core` built that union and
aiecc rejected it with "all 8 ShimNOCTile(s) are at 9/16 input, 16/16 output channels used". This
design claims 9 input and 8 output by giving every stage one column meaning, and that claim is
either true here or it is not.

Placement is NOT correctness. This gate cannot see a runtime-sequence deadlock (qkv_head_dp's
first device run hung with ERT_CMD_STATE_TIMEOUT off one) and it cannot see a wrong answer. Both
need the device.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from iron.common import AIEContext
from iron.operators.attn_block_dp.op import AttnBlockDataParallel

PROGRAM_MEM_BYTES = 0x4000  # AIETargetModel.h getProgramMemorySize() for AIE2/AIE2P.

# Measured .text of the four designs this replaces, from the shipped build
# build/qwen3_0_6b_decode_mlpdp4_mlpo_qkvdp_shared_fused.mlir.d (llvm-size -A, pinned Peano).
SHIPPED_TEXT = {"op0 QKVHeadDataParallel": 8000, "op1 GEMV scores": 1760,
                "op2 Softmax": 1712, "op3 TMatVec ctx": 1664}


def test_window_parameter_defaults_off_and_is_not_in_the_name():
    """The switch must be invisible when unused: same operator name, so a shared build dir
    cannot let a windowed build silently satisfy a plain one. Mirrors op.py's kv_alloc rule."""
    from iron.operators.attn_block_dp.op import AttnBlockDataParallel
    common = dict(D=1024, HD=128, Hq=16, Hkv=8, max_seq=4096, num_aie_columns=8,
                  tile_size_input=4)
    plain = AttnBlockDataParallel(**common)
    off = AttnBlockDataParallel(**common, window_parameter=None)
    assert off.name == plain.name
    on = AttnBlockDataParallel(**common, window_parameter="attn_window")
    assert on.name != plain.name, "a dynamic-window build must not share a name with a plain one"


def test_dynamic_window_makes_the_core_elf_window_independent(tmp_path):
    """The whole point: with the trip count read at runtime, two windows must compile to the
    SAME core program. The second half is the negative control -- without the parameter they
    MUST differ, or this gate is vacuous."""
    import hashlib
    from pathlib import Path
    from iron.common import AIEContext
    from iron.operators.attn_block_dp.op import AttnBlockDataParallel

    def core_elf_hashes(S, window_parameter):
        op = AttnBlockDataParallel(
            D=1024, HD=128, Hq=16, Hkv=8, max_seq=S, num_aie_columns=8, tile_size_input=4,
            kv_alloc=4096, kv_block_size=128, window_parameter=window_parameter,
            context=AIEContext(build_dir=tmp_path / f"S{S}_{window_parameter}"))
        op.compile()
        d = Path(op.xclbin_artifact.mlir_input.filename)
        d = d.parent / f"{d.stem}.mlir.d"
        return sorted(hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in d.glob("elfs_main_core_*/*.elf"))

    assert core_elf_hashes(1024, "attn_window") == core_elf_hashes(4096, "attn_window"), \
        "dynamic window still bakes the trip count into the core"
    assert core_elf_hashes(1024, None) != core_elf_hashes(4096, None), \
        "negative control: without the parameter the two MUST differ"


def main():
    D, HD, Hq, Hkv, N = 1024, 128, 16, 8, 8      # Qwen3-0.6B decode shapes
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    tsi = int(argv[0]) if argv else 4
    S = int(argv[1]) if len(argv) > 1 else 2048
    hm = "--head-major" in sys.argv

    build_dir = Path(__file__).resolve().parents[4] / "build" / f"attn_block_dp_tsi{tsi}_S{S}{'_hm' if hm else ''}"
    op = AttnBlockDataParallel(D=D, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S, num_aie_columns=N,
                               tile_size_input=tsi, wqkv_head_major=hm,
                               context=AIEContext(build_dir=build_dir))
    print(f"operator: {op.name}")
    print(f"wqkv layout: {'head-major (1 fill/core)' if hm else 'stock [Wq|Wk|Wv] (3 fills/core)'}")
    op.compile()   # raises on any failed command, aiecc placement included

    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    text = mlir_path.read_text()
    n_devices = len(re.findall(r"\baie\.device\b", text))
    n_cores = len(re.findall(r"\baie\.core\(", text))
    assert n_devices == 1, f"expected exactly one aie.device, found {n_devices}"
    assert n_cores == N, f"expected {N} aie.core ops (one per column), found {n_cores}"

    build_subdir = mlir_path.parent / f"{mlir_path.stem}.mlir.d"
    tiles = sorted(tuple(int(x) for x in d.name.removeprefix("elfs_main_core_").split("_"))
                   for d in build_subdir.glob("elfs_main_core_*") if d.is_dir())
    print(f"placed compute tiles (col,row): {tiles}")

    max_text = 0
    if shutil.which("llvm-size"):
        for col, row in tiles:
            elf = build_subdir / f"elfs_main_core_{col}_{row}" / f"elfs_main_core_{col}_{row}.elf"
            out = subprocess.run(["llvm-size", "-A", str(elf)], capture_output=True, text=True)
            tb = next((int(ln.split()[1]) for ln in out.stdout.splitlines()
                       if ln.startswith(".text")), None)
            if tb is not None:
                max_text = max(max_text, tb)
                print(f"  ({col},{row}): .text={tb} B ({100.0*tb/PROGRAM_MEM_BYTES:.1f}%)")
        # The sum is an UPPER bound on what fusion should cost -- shared runtime/library code is
        # counted four times in it -- so print the comparison rather than asserting on it.
        s = sum(SHIPPED_TEXT.values())
        print(f"  shipped four designs sum to {s} B "
              f"({', '.join(f'{k}={v}' for k, v in SHIPPED_TEXT.items())}); "
              f"fused max {max_text} B = {100.0*max_text/s:.0f}% of that sum")
    else:
        print("llvm-size not on PATH -- skipping the per-core .text measurement")

    print(f"PLACES: one aie.device, {n_cores} cores over {len(tiles)} tiles, "
          f"tile_size_input={tsi}, max_seq={S}, max .text {max_text} B "
          f"({100.0*max_text/PROGRAM_MEM_BYTES:.1f}% of {PROGRAM_MEM_BYTES} B).")
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
