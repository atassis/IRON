#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only gate for the fused SwiGLU MLP block: does aiecc PLACE and BUILD one `aie.device`
covering residual-add + weighted RMSNorm + gate/up GEMV + SiLU + mul + down GEMV + residual-add?

Deliberately does NOT touch /dev/accel -- no xclbin load, no NPUKernel, no run_test(). Another
session owns the on-device gate; this only answers "does it place" (CPU-only aiecc).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from iron.common import AIEContext
from iron.operators.swiglu_mlp_fused.op import SwiGLUMLPFused
from iron.operators.swiglu_mlp_fused.reference import generate_golden_reference

PROGRAM_MEM_BYTES = 0x4000  # AIETargetModel.h getProgramMemorySize() for AIE2/AIE2P -- NOT 0x20000,
                             # that figure is a dead hardcode in AIETargetLdScript.cpp's ld-script
                             # emission and does not bound what a core can actually hold.


def main():
    D, FF = 1024, 3072  # Qwen3-0.6B decode shapes (d_model, ffn)

    build_dir = Path(__file__).resolve().parents[4] / "build" / "swiglu_mlp_fused"
    ctx = AIEContext(build_dir=build_dir)
    op = SwiGLUMLPFused(D=D, FF=FF, context=ctx)
    print(f"operator: {op.name}")
    print(f"build dir: {build_dir}")

    op.compile()  # raises RuntimeError on any failed compilation command (incl. aiecc placement)

    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    mlir_text = mlir_path.read_text()
    # This is the PRE-placement generator output (design.py's `aie.logical_tile`s, not yet
    # resolved to physical coordinates), so it counts devices/cores but not tile placement.
    n_devices = len(re.findall(r"\baie\.device\b", mlir_text))
    n_cores = len(re.findall(r"\baie\.core\(", mlir_text))
    print(f"aie.device count: {n_devices}")
    print(f"aie.core count:   {n_cores}")
    assert n_devices == 1, f"expected exactly one aie.device, found {n_devices}"

    # Placed compute-tile occupancy: aiecc's per-core work dirs are named elfs_main_core_<col>_<row>.
    build_subdir = mlir_path.parent / f"{mlir_path.stem}.mlir.d"
    placed_tiles = sorted(
        tuple(int(x) for x in d.name.removeprefix("elfs_main_core_").split("_"))
        for d in build_subdir.glob("elfs_main_core_*") if d.is_dir()
    )
    cols_used = sorted({c for c, r in placed_tiles})
    print(f"placed compute tiles (col,row): {placed_tiles}")
    print(f"columns used: {cols_used}")

    # .text per core against the REAL 16 KB program-memory ceiling. Identify each core's role by
    # its `call` targets in the optimized LLVM IR, not by buffer-symbol presence: the linker
    # script lists neighbour-tile buffers too (address-space sharing), so that name search matches
    # cores that never actually call the kernel.
    llvm_size = shutil.which("llvm-size")
    if llvm_size:
        print(f"program memory per core: {PROGRAM_MEM_BYTES} B (0x{PROGRAM_MEM_BYTES:x})")
        for col, row in placed_tiles:
            elf = build_subdir / f"elfs_main_core_{col}_{row}" / f"elfs_main_core_{col}_{row}.elf"
            ll = build_subdir / f"opted_main_core_{col}_{row}.ll"
            calls = sorted(set(re.findall(r"call [\w ]*@(\w+)\(", ll.read_text()))) if ll.exists() else []
            out = subprocess.run([llvm_size, "-A", str(elf)], capture_output=True, text=True)
            text_bytes = next(
                (int(line.split()[1]) for line in out.stdout.splitlines() if line.startswith(".text")),
                None,
            )
            if text_bytes is not None:
                pct = 100.0 * text_bytes / PROGRAM_MEM_BYTES
                print(f"  ({col},{row}) [{','.join(calls) or '?'}]: .text={text_bytes} B "
                      f"({pct:.1f}% of {PROGRAM_MEM_BYTES})")
    else:
        print("llvm-size not found on PATH -- skipping per-core .text measurement")

    xclbin = Path(op.xclbin_artifact.filename)
    insts = Path(op.insts_artifact.filename)
    print(f"xclbin: {xclbin} ({xclbin.stat().st_size} bytes)")
    print(f"insts:  {insts} ({insts.stat().st_size} bytes)")

    # Host-side (f32) reference -- exercises the SAME math the device computes, no device
    # involved. Confirms the Python model is self-consistent; it is NOT a device correctness
    # check (that needs the on-device gate another session owns).
    golden = generate_golden_reference(D, FF)
    print(f"host reference nxt[:8] = {golden['nxt'][:8]}")

    print(f"PASS: one aie.device, aiecc placed and built it ({n_cores} aie.core across "
          f"{len(placed_tiles)} tiles, {len(cols_used)} column(s): {cols_used}).")


if __name__ == "__main__":
    sys.exit(main())
