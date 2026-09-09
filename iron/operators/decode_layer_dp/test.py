#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only gate for the whole decoder layer as ONE `aie.device`.

THE QUESTION IS PLACEMENT, and it is the only thing this gate can answer. Both halves' per-core
budgets are already measured and neither is at risk: `.text` 11,760 B and 8,048 B against a 16 KB
region, L1 45,568 B and 50,180 B against 64 KB. What is NOT known in advance is whether aiecc can
place and route 12 workers over two compute rows at two different column counts while the two
halves' fifos together spend 14 of the device's 16 shim input channels and 12 of its 16 output.

Placement is NOT correctness, and it is not freedom from deadlock. Both need the device.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

from iron.common import AIEContext
from iron.operators.decode_layer_dp.op import DecodeLayerDataParallel

PROGRAM_MEM_BYTES = 0x4000


def main():
    D, FF, HD, Hq, Hkv = 1024, 3072, 128, 16, 8          # Qwen3-0.6B
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    build_dir = Path(__file__).resolve().parents[4] / "build" / f"decode_layer_dp_S{S}"
    op = DecodeLayerDataParallel(D=D, FF=FF, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S,
                                 attn_cols=8, mlp_cols=4,
                                 context=AIEContext(build_dir=build_dir))
    print(f"operator: {op.name}")
    op.compile()

    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    text = mlir_path.read_text()
    n_dev = len(re.findall(r"\baie\.device\b", text))
    n_cores = len(re.findall(r"\baie\.core\(", text))
    n_fifo = len(re.findall(r"aie\.objectfifo @", text))
    assert n_dev == 1, f"expected ONE aie.device (that is the whole point), found {n_dev}"
    print(f"aie.device={n_dev}  aie.core={n_cores}  objectfifos={n_fifo}")

    sub = mlir_path.parent / f"{mlir_path.stem}.mlir.d"
    tiles = sorted(tuple(int(x) for x in d.name.removeprefix("elfs_main_core_").split("_"))
                   for d in sub.glob("elfs_main_core_*") if d.is_dir())
    rows = sorted({r for _, r in tiles})
    print(f"placed compute tiles (col,row): {tiles}")
    print(f"compute ROWS used: {rows}")

    if shutil.which("llvm-size"):
        by_row = {}
        for col, row in tiles:
            elf = sub / f"elfs_main_core_{col}_{row}" / f"elfs_main_core_{col}_{row}.elf"
            out = subprocess.run(["llvm-size", "-A", str(elf)], capture_output=True, text=True)
            tb = next((int(ln.split()[1]) for ln in out.stdout.splitlines()
                       if ln.startswith(".text")), None)
            by_row.setdefault(row, []).append((col, tb))
        for row in sorted(by_row):
            ts = [t for _, t in by_row[row] if t]
            print(f"  row {row}: {len(by_row[row])} cores, .text "
                  f"{min(ts)}-{max(ts)} B ({100.0*max(ts)/PROGRAM_MEM_BYTES:.1f}% of region)")

    print(f"PLACES: ONE aie.device, {n_cores} cores over {len(rows)} compute rows, max_seq={S}")
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
