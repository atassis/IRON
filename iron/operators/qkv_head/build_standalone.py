#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-free placement gate for the fused QKV-head operator: aiecc build, no XRT/NPU.

Run with THIS worktree's own `iron` package on sys.path (prepended ahead of any editable
install of the shared checkout), against the qkv_head operator dimensions for Qwen3-0.6B decode
(D=1024, HD=128, Hq=16, Hkv=8, QD=2048, KVD=1024, 8 AIE columns).

Only calls `operator.compile()` -- never `get_callable()`/`run_test()`, which need XRT+the NPU.
"""
import sys
from pathlib import Path

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE_ROOT))

import aie.utils as aie_utils  # noqa: E402
from iron.common import AIEContext  # noqa: E402
from iron.operators.qkv_head.op import QKVHead  # noqa: E402


def main():
    dev = aie_utils.get_current_device()
    print(f"iron package: {sys.modules['iron'].__file__}")
    print(f"device: {dev}, cols={getattr(dev, 'cols', '?')}")

    ctx = AIEContext(build_dir=WORKTREE_ROOT / "build" / "qkv_head", mlir_verbose=True)

    op = QKVHead(
        D=1024, HD=128, Hq=16, Hkv=8, QD=2048, KVD=1024,
        num_aie_columns=8, epsilon=1e-6,
        context=ctx,
    )

    print(f"operator name: {op.name}")
    op.compile()
    print("COMPILE OK")

    mlir_path = ctx.build_dir / op.get_mlir_artifact().filename
    if mlir_path.exists():
        text = mlir_path.read_text()
        n_devices = text.count("aie.device")
        n_cores = text.count("aie.core(")
        n_tiles = text.count("aie.tile(")
        print(f"mlir file: {mlir_path} ({len(text)} bytes)")
        print(f"aie.device count: {n_devices}")
        print(f"aie.core count:   {n_cores}")
        print(f"aie.tile count:   {n_tiles}")
    else:
        print(f"WARNING: expected MLIR file not found at {mlir_path}")

    print("build dir contents:")
    for p in sorted(ctx.build_dir.glob(f"{op.name}*")):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
