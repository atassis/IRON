#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free build gate for T2.1a (g4-t2-mlp-l1-floor): does swiglu_mlp_dp's chunked-K down
projection (`gh_chunks`) fit L1 and build at Gemma-4-12B's shape, where the UNCHUNKED design does
not fit at any setting (bf16 or quantized)? CPU-only aiecc, like test.py's own build-only gate.

Three arms:
  regression     D=1024 FF=3072 (Qwen3-0.6B), gh_chunks=1 (default) -- must be MLIR-md5-identical
                 to a build with no T2.1a changes present, proving the new code path is inert when
                 unused.
  gemma4         D=3840 FF=15360 (Gemma-4-12B), int8 g64, gh_chunks=4 (=FF/D), tile_rows_gu=1,
                 weight_depth=1 -- the first build of this operator at ANY quantized shape (the
                 plain bf16/int4/int8 paths were plumbed through op.py/design.py but never
                 build-tested here). gh_chunks=1 at this shape does not fit L1 at any weight_dtype;
                 see g4-t2-mlp-l1-floor's worklog for the arithmetic.
  qwen3_chunked  D=1024 FF=3072 (Qwen3-0.6B, which fits L1 unchunked), int8/int4 g64,
                 gh_chunks=3 (=FF/D) -- qwen3-mlp-gh-chunks-numerics: device-free build gate for
                 the arm a real device A/B (paired against gh_chunks=1) was run against. Confirms
                 it builds at production's own MLP_DP_COLS=4, not gemma4()'s 8.

    python iron/tests/operators/swiglu_mlp_dp_gh_chunks.py [regression|gemma4|qwen3_chunked|all]
"""
import hashlib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from iron.common import AIEContext
from iron.operators.swiglu_mlp_dp.op import SwiGLUMLPDataParallel

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def build(name, build_dir, **kwargs):
    ctx = AIEContext(build_dir=Path(build_dir))
    op = SwiGLUMLPDataParallel(context=ctx, **kwargs)
    print(f"=== {name}: {op.name} ===")
    op.compile()
    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    mlir_text = mlir_path.read_text()
    n_cores = len(re.findall(r"\baie\.core\(", mlir_text))
    md5 = hashlib.md5(mlir_text.encode()).hexdigest()
    xclbin = Path(op.xclbin_artifact.filename)
    print(f"  aie.core={n_cores} mlir_md5={md5} xclbin={xclbin.stat().st_size}B")
    return md5, n_cores


def regression(base):
    # No gh_chunks kwarg at all (not just gh_chunks=1): exercises the exact call shape every
    # existing caller uses, so a regression here means the new parameter's DEFAULT broke someone
    # who never asked for it.
    md5, n_cores = build("qwen3 regression (gh_chunks unset)", f"{base}/regr",
                         D=1024, FF=3072, num_aie_columns=8, num_aie_rows=1)
    check("qwen3 shape still builds 8 cores", n_cores == 8)
    # The known-good md5 from the pre-T2.1a baseline (worktree at ef5dd58, this file's own
    # verification pass) -- see g4-t2-mlp-l1-floor's report for how this was cross-checked.
    check("qwen3 MLIR byte-identical to the pre-T2.1a baseline",
          md5 == "1c11431f65650f62d0379bc1caf0e91a", md5)


def gemma4(base):
    md5, n_cores = build("gemma4 int8g64 gh_chunks=4", f"{base}/gemma4",
                         D=3840, FF=15360, num_aie_columns=8, num_aie_rows=1,
                         weight_dtype="int8", group_size=64, gh_chunks=4,
                         tile_rows_gu=1, weight_depth=1)
    check("gemma4 shape builds 8 cores", n_cores == 8)


def qwen3_chunked(base):
    # qwen3-mlp-gh-chunks-numerics: gh_chunks == R == FF/D == 3 is the only legal value (design.py's
    # own assert). Qwen3-0.6B does not NEED chunking to fit L1 (unlike Gemma-4) -- this arm exists to
    # confirm the chunked path builds cleanly at Qwen3's own shape and DP_COLS, matching production's
    # MLP_DP_COLS=4 default (gen_llm_decode.py), not the 8-column shape gemma4() above uses.
    # Real device A/B (not run here): both weight_dtypes came back bit-identical to gh_chunks=1 at
    # this shape -- Qwen3's unchunked down-projection is a single full-K matvec per output tile
    # (one bf16 round already), unlike the graph-level K-split baseline gh_chunks improved on for
    # Gemma-4, so there is no per-partial-round defect here for chunking to fix.
    for dt in ("int8", "int4"):
        md5, n_cores = build(f"qwen3 {dt}g64 gh_chunks=3", f"{base}/qwen3_{dt}",
                             D=1024, FF=3072, num_aie_columns=4, num_aie_rows=1,
                             weight_dtype=dt, group_size=64, gh_chunks=3,
                             tile_rows_gu=8, weight_depth=2)
        check(f"qwen3 {dt} shape builds 4 cores", n_cores == 4)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    base = os.environ.get("IRON_TEST_BUILD_ROOT", str(Path.cwd() / "build"))
    if which in ("regression", "all"):
        regression(base)
    if which in ("gemma4", "all"):
        gemma4(base)
    if which in ("qwen3_chunked", "all"):
        qwen3_chunked(base)
    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
