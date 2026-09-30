#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free DISPATCH-level regression test for swiglu_mlp_dp's `gh_chunks>1` Wd wire format.

g4-t2-mlp-l1-floor's device gate (2026-09-14) built and ran cleanly against
`iron/tests/mv_quant_taccum.py`'s host contract test, then came back 75% NaN on real hardware (a
strict 1-real-3-NaN stride at gh_chunks=4) -- see gh-chunks-device-gate-fails-with-a-1-in-4-nan-
stride. Root cause: the device gate packed `Wd` with the plain `quantize_weight(Wd, ...)` (one
header per FULL FF-wide row), but design.py's `gh_chunks>1` fill (`sequence()`'s `tg3`, the Wd loop
under `if gh_chunks > 1:`) reads the L3 buffer as `gh_chunks` INDEPENDENTLY-quantized [D, D] planar
blocks -- `quantize_weight_chunked`'s format, not `quantize_weight`'s. `mv_quant_taccum.py` already
verified the KERNEL's own arithmetic against the correct (chunked) format; it never touches
design.py's fill offsets, so it could not have caught a mismatch between THOSE offsets and what a
caller's packer actually produces. This test closes that gap by reimplementing design.py's own
fill-offset formula (not re-deriving it -- see `wd_chunk_byte_offset`'s docstring) and checking it
against both packings.

    python iron/tests/operators/swiglu_mlp_dp_gh_chunks_wd_layout.py
"""
import importlib.util
import pathlib
import sys

import numpy as np

# Loaded by FILE PATH, not `from iron.common.quant import ...` / `from iron.tests...` -- same
# reason mv_quant_taccum.py loads quant.py this way: the iron package's __init__.py eagerly
# imports aie.utils, which needs a built toolchain instance this device-free test has no need of.
_here = pathlib.Path(__file__).resolve()
_quant_path = _here.parents[2] / "common" / "quant.py"
_taccum_path = _here.parents[1] / "mv_quant_taccum.py"

_qspec = importlib.util.spec_from_file_location("_wd_layout_quant", _quant_path)
quant = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(quant)

_tspec = importlib.util.spec_from_file_location("_wd_layout_taccum", _taccum_path)
taccum = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(taccum)  # defines kernel_chunk_dot; also runs its own module-level code
                                    # (imports only -- gated `main()` is not exec'd -- including
                                    # taccum's own ml_dtypes-or-SystemExit guard, so this test needs
                                    # no separate one)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def wd_chunk_byte_offset(r, g, row, D, D_PER_CORE, WROW_D):
    """swiglu_mlp_dp/design.py's `sequence()`, the `gh_chunks > 1` arm of `tg3`'s Wd fill:

        for r in range(gh_chunks):
            for g in range(n_aie_cols):
                gweight_ps[g].fill(Wd, _flat_tap(D * WROW_FF, D_PER_CORE * WROW_D,
                                    r * D * WROW_D + g * D_PER_CORE * WROW_D), ...)

    followed by core_fn's row-batch loop, which (at TSI_GU=1) acquires one WROW_D-byte WTILE_ty
    tile per row off that fill in row order. So byte `row` (0..D_PER_CORE-1) of core g's chunk r
    tile sits at this offset into the L3 Wd buffer -- the SAME formula, reproduced here rather than
    imported, because design.py's module scope needs the built aie.iron toolchain and this test
    must not (see the file-path import above)."""
    return r * D * WROW_D + g * D_PER_CORE * WROW_D + row * WROW_D


def decode_row(buf_i8, off, K, group_size, weight_dtype):
    """Parse one WROW_D-byte row at `off` exactly like `taccum_chunk_int8`/`_int4` (mv_quant_taccum.cc):
    `n_groups = K/group_size` header scales, then a K-wide payload, scale broadcast per group."""
    stride = quant.row_stride_bytes(K, group_size, weight_dtype)
    row_bytes = np.asarray(buf_i8[off:off + stride]).view(np.uint8)
    return taccum.kernel_chunk_dot(row_bytes, 1, K, group_size, weight_dtype,
                                   np.ones(K, dtype=np.float32))[0]


def main():
    rng = np.random.default_rng(7)
    D, FF, G, WDT, N = 3840, 15360, 64, "int8", 8
    R = FF // D
    D_PER_CORE = D // N
    WROW_D = quant.row_stride_bytes(D, G, WDT)

    Wd_f32 = (rng.standard_normal((D, FF)) * 0.02).astype(np.float32)

    # --- GREEN: quantize_weight_chunked is the format design.py's fill offsets expect. Reading
    # every (round, core, row) via wd_chunk_byte_offset and decoding it must reproduce the SAME
    # per-row dot product `dequantize_weight`ing the true column-slice and dotting directly gives
    # -- checked on core 0 and the last core (boundary), every gh_chunks round. ---
    correct_packed = quant.quantize_weight_chunked(Wd_f32, G, WDT, n_chunks=R)
    totals, refs = [], []
    for g in (0, N - 1):
        for row in range(0, D_PER_CORE, 37):  # sparse sweep, full core-boundary coverage
            out_row = g * D_PER_CORE + row
            total = 0.0
            for r in range(R):
                off = wd_chunk_byte_offset(r, g, row, D, D_PER_CORE, WROW_D)
                total += decode_row(correct_packed, off, D, G, WDT)
            totals.append(total)
            refs.append(float(Wd_f32[out_row].sum()))
    # rel-L2 over the whole sampled-row VECTOR, not a per-row relative error: an individual row's
    # true sum (15360 near-zero-mean terms) can itself land near zero by chance -- e.g. -0.12 here
    # -- which inflates a PER-ROW relative error from ordinary bf16/int8 quantization noise (~0.015
    # absolute) without signaling any real defect. mv_quant_taccum.py's own contract test uses the
    # same aggregate-rel-L2-over-many-rows convention for the identical reason.
    totals, refs = np.asarray(totals), np.asarray(refs)
    any_nan = bool(np.isnan(totals).any())
    rel_l2 = float(np.linalg.norm(totals - refs) / (np.linalg.norm(refs) + 1e-30))
    check("correct (chunked) packing: design.py's own offsets reconstruct the true row",
          not any_nan and rel_l2 < 1e-2, f"rel_l2={rel_l2:.4e} any_nan={any_nan}")

    # --- RED: the device gate's actual (wrong) packing -- ONE quantize_weight call over the full
    # [D, FF] row, exactly what the standalone device-gate script built before this fix. Reading it
    # through the SAME design.py offsets must NOT reproduce the true row, and must reproduce the
    # measured failure signature: row%R==0 decodes to a FINITE but wrong value (it lands exactly on
    # a real row's own header, just the wrong row/groups), row%R!=0 decodes to NaN (it lands inside
    # a real row's payload, and re-reading int8 payload bytes as float32 "scale" is NaN far more
    # often than not) -- see gh-chunks-device-gate-fails-with-a-1-in-4-nan-stride's device[:16]. ---
    naive_packed = quant.quantize_weight(Wd_f32, G, WDT)
    core0_vals = []
    for row in range(16):
        total = 0.0
        for r in range(R):
            off = wd_chunk_byte_offset(r, 0, row, D, D_PER_CORE, WROW_D)
            total += decode_row(naive_packed, off, D, G, WDT)
        core0_vals.append(total)
    core0_vals = np.array(core0_vals)
    nan_at_wrong_stride = np.isnan(core0_vals[1::4]).all() and np.isnan(core0_vals[2::4]).all() \
        and np.isnan(core0_vals[3::4]).all()
    finite_at_multiple_of_r = not np.isnan(core0_vals[0::4]).any()
    check("naive (unchunked) packing reproduces the measured 1-real-3-NaN, period-R stride",
          nan_at_wrong_stride and finite_at_multiple_of_r,
          f"vals[:8]={core0_vals[:8]}")
    check("naive packing does NOT reconstruct the true row (guards the format contract)",
          np.isnan(core0_vals).any(), "expected at least one NaN from the wrong-layout read")

    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
