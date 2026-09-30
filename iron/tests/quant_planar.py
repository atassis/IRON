# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract test for the `row_group_planar` weight layout.

Each check stands for a property a measurement or a device failure paid for:

  value identity      planar is a REORDER, not a second packing path. If a value ever differs
                      between the layouts, the two have forked and every accuracy result taken
                      under one stops applying to the other.
  byte identity       a planar buffer is EXACTLY as large as a header-first one at every shape.
                      That is what leaves arena sizing, the byte census and every /token figure
                      untouched, so a layout flip cannot be read as a byte win or loss.
  offset agreement    `row_offsets` is the one owner of the arithmetic mv_quant.cc reimplements
                      per dtype. This reads each row through those offsets and compares against
                      the row-major view, which is the check that would have caught K021's
                      original defect (ppl 4.25e9, top-1 0.00% on device, while compiling,
                      linking, passing a numpy round-trip and running 5/5 bit-identically).
  alignment           the point of the layout. Every payload pointer must clear the derived load
                      width, for the shapes this tree actually builds.
  the group still caps mv_quant.cc asserts `g % r == 0` -- the scale meets a chunk as one
                      broadcast -- so planar does NOT reach 512 bits at every group. Asserted
                      here so nobody re-reads the layout as a free width doubling.
  shared tile         `swiglu_mlp_dp` needs `TSI_GU*WROW_D == TSI_D*WROW_FF`, which holds because
                      the stride is affine in K with a proportional header. This is the invariant
                      that made header PADDING unusable; planar must not break it too.
  refusals            a partial block and an unimplemented payload order must fail loudly.
  device agreement    the kernel's own offsets, compiled from quant_row_layout.h by a host
                      compiler, must equal the packer's. Skipped (not failed) when no C++ compiler
                      is on PATH, and it SAYS so -- a check that silently does not run is worse
                      than one that is absent.

    python iron/tests/quant_planar.py
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iron.common.quant import (widest_chunk,  # noqa: F401
      # noqa: E402
    block_stride_bytes, dequantize_weight, derive_row_group, header_bytes,
    max_legal_vec_size, payload_bytes, quantize_weight, row_offsets, row_stride_bytes)

PLANAR = "row_group_planar"
G = 4   # what Gemma-4's planar+g64 target derives; the tests pass it explicitly
DTYPES = ("int8", "int4", "int8a", "int4a")
GROUPS = (32, 64, 128)
# (M, K) pairs the shipped graphs actually build, plus one small shape for the fast checks.
SHAPES = ((32, 256), (3840, 3840), (15360, 3840), (8704, 3840), (3840, 4096), (3840, 8192),
          (2048, 640))
FAIL = []


def check(ok, what):
    if not ok:
        FAIL.append(what)


def _load_bytes(dtype, vec):
    return vec // 2 if dtype in ("int4", "int4a") else vec


# ---- value and byte identity -------------------------------------------------------------------
rng = np.random.default_rng(11)
for dtype in DTYPES:
    for g in (32, 64):
        W = (rng.standard_normal((G * 4, 256)) * 0.02).astype(np.float32)
        a = quantize_weight(W, g, dtype)
        b = quantize_weight(W, g, dtype, layout=PLANAR, row_group=G)
        check(a.nbytes == b.nbytes, f"{dtype}/g{g}: planar is {b.nbytes}B against {a.nbytes}B")
        da = dequantize_weight(a, W.shape[0], W.shape[1], g, dtype)
        db = dequantize_weight(b, W.shape[0], W.shape[1], g, dtype, layout=PLANAR, row_group=G)
        check(np.array_equal(da, db), f"{dtype}/g{g}: the layouts dequantize differently")
        # Not the same bytes in the same places -- if they were, the reorder did nothing.
        check(not np.array_equal(a, b), f"{dtype}/g{g}: planar emitted header-first bytes")

# ---- row_offsets agrees with where the bytes are ------------------------------------------------
# Read every row's header and payload through row_offsets and rebuild the [M, stride] row-major
# view; it must equal the header-first buffer exactly. This is the property the kernel relies on.
for dtype in ("int8", "int4"):
    for g in (32, 64):
        M, K = G * 3, 512
        W = (rng.standard_normal((M, K)) * 0.02).astype(np.float32)
        flat = np.asarray(quantize_weight(W, g, dtype, layout=PLANAR, row_group=G)).view(np.uint8)
        ref = np.asarray(quantize_weight(W, g, dtype)).view(np.uint8).reshape(M, -1)
        hdr, pay = header_bytes(K, g, dtype), payload_bytes(K, dtype)
        for row in range(M):
            p_off, h_off = row_offsets(row, K, g, dtype, layout=PLANAR, row_group=G)
            check(np.array_equal(flat[h_off:h_off + hdr], ref[row, :hdr]),
                  f"{dtype}/g{g} row {row}: header at {h_off} does not match")
            check(np.array_equal(flat[p_off:p_off + pay], ref[row, hdr:]),
                  f"{dtype}/g{g} row {row}: payload at {p_off} does not match")

# ---- alignment, at the shapes this tree builds ---------------------------------------------------
for dtype in DTYPES:
    for g in GROUPS:
        for M, K in SHAPES:
            if K % g or M % G:
                continue
            try:
                vec = max_legal_vec_size([K], g, dtype, layout=PLANAR, row_group=G)
            except ValueError:
                continue
            load = _load_bytes(dtype, vec)
            for row in (0, 1, G - 1, G, M - 1):
                p_off, _ = row_offsets(row, K, g, dtype, layout=PLANAR, row_group=G)
                check(p_off % load == 0,
                      f"{dtype}/g{g} K={K}: row {row} payload at {p_off} is not {load}B aligned")

# ---- the group caps the width at TWO groups, and alignment reduces from there ---------------------
# A chunk sits inside one group or spans exactly two -- mv_quant.cc's `chunk_scales` builds the
# second case as two half-broadcasts concatenated. It is NOT unbounded: a wider chunk would need a
# scale per lane-run the kernel cannot form. widest_chunk is the one owner of that bound, because
# the packer and the operator derive row_group independently and never compare.
for K in (640, 1024, 3840, 4096, 8192, 15360):
    for g in GROUPS:
        want = widest_chunk(g, "int8")
        while want >= 8 and (payload_bytes(K, "int8") % want
                             or block_stride_bytes(K, g, "int8", layout=PLANAR, row_group=G) % want):
            want //= 2
        check(max_legal_vec_size([K], g, "int8", layout=PLANAR, row_group=G) == want,
              f"int8/g{g} K={K}: planar width is not widest_chunk reduced to alignment")
check(widest_chunk(32, "int8") == 64,
      "a g32 chunk may span two groups -- this is what stops g32 running 32 lanes on a 64-lane core")
check(widest_chunk(32, "int4a") == 32,
      "the affine kernels have no chunk_scales and keep the one-group rule")
check(max_legal_vec_size([3840], 64, "int8", layout=PLANAR, row_group=G) == 64,
      "planar at K=3840 g64 should be 64 elements (512b) -- this is what planar buys")
check(max_legal_vec_size([3840], 64, "int8") == 16,
      "header-first at K=3840 g64 should be 16 elements (128b) -- the case planar fixes")

# ---- swiglu_mlp_dp's shared weight tile stays expressible -----------------------------------------
# TSI_GU*WROW_D == TSI_D*WROW_FF holds iff the stride is proportional in K. Checked as a ratio so
# a future change that adds any INTERCEPT (a pad, an alignment round-up) fails here rather than in
# an operator assert nobody connects back to the packer.
for dtype in DTYPES:
    for g in (32, 64):
        for D, FF in ((3840, 15360), (1024, 3072)):
            for layout in ("header_first", PLANAR):
                s_d = block_stride_bytes(D, g, dtype, layout=layout, row_group=G)
                s_ff = block_stride_bytes(FF, g, dtype, layout=layout, row_group=G)
                check(s_ff * D == s_d * FF,
                      f"{dtype}/g{g} {layout}: stride not proportional in K "
                      f"({s_d} at {D}, {s_ff} at {FF})")

# ---- ROW_GROUP is derived, and the binding constraint is the L1 tile ----------------------------
# Gemma-4's quantized GEMV shapes, with the tile_size_output each one is built at. The A ObjectFifo
# is 2*n_vec*tile_size_input*row_stride bytes, so a coarser block is paid for in L1 -- which is the
# constraint a packer-only derivation cannot see and the one that decides the constant.
GEMV_SITES = (("mlp gate/up", 3840, 1920), ("mlp down chunk", 3840, 480),
              ("qkv global", 3840, 1024), ("qkv sliding", 3840, 1088),
              ("attn_o", 4096, 480), ("head", 3840, 4096))
L1_BYTES = 65536


def a_fifo_l1(tsi, tso, K, stride, n_vec=1):
    """iron/operators/gemv/design.py::l1_footprint_bytes, at this design's terms."""
    return 2 * n_vec * tsi * stride + n_vec * K * 2 + 2 * n_vec * tso * 2


for gsz, want in ((32, 1), (64, 4)):
    got = derive_row_group([3840], gsz, "int8", min(64, gsz))
    check(got == want, f"int8/g{gsz} K=3840: derived row group {got}, expected {want}")

for name, K, tso in GEMV_SITES:
    for gsz in (32, 64):
        vec = min(64, gsz)
        rg = derive_row_group([K], gsz, "int8", vec)
        stride = row_stride_bytes(K, gsz, "int8")
        used = a_fifo_l1(rg, tso, K, stride)
        check(used <= L1_BYTES,
              f"{name} K={K} g{gsz}: derived row group {rg} needs {used}B of L1")
        # and the constant someone would reach for -- aie::mmul's output-row count -- does not fit.
        check(a_fifo_l1(8, tso, K, stride) > L1_BYTES,
              f"{name} K={K} g{gsz}: an 8-row block now fits L1; re-check whether it is the better "
              f"constant, since it is also aie::mmul's tile height")

try:
    derive_row_group([3840], 64, "int8", 64, max_rows=2)
    check(False, "derive_row_group ignored its tile budget")
except ValueError as e:
    check("tile holds 2" in str(e), f"tile-budget refusal has the wrong message: {e}")

# ---- refusals -------------------------------------------------------------------------------------
W = np.zeros((G * 2 + 1, 256), np.float32)
try:
    quantize_weight(W, 32, "int8", layout=PLANAR, row_group=G)
    check(False, "a partial row block was accepted")
except ValueError as e:
    check("whole number" in str(e), f"partial-block refusal has the wrong message: {e}")
try:
    quantize_weight(np.zeros((8, 256), np.float32), 32, "int8", payload_order="mmul_8x8")
    check(False, "payload_order=mmul_8x8 was accepted")
except NotImplementedError:
    pass
try:
    quantize_weight(np.zeros((8, 256), np.float32), 32, "int8", layout="planar")
    check(False, "an unknown layout name was accepted")
except ValueError:
    pass

# ---- the kernel's own arithmetic, compiled from the header mv_quant.cc includes -----------------
import shutil        # noqa: E402 -- local to this check
import subprocess    # noqa: E402
import tempfile      # noqa: E402

CXX = shutil.which("clang++") or shutil.which("g++")
if CXX is None:
    print("quant_planar: SKIPPED the device-side offset check -- no clang++/g++ on PATH")
else:
    root = pathlib.Path(__file__).resolve().parents[2]
    src = root / "iron" / "tests" / "quant_row_layout_host.cc"
    inc = root / "aie_kernels" / "generic"
    with tempfile.TemporaryDirectory() as td:
        for layout, planar in (("header_first", 0), (PLANAR, 1)):
            exe = pathlib.Path(td) / f"qrl{planar}"
            cp = subprocess.run([CXX, "-std=c++17", f"-DPLANAR={planar}", f"-DROW_GROUP={G}", "-I", str(inc),
                                 str(src), "-o", str(exe)], capture_output=True, text=True)
            if cp.returncode:
                check(False, f"{layout}: the host build of quant_row_layout.h failed: "
                             f"{cp.stderr.strip().splitlines()[:2]}")
                continue
            for dtype, bits in (("int8", 8), ("int4", 4)):
                for K, g in ((3840, 32), (3840, 64), (640, 32), (4096, 64)):
                    rows = G * 3
                    out = subprocess.run([str(exe), str(K), str(g), str(bits), str(rows)],
                                         capture_output=True, text=True)
                    check(out.returncode == 0,
                          f"{layout} {dtype} K={K} g{g}: host runner exited {out.returncode}")
                    for line in out.stdout.split("\n"):
                        if not line.strip():
                            continue
                        row, h, p = (int(x) for x in line.split())
                        want_p, want_h = row_offsets(row, K, g, dtype, layout=layout, row_group=G)
                        check((h, p) == (want_h, want_p),
                              f"{layout} {dtype} K={K} g{g} row {row}: kernel says "
                              f"header {h} payload {p}, packer says {want_h} / {want_p}")

print(f"quant_planar: {len(FAIL)} failure(s)")
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
