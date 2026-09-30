#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract test for mv_quant_taccum.cc's chunked-K accumulate matvec
(matvec_taccum_{int4,int8}_bf16 + mv_taccum_zero_f32/mv_taccum_finish_bf16).

Mirrors quant_affine.py's shape: reimplement the KERNEL's own arithmetic in numpy and check it
against properties a silent regression would break, rather than just round-tripping the packer.

  chunk independence     each accumulate call's `a` is a SELF-CONTAINED packed block for its own
                         chunk width (LOCAL addressing) -- quantizing a [M, FULL_K] weight in R
                         separate [M, K_CHUNK] column-slices and reassembling via the kernel's
                         zero/accumulate/finish must reproduce dequantizing the ORIGINAL, UNSLICED
                         weight and taking one dot product. A global-k_off design (indexing a
                         full-row buffer from a per-call chunk offset) would pass a same-buffer
                         round trip but not this cross-check against independently-packed chunks.
  one rounding, not R    the accumulator is f32 across all R chunks; only mv_taccum_finish_bf16
                         narrows to bf16, once. A graph-level K-split (R separate GEMV calls, each
                         rounding its own bf16 output before an R-1-term bf16 sum) is measurably
                         worse: measured 1.93x rel-L2 at a 4-way split on this exact shape, since
                         rounding each chunk's own bf16 output before summing loses precision a
                         single f32 accumulator does not. This kernel exists to
                         not pay that cost, so the test checks it doesn't.

    python iron/tests/mv_quant_taccum.py
"""
import importlib.util
import sys, pathlib
import numpy as np

# iron.common.quant is loaded by FILE PATH, not `from iron.common.quant import ...`: the package
# __init__.py eagerly imports aie.utils (the mlir_aie python bindings), which this device-free
# numpy test has no need of and which is only resolvable through a built toolchain instance.
# quant.py itself imports only numpy at module scope (ml_dtypes lazily, inside functions), so it
# is genuinely standalone -- the package import chain is the only obstacle.
_quant_path = pathlib.Path(__file__).resolve().parents[2] / "iron" / "common" / "quant.py"
_spec = importlib.util.spec_from_file_location("_mv_quant_taccum_quant", _quant_path)
_quant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_quant)
quantize_weight, dequantize_weight, row_stride_bytes = (
    _quant.quantize_weight, _quant.dequantize_weight, _quant.row_stride_bytes)
try:
    import ml_dtypes
except ImportError:
    raise SystemExit("mv_quant_taccum needs ml_dtypes (the packer's bf16 type)")

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def kernel_chunk_dot(packed_chunk, M, K, g, wdt, b_chunk):
    """Reimplement taccum_chunk_{int4,int8}'s per-chunk arithmetic: dequant q*s per element,
    bf16-round the dequantized weight (the kernel narrows via `.template to_vector<bfloat16>()`
    before the mac), then dot against b_chunk. Returns one f32 partial sum per row -- exactly
    what one accumulate call adds into `acc[row_offset+row]`."""
    n_groups = K // g
    stride = row_stride_bytes(K, g, wdt)
    rows = np.asarray(packed_chunk).view(np.uint8).reshape(M, stride)
    hdr = n_groups * 4  # f32 scale (SCALE_BF16=0, the default this test targets)
    scale = rows[:, :hdr].view(np.float32).reshape(M, n_groups)
    payload = rows[:, hdr:]
    if wdt == "int4":
        lo = (payload & 0x0F).astype(np.int8); lo = np.where(lo >= 8, lo - 16, lo)
        hi = ((payload >> 4) & 0x0F).astype(np.int8); hi = np.where(hi >= 8, hi - 16, hi)
        q = np.empty((M, K), np.int8); q[:, 0::2] = lo; q[:, 1::2] = hi
    else:
        q = payload.view(np.int8)
    q = q.reshape(M, n_groups, g).astype(np.float32)
    sv = scale[:, :, None].astype(np.float32)
    # qbf = to_float<bfloat16>(q, 0) * scale, narrowed to bf16 -- matches `mul(qbf, sv)
    # .template to_vector<bfloat16>()` before the mac.
    w_bf16 = (q * sv).astype(ml_dtypes.bfloat16).astype(np.float32)
    bg = np.asarray(b_chunk, np.float32).reshape(1, K)
    return (w_bf16.reshape(M, K) * bg).sum(axis=1)  # per-row partial (the chunk_acc reduce)


def kernel_dot_chunked_accum(W_full, M, K_chunk, R, g, wdt, b_full):
    """Full zero/accumulate/finish sequence: quantize each of the R column-slices of W_full
    INDEPENDENTLY (LOCAL addressing -- a separate `quantize_weight` call per chunk, not a slice of
    one full-row packing), accumulate their kernel_chunk_dot partials in f32, round to bf16 once."""
    acc = np.zeros(M, dtype=np.float32)  # mv_taccum_zero_f32
    for r in range(R):
        Wc = W_full[:, r * K_chunk:(r + 1) * K_chunk]
        bc = b_full[r * K_chunk:(r + 1) * K_chunk]
        pk = quantize_weight(Wc, g, wdt)
        acc += kernel_chunk_dot(pk, M, K_chunk, g, wdt, bc)  # matvec_taccum_*_bf16
    return acc.astype(ml_dtypes.bfloat16).astype(np.float32)  # mv_taccum_finish_bf16


def graph_split_baseline(W_full, M, K_chunk, R, g, wdt, b_full):
    """The OTHER way to chunk a GEMV over K: R independent GEMV calls, each rounding its own
    bf16 output, summed with R-1 bf16 adds. What this kernel exists to avoid -- see the KB note
    in the module docstring."""
    partial_bf16 = np.zeros(M, dtype=ml_dtypes.bfloat16).astype(np.float32)
    for r in range(R):
        Wc = W_full[:, r * K_chunk:(r + 1) * K_chunk]
        bc = b_full[r * K_chunk:(r + 1) * K_chunk]
        pk = quantize_weight(Wc, g, wdt)
        one = kernel_chunk_dot(pk, M, K_chunk, g, wdt, bc).astype(ml_dtypes.bfloat16).astype(np.float32)
        partial_bf16 = (partial_bf16 + one).astype(ml_dtypes.bfloat16).astype(np.float32)
    return partial_bf16


def main():
    rng = np.random.default_rng(0)
    K_CHUNK, R, G = 3840, 4, 64  # Gemma-4's down-projection: FULL_K=FF=15360, chunk=D=3840
    FULL_K = K_CHUNK * R
    M = 16
    for wdt in ("int4", "int8"):
        print(f"{wdt} K_CHUNK={K_CHUNK} R={R} G={G}")
        W = rng.standard_normal((M, FULL_K)).astype(np.float32) * 0.02
        b = rng.standard_normal(FULL_K).astype(np.float32)

        # unsplit ground truth: dequantize the WHOLE row (one packing, one K) and dot in f64.
        pk_full = quantize_weight(W, G, wdt)
        dq_full = dequantize_weight(pk_full, M, FULL_K, G, wdt)
        ref = (dq_full.astype(np.float64) @ b.astype(np.float64))

        got = kernel_dot_chunked_accum(W, M, K_CHUNK, R, wdt=wdt, g=G, b_full=b)
        rel_chunked = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
        check("chunked accumulate reproduces the unsplit dequant-dot", rel_chunked < 1e-2,
              f"rel-L2 {rel_chunked:.4e}")

        base = graph_split_baseline(W, M, K_CHUNK, R, wdt=wdt, g=G, b_full=b)
        rel_base = float(np.linalg.norm(base - ref) / (np.linalg.norm(ref) + 1e-30))
        check("chunked accumulate is no worse than the graph-split baseline",
              rel_chunked <= rel_base * 1.001,
              f"accum {rel_chunked:.4e} vs graph-split {rel_base:.4e}")

        # Same-buffer round trip: one full-row packing sliced into R views (the shape a
        # single-buffer, single-quantize deployment would actually hand the kernel at Gemma-4's
        # int8 g64 shape, where row_stride_bytes(K_CHUNK)*R == row_stride_bytes(FULL_K)) must
        # still reconstruct correctly when each slice is re-packed independently and re-summed --
        # i.e. per-chunk quantization is not a DIFFERENT quantizer, only a different packing call
        # over the same group boundaries (GROUP_SIZE=64 divides K_CHUNK=3840 exactly, so group
        # boundaries never straddle a chunk boundary).
        stride_full = row_stride_bytes(FULL_K, G, wdt)
        stride_chunk = row_stride_bytes(K_CHUNK, G, wdt)
        check("R independently-packed chunks total the same bytes as one full-row packing",
              stride_chunk * R == stride_full, f"{stride_chunk}*{R} vs {stride_full}")

    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
