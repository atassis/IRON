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


def _extract_core_regions(mlir_text):
    """Every `aie.core(...) { ... }` op body, source order. Brace-matched by hand -- the .mlir
    is text, not a parsed module, and a core body nests its own scf.for/if blocks that also use
    `{ }`, so a regex match on the closing brace alone would stop at the first nested one."""
    regions = []
    for m in re.finditer(r"aie\.core\(", mlir_text):
        brace = mlir_text.index("{", m.start())
        depth, i = 0, brace
        while True:
            if mlir_text[i] == "{":
                depth += 1
            elif mlir_text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        regions.append(mlir_text[m.start():i + 1])
    return regions


_ROWLEN_CALL_RE = re.compile(r"func\.call @\w*(mask_bf16|softmax_bf16|taccum_rows_bf16_f32)\(")

# The SECOND operand is deliberately NOT captured: it is the runtime chunks value the compute
# loop's own bound already traces to (see test_dynamic_window_drain_conserves_stream_acquires),
# and leaving it alone is what stops this rule from also hiding a broken compute-loop bound.
_SUBI_DRAIN_RE = re.compile(r"^\s*%\S+ = arith\.subi %(c\d+_i32(?:_\d+)?), %\S+ : i32\s*$")


def _normalize_core_region(region):
    """Strip the four terms that legitimately vary with max_seq (or, for (4), with something
    outside this text entirely) but are NOT the trip count:
    (1) the row-length CONSTANT fed to mask_bf16/softmax_bf16/taccum_rows_bf16_f32 -- a fresh
    `%cN_i32... = arith.constant N : i32` immediately followed by its sole use as that call's
    row-length operand; (2) the sc/sw buffer WIDTH (S, the `2*gqa*S*2` L1 bytes design.py costs
    to the window) wherever it appears as a `memref<Nxbf16>` type in one of the four kernel call
    signatures; (3) the drain loop's TOTAL operand -- design.py's
    `arith.subi(TOTAL, arith.muli(nsplits, PER_SPLIT))`, the TOTAL constant only, matched the same
    way as (1): a fresh `%cN_i32... = arith.constant N : i32` whose sole use is arith.subi's FIRST
    operand. (1) and (2) are anchored on the KERNEL NAME, never the numeric value: tile_elems
    (tsi*D) equals max_seq by coincidence at this test's S=4096 (tsi=4, D=1024), and a blind
    "replace this number" pass would conflate the two and silently launder a real difference. (3)
    has no kernel name to anchor on, so it is anchored on the OP instead. arith.subi now appears
    TWICE in core_fn -- split-K's seg_unmasked clamp is the other one (`arith.subi(mask_len,
    seg_lo)`) -- but only TOTAL's use has a fresh CONSTANT as its first operand, which is what the
    match requires, so the anchor still picks out exactly one line. Only the constant TOTAL is
    stripped; the muli's own PER_SPLIT operand and the nsplits value being subtracted from are
    untouched, so a broken segment-loop trip count still shows up as a diff here. TOTAL's use sits
    TWO lines below its definition rather than one -- an unrelated `arith.muli` (consuming a
    DIFFERENT, invariant PER_SPLIT constant) sits in between -- so this rule looks one line further
    than (1) and (2) do.

    (3) is TRANSITIONAL, not a permanent carve-out: it exists only because the K/V-cache fill still
    streams every built segment regardless of `nsplits`. Once the fill itself reads a runtime
    length -- streaming exactly `nsplits` segments -- the drain loop this rule normalizes
    disappears, and this rule should be deleted with it, not carried forward.

    (4) FOUND while adding attn_split to this test, not designed in advance: `%cV_TYPE_N`-style
    constant register NAMES carry a disambiguation suffix N that is NOT purely positional --
    Measured 2026-09-11, two attn_split=128 builds differing only in max_seq (1024 vs 4096) produce
    core bodies identical in every operation but ONE register suffix (73 vs 74), with no textual
    difference anywhere before it (confirmed via a full-file diff against the un-normalized MLIR,
    first divergence inside `aie.core`'s own body, not in any preceding shared declaration).
    So the disambiguation counter depends on something outside the printed region -- unidentified,
    and out of scope to chase here. Every constant name of this shape is stripped to its base (the
    literal digits and the optional `_TYPE` word, dropping the trailing `_N`) -- safe because this
    generated code always defines a constant immediately before its single use (verified against
    the emitted MLIR throughout this design), so no result name is read across the boundary a
    collapsed suffix could blur. This is NOT anchored to the drain loop like (1)-(3); it is a
    blanket textual pass, because the suffix drift itself is not proven to originate from split-K
    or from the drain specifically."""
    lines = region.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^(\s*)%(c\d+_i32(?:_\d+)?) = arith\.constant (\d+) : i32\s*$", lines[i])
        if m and i + 1 < len(lines) and re.search(rf"%{re.escape(m.group(2))}\b", lines[i + 1]):
            if _ROWLEN_CALL_RE.search(lines[i + 1]):
                out.append(f"{m.group(1)}%ROWLEN = arith.constant ROWLEN : i32")
                out.append(re.sub(rf"%{re.escape(m.group(2))}\b", "%ROWLEN", lines[i + 1]))
                i += 2
                continue
            if _SUBI_DRAIN_RE.match(lines[i + 1]):
                out.append(f"{m.group(1)}%NKVCHUNKS = arith.constant NKVCHUNKS : i32")
                out.append(re.sub(rf"%{re.escape(m.group(2))}\b", "%NKVCHUNKS", lines[i + 1]))
                i += 2
                continue
        if (m and i + 2 < len(lines) and not re.search(rf"%{re.escape(m.group(2))}\b", lines[i + 1])
                and re.search(rf"%{re.escape(m.group(2))}\b", lines[i + 2])
                and _SUBI_DRAIN_RE.match(lines[i + 2])):
            out.append(f"{m.group(1)}%NKVCHUNKS = arith.constant NKVCHUNKS : i32")
            out.append(lines[i + 1])
            out.append(re.sub(rf"%{re.escape(m.group(2))}\b", "%NKVCHUNKS", lines[i + 2]))
            i += 3
            continue
        out.append(lines[i])
        i += 1
    text = "\n".join(out)
    for pat in (
        r"(func\.call @\w*sc_matvec_vectorized_bf16_bf16\([^)]*\) : "
        r"\(i32, i32, memref<\d+xbf16>, memref<\d+xbf16>, memref<)\d+(xbf16>\) -> \(\))",
        r"(func\.call @\w*mask_bf16\([^)]*\) : \(memref<)\d+(xbf16>, i32, i32\) -> \(\))",
        r"(func\.call @\w*taccum_rows_bf16_f32\([^)]*\) : "
        r"\(i32, i32, i32, i32, memref<\d+xbf16>, memref<)\d+(xbf16>, memref<\d+xf32>\) -> \(\))",
    ):
        text = re.sub(pat, r"\1SROW\2", text)
    text = re.sub(
        r"(func\.call @\w*softmax_bf16\([^)]*\) : \(memref<)\d+(xbf16>, memref<)\d+(xbf16>, i32\) -> \(\))",
        r"\1SROW\2SROW\3", text,
    )
    return re.sub(r"%(c\d+(?:_[a-zA-Z][a-zA-Z0-9]*)?)_\d+\b", r"%\1", text)


def _extract_scf_for_blocks(region):
    """Every LEAF `scf.for ... { ... }` in one core region, as (bound-operand, body-text) -- i.e.
    every step loop, but not core_fn's own outer Worker dispatch loop, which wraps the WHOLE body
    in one `scf.for %arg0 = %c0 to %c9223372036854775807 ...` (run forever, one dispatch per
    iteration) and would otherwise swallow every other match into a single block. Brace-matched
    like `_extract_core_regions`; a block whose own body contains another `scf.for` is a wrapper,
    not a step loop, and is dropped -- core_fn nests loops no deeper than that one level today."""
    blocks = []
    for m in re.finditer(r"scf\.for %\S+ = %\S+ to %(\S+) step %\S+ \{", region):
        brace = region.index("{", m.start())
        depth, i = 0, brace
        while True:
            if region[i] == "{":
                depth += 1
            elif region[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        blocks.append((m.group(1), region[brace + 1:i]))
    return [(b, body) for b, body in blocks if "scf.for " not in body]


_ACQUIRE_RE = re.compile(r"^%\S+ = aie\.objectfifo\.acquire @(\S+)\(Consume, 1\) : memref<\d+xbf16>$")
_RELEASE_RE = re.compile(r"^aie\.objectfifo\.release @(\S+)\(Consume, 1\)$")


def _drain_block_fifo(body):
    """A drain loop's body is EXACTLY one acquire and one release of the SAME fifo and nothing
    else -- no func.call between them. That is what makes "acquire without computing" structurally
    detectable in the IR text rather than argued from the Python source. Returns the fifo name, or
    None if the block is not this shape (e.g. a compute loop, which always calls a kernel)."""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) != 2:
        return None
    acq, rel = _ACQUIRE_RE.match(lines[0]), _RELEASE_RE.match(lines[1])
    return acq.group(1) if (acq and rel and acq.group(1) == rel.group(1)) else None


def _chunks_ssa_for_bound(region, bound_operand):
    """`bound_operand` is a loop's `to %X`. On a windowed build X is `arith.index_cast` off the
    i32 `chunks`/`nsplits` value read from the scratchpad parameter -- return that i32 SSA name.
    None on a compile-time build, where the operand is a bare index constant instead."""
    m = re.search(rf"%{re.escape(bound_operand)} = arith\.index_cast %(\S+) : i32 to index", region)
    return m.group(1) if m else None


def _all_scf_for_spans(region):
    """Every `scf.for` in `region`, brace-matched, as (start, end, bound-operand, body-text,
    nesting-depth) -- unlike `_extract_scf_for_blocks`, this keeps WRAPPER loops (a body containing
    another `scf.for`) rather than dropping them. Split-K nests one level deeper than the pre-split
    design did (see test_dynamic_window_drain_conserves_stream_acquires): the segment loop IS a
    wrapper now, and its own bound is exactly the runtime trip count this file exists to check."""
    spans = []
    for m in re.finditer(r"scf\.for %\S+ = %\S+ to %(\S+) step %\S+ \{", region):
        brace = region.index("{", m.start())
        depth, i = 0, brace
        while True:
            if region[i] == "{":
                depth += 1
            elif region[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append((m.start(), i, m.group(1), region[brace + 1:i]))
    return [(s, e, b, body, sum(1 for s2, e2, *_ in spans if s2 < s and e < e2))
            for s, e, b, body in spans]


def _drain_total_and_per_split(region, bound_operand, chunks_ssa):
    """`bound_operand` is the drain loop's `to %X`. Confirms X traces through `arith.index_cast`
    off `arith.subi(TOTAL, arith.muli(chunks_ssa, PER_SPLIT))`, where `chunks_ssa` is the SAME
    value the sibling segment loop's own bound traced to -- the identity that makes
    `segment_acquires + drain_acquires == TOTAL` hold for every runtime value of `nsplits`, not
    just the one this particular build happens to carry. Returns (TOTAL, PER_SPLIT) read out of
    the IR text, or None if the shape does not match."""
    m = re.search(rf"%{re.escape(bound_operand)} = arith\.index_cast %(\S+) : i32 to index", region)
    if not m:
        return None
    m2 = re.search(rf"%{re.escape(m.group(1))} = arith\.subi %(\S+), %(\S+) : i32", region)
    if not m2:
        return None
    m3 = re.search(rf"%{re.escape(m2.group(2))} = arith\.muli %(\S+), %(\S+) : i32", region)
    if not m3 or m3.group(1) != chunks_ssa:
        return None
    m4 = re.search(rf"%{re.escape(m2.group(1))} = arith\.constant (\d+) : i32\b", region)
    m5 = re.search(rf"%{re.escape(m3.group(2))} = arith\.constant (\d+) : i32\b", region)
    return (int(m4.group(1)), int(m5.group(1))) if (m4 and m5) else None


def test_dynamic_window_drain_conserves_stream_acquires(tmp_path):
    """The fill streams every built segment's (K, V) tiles into `stream_c` regardless of
    `nsplits` (design.py's drain_remainder comment); the segment loop and the ONE drain loop that
    follows it must together acquire exactly that many, for every value `nsplits` could take at
    runtime -- not just the one this build's dispatch would carry, which this test never runs.

    Split-K nests one level deeper than the pre-split design: the runtime trip count now lives on
    the SEGMENT loop (`for sp in range_(nsplits)`, a WRAPPER around the two per-marker loops), not
    on the sc_matvec/taccum_rows loops directly -- their own bound, SPLIT_CHUNKS, is a build-time
    literal by construction, since attn_split is fixed at build time. Proved algebraically from the
    IR text: the segment loop's bound and the ONE combined drain loop's bound both trace back to
    the SAME i32 SSA value (nsplits), and the drain's is
    `arith.subi(TOTAL, arith.muli(nsplits, PER_SPLIT))` -- so the two bounds sum to TOTAL by
    construction, whatever `nsplits` is. TOTAL is checked against 2 * N_KV_CHUNKS (N_KV_CHUNKS =
    S // rpc, doubled because one drain now covers BOTH caches the pre-split design drained
    separately) and PER_SPLIT against 2 * SPLIT_CHUNKS (SPLIT_CHUNKS = rpc at this test's S, since
    the window fits L1 in one segment and attn_split derives to S itself).

    Control: a window_parameter=None build must contain NO acquire-only/release-only loop at all
    (`_drain_block_fifo` finds none) -- the compute loops' own bounds are already build-time
    literals, so nothing is left over to drain. Manually verified this control is not vacuous: with
    `drain_remainder()`'s two call sites deleted from design.py, this same assertion on the SAME
    "attn_window" build fails with "expected exactly one drain loop ... found 0" instead of passing.
    """
    D, HD, Hq, Hkv, S, tsi = 1024, 128, 16, 8, 1024, 4
    rpc = (tsi * D) // HD               # TILE_ELEMS // HD, design.py's own derivation
    n_kv_chunks = S // rpc
    split_chunks = rpc                  # S=1024 fits L1 in one segment: attn_split derives to S

    from iron.common import AIEContext
    from iron.operators.attn_block_dp.op import AttnBlockDataParallel

    def core_regions(window_parameter):
        op = AttnBlockDataParallel(
            D=D, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S, num_aie_columns=8, tile_size_input=tsi,
            kv_alloc=4096, kv_block_size=128, window_parameter=window_parameter,
            context=AIEContext(build_dir=tmp_path / f"drain_{window_parameter}"))
        op.compile()
        text = Path(op.xclbin_artifact.mlir_input.filename).read_text()
        return _extract_core_regions(text)

    on_regions = core_regions("attn_window")
    assert len(on_regions) == 8
    for region in on_regions:
        spans = _all_scf_for_spans(region)
        # The segment loop: a WRAPPER whose DIRECT children (depth+1) include a LEAF (no further
        # nested scf.for) containing each marker -- "in body" alone is not enough, since the outer
        # Worker dispatch loop also transitively contains both markers in ITS body text and would
        # otherwise false-match here too.
        segment = [(s, e, b, body, d) for s, e, b, body, d in spans
                   if all(any(s2 > s and e2 < e and d2 == d + 1 and marker in body2
                              and "scf.for " not in body2
                              for s2, e2, _, body2, d2 in spans)
                          for marker in ("sc_matvec_vectorized_bf16_bf16", "taccum_rows_bf16_f32"))]
        assert len(segment) == 1, f"expected exactly one segment loop, found {len(segment)}"
        seg_s, seg_e, seg_bound, _, seg_depth = segment[0]
        chunks_ssa = _chunks_ssa_for_bound(region, seg_bound)
        assert chunks_ssa is not None, "segment loop's trip count is not runtime-derived"

        fifo = None
        for marker in ("sc_matvec_vectorized_bf16_bf16", "taccum_rows_bf16_f32"):
            inner = [(s, e, b, body) for s, e, b, body, d in spans
                     if d == seg_depth + 1 and s > seg_s and e < seg_e and marker in body]
            assert len(inner) == 1, (
                f"expected exactly one {marker} loop inside the segment loop, found {len(inner)}")
            inner_bound, inner_body = inner[0][2], inner[0][3]
            assert _chunks_ssa_for_bound(region, inner_bound) is None, (
                f"{marker} loop's own bound is runtime-derived -- SPLIT_CHUNKS should be a "
                "build-time literal now that the segment loop carries the runtime trip count")
            # Both marker loops acquire the SAME `stream_c` fifo (K for scores, V for context,
            # interleaved on one objectFifo) -- read its actual generated name here rather than
            # assume it matches design.py's `stream_c` Python variable name.
            m = re.search(r"aie\.objectfifo\.acquire @(\S+)\(Consume, 1\)", inner_body)
            assert m, f"{marker} loop has no objectfifo acquire: {inner_body!r}"
            assert fifo is None or fifo == m.group(1), (
                f"scores and context loops acquire DIFFERENT fifos ({fifo} vs {m.group(1)})")
            fifo = m.group(1)

        # The ONE drain loop: the sibling `scf.for` immediately after the segment loop closes, at
        # the SAME nesting depth -- draining BOTH caches' leftover tiles together (design.py's
        # drain_remainder comment), unlike the pre-split design's two separate per-marker drains.
        siblings = sorted((s, e, b, body) for s, e, b, body, d in spans if d == seg_depth)
        idx = next(i for i, (s, e, b, body) in enumerate(siblings) if s == seg_s)
        assert idx + 1 < len(siblings), f"no loop follows the segment loop to drain {fifo}"
        drain_bound, drain_body = siblings[idx + 1][2], siblings[idx + 1][3]
        assert _drain_block_fifo(drain_body) == fifo, (
            f"loop after the segment loop is not a pure {fifo} drain: {drain_body!r}")
        totals = _drain_total_and_per_split(region, drain_bound, chunks_ssa)
        assert totals is not None, (
            "drain loop bound does not trace to arith.subi(TOTAL, arith.muli(nsplits, PER_SPLIT))")
        total, per_split = totals
        assert total == 2 * n_kv_chunks, (
            f"segment + drain trip counts sum to {total}, expected 2*N_KV_CHUNKS={2 * n_kv_chunks}")
        assert per_split == 2 * split_chunks, (
            f"drain PER_SPLIT is {per_split}, expected 2*SPLIT_CHUNKS={2 * split_chunks}")

    off_regions = core_regions(None)
    assert len(off_regions) == 8
    for region in off_regions:
        drains = [body for _, body in _extract_scf_for_blocks(region) if _drain_block_fifo(body)]
        assert not drains, f"window_parameter=None must emit no drain loop, found {len(drains)}"


def test_dynamic_window_makes_the_core_trip_count_independent(tmp_path):
    """The whole point: with the trip count read at runtime, two windows must compile to the SAME
    core PROGRAM -- AT A SHARED attn_split. Two different max_seq no longer share a program on
    their own: attn_split=None DERIVES the split from max_seq, and NSPLIT/SPLIT_CHUNKS become baked
    loop-nesting structure rather than a stripped constant, so two capacities can legitimately
    place different programs (same .text SIZE, different immediate operands -- a real regression
    this test caught: identical size is not identical program). Passing the SAME explicit
    attn_split to both builds is what the production rung ladder relies on (one_design_rung_ladder
    needs exactly this), and it is what restores the test's original point.

    This test's predecessor compared whole core ELFs and was vacuous: two OTHER terms scale with
    max_seq regardless of the trip count -- the sc/sw buffer size and the row-length literal handed
    to mask_k/softmax_k/tr_k -- so it would have passed with the runtime-trip-count mechanism
    absent, present, or broken. This instead diffs the `aie.core` MLIR text with
    `_normalize_core_region`'s terms stripped out, so what remains is the trip count alone (a
    fourth term, the drain loop's TOTAL literal, joined the first three once split-K's fill needed
    draining across segments -- see that function's docstring for why it is transitional).

    Second half is the negative control -- reverting to window_parameter=None must still differ
    under the SAME normalisation, or this gate is exactly as vacuous as its predecessor."""
    from iron.common import AIEContext
    from iron.operators.attn_block_dp.op import AttnBlockDataParallel

    # A multiple of lcm(rpc=32, kv_block_size=128, FLASH_SM_VEC_LEN=64) = 128 that divides both
    # 1024 and 4096 -- design.py's own attn_split legality rule (see derive_attn_split).
    ATTN_SPLIT = 128

    def core_regions(S, window_parameter):
        op = AttnBlockDataParallel(
            D=1024, HD=128, Hq=16, Hkv=8, max_seq=S, num_aie_columns=8, tile_size_input=4,
            kv_alloc=4096, kv_block_size=128, window_parameter=window_parameter,
            attn_split=ATTN_SPLIT if window_parameter else None,
            context=AIEContext(build_dir=tmp_path / f"S{S}_{window_parameter}"))
        op.compile()
        mlir_text = Path(op.xclbin_artifact.mlir_input.filename).read_text()
        return [_normalize_core_region(r) for r in _extract_core_regions(mlir_text)]

    on_1024, on_4096 = core_regions(1024, "attn_window"), core_regions(4096, "attn_window")
    assert len(on_1024) == 8 and len(on_4096) == 8
    assert on_1024 == on_4096, "dynamic window still bakes the trip count into the core"

    off_1024, off_4096 = core_regions(1024, None), core_regions(4096, None)
    assert off_1024 != off_4096, (
        "negative control: a compile-time trip count must still differ under the SAME "
        "normalisation, or the assertion above cannot fail"
    )


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
