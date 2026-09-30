# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Does `Softmax(segment=L)` go FLAT in the context length, and how far does it place?

The unchunked op holds four `memref<cols x bf16>` buffers in one core's L1 (`in1`/`out` at depth 2),
so it stops at 8063 positions and trips a second wall, the 16383-word core-tile `aie.dma_bd` length
field, at ~32768 on the same buffer. Both are properties of the TILE.

    l1_sweep()      pure arithmetic, both modes across the S ladder
    build_check()   device-free aiecc placement at each rung -- needs an activated toolchain env

    python softmax_chunk_l1_check.py [--segment 1024] [--rows 16] [--cols 8]
"""
import argparse
import sys
import tempfile
import os
from pathlib import Path

L1_BYTES = 65536
BD_MAX_WORDS = 16383   # aie.dma_bd's length field, in 32-bit words
STACK = 1024           # aie.core's default stack_size, 0x400
SCRATCHPAD_PARAM = 8   # the sm_mask ScratchpadParameter buffer
WIDTH_FIFO = 8         # of_widths: one int32 at ObjectFifo's default depth=2 (rows/rows-chunked mode)
SM_VEC_LEN = 64


def unchunked_l1(cols):
    """in1 and out, each depth 2, each `cols` bf16 -- the buffers the S<=8063 arithmetic counts."""
    return 4 * (cols * 2) + STACK + SCRATCHPAD_PARAM


def chunked_l1(segment, rows_per_core):
    """Same two fifos at `segment`, plus the per-row running state the three passes carry."""
    return (
        4 * (segment * 2)
        + rows_per_core * SM_VEC_LEN * 4   # the 64-lane f32 exp accumulator, one per row
        + rows_per_core * 4                # the running max, one per row
        + STACK
        + SCRATCHPAD_PARAM
    )


def chunked_rows_l1(segment, rows_per_core):
    """`chunked_l1`, minus the scratchpad scalar, plus `of_widths` (rows composition): the
    causal-mask width is now a per-row fifo element instead of one scalar per core, and the
    rows composition skips the write-RTP buffer entirely -- see design.py's `per_row_vector_size`
    branch in `_softmax_chunked`."""
    return (
        4 * (segment * 2)
        + rows_per_core * SM_VEC_LEN * 4   # the 64-lane f32 exp accumulator, one per row
        + rows_per_core * 4                # the running max, one per row
        + WIDTH_FIFO
        + STACK
    )


def l1_sweep(rows, cols_split, segment, ladder):
    rows_per_core = rows // cols_split
    print(f"=== L1 per core, rows={rows} on {cols_split} columns ({rows_per_core} rows/core), "
          f"segment={segment} ===")
    print(f"{'S':>8}  {'unchunked B':>12} {'fits':>5} {'core BD words':>14} {'BD ok':>6}   "
          f"{'chunked B':>10} {'fits':>5} {'core BD words':>14} {'BD ok':>6}   "
          f"{'chunked+rows B':>14} {'fits':>5}")
    for S in ladder:
        u, c, cr = unchunked_l1(S), chunked_l1(segment, rows_per_core), chunked_rows_l1(segment, rows_per_core)
        ub, cb = S * 2 // 4, segment * 2 // 4
        print(f"{S:>8}  {u:>12} {'yes' if u <= L1_BYTES else 'NO':>5} {ub:>14} "
              f"{'yes' if ub <= BD_MAX_WORDS else 'NO':>6}   "
              f"{c:>10} {'yes' if c <= L1_BYTES else 'NO':>5} {cb:>14} "
              f"{'yes' if cb <= BD_MAX_WORDS else 'NO':>6}   "
              f"{cr:>14} {'yes' if cr <= L1_BYTES else 'NO':>5}")
    print(f"(L1 budget {L1_BYTES} B, aie.dma_bd max {BD_MAX_WORDS} 32-bit words. The chunked "
          f"columns are constant in S -- that is the whole result.)")


def build_check(rows, cols_split, segment, ladder, chunked=True, causal_rows=False):
    import os
    from iron.common import AIEContext
    from iron.operators.softmax.op import Softmax

    # Bind the target instead of letting op.name/op.compile() probe the runtime (which OPENS
    # /dev/accel even for a compile-only build -- kb: method-keep-a-compile-only-build-off-the-device-lock).
    if os.environ.get("AIE_DEVICE"):
        import aie.utils as aie_utils
        from aie.iron.device import from_name

        aie_utils.set_current_device(from_name(os.environ["AIE_DEVICE"], n_cols=None))

    mode = f"segment={segment}" if chunked else "unchunked"
    mode += "+rows" if causal_rows else ""
    print(f"\n=== device-free aiecc placement, {mode}, rows={rows}, {cols_split} columns ===")
    for S in ladder:
        with tempfile.TemporaryDirectory(dir=os.environ.get("IRON_TEST_BUILD_ROOT")) as td:
            kw = dict(rows=rows, cols=S, num_aie_columns=cols_split, num_channels=1,
                      context=AIEContext(build_dir=Path(td)))
            if causal_rows:
                kw["vector_size_source"] = "rows"
            else:
                kw["rtp_vector_size"] = S
                kw["vector_size_parameter"] = "sm_mask"
            if chunked:
                kw["segment"] = segment
            try:
                op = Softmax(**kw)
                op.compile()
            except Exception as e:  # aiecc failures surface as a generic build error
                msg = " | ".join(
                    ln.strip() for ln in str(e).splitlines()
                    if ln.strip() and ("error" in ln.lower() or "exceed" in ln.lower())
                )[:400]
                print(f"  S={S:>7}: FAILED  {msg or str(e).splitlines()[0][:200]}")
                continue
            print(f"  S={S:>7}: PLACED  ({op.name})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)        # gemma4-12b n_q_heads
    ap.add_argument("--cols", type=int, default=8)         # sp.softmax_cols(COLS)
    ap.add_argument("--segment", type=int, default=1024)
    ap.add_argument("--ladder", type=int, nargs="*",
                    default=[8192, 16384, 32768, 65536, 131072, 262144])
    ap.add_argument("--unchunked", action="store_true", help="place the current op instead")
    ap.add_argument("--rows-mode", action="store_true",
                     help="vector_size_source='rows' instead of the shared scratchpad scalar")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    l1_sweep(a.rows, a.cols, a.segment, [6912, 7936] + a.ladder)
    if not a.no_build:
        try:
            build_check(a.rows, a.cols, a.segment, a.ladder, chunked=not a.unchunked,
                        causal_rows=a.rows_mode)
        except ImportError as e:
            print(f"=== build check SKIPPED: toolchain env not active ({e}) ===", file=sys.stderr)
