# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel decode SwiGLU MLP: every core runs every stage on its own 1/N slice, unlike
fuse/mlp-block's spatial 5-core PIPELINE (measured +45% slower -- one core streamed all 12.58 MB
of gate+up weights while 27 of 32 cores sat idle). Fusion stays TEMPORAL (one aie.device, one
aiex.configure); parallelism is SPATIAL, across N cores, at every stage:

    x1  = cur + a                             every core, full D (replicated -- cheap, 2 KB)
    hf  = RMSNorm_weighted(x1, n_pf)          every core, full D (replicated -- avoids a reduction)
    g   = Wg[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    u   = Wu[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    gh  = silu(g) * u                         core c's own FF/N slice
    -- ALL-GATHER gh (the one unavoidable exchange -- down's K is the full FF) --
    d   = Wd[c*D/N:(c+1)*D/N] @ gh            core c's own D/N output rows
    nxt[c*D/N:(c+1)*D/N] = x1[same] + d       core c's own D/N output rows

N compute tiles means N cores each with the SAME 2-input/2-output DMA-channel budget the
single-core fuse/mlp-block attempt tripped over. Three consolidations make N cores fit it:

  MISC (1 in): cur, a, n_pf are D-shaped, needed once each. The all-gathered gh is FF-shaped, but
  FF is a whole multiple of D (ratio R = FF/D), so it comes back as R separate D-sized reads on
  the SAME channel and is reassembled by an explicit-offset copy kernel. depth=2 lets the core
  hold cur+a simultaneously (`.acquire(2)`) for the first add; every other use is one at a time.
  Broadcast to all N cores via N `.cons()` handles on one producer -- fan-out is in the
  stream-switch fabric, not the source's own DMA (the mechanism fuse/mlp-block's P1 already uses
  to feed both P2 and P3 from one output port).

  WEIGHT (1 in): Wg, Wu (row width D) and Wd (row width FF) are three different L3 buffers, but
  the shared ObjectFifo tile is sized so ONE flat shape serves all three: TSI_GU rows of D
  elements == TSI_D rows of FF elements (TSI_D = TSI_GU // R). Reused sequentially for Wg's tiles,
  then Wu's, then Wd's -- the same "same fifo, several fill() calls" idiom fuse/mlp-block's
  gate/up A-tile stream already uses.

  OUTPUT (1 of the 2 available, one spare): gh's own FF/N slice is emitted in R D/N-sized chunks
  (an offset-mul kernel reading straight out of the g/u buffers, no separate gh-sized scratch) onto
  the SAME small ObjectFifo the final residual reuses for nxt -- R+1 sequential produce/drain
  rounds share one channel.

The gather has no native all-to-all primitive: ObjectFifoLink is many-to-one XOR one-to-many,
never both (confirmed against attn_core's identical wall in this same codebase). It round-trips
through an internal DRAM scratch buffer instead: each core drains its own gh slice in R chunks
(N*R small, disjoint, offset-addressed drains -- no join, no memtile fan-in, since at n_aie_rows=1
each core already owns a whole column's slice), then every core re-reads the FULL scratch buffer
back over the MISC channel. Cost is trivial (2*FF elements moved, ~0.03% of the layer's traffic);
correctness rests only on a TaskGroup barrier separating the drains from the refill.

MEASURED (device-free, aiecc placement): n_aie_rows=1 (N=n_aie_cols<=8) places with plain flat
per-core ObjectFifos -- no explicit MemTile step needed, the automatic placer inserts whatever
staging one column needs (it column-major-fills 4 rows before moving to the next column, so N=8
lands on physical columns 0-1, not 0-7, and that is fine -- the design never assumes a "logical
core c" is "physical column c"). Both N=16 and N=32 fail there: aiecc's error is explicit --
"no ShimNOCTile ... free: all 8 ShimNOCTile(s) are at 16/16 input... channels used" -- the
DEVICE-WIDE ShimDMA budget (16, matching get_shim_dma_limit()) is spent one channel per DISTINCT
shim-facing ObjectFifo, not "2 per tile" as the compute-tile figure might suggest; misc(1) +
weight(N) already exceeds it at N=16.

n_aie_rows>1 fixes this with the SAME two combinators fuse/mlp-block's own report cites as the
proven multi-row pattern in this codebase (whole_array_silu_iron.py's A-split / C-join): per
GROUP of n_aie_rows cores sharing one shim source,
  - WEIGHT: one group-level ObjectFifo (n_aie_rows*WTILE_UNITS per fill) is `.split()` into
    n_aie_rows row sub-fifos (WTILE_ty each) at a MemTile. One fill per weight-tile ROUND gathers
    all n_aie_rows rows' data for that round with a strided TAP (rows are NOT adjacent in Wg/Wd's
    own row-major layout -- consecutive rows in one group are FF_PER_CORE, resp. D_PER_CORE,
    elements apart), correct because ObjectFifoLink's own offsets place them contiguously in the
    fetched tile in row order, matching the `.split()` offsets below.
  - OUTPUT: one group-level ObjectFifo is `.join()` from n_aie_rows row sub-fifos (DPC_ty each);
    cores write into the row sub-fifo exactly as at n_aie_rows=1. Draining gh needs the same
    strided TAP (destination rows are FF_PER_CORE apart) since the join's own buffer is
    contiguous by row; draining the final nxt round does not (D_PER_CORE apart on both sides).
This only reduces the number of DISTINCT shim-facing ObjectFifos from N to n_aie_cols for both
weight and output -- misc is already 1 regardless of n_aie_rows (see the MISC paragraph above).

FUSE_O (fuse_o=True, n_aie_rows=1 only -- see below): folds the attention output projection
`a = Wo @ cx` into this design too, deleting a whole standalone GEMV design/configure/run from the
decode runlist. `a` was the ONLY external input this design didn't already compute on-chip; now
`cx` (QD-wide) and `Wo` ([D, QD]) arrive instead, and every core computes its own D/N slice of `a`
from its own row-slice of Wo, exactly like the existing Wd/d_buf step. Two wrinkles this adds:

  cx reassembly: QD is a whole multiple of D (R_CX = QD/D), so cx arrives as R_CX D-sized misc
  broadcasts and is reassembled by the SAME explicit-offset-copy idiom gh already uses -- no new
  channel, just more rounds through the existing one.

  Wo's row-tile CANNOT share Wg/Wu/Wd's byte-identical WTILE_ty tile cleanly: the shared tile size
  is forced to lcm(D, FF, QD) = 6*D (FF=3D, QD=2D here), which makes TSI_O = 6*D/QD a multiple of
  3, and D_PER_CORE = D/N a power of two for every N this codebase places -- a multiple of 3 can
  never divide a power of two, so a uniform TSI_O-row tiling of D_PER_CORE always leaves a
  remainder, for ANY N. Reusing the channel anyway with a "short" final fill was rejected: nothing
  in this codebase does a partial-tile fill into a fixed-shape ObjectFifo object (see
  qkv_head_dp/design.py's identical refusal, "a D-wide tile would need a 128-of-1024 partial fill
  ... which nothing in this codebase does"), and a dedicated second weight channel for Wo is a
  non-starter on ITS OWN merits: misc(1)+weight(N)+weight_o(N) is 17 input channels at N=8, one
  over the same 16-channel device-wide budget that already caps this design at N=8 (see above) --
  confirmed independently by attn_core's fusion attempt, which hit exactly this wall trying to
  fold op_o in elsewhere ("all 8 ShimNOCTile(s) are at 9/16 input, 16/16 output channels used").

  The fix is a 1-row OVERLAP, not a partial fill: every core reads ceil(D_PER_CORE/TSI_O) FULL
  TSI_O-row tiles (a window of N_O_TILES*TSI_O rows, >= D_PER_CORE), starting at its own
  c*D_PER_CORE offset in Wo. For every core but the last this window simply reads a few of the
  NEXT core's real rows too (harmless -- Wo is read-only, and the extra rows are computed but
  never drained). Only the LAST core's window would run past Wo's true D rows, so Wo is padded
  with O_OVERLAP (< TSI_O) zero rows at the very end -- a single, tiny, explicit append, not an
  assumption about stale buffer contents. Every fill is a full, byte-identical WTILE_ty tile,
  identical in shape to the existing Wg/Wu/Wd fills; only the LAST core's window ever touches a
  padding row, and that row's own output (computed, never drained) is exactly zero. The per-core
  matvec output lands in a plain (n_aie_rows=1-scoped) O_WINDOW-sized scratch buffer, not
  ObjectFifo-backed, so the overlap/pad tail costs nothing beyond that buffer's own bytes; only
  its first D_PER_CORE elements -- this core's real slice -- are drained.

  `a`'s own all-gather reuses gh's exact mechanism: each core drains its D_PER_CORE-wide real
  slice onto the shared OUTPUT channel (now a NEW, first round ahead of gh's own R rounds), then
  every core re-reads the full D-wide result over MISC once the drains are barriered -- structured
  as its own TaskGroup pair (tg_a_drain/tg_a_refill) ahead of the existing gh pair, because this
  core now produces its FIRST output (the a-slice drain) before consuming cur/n_pf, not after (see
  op.py's Runtime docstring for why a stale two-group split would deadlock here).

GH_CHUNKS (gh_chunks>1, quantized weights only, n_aie_rows=1, fuse_o/post_norm off -- T2.1a,
g4-t2-mlp-l1-floor): at Gemma-4-12B's shape the UNCHUNKED down projection does not fit L1 at any
weight_dtype -- `gh` alone (FF-wide) is 47% of the 64 KB budget, and a full Wd row is most of the
rest. Only gh_chunks == R = FF/D is derived: it makes a chunk-row of Wd exactly WROW_D bytes, the
SAME size as a Wg/Wu row at the same TSI_GU, so the down projection's chunks ride weight_ofs[c]'s
EXISTING channel/tile -- no new shim-facing ObjectFifo, which the 16-channel device-wide budget has
no room for at cols=8 (see the MEASURED paragraph above). This also frees TSI_GU from the
`% R == 0` coupling the unchunked path needs (Wd's chunk-row is byte-identical to a Wg/Wu row
regardless of TSI_GU), so TSI_GU=1 is legal here and is what makes the shape fit at all.

The down-projection loop (core_fn) nests chunk-outer / row-batch-inner: for each of gh_chunks
rounds, gh's CHUNK_WIDTH(==D)-wide slice is read ONCE (one misc acquire, matching the unchunked
path's own R misc reads exactly) and reused across every TSI_GU-row batch of THAT chunk's Wd
weight, accumulating into a persistent f32 `acc` (mv_quant_taccum.cc's zero/accumulate/finish,
mv_taccum.cc's pattern applied to mv_quant.cc's dequant). Total gh and weight bytes moved are
unchanged from the unchunked path -- only gh's and Wd's PER-FETCH L1 footprint shrinks, from
FF-wide/one-full-row to CHUNK_WIDTH-wide/one-chunk-row. The opposite loop nesting (row-outer,
chunk-inner, reusing a fully-resident weight tile across chunks) does not save L1: gh's chunks
would then all need to be simultaneously resident for the whole row loop, R separate CHUNK_WIDTH
buffers being the same total bytes as one FF-wide buffer.

ROW_PARALLEL_DOWN (row_parallel_down=True, quantized+header_first weights only, n_aie_rows=1,
fuse_o off, post_norm REQUIRED, gh_chunks=1): GH_CHUNKS still holds gh FF-wide in spirit -- every
core re-reads the FULL all-gathered gh, just in smaller pieces. This axis removes the all-gather
instead: each core keeps ONLY its own FF/N slice of gh (no gh_scratch round trip at all) and
computes a PARTIAL, FULL-D-wide down-projection output from it -- a column-slice of Wd
(`quantize_weight_chunked(Wd, ..., n_chunks=N)`, N independently-quantized [D, FF/N] blocks, one
per core) against that local gh, chunked over D (`d_chunks` rounds of D_CHUNK=D/d_chunks each,
this build's D_CHUNK forced equal to D_PER_CORE so a chunk's finished result is byte-identical to
a normal output tile). The N cores' partials are summed by the AIE hardware CASCADE (put_mcd/
get_scd, cascade_reduce_f32.cc): a linear chain core[0]->core[1]->...->core[N-1] (`CascadeFlow`,
placed by aiecc's own cascade-adjacency search -- no explicit `Tile()` pinning needed), core[0]
put-only, core[N-1] get-only. Only core[N-1] ends up holding the full sum; it converts to bf16
(mv_quant_taccum.cc's `finish`) straight into its own out-fifo tile and drains it, reusing
post_norm's EXISTING raw-`d` gh_scratch round trip to broadcast the result back to every core --
which is why post_norm=True is required: post_norm=False has no such round trip to repurpose, and
building one is out of scope here. Wd's column-shard format is a DIFFERENT total byte count than
the row-sharded format (N separate per-block quant headers instead of one row-wide header), so its
arg-spec size and object name both carry the row_parallel_down axis.
"""

from ml_dtypes import bfloat16
import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.iron import (
    Buffer, CascadeFlow, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

from iron.operators._trace import maybe_enable_trace

# Shared weight-tile row counts (see module docstring, WEIGHT channel). Fixed, not searched: this
# design is gated at Qwen3-0.6B's D=1024/FF=3072 (R=3) shape only, and 6/2 is verified below to
# fit L1 at every N in {8, 16, 32} this file is built against.
TSI_GU = 6
TSI_D = 2


def _flat_tap(total, size, offset=0):
    """A contiguous [offset:offset+size) read/write into an L3 buffer of `total` elements.
    `total` is the FULL buffer's own declared size (TensorAccessPattern validates offset+extent
    against it), which is why a bare (size,) tensor_dims -- correct only at offset=0 -- silently
    rejects every sliced fill/drain this design needs once `total` differs from `size`."""
    return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def _group_tap(total, offset, n_rows, row_stride, run_hi, run_lo):
    """A group-of-`n_rows` gather/scatter: row r's `run_hi*run_lo`-element contiguous run sits
    `r*row_stride` elements apart in the L3 buffer, but CONTIGUOUS (row order) in the L2/L1 tile
    on the other end of the ObjectFifoLink -- exactly what `.split()`/`.join()`'s own `offsets=`
    (row r at r*run_hi*run_lo in the fetched/joined tile) assume. `run_hi*run_lo` splits a
    per-row run that exceeds the shim's 1023-element wrap cap into two dims (see _split_run) --
    at n_rows==1 this degenerates to _flat_tap's own [0,1,1,size] shape."""
    return TensorAccessPattern(
        (1, total), offset, [1, n_rows, run_hi, run_lo], [0, row_stride, run_lo, 1]
    )


def _split_run(total, lim=1023, gran=2):
    """Largest (hi, lo) with hi*lo == total, lo <= lim, lo a multiple of `gran` -- the shim BD
    4-dim wrap-size cap (mlir-aie's verifyStridesWraps), same constraint gemv/design.py's own
    split_run guards. Raises if no such split exists."""
    for lo in range(lim - (lim % gran), 0, -gran):
        if total % lo == 0:
            return total // lo, lo
    raise ValueError(f"{total} has no wrap-legal split (lim={lim}, gran={gran})")


def my_swiglu_mlp_dp(
    dev, D, FF, epsilon=1e-5, stack_size=0x800, func_prefix="", n_aie_cols=8, n_aie_rows=1,
    QD=None, fuse_o=False, trace_size=0, weight_dtype="bf16", group_size=0,
    weight_depth=2, tile_rows_gu=None, fifo_prefix="", parts_only=False, split_gh=1,
    act="silu", post_norm=False, gh_chunks=1,
    layout="header_first", row_group=None, scale_dtype="f32",
    row_parallel_down=False, d_chunks=1,
):
    """`func_prefix` is required (not optional) by iron.common.sequence.FusedDispatch the moment
    this design is placed in an OperatorSequence -- see gemv/design.py's identical parameter for
    the same reason. N = n_aie_cols * n_aie_rows; n_aie_rows=1 is the plain-ObjectFifo topology,
    n_aie_rows>1 uses the MemTile split/join topology -- see the module docstring for both.

    `fuse_o=True` folds `a = Wo @ cx` into this design (see module docstring's FUSE_O section);
    it needs `QD` (the attention context width) and is currently n_aie_rows==1 only -- the
    overlap/pad arithmetic below is derived for the plain per-core-direct-fill topology and has
    not been re-derived for the MemTile split/join one.

    `act` picks the gated-FFN nonlinearity: "silu" (default, byte-identical to every prior build)
    or "gelu_tanh" (Gemma). `silu_tile_bf16` (aie2p/silu.cc) and `gelu_tile_bf16` (aie2p/gelu.cc)
    share one ABI -- `(uint32_t n, bfloat16 *restrict c)`, in place -- so this is a SYMBOL SWAP:
    core_fn's call site is unchanged, only which object op.py compiles and which name design.py
    binds move. `_ACT_TAG` is empty at the default so every existing archive/operator name is
    unchanged; see op.py's `get_kernel_artifacts` for the matching object-name half of this.

    `post_norm=True` adds the sandwich post-block norm slot (Gemma): a weighted RMSNorm
    application to the all-gathered, full-D down-projection output `d` before the final residual,
    and -- only under `fuse_o=True` -- the SAME to the all-gathered `a` before `x1 = cur + a`.
    `a`'s gain (pa_gain) feeds x1_buf, replicated full-D on every core, so it reuses
    `wnorm_kernel` (D-wide gain, already bound for the pre-FFN norm) unchanged. `d`'s gain
    (pff_gain) only ever multiplies THIS core's own D_PER_CORE output slice, so it skips
    `wnorm_kernel` entirely: `d` is normalised unweighted at full D (`rms_norm_bf16_vector`), then
    the gain is read straight off its misc tile for a narrow multiply (`mul_pn_kernel`) -- an L1
    saving, since a D-wide gain buffer per core does not fit Gemma-4's shape. `d` has no existing
    all-gather (unlike `gh`/`a`), so one is added here, reusing
    `gh_scratch`'s own L3 allocation as scratch (dead for the rest of this call once Wd's fill has
    been consumed) rather than adding a 16th/17th host buffer argument -- see decode_layer_dp's own
    HOST-BO accounting. The new gain(s) ride ONE packed `post_norms` argument (mirrors
    attn_block_dp's `norms_packed`): `[pff]` at D elements, or `[pa | pff]` at 2*D under fuse_o.
    Default False keeps the arg list, archive and emitted MLIR byte-identical.

    `row_parallel_down=True` swaps the down projection for the ROW_PARALLEL_DOWN scheme (module
    docstring): no gh all-gather, a per-core column-shard of Wd, and a cascade sum across all N
    cores in place of gh_chunks' K-chunked local reduction. `d_chunks` splits that sum's D axis
    into rounds (this build requires d_chunks == N, one D_PER_CORE-wide round per core). Default
    False keeps the arg list, archive and emitted MLIR byte-identical.
    """
    assert act in ("silu", "gelu_tanh"), f"unknown act {act!r}"
    # Local shadowing of the module defaults, so an arm can trade tile ROWS against fifo DEPTH at
    # constant L1: depth * TSI_GU * D * 2 bytes is what the budget below actually sees.
    TSI_GU = tile_rows_gu if tile_rows_gu else globals()["TSI_GU"]
    # TSI_D (Wd's own row-batch size) only exists on the UNCHUNKED path, where it derives from
    # TSI_GU via the shared-tile invariant below. gh_chunks>1 batches Wd BY TSI_GU directly (a
    # chunk-row is already WROW_D bytes, same as a Wg/Wu row -- see the T2.1a block further down),
    # so TSI_GU is not forced to be a multiple of R there.
    if gh_chunks == 1 and not row_parallel_down:
        TSI_D = TSI_GU // (FF // D)
        assert TSI_D >= 1 and TSI_GU % (FF // D) == 0, (
            f"tile_rows_gu={TSI_GU} must be a multiple of R=FF/D={FF // D}")
    else:
        # gh_chunks>1 batches Wd rows by TSI_GU directly (see below); row_parallel_down doesn't
        # use TSI_D or the old down-projection path at all, so the R-divisibility coupling above
        # is not needed either.
        TSI_D = None

    N = n_aie_cols * n_aie_rows
    assert FF % D == 0, f"this design assumes FF ({FF}) is a whole multiple of D ({D})"
    R = FF // D  # =3 at Qwen3-0.6B's shape; also N_GH_CHUNKS (misc) and N_GH_ROUNDS (output)
    assert D % N == 0 and FF % N == 0, f"D={D}, FF={FF} must both be divisible by N={N}"
    D_PER_CORE = D // N
    FF_PER_CORE = FF // N
    assert FF_PER_CORE % 32 == 0, (
        f"both silu_tile_bf16 and gelu_tile_bf16 walk their buffer 32 lanes at a time with no "
        f"tail handling; FF/N ({FF_PER_CORE}) must be a multiple of 32"
    )
    # WEIGHT WIRE UNITS. bf16 weights are addressed in ELEMENTS; a group-quantized weight is a flat
    # byte row -- [n_groups x f32 scale][packed payload] for the symmetric dtypes, and
    # [n_groups x bf16 scale][n_groups x bf16 min][packed payload] for the affine ones ("int4a" /
    # "int8a") -- so every weight size, offset and stride
    # below is in whatever unit the wire format uses. Activations (hf, gh, nxt, cx) are ALWAYS bf16
    # and keep their element units; mixing the two is exactly the bytes-vs-elements seam that has
    # no owner, so the weight quantities are named WROW_* and nothing else changes.
    if weight_dtype == "bf16":
        WDT, WUNIT = bfloat16, 2
        WROW_D, WROW_FF = D, FF
        WROW_QD = QD
    else:
        from iron.common.quant import row_stride_bytes
        assert weight_dtype in ("int4", "int8", "int4a", "int8a"), \
            f"unknown weight_dtype {weight_dtype!r}"
        assert group_size > 0, "weight_dtype != 'bf16' needs an explicit group_size > 0"
        assert n_aie_rows == 1, (
            "quantized weights are only derived for the plain (n_aie_rows=1) topology -- the "
            "MemTile split/join TAPs below still carry element strides"
        )
        for name, K in (("D", D), ("FF", FF)) + ((("QD", QD),) if fuse_o else ()):
            assert K % group_size == 0, f"{name}={K} must be a whole number of groups ({group_size})"
        WDT, WUNIT = np.int8, 1
        WROW_D = row_stride_bytes(D, group_size, weight_dtype, scale_dtype)
        WROW_FF = row_stride_bytes(FF, group_size, weight_dtype, scale_dtype)
        WROW_QD = (row_stride_bytes(QD, group_size, weight_dtype, scale_dtype)
                   if fuse_o else None)
        # A planar row's groups sit row_group*payload apart, so an L1 tile must hold a whole
        # number of blocks -- the same constraint GEMV enforces on tile_size_input.
        if layout == "row_group_planar":
            assert row_group and row_group >= 1, (
                f"layout='row_group_planar' needs a derived row_group (got {row_group!r})")

    if gh_chunks == 1 and not row_parallel_down:
        # The shared-tile invariant is what lets Wg/Wu (row width D) and Wd (row width FF) ride
        # ONE ObjectFifo. It survives quantization because row_stride_bytes is affine in K with
        # the same group size, so the R = FF/D ratio is preserved: at int4 g128,
        # 6*544 == 2*1632 == 3264 B.
        assert TSI_GU * WROW_D == TSI_D * WROW_FF, (
            f"shared weight tile must be identical for gate/up and down: "
            f"{TSI_GU}*{WROW_D} != {TSI_D}*{WROW_FF} (weight_dtype={weight_dtype})"
        )
        assert FF_PER_CORE % TSI_GU == 0 and D_PER_CORE % TSI_D == 0, (
            f"N={N}: FF/N ({FF_PER_CORE}) must divide by TSI_GU ({TSI_GU}) and "
            f"D/N ({D_PER_CORE}) must divide by TSI_D ({TSI_D})"
        )
        N_D_TILES = D_PER_CORE // TSI_D
    else:
        # A chunk-row of Wd is WROW_D bytes by construction (K=D, same as a Wg/Wu row) -- no
        # separate invariant to assert; N_D_TILES_CHUNK (below) is the chunked path's analogue.
        # row_parallel_down has its OWN shared-tile invariant (WROW_FFPC/TSI_RP, derived below)
        # and never uses N_D_TILES/mv_d_kernel at all.
        assert FF_PER_CORE % TSI_GU == 0, (
            f"N={N}: FF/N ({FF_PER_CORE}) must divide by TSI_GU ({TSI_GU})"
        )
        N_D_TILES = None
    N_GU_TILES = FF_PER_CORE // TSI_GU
    WTILE_UNITS = TSI_GU * WROW_D

    # T2.1a (g4-t2-mlp-l1-floor): chunk the down projection's K (=FF) reduction into `gh_chunks`
    # rounds so gh and Wd's weight tile never hold more than one chunk. Only gh_chunks == R is
    # derived: that makes a chunk-row of Wd exactly WROW_D bytes, byte-identical to a Wg/Wu row at
    # the same TSI_GU, so the chunk rides weight_ofs[c]'s EXISTING tile/channel -- a different
    # divisor of R would need its own weight channel, which the 16-channel device-wide shim budget
    # does not have room for at cols=8 (misc(1)+weight_gu(N)+weight_d(N) already exceeds it).
    if gh_chunks > 1:
        assert weight_dtype in ("int4", "int8"), (
            f"gh_chunks>1 needs weight_dtype in ('int4','int8'), not {weight_dtype!r} -- a bf16 "
            f"Wd row alone is already most of L1, and mv_quant_taccum.cc has no affine form"
        )
        assert not fuse_o and not post_norm and n_aie_rows == 1, (
            "gh_chunks>1 is only derived for fuse_o=False, post_norm=False, n_aie_rows=1 -- "
            "post_norm's raw-d all-gather (step 6b/7a/7b below) is not re-derived against an "
            "f32 acc_buf finish target")
        assert gh_chunks == R, (
            f"gh_chunks={gh_chunks} must equal R=FF/D={R}: chunk width FF/gh_chunks must equal D "
            f"exactly for the chunk-row/Wg-row byte-size coincidence above to hold"
        )
        assert D_PER_CORE % TSI_GU == 0, (
            f"gh_chunks>1 batches Wd rows by TSI_GU ({TSI_GU}), not TSI_D; D/N ({D_PER_CORE}) "
            f"must divide by it"
        )
    CHUNK_WIDTH = D if gh_chunks > 1 else FF  # == FF/gh_chunks; D is asserted equal above
    N_D_TILES_CHUNK = D_PER_CORE // TSI_GU if gh_chunks > 1 else None

    # ROW_PARALLEL_DOWN (module docstring): Wd's column-shard tile shares weight_ofs[c] with
    # Wg/Wu by the same byte-matching trick FUSE_O uses for Wo (TSI_O there, TSI_RP here) --
    # WTILE_UNITS must be a whole number of column-shard rows.
    if row_parallel_down:
        assert weight_dtype in ("int4", "int8"), (
            f"row_parallel_down needs weight_dtype in ('int4','int8'), not {weight_dtype!r} -- "
            f"mv_quant_taccum.cc (the local matvec this reuses) has no bf16/affine form"
        )
        assert layout == "header_first", (
            "row_parallel_down needs layout='header_first' -- mv_quant_taccum.cc computes row "
            "stride as header-then-payload and has no row_group_planar form"
        )
        assert not fuse_o and post_norm and n_aie_rows == 1 and gh_chunks == 1, (
            "row_parallel_down is only derived for fuse_o=False, post_norm=True, n_aie_rows=1, "
            "gh_chunks=1 -- post_norm=True is REQUIRED (not forbidden, unlike gh_chunks): the "
            "cascade-summed result reaches every core by reusing post_norm's own raw-d gh_scratch "
            "round trip, and post_norm=False has no such round trip to repurpose"
        )
        WROW_FFPC = row_stride_bytes(FF_PER_CORE, group_size, weight_dtype, scale_dtype)
        assert WTILE_UNITS % WROW_FFPC == 0, (
            f"row_parallel_down needs the shared weight tile ({WTILE_UNITS} units) to be a whole "
            f"number of Wd column-shard rows ({WROW_FFPC} units each); it isn't, so Wd's shard "
            f"can't share this channel -- try a different tile_rows_gu"
        )
        TSI_RP = WTILE_UNITS // WROW_FFPC
        assert D % d_chunks == 0, f"D={D} must be a whole multiple of d_chunks={d_chunks}"
        D_CHUNK = D // d_chunks
        assert d_chunks == N, (
            f"d_chunks={d_chunks} must equal N={N} in this build -- each round's finished chunk "
            f"drains through a core's existing D_PER_CORE-wide out-fifo tile with no partial-fill "
            f"support, which forces D_CHUNK ({D_CHUNK}) == D_PER_CORE ({D_PER_CORE}); a finer "
            f"d_chunks needs an accumulate-into-an-offset step this build does not add"
        )
        assert D_CHUNK % TSI_RP == 0, (
            f"row_parallel_down batches a column-shard's rows by TSI_RP ({TSI_RP}); "
            f"D_CHUNK ({D_CHUNK}) must divide by it"
        )
        N_RP_TILES = D_CHUNK // TSI_RP

    if fuse_o:
        assert n_aie_rows == 1, "fuse_o is only derived for the plain (n_aie_rows=1) topology"
        assert QD is not None, "fuse_o needs QD (the attention context width)"
        assert QD % D == 0, f"fuse_o assumes QD ({QD}) is a whole multiple of D ({D})"
        R_CX = QD // D
        assert WTILE_UNITS % WROW_QD == 0, (
            f"fuse_o needs the shared weight tile ({WTILE_UNITS} units) to be a whole number of "
            f"Wo rows ({WROW_QD} units each); it isn't, so Wo can't share this channel"
        )
        TSI_O = WTILE_UNITS // WROW_QD
        N_O_TILES = -(-D_PER_CORE // TSI_O)          # ceil division
        O_WINDOW = N_O_TILES * TSI_O                 # rows actually read per core (>= D_PER_CORE)
        O_OVERLAP = O_WINDOW - D_PER_CORE             # extra rows read past this core's own slice
        assert O_OVERLAP < TSI_O                      # ceil() guarantees this; sanity check
        WO_ROWS_PADDED = D + O_OVERLAP                # Wo's own arg spec size, in rows

    if post_norm:
        # The post-attn-norm arm reuses fuse_o's own all-gathered `a` (see the docstring); its
        # pairing/derivation is only done for n_aie_rows==1, same restriction as fuse_o itself.
        assert n_aie_rows == 1, "post_norm is only derived for the plain (n_aie_rows=1) topology"
        # POST_NORMS packs the new gain(s): [pff] at D, or [pa | pff] at 2*D under fuse_o. Mirrors
        # attn_block_dp's norms_packed NORMS blob (D + 2*HD there).
        POST_NORMS_UNITS = 2 * D if fuse_o else D
        PFF_OFF = D if fuse_o else 0

    # The affine kernels keep one float per quant group on the STACK (mv_quant.cc's
    # `float bsum[n_groups]`, the per-group sums of B). It is the only stack term this design
    # controls, and stack_size is otherwise an opaque constant the budget below just adds -- so
    # size it here rather than let it be a hanging number. Worst case is the widest K, since
    # n_groups = K/group_size: at FF=3072 group_size=32 that is 96 floats = 384 B of the 2048 B
    # default. The 512 B floor left for everything else (two accums, the ones vector, the frame)
    # is a policy, not a measurement; aiecc validates the real requirement against stack_size per
    # core and fails the build if it is short, so this assert exists to fail EARLIER and to name
    # the term, not to be the only guard.
    if weight_dtype in ("int4a", "int8a"):
        bsum_bytes = 4 * (max(K for K in (D, FF) + ((QD,) if fuse_o else ())) // group_size)
        assert bsum_bytes + 512 <= stack_size, (
            f"affine bsum[] needs {bsum_bytes} B of the {stack_size} B core stack at "
            f"group_size={group_size}; raise stack_size or the group"
        )

    # L1 budget check (64 KB/core) -- see module docstring's channel accounting for what each
    # buffer is. Computed, not guessed: this is exactly the "hanging numbers are bugs" rule.
    L1_BYTES = 65536
    # misc_of is declared depth=2, but its mixed acquire(2)/acquire(1) pattern (the cur+a pair vs
    # n_pf, pff_gain+d, ...) makes aiecc's objectFifo lowering allocate a THIRD physical buffer
    # (misc_*_cons_buff_2) UNCONDITIONALLY -- MEASURED (real aiecc buffer dump) on both the plain
    # default build and post_norm=True.
    misc_bytes = 3 * (D * 2)
    weight_bytes = weight_depth * (WTILE_UNITS * WUNIT)
    out_bytes = 2 * (D_PER_CORE * 2)  # depth=2
    # gh_buf costs 0 EXTRA bytes under row_parallel_down (aliased onto g_buf, in place -- see the
    # workers loop), CHUNK_WIDTH otherwise (FF unchunked, D at gh_chunks>1).
    _gh_buf_width = 0 if row_parallel_down else CHUNK_WIDTH
    persistent_bytes = 2 * (D * 2) + (_gh_buf_width * 2) + 2 * (FF_PER_CORE * 2) + (D_PER_CORE * 2)
    # x1_buf + hf_buf         gh_buf                    g_buf + u_buf          d_buf
    if gh_chunks > 1:
        persistent_bytes += D_PER_CORE * 4  # acc_buf: f32, one slot per this core's own output row
    if row_parallel_down:
        persistent_bytes += D_CHUNK * 4     # acc_buf: f32, this round's D-wide partial/sum slot
    if fuse_o:
        persistent_bytes += (QD * 2) + (O_WINDOW * 2)
        # cx_buf                a_slice_buf
    if post_norm:
        # pff_gain only ever multiplies THIS core's own D_PER_CORE output slice (core_fn reads
        # it straight off the misc tile, never buffers it D-wide -- see below); d_norm_buf stays
        # D-wide because the RMSNorm reduction needs the whole vector.
        persistent_bytes += D * 2                      # d_norm_buf
        if fuse_o:
            # pa_gain/a_norm both feed x1_buf, replicated FULL-D on every core for step 2's own
            # full-D reduction -- every element is consumed, so neither shards.
            persistent_bytes += 2 * (D * 2)             # pa_gain_buf + a_norm_buf
    total = misc_bytes + weight_bytes + out_bytes + persistent_bytes + stack_size
    assert total <= L1_BYTES, (
        f"N={N}: estimated L1 use {total} B exceeds {L1_BYTES} B "
        f"(misc={misc_bytes} weight={weight_bytes} out={out_bytes} "
        f"persistent={persistent_bytes} stack={stack_size})"
    )

    D_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    FF_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    DPC_ty = np.ndarray[(D_PER_CORE,), np.dtype[bfloat16]]
    FFPC_ty = np.ndarray[(FF_PER_CORE,), np.dtype[bfloat16]]
    WTILE_ty = np.ndarray[(WTILE_UNITS,), np.dtype[WDT]]
    Wg_L3_ty = np.ndarray[(FF * WROW_D,), np.dtype[WDT]]
    # Wd_L3_ty's TOTAL size is unchanged by gh_chunks: gh_chunks == R (asserted above) makes
    # gh_chunks*D*WROW_D == D*WROW_FF exactly (WROW_D and WROW_FF are the same row_stride_bytes
    # formula at K=D and K=FF=gh_chunks*D, affine in K at one group_size). Only the READ pattern
    # below differs -- the buffer is reinterpreted as gh_chunks independently-packed [D, WROW_D]
    # planar blocks instead of one [D, WROW_FF] block, never resized.
    #
    # row_parallel_down's Wd_L3_ty is a GENUINELY different total: N independently-quantized
    # [D, FF/N] column-shard blocks (quantize_weight_chunked(..., n_chunks=N)) each carry their
    # OWN header, so N*D*WROW_FFPC != D*WROW_FF (unlike gh_chunks' row-sharded total above).
    Wd_L3_ty = (np.ndarray[(N * D * WROW_FFPC,), np.dtype[WDT]] if row_parallel_down
               else np.ndarray[(D * WROW_FF,), np.dtype[WDT]])
    GH_SCRATCH_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    GH_ty = FFPC_ty if row_parallel_down else (D_ty if gh_chunks > 1 else FF_ty)
    if gh_chunks > 1:
        ACC_ty = np.ndarray[(D_PER_CORE,), np.dtype[np.float32]]
    elif row_parallel_down:
        ACC_ty = np.ndarray[(D_CHUNK,), np.dtype[np.float32]]
    if fuse_o:
        QD_ty = np.ndarray[(QD,), np.dtype[bfloat16]]
        OWIN_ty = np.ndarray[(O_WINDOW,), np.dtype[bfloat16]]
        Wo_L3_ty = np.ndarray[(WO_ROWS_PADDED * WROW_QD,), np.dtype[WDT]]
        A_SCRATCH_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    if post_norm:
        POST_NORMS_ty = np.ndarray[(POST_NORMS_UNITS,), np.dtype[bfloat16]]

    # ---- kernels (one archive per core -- every core plays every role) ----
    # The weight dtype AND the activation are IN the archive name. Without them a bf16/silu build
    # silently reuses a cached int4 (or gelu_tanh) archive built earlier under the same name and
    # dies at link with "undefined symbol: <prefix>matvec_vectorized_bf16_bf16" (or the activation
    # symbol) -- the artifact-key collision this tree already documents for the fused sequence
    # name. `_ACT_TAG` is empty at act="silu" so every existing archive name is unchanged.
    _WTAG = "" if weight_dtype == "bf16" else f"_{weight_dtype}g{group_size}"
    _ACT_TAG = "" if act == "silu" else f"_{act}"
    # post_norm adds an extra linked object (the pn_ copy) without changing any EXISTING symbol,
    # so a stale post_norm=False archive under the same name would fail loud (undefined symbol at
    # link) rather than silently -- still exactly the collision class this tag exists to avoid.
    _PN_TAG = "_pn" if post_norm else ""
    _GHC_TAG = f"_ghc{gh_chunks}" if gh_chunks > 1 else ""
    _RPD_TAG = "_rpd" if row_parallel_down else ""
    CORE_ARCHIVE = f"{func_prefix}swiglu_mlp_dp_core{_WTAG}{_ACT_TAG}{_PN_TAG}{_GHC_TAG}{_RPD_TAG}.a"
    # Two DIFFERENT bindings, not one reused: a Kernel() fixes ONE func.func signature for its
    # symbol, and the two call sites acquire differently-sized buffers (x1=cur+a is full-D; the
    # final residual is D/N-sized). The plain add costs nothing extra -- eltwise_add_bf16_vector
    # is already linked into every other design that touches add.cc.
    add_kernel = Kernel(
        f"{func_prefix}eltwise_add_bf16_vector", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.int32]
    )
    add_off_kernel = Kernel(
        f"{func_prefix}eltwise_add_offset_a_bf16_vector", CORE_ARCHIVE,
        [D_ty, DPC_ty, DPC_ty, np.int32, np.int32],
    )
    wnorm_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_fixed", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.float32]
    )
    MV = f"matvec_vectorized_{weight_dtype}_bf16"
    mv_gu_kernel = Kernel(
        f"{func_prefix}{MV}", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, D_ty, FFPC_ty],
    )
    # Down's own matvec: different DIM_K, same extern "C" name as mv_gu_kernel -- symbol
    # uniqueness is device-wide (one aie.device, one symbol table), so op.py compiles this one
    # from a prefixed object (see fuse/mlp-block's identical mv.cc reuse for the same reason).
    mv_d_kernel = Kernel(
        f"{func_prefix}down_{MV}", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, FF_ty, DPC_ty],
    )
    # Activation SYMBOL only moves with `act` -- silu_tile_bf16/gelu_tile_bf16 share one ABI (see
    # the docstring), so this binding is the entire swap.
    _ACT_SYM = "silu_tile_bf16" if act == "silu" else "gelu_tile_bf16"
    act_kernel = Kernel(f"{func_prefix}{_ACT_SYM}", CORE_ARCHIVE, [np.int32, FFPC_ty])
    mul_off_kernel = Kernel(
        f"{func_prefix}eltwise_mul_offset_ab_bf16_vector", CORE_ARCHIVE,
        [FFPC_ty, FFPC_ty, DPC_ty, np.int32, np.int32],
    )
    copy_off_kernel = Kernel(
        f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE, [FF_ty, D_ty, np.int32, np.int32]
    )
    if gh_chunks > 1:
        # T2.1a chunk-accumulate down matvec (mv_quant_taccum.cc): zero/accumulate/finish over
        # gh_chunks rounds, each against a TSI_GU-row batch of ONE chunk's weight bytes -- see
        # core_fn's down-projection section and the module docstring.
        taccum_zero_kernel = Kernel(
            f"{func_prefix}mv_taccum_zero_f32", CORE_ARCHIVE, [np.int32, ACC_ty]
        )
        MVT = f"matvec_taccum_{weight_dtype}_bf16"
        mv_d_taccum_kernel = Kernel(
            f"{func_prefix}{MVT}", CORE_ARCHIVE,
            [np.int32, np.int32, WTILE_ty, D_ty, ACC_ty],
        )
        taccum_finish_kernel = Kernel(
            f"{func_prefix}mv_taccum_finish_bf16", CORE_ARCHIVE, [np.int32, ACC_ty, DPC_ty]
        )
        # `gc_copy_offset_bf16_vector` is a renamed recompile of copy_offset_bf16_vector at the
        # D_ty->D_ty shape gh_chunk_buf needs -- same "own symbol per call-site shape" reasoning
        # as pn_/cx_/oa_ above (op.py's gc_copy_obj, prefix_symbols="gc_").
        gc_copy_kernel = Kernel(
            f"{func_prefix}gc_copy_offset_bf16_vector", CORE_ARCHIVE,
            [D_ty, D_ty, np.int32, np.int32],
        )
    if row_parallel_down:
        # Local zero/accumulate/finish (mv_quant_taccum.cc again, DIM_K=FF_PER_CORE this time --
        # a single, unchunked local reduction, not gh_chunks' multi-round one) plus the
        # cross-core cascade sum (cascade_reduce_f32.cc). Both objects are compiled with their
        # OWN symbol prefix ("rp_") since matvec_taccum_*/mv_taccum_zero_f32/mv_taccum_finish_bf16
        # would otherwise collide with a gh_chunks build at a different DIM_K -- same reasoning as
        # mv_d_kernel's "down_" prefix. gh_mul_kernel replaces the R-way emit (mul_off_kernel) with
        # one plain full-FFPC multiply: no all-gather, so no chunked offsets to write.
        rp_taccum_zero_kernel = Kernel(
            f"{func_prefix}rp_mv_taccum_zero_f32", CORE_ARCHIVE, [np.int32, ACC_ty]
        )
        RPMVT = f"rp_matvec_taccum_{weight_dtype}_bf16"
        rp_mv_taccum_kernel = Kernel(
            f"{func_prefix}{RPMVT}", CORE_ARCHIVE,
            [np.int32, np.int32, WTILE_ty, FFPC_ty, ACC_ty],
        )
        rp_taccum_finish_kernel = Kernel(
            f"{func_prefix}rp_mv_taccum_finish_bf16", CORE_ARCHIVE, [np.int32, ACC_ty, DPC_ty]
        )
        gh_mul_kernel = Kernel(
            f"{func_prefix}eltwise_mul_bf16_vector", CORE_ARCHIVE,
            [FFPC_ty, FFPC_ty, FFPC_ty, np.int32],
        )
        cr_put_only_kernel = Kernel(
            f"{func_prefix}cascade_reduce_put_only_f32", CORE_ARCHIVE, [np.int32, ACC_ty]
        )
        cr_put_get_kernel = Kernel(
            f"{func_prefix}cascade_reduce_put_get_f32", CORE_ARCHIVE, [np.int32, ACC_ty]
        )
        cr_get_only_kernel = Kernel(
            f"{func_prefix}cascade_reduce_get_only_f32", CORE_ARCHIVE,
            [np.int32, ACC_ty, ACC_ty],
        )
    if post_norm:
        # pff_gain is read straight off the misc tile in core_fn (never copied D-wide -- see the
        # step 7 comment there); only the unweighted reduction and the narrow gain multiply need
        # binding here.
        rms_unweighted_kernel = Kernel(
            f"{func_prefix}rms_norm_bf16_vector", CORE_ARCHIVE, [D_ty, D_ty, np.int32, np.float32]
        )
        # Renamed recompile of eltwise_mul_offset_ab_bf16_vector at D_ty/D_ty/DPC_ty -- gh's own
        # binding (mul_off_kernel below) is FFPC_ty/FFPC_ty/DPC_ty, a different shape sharing the
        # unprefixed symbol (see mv_d_kernel's "down_" prefix for why that needs a rename).
        mul_pn_kernel = Kernel(
            f"{func_prefix}pn_eltwise_mul_offset_ab_bf16_vector", CORE_ARCHIVE,
            [D_ty, D_ty, DPC_ty, np.int32, np.int32],
        )
    if fuse_o:
        # o's own matvec: DIM_K=QD, distinct from mv_gu (DIM_K=D) and mv_d (DIM_K=FF) -- same
        # symbol-uniqueness reasoning as mv_d_kernel above.
        mv_o_kernel = Kernel(
            f"{func_prefix}o_{MV}", CORE_ARCHIVE,
            [np.int32, np.int32, WTILE_ty, QD_ty, OWIN_ty],
        )
        if post_norm:
            # pa_gain feeds x1_buf, replicated full-D on every core (see the L1 comment above),
            # so -- unlike pff_gain -- it IS copied out D-wide, before the cur/a pair below.
            copy_pn_kernel = Kernel(
                f"{func_prefix}pn_copy_offset_bf16_vector", CORE_ARCHIVE,
                [D_ty, D_ty, np.int32, np.int32],
            )
        # copy_offset_bf16_vector is (dst, src, size, dst_offset) over raw pointers -- no
        # compile-time size baked in -- but a func.func symbol is keyed by NAME only, and MLIR's
        # verifier refuses two declarations of the same symbol with different memref types
        # ("redefinition of symbol"), so each new call-site shape needs its own renamed object
        # (op.py's cx_copy_obj/oa_copy_obj), exactly like mv_d_kernel's "down_" prefix below.
        copy_off_cx_kernel = Kernel(
            f"{func_prefix}cx_copy_offset_bf16_vector", CORE_ARCHIVE,
            [QD_ty, D_ty, np.int32, np.int32],
        )
        copy_off_a_kernel = Kernel(
            f"{func_prefix}oa_copy_offset_bf16_vector", CORE_ARCHIVE,
            [DPC_ty, OWIN_ty, np.int32, np.int32],
        )

    # ---- ObjectFifos: misc(1, always) + weight(n_aie_cols groups) + output(n_aie_cols groups).
    # n_aie_rows==1: plain per-core ObjectFifos, direct L3<->L1 (no MemTile step in this file --
    # the automatic placer inserts whatever staging one column needs). n_aie_rows>1: one
    # group-level ObjectFifo per column, split (weight) / joined (output) into n_aie_rows row
    # sub-fifos at a MemTile -- see module docstring. Either way `weight_ofs[c]`/`out_ofs[c]`
    # (c = g*n_aie_rows + r) end up as the per-core handles core_fn acquires/releases from; it
    # does not know or care which path built them. fuse_o adds no new shim-facing ObjectFifo: Wo
    # rides the SAME weight_ofs/gweight_ps channel as Wg/Wu/Wd, and `a`'s all-gather rides the
    # SAME out_ofs/gout_cs channel gh's all-gather already uses (see module docstring). ----
    misc_of = ObjectFifo(D_ty, name=f"{fifo_prefix}misc", depth=2)
    weight_ofs = [None] * N
    out_ofs = [None] * N
    if n_aie_rows == 1:
        for c in range(N):
            weight_ofs[c] = ObjectFifo(WTILE_ty, name=f"{fifo_prefix}weight_{c}", depth=weight_depth)
            out_ofs[c] = ObjectFifo(DPC_ty, name=f"{fifo_prefix}out_{c}", depth=2)
        group_weight_ofs = weight_ofs  # sequence() fills/drains these directly, one per "group"
        group_out_ofs = out_ofs
    else:
        RUN_HI, RUN_LO = _split_run(WTILE_UNITS)
        GROUP_WTILE_ty = np.ndarray[(n_aie_rows * WTILE_UNITS,), np.dtype[WDT]]
        GROUP_OTILE_ty = np.ndarray[(n_aie_rows * D_PER_CORE,), np.dtype[bfloat16]]
        group_weight_ofs = []
        group_out_ofs = []
        for g in range(n_aie_cols):
            gw = ObjectFifo(GROUP_WTILE_ty, name=f"{fifo_prefix}weight_g{g}", depth=weight_depth)
            sub_w = gw.cons().split(
                [r * WTILE_UNITS for r in range(n_aie_rows)],
                obj_types=[WTILE_ty] * n_aie_rows,
                names=[f"{fifo_prefix}weight_{g}_{r}" for r in range(n_aie_rows)],
                depths=[2] * n_aie_rows,
            )
            go = ObjectFifo(GROUP_OTILE_ty, name=f"{fifo_prefix}out_g{g}", depth=2)
            sub_o = go.prod().join(
                [r * D_PER_CORE for r in range(n_aie_rows)],
                obj_types=[DPC_ty] * n_aie_rows,
                names=[f"{fifo_prefix}out_{g}_{r}" for r in range(n_aie_rows)],
                depths=[2] * n_aie_rows,
            )
            for r in range(n_aie_rows):
                weight_ofs[g * n_aie_rows + r] = sub_w[r]
                out_ofs[g * n_aie_rows + r] = sub_o[r]
            group_weight_ofs.append(gw)
            group_out_ofs.append(go)

    def core_fn(misc_c, weight_c, out_p,
                x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
                add_k, add_off_k, wnorm_k, mv_gu_k, mv_d_k, act_k, mul_off_k, copy_off_k,
                core_id, *extra):
        idx = 0
        if fuse_o:
            (cx_buf, a_slice_buf, mv_o_k, copy_off_cx_k, copy_off_a_k) = extra[idx:idx + 5]
            idx += 5
            if post_norm:
                (pa_gain_buf, a_norm_buf, copy_pn_k) = extra[idx:idx + 3]
                idx += 3
        if post_norm:
            (d_norm_buf, rms_unweighted_k, mul_pn_k) = extra[idx:idx + 3]
            idx += 3
        if gh_chunks > 1:
            (acc_buf, taccum_zero_k, mv_d_taccum_k, taccum_finish_k, gc_copy_k) = extra[idx:idx + 5]
            idx += 5
        if row_parallel_down:
            (acc_buf, rp_zero_k, rp_taccum_k, rp_finish_k, gh_mul_k,
             cr_put_only_k, cr_put_get_k, cr_get_only_k) = extra[idx:idx + 8]
            idx += 8

        if fuse_o:
            if post_norm:
                # step -1a: pa_gain (post-attn-norm gain), broadcast full-D, copied into a
                # persistent buffer immediately so its misc slot is free before `pair` below --
                # the two are never held open concurrently (stays inside misc_of's depth=2).
                pa_tile = misc_c.acquire(1)
                copy_pn_k(pa_gain_buf, pa_tile, D, 0)
                misc_c.release(1)

            # step -1: reassemble cx (QD-wide) from R_CX D-sized misc broadcasts.
            for i in range(R_CX):
                chunk = misc_c.acquire(1)
                copy_off_cx_k(cx_buf, chunk, D, i * D)
                misc_c.release(1)

            # step 0: a_slice[0:O_WINDOW) = Wo[my window] @ cx -- a window of N_O_TILES full
            # TSI_O-row tiles, always >= D_PER_CORE rows (see module docstring's FUSE_O section).
            for j in range_(N_O_TILES):
                j32 = index.casts(T.i32(), j)
                row_off = j32 * TSI_O
                wt = weight_c.acquire(1)
                mv_o_k(TSI_O, row_off, wt, cx_buf, a_slice_buf)
                weight_c.release(1)

            # step 0b: drain only this core's real D_PER_CORE-wide prefix (discard the overlap
            # tail) onto the shared output channel -- the FIRST round through it now, ahead of
            # gh's own R rounds.
            ot = out_p.acquire(1)
            copy_off_a_k(ot, a_slice_buf, D_PER_CORE, 0)
            out_p.release(1)

            # step 0c: refill full `a` (barriered by the caller between 0b and here -- see
            # sequence()'s tg_a_drain/tg_a_refill split) and `cur`, adjacent in the misc queue by
            # construction (sequence() fills them as the last two items before this barrier and
            # the first item after it), so one acquire(2) still returns them as a pair exactly
            # like the non-fused-o arm below. post_norm: normalise `a` (pair[1]) with pa_gain
            # BEFORE the residual add -- same full-D wnorm_kernel binding the pre-FFN norm uses.
            pair = misc_c.acquire(2)
            if post_norm:
                wnorm_k(pair[1], pa_gain_buf, a_norm_buf, epsilon)
                add_k(pair[0], a_norm_buf, x1_buf, D)
            else:
                add_k(pair[0], pair[1], x1_buf, D)
            misc_c.release(2)
        else:
            # step 1: x1 = cur + a, full D, replicated on every core. `a` arrives already
            # sandwich-normalised from outside when post_norm is set (see design.py's module
            # docstring / op.py's reference()) -- fuse_o is the only arm that computes `a` ON-CHIP,
            # so it is the only arm that must normalise it here.
            pair = misc_c.acquire(2)
            add_k(pair[0], pair[1], x1_buf, D)
            misc_c.release(2)

        # step 2: hf = weighted_rms_norm(x1, n_pf), full D, replicated.
        npf = misc_c.acquire(1)
        wnorm_k(x1_buf, npf, hf_buf, epsilon)
        misc_c.release(1)

        # step 3: g = Wg[my rows] @ hf, then u = Wu[my rows] @ hf -- same shared weight channel,
        # continued (Wg's N_GU_TILES tiles, then Wu's).
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, g_buf)
            weight_c.release(1)
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, u_buf)
            weight_c.release(1)

        # step 4: g = act(g), in place over the whole FF/N slice (silu_tile_bf16 or
        # gelu_tile_bf16, same in-place ABI -- see the docstring).
        act_k(FF_PER_CORE, g_buf)

        if row_parallel_down:
            # step 5 (row-parallel): gh = silu(g)*u, kept CORE-LOCAL -- no all-gather, this
            # core's own FF/N slice is exactly what its own Wd column-shard needs.
            gh_mul_k(g_buf, u_buf, gh_buf, FF_PER_CORE)
        else:
            # step 5: emit gh = silu(g)*u in R chunks of D/N, straight onto the shared output
            # fifo (no separate gh-slice buffer -- the offset read comes out of g_buf/u_buf
            # directly).
            for r in range(R):
                ot = out_p.acquire(1)
                mul_off_k(g_buf, u_buf, ot, D_PER_CORE, r * D_PER_CORE)
                out_p.release(1)

        if row_parallel_down:
            # step 6 (row-parallel + cascade, module docstring): d_chunks (== N) rounds, each a
            # local reduction over THIS core's own FF/N-wide gh against its own Wd column-shard,
            # summed across all N cores by the hardware cascade. Only core N-1 (get_only, the
            # chain's end) ends up holding the full D_CHUNK-wide sum; it finishes straight into
            # its own out-fifo tile and drains it (sequence()'s tg_d_drain, replacing the
            # per-core drain the other arms use). core 0 (put_only) and the middle cores
            # (put_get) never touch out_p here.
            for cd in range(d_chunks):
                rp_zero_k(D_CHUNK, acc_buf)
                for j in range_(N_RP_TILES):
                    j32 = index.casts(T.i32(), j)
                    row_off = j32 * TSI_RP
                    wt = weight_c.acquire(1)
                    rp_taccum_k(TSI_RP, row_off, wt, gh_buf, acc_buf)
                    weight_c.release(1)
                if core_id == 0:
                    cr_put_only_k(D_CHUNK, acc_buf)
                elif core_id == N - 1:
                    d_tile = out_p.acquire(1)
                    cr_get_only_k(D_CHUNK, acc_buf, acc_buf)
                    rp_finish_k(D_CHUNK, acc_buf, d_tile)
                    out_p.release(1)
                else:
                    cr_put_get_k(D_CHUNK, acc_buf)
        elif gh_chunks > 1:
            # step 5b/6 (T2.1a, chunked): gh is never assembled FF-wide. Each of gh_chunks rounds
            # refills gh_buf (now CHUNK_WIDTH==D wide) from ONE misc read, then immediately
            # consumes it against every TSI_GU-row batch of THIS round's Wd chunk, accumulating
            # into acc_buf (persistent f32, one slot per this core's own D_PER_CORE output rows).
            # The misc read count (gh_chunks==R) and the total weight bytes read (N_D_TILES_CHUNK
            # batches * gh_chunks rounds, same TSI_GU granularity as Wg/Wu) both match the
            # unchunked arm exactly -- only gh_buf's and Wd's per-fetch L1 footprint shrinks. See
            # the module docstring's T2.1a section.
            taccum_zero_k(D_PER_CORE, acc_buf)
            for r in range(gh_chunks):
                chunk = misc_c.acquire(1)
                gc_copy_k(gh_buf, chunk, D, 0)
                misc_c.release(1)
                for j in range_(N_D_TILES_CHUNK):
                    j32 = index.casts(T.i32(), j)
                    row_off = j32 * TSI_GU
                    wt = weight_c.acquire(1)
                    mv_d_taccum_k(TSI_GU, row_off, wt, gh_buf, acc_buf)
                    weight_c.release(1)
            taccum_finish_k(D_PER_CORE, acc_buf, d_buf)
        else:
            # step 5b: reassemble the all-gathered gh from R D-sized misc reads (the Runtime's
            # sequence issues these only AFTER every core's R output drains land in gh_scratch --
            # see the TaskGroup barrier in sequence()).
            for i in range(R):
                chunk = misc_c.acquire(1)
                copy_off_k(gh_buf, chunk, D, i * D)
                misc_c.release(1)

            # step 6: d = Wd[my rows] @ gh -- same shared weight channel, continued (Wd's
            # N_D_TILES). post_norm: mv_d_k writes directly into the acquired output tile instead
            # of d_buf, since that tile is about to be drained RAW (pre-norm) rather than combined
            # with x1 here -- see step 7 below. d_buf stays allocated but unused in that arm
            # (cheap, keeps this loop's shape identical either way).
            if post_norm:
                d_tile = out_p.acquire(1)
            for j in range_(N_D_TILES):
                j32 = index.casts(T.i32(), j)
                row_off = j32 * TSI_D
                wt = weight_c.acquire(1)
                mv_d_k(TSI_D, row_off, wt, gh_buf, d_tile if post_norm else d_buf)
                weight_c.release(1)

        if post_norm:
            if not row_parallel_down:
                # step 6b: hand off the RAW (pre-norm) d slice -- sequence()'s tg_d_drain/
                # tg_d_refill all-gather it through gh_scratch (dead for the rest of this call
                # once step 5b has consumed it) and broadcast the full-D result back over misc,
                # mirroring gh's own all-gather. d has no such round-trip otherwise: every core
                # only ever computes its own D/N output slice, but a real RMSNorm needs the full
                # D vector's sum of squares. row_parallel_down's own step 6 already closed every
                # acquire it opened (only core N-1 opens one, per d_chunks round), so there is
                # nothing pending to release here.
                out_p.release(1)

            # step 7a: pff_gain (post-ffn-norm gain) + the refilled full-D raw d, paired. The
            # RMSNorm reduction needs the whole D-wide `d`, but the gain only ever multiplies
            # this core's own D_PER_CORE output slice -- so `d` is normalised unweighted at full
            # D, and the gain is read straight off the still-open misc tile for that one slice,
            # never copied to a D-wide buffer first (unlike pa_gain above).
            pair = misc_c.acquire(2)
            rms_unweighted_k(pair[1], d_norm_buf, D, epsilon)
            mul_pn_k(d_norm_buf, pair[0], d_buf, D_PER_CORE, core_id * D_PER_CORE)
            misc_c.release(2)

            # step 7b: nxt[my rows] = x1[my rows] + (d_norm*pff_gain)[my rows] -- d_buf is unused
            # elsewhere in this arm (step 6 wrote the raw slice straight into d_tile instead), so
            # this reuses it rather than adding a buffer.
            ot = out_p.acquire(1)
            add_off_k(x1_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
            out_p.release(1)
        else:
            # step 7: nxt[my rows] = x1[my rows] + d, onto the shared output fifo's (R+1)-th round.
            ot = out_p.acquire(1)
            add_off_k(x1_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
            out_p.release(1)

    workers = []
    for c in range(N):
        x1_buf = Buffer(D_ty, name=f"{fifo_prefix}x1_{c}")
        hf_buf = Buffer(D_ty, name=f"{fifo_prefix}hf_{c}")
        g_buf = Buffer(FFPC_ty, name=f"{fifo_prefix}g_{c}")
        # row_parallel_down computes gh = silu(g)*u IN PLACE over g_buf (gh_mul_k(g_buf, u_buf,
        # g_buf, ...), same object bound to both the gh_buf and g_buf slots below) -- an
        # elementwise map is alias-safe (index i's store depends only on index i's own loads),
        # and it is a whole FF/N-wide buffer this shape cannot otherwise fit L1 with (see the
        # module docstring). Every other arm keeps its own gh_buf, sized GH_ty (CHUNK_WIDTH at
        # gh_chunks>1, FF unchunked).
        gh_buf = g_buf if row_parallel_down else Buffer(GH_ty, name=f"{fifo_prefix}gh_{c}")
        u_buf = Buffer(FFPC_ty, name=f"{fifo_prefix}u_{c}")
        d_buf = Buffer(DPC_ty, name=f"{fifo_prefix}d_{c}")
        core_args = [
            misc_of.cons(), weight_ofs[c].cons(), out_ofs[c].prod(),
            x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
            add_kernel, add_off_kernel, wnorm_kernel, mv_gu_kernel, mv_d_kernel,
            act_kernel, mul_off_kernel, copy_off_kernel,
            c,
        ]
        if fuse_o:
            cx_buf = Buffer(QD_ty, name=f"{fifo_prefix}cx_{c}")
            a_slice_buf = Buffer(OWIN_ty, name=f"{fifo_prefix}aslice_{c}")
            core_args += [cx_buf, a_slice_buf, mv_o_kernel, copy_off_cx_kernel, copy_off_a_kernel]
            if post_norm:
                pa_gain_buf = Buffer(D_ty, name=f"{fifo_prefix}pagain_{c}")
                a_norm_buf = Buffer(D_ty, name=f"{fifo_prefix}anorm_{c}")
                core_args += [pa_gain_buf, a_norm_buf, copy_pn_kernel]
        if post_norm:
            d_norm_buf = Buffer(D_ty, name=f"{fifo_prefix}dnorm_{c}")
            core_args += [d_norm_buf, rms_unweighted_kernel, mul_pn_kernel]
        if gh_chunks > 1:
            acc_buf = Buffer(ACC_ty, name=f"{fifo_prefix}acc_{c}")
            core_args += [acc_buf, taccum_zero_kernel, mv_d_taccum_kernel, taccum_finish_kernel,
                          gc_copy_kernel]
        if row_parallel_down:
            acc_buf = Buffer(ACC_ty, name=f"{fifo_prefix}acc_{c}")
            core_args += [acc_buf, rp_taccum_zero_kernel, rp_mv_taccum_kernel,
                          rp_taccum_finish_kernel, gh_mul_kernel,
                          cr_put_only_kernel, cr_put_get_kernel, cr_get_only_kernel]
        workers.append(Worker(core_fn, core_args, stack_size=stack_size))

    if row_parallel_down:
        # Cascade sum, module docstring: a linear chain core[0] -> core[1] -> ... -> core[N-1].
        # aiecc's own place-tiles pass is cascade-aware (S/E adjacency search) -- no explicit
        # Tile() pinning needed, same bare placement every other N-way arm in this file uses.
        for c in range(N - 1):
            CascadeFlow(workers[c], workers[c + 1])

    def sequence(*args):
        post_norms = None
        if fuse_o and post_norm:
            (cur, cx, npf, Wo, Wg, Wu, Wd, gh_scratch, a_scratch, post_norms, nxt,
             misc_p, gweight_ps, gout_cs) = args
        elif fuse_o:
            (cur, cx, npf, Wo, Wg, Wu, Wd, gh_scratch, a_scratch, nxt,
             misc_p, gweight_ps, gout_cs) = args
        elif post_norm:
            (cur, a, npf, Wg, Wu, Wd, gh_scratch, post_norms, nxt,
             misc_p, gweight_ps, gout_cs) = args
        else:
            (cur, a, npf, Wg, Wu, Wd, gh_scratch, nxt,
             misc_p, gweight_ps, gout_cs) = args
        # `wait=True` EVERYWHERE, not just on the drains: a plain TaskGroup.finish() with no
        # wait=True lowers to dma_free_task, which is compile-time BD-ID recycling ONLY -- no
        # hardware wait is emitted (AIEAssignRuntimeSequenceBDIDs.cpp; AIEDMATasksToNPU.cpp never
        # even sees dma_free_task). BD-ID pools are allocated PER SHIM TILE, shared across every
        # ObjectFifo mapped to that tile, so a later fill/drain on the SAME tile (misc, weight and
        # output are only 1-2 distinct shim tiles at N=8, since the placer fills 4 rows per column
        # before moving on) can get a recycled BD ID reprogrammed while the freed one's transfer
        # is still in flight -- a lock-count race, not a copy race, so it does not corrupt data,
        # it desyncs an ObjectFifo's acquire()/release() and hangs. Only wait=True lowers to a
        # real `dma_await_task`/NpuSyncOp barrier. MEASURED: without this, every N (including
        # N=8, which has no split/join to blame) hit a genuine device-side TDR
        # (aie2_tdr_detect, journalctl -k) and ERT_CMD_STATE_TIMEOUT, not just a host illusion.
        #
        # fuse_o inserts tg_a_drain/tg_a_refill AHEAD of this group (not merged into it): this
        # core now produces its FIRST output (the a-slice drain) before it has consumed cur/n_pf,
        # where the un-fused-o arm produces its first output (gh) only after consuming ALL of its
        # input. A TaskGroup boundary is a hard barrier (Runtime.finish_task_group awaits at group
        # close), so folding a's fills into tg1 unchanged while a's drain waits behind it would be
        # fine -- but folding the REFILL in with it would not: cur/n_pf are needed for step 1/2,
        # AFTER a is refilled, so their fill has to land in the group that FOLLOWS the a-slice
        # drain barrier, not the one that precedes it. Splitting fills vs drains across groups is
        # always safe; it is only unsafe to place a drain that depends on a not-yet-issued fill in
        # the SAME or an EARLIER group than that fill.
        tg1 = TaskGroup()
        if fuse_o:
            if post_norm:
                # pa_gain (post-attn-norm gain), first in the misc queue -- core_fn's step -1a
                # acquires it before the cx reassembly below, at PA_OFF=0 in the packed buffer.
                misc_p.fill(post_norms, _flat_tap(POST_NORMS_UNITS, D, 0), wait=True, group=tg1)
            for i in range(R_CX):
                misc_p.fill(cx, _flat_tap(QD, D, i * D), wait=True, group=tg1)
            # cur is filled here (fills-only group, before the a barrier) but not CONSUMED until
            # after a is refilled -- see core_fn's step 0c. It stays adjacent to a's own refill in
            # the misc queue only because nothing else is filled into misc between here and
            # tg_a_refill below (n_pf is deliberately deferred to that same later group).
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg1)
            for g in range(n_aie_cols):
                gweight_ps[g].fill(
                    Wo, _flat_tap(WO_ROWS_PADDED * WROW_QD, O_WINDOW * WROW_QD, g * D_PER_CORE * WROW_QD),
                    wait=True, group=tg1,
                )
        else:
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg1)
            misc_p.fill(a, _flat_tap(D, D), wait=True, group=tg1)
            misc_p.fill(npf, _flat_tap(D, D), wait=True, group=tg1)
        if n_aie_rows == 1:
            # Wg/Wu belong in tg1 ONLY when nothing barriers the core between its Wo reads and its
            # Wg reads. With fuse_o the core PRODUCES its a-slice in between, and that drain is in
            # tg_a_drain -- so a Wg fill here can never complete: the core cannot reach step 3 to
            # consume it until a drain that tg1.finish() is itself blocking gets issued.
            # MEASURED as ERT_CMD_STATE_TIMEOUT with `Fatal error type: 0x0`. They are issued
            # after tg_a_refill instead; see the invariant note there.
            if not fuse_o:
                for g in range(n_aie_cols):
                    gweight_ps[g].fill(
                        Wg, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                        wait=True, group=tg1,
                    )
                for g in range(n_aie_cols):
                    gweight_ps[g].fill(
                        Wu, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                        wait=True, group=tg1,
                    )
            tg1.finish()
        else:
            tg1.finish()
            # Round-major, group-minor, with a finish() per round: each round issues exactly
            # n_aie_cols fills (one per shim tile), so no tile ever has more than 1 in flight.
            # Batching all rounds into one TaskGroup instead hit aiecc's real per-tile BD queue
            # depth (16) -- shim (0,0) alone would have queued N_GU_TILES*2 (Wg+Wu) unfreed
            # descriptors for group 0. Gathers all n_aie_rows rows' data for one round with a
            # strided TAP (see _group_tap / module docstring).
            for i in range(N_GU_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * FF_PER_CORE * D
                    gweight_ps[g].fill(
                        Wg,
                        _group_tap(FF * D, base + i * TSI_GU * D, n_aie_rows,
                                   FF_PER_CORE * D, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()
            for i in range(N_GU_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * FF_PER_CORE * D
                    gweight_ps[g].fill(
                        Wu,
                        _group_tap(FF * D, base + i * TSI_GU * D, n_aie_rows,
                                   FF_PER_CORE * D, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()

        if fuse_o:
            # tg_a_drain: every core's real D_PER_CORE-wide a-slice, the FIRST round through the
            # shared output channel (gh's own R rounds and the final residual follow it).
            tg_a_drain = TaskGroup()
            for g in range(n_aie_cols):
                gout_cs[g].drain(
                    a_scratch, _flat_tap(D, D_PER_CORE, g * D_PER_CORE),
                    wait=True, group=tg_a_drain,
                )
            tg_a_drain.finish()

            # tg_a_refill: full `a` back to every core (misc), plus n_pf (deferred here so it
            # stays AFTER cur in the misc queue -- core_fn's pair-acquire needs cur and this fill
            # adjacent, and n_pf is consumed only after that pair, so its position here is fine).
            tg_a_refill = TaskGroup()
            misc_p.fill(a_scratch, _flat_tap(D, D), wait=True, group=tg_a_refill)
            misc_p.fill(npf, _flat_tap(D, D), wait=True, group=tg_a_refill)
            tg_a_refill.finish()

            # Wg/Wu, moved here from tg1. THE INVARIANT, stated one-directionally in TIME rather
            # than by task kind: every task in group k must be reachable by the core using only
            # groups <= k. A group is unsafe both when it holds a drain waiting on a later fill
            # AND -- the case that hung this design -- when it holds a fill the core cannot reach
            # until a later group's drain is issued.
            tg_gu = TaskGroup()
            for g in range(n_aie_cols):
                gweight_ps[g].fill(
                    Wg, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                    wait=True, group=tg_gu,
                )
            for g in range(n_aie_cols):
                gweight_ps[g].fill(
                    Wu, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                    wait=True, group=tg_gu,
                )
            tg_gu.finish()

        # Barrier: gh_scratch must be fully written before any core reads it back. Every core's R
        # output-fifo drains for gh land at disjoint, contiguous offsets that together cover all
        # of gh_scratch exactly once, in the natural FF order Wd's rows expect.
        #
        # `split_gh` is an INSTRUMENT, not a feature. It chops this ONE group into k groups over
        # the same drains in the same order, so bytes, shim tasks, BDs, configures and designs are
        # all byte-identical and the ONLY thing that moves is the number of sync points: +(k-1) per
        # layer. That is the one axis left after BD-chaining refuted the per-task model on device
        # (105 -> 64 tasks/layer at constant bytes bought +0.209 ms, the wrong sign): per-task and
        # per-sync-point were confounded until something moved them independently, and only this
        # does.
        # Splitting HERE is safe in the invariant this file states one-directionally in time
        # (every task in group k reachable by the core using only groups <= k): these are all
        # DRAINS whose fills were issued in strictly earlier groups, and each column's output fifo
        # is its own, so an undrained column stalls its own core without blocking another
        # column's drain. Contrast attn_block_dp's tg3, where splitting fills from drains
        # deadlocks a core that interleaves them.
        # row_parallel_down has no gh all-gather at all (module docstring): every core keeps its
        # own FF/N slice core-local, so neither the drain-to-scratch below nor its misc refill
        # (gh_buf is never read from misc in that arm) has anything to do.
        if not row_parallel_down:
            gh_drains = []
            for g in range(n_aie_cols):
                for r in range(R):
                    if n_aie_rows == 1:
                        tap = _flat_tap(FF, D_PER_CORE, g * FF_PER_CORE + r * D_PER_CORE)
                    else:
                        tap = _group_tap(
                            FF, g * n_aie_rows * FF_PER_CORE + r * D_PER_CORE,
                            n_aie_rows, FF_PER_CORE, 1, D_PER_CORE,
                        )
                    gh_drains.append((g, tap))
            k = max(1, min(int(split_gh), len(gh_drains)))
            per = -(-len(gh_drains) // k)
            for lo in range(0, len(gh_drains), per):
                tg2 = TaskGroup()
                for g, tap in gh_drains[lo:lo + per]:
                    gout_cs[g].drain(gh_scratch, tap, wait=True, group=tg2)
                tg2.finish()

        # gh_scratch refill AND Wd share this group: both are exactly what the core's down-matvec
        # step needs next, and neither has an ordering hazard against anything still pending.
        tg3 = TaskGroup()
        if not row_parallel_down:
            for i in range(R):
                misc_p.fill(gh_scratch, _flat_tap(FF, D, i * D), wait=True, group=tg3)
        if n_aie_rows == 1:
            if row_parallel_down:
                # Wd's L3 buffer holds N independently-quantized [D, WROW_FFPC] column-shard
                # blocks, one per core (Wd_L3_ty comment above) -- block c is entirely core c's
                # own, so its d_chunks*N_RP_TILES fills never touch another core's bytes. Order
                # matches core_fn's chunk-outer/row-batch-inner acquire order on weight_ofs[c].
                #
                # Round-major (one TaskGroup per (cd, j)), not batched into tg3: at cols=8 this
                # is d_chunks*N_RP_TILES rounds, each 8 fills (one per shim tile) -- batching them
                # all into one group hits aiecc's real per-tile BD queue depth (16), the same wall
                # the n_aie_rows>1 Wg/Wu loop below already works around this way (MEASURED,
                # device-free: aiecc's own "too many simultaneously active buffer descriptors on
                # tile" error).
                for cd in range(d_chunks):
                    for j in range(N_RP_TILES):
                        tgw = TaskGroup()
                        row_off = (cd * D_CHUNK + j * TSI_RP) * WROW_FFPC
                        for c in range(n_aie_cols):
                            gweight_ps[c].fill(
                                Wd,
                                _flat_tap(N * D * WROW_FFPC, TSI_RP * WROW_FFPC,
                                         c * D * WROW_FFPC + row_off),
                                wait=True, group=tgw,
                            )
                        tgw.finish()
            elif gh_chunks > 1:
                # T2.1a: Wd's L3 buffer holds gh_chunks independently-packed [D, WROW_D] planar
                # blocks (see the Wd_L3_ty comment above) -- one fill per (chunk, core), matching
                # core_fn's chunk-outer/row-batch-inner acquire order on weight_ofs[c]. Total
                # elements moved are unchanged from the unchunked fill below
                # (gh_chunks*D*WROW_D == D*WROW_FF), just issued as gh_chunks smaller reads; no
                # extra barrier needed since Wd never depends on gh (gh_scratch's own refill above
                # is independent of it).
                for r in range(gh_chunks):
                    for g in range(n_aie_cols):
                        gweight_ps[g].fill(
                            Wd,
                            _flat_tap(D * WROW_FF, D_PER_CORE * WROW_D,
                                     r * D * WROW_D + g * D_PER_CORE * WROW_D),
                            wait=True, group=tg3,
                        )
            else:
                for g in range(n_aie_cols):
                    gweight_ps[g].fill(
                        Wd, _flat_tap(D * WROW_FF, D_PER_CORE * WROW_FF, g * D_PER_CORE * WROW_FF),
                        wait=True, group=tg3,
                    )
            tg3.finish()
        else:
            tg3.finish()
            for i in range(N_D_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * D_PER_CORE * FF
                    gweight_ps[g].fill(
                        Wd,
                        _group_tap(D * FF, base + i * TSI_D * FF, n_aie_rows,
                                   D_PER_CORE * FF, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()

        if post_norm:
            # d has no existing all-gather (unlike gh/a above): every core drains its own RAW
            # D/N-wide slice into gh_scratch's first D elements (dead for the rest of this call --
            # its own content was already consumed into gh_buf by step 5b, strictly before this
            # point), then every core reads the assembled full-D vector back, mirroring gh's own
            # drain-then-refill shape one axis over. Reuses gh_scratch's own L3 allocation rather
            # than adding a 16th/17th host buffer argument -- see the module docstring.
            tg_d_drain = TaskGroup()
            if row_parallel_down:
                # Only core N-1 (the cascade chain's end) produced anything for this step -- its
                # own out-fifo channel got d_chunks sequential D_CHUNK-wide tiles (core_fn's step
                # 6), drained here in the same order.
                sink = n_aie_cols - 1
                for cd in range(d_chunks):
                    gout_cs[sink].drain(
                        gh_scratch, _flat_tap(D, D_CHUNK, cd * D_CHUNK),
                        wait=True, group=tg_d_drain,
                    )
            else:
                for g in range(n_aie_cols):
                    gout_cs[g].drain(
                        gh_scratch, _flat_tap(FF, D_PER_CORE, g * D_PER_CORE),
                        wait=True, group=tg_d_drain,
                    )
            tg_d_drain.finish()

            tg_d_refill = TaskGroup()
            # pff_gain first (core_fn's step 7a acquires it before the raw-d refill below).
            misc_p.fill(post_norms, _flat_tap(POST_NORMS_UNITS, D, PFF_OFF),
                        wait=True, group=tg_d_refill)
            misc_p.fill(gh_scratch, _flat_tap(FF, D, 0), wait=True, group=tg_d_refill)
            tg_d_refill.finish()

        tg4 = TaskGroup()
        for g in range(n_aie_cols):
            # Final residual: joined-buffer row order and nxt's own indexing both step by
            # D_PER_CORE, so this drain -- unlike gh's -- is a plain contiguous run even at
            # n_aie_rows>1.
            gout_cs[g].drain(
                nxt, _flat_tap(D, n_aie_rows * D_PER_CORE, g * n_aie_rows * D_PER_CORE),
                wait=True, group=tg4,
            )
        tg4.finish()

    if fuse_o:
        rt_args = [
            D_ty, QD_ty, D_ty, Wo_L3_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, GH_SCRATCH_ty, A_SCRATCH_ty,
            *([POST_NORMS_ty] if post_norm else []),
            D_ty,
            misc_of.prod(),
            [of.prod() for of in group_weight_ofs], [of.cons() for of in group_out_ofs],
        ]
    else:
        rt_args = [
            D_ty, D_ty, D_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, GH_SCRATCH_ty,
            *([POST_NORMS_ty] if post_norm else []),
            D_ty,
            misc_of.prod(),
            [of.prod() for of in group_weight_ofs], [of.cons() for of in group_out_ofs],
        ]
    if parts_only:
        # See attn_block_dp/design.py's identical hook: hand the pieces to a caller assembling a
        # larger aie.device. `rt_args` is already [L3 types ..., handles ...]; the split is at -3
        # (misc producer, the per-group weight producers, the per-group output consumers).
        return dict(workers=workers, seq=sequence,
                    l3_types=rt_args[:-3], handles=rt_args[-3:])

    rt = Runtime(sequence, rt_args)

    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
