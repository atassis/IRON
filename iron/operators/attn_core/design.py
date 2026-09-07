# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)

"""
Fused attention core: KV-append, scores, softmax, context, output projection -- as ONE aie.device.

Fuses 7 consecutive runlist entries of the Qwen3-0.6B fused decode (xdna-engine's
designs/decode_fused/gen_llm_decode.py, defaults TMV_CTX=1 GQA_GROUPED_K=1) that today cost 7
`aiex.configure`/layer into ONE design (1 configure/layer):

    op_sck     StridedCopy   k        -> kc[kv_off]   append this token's K row into the cache
    op_scv     StridedCopy   v        -> vc[kv_off]   append this token's V row into the cache
    op_scores  GEMV          kc, q    -> sc           S=2048-wide scores per head, GQA-grouped
    op_scale   ElementwiseMul sc      *= attn_scale   FOLDED into op_softmax below (see there)
    op_softmax Softmax        sc      -> sw
    op_ctx     TMatVec        vc, sw  -> cx           transposed-A matvec: context vector per head
    op_o       GEMV           Wo, cx  -> a            M=1024 K=2048 output projection

Shapes are Qwen3-0.6B's (llm_decode_spec.py QWEN3_0_6B): D=1024, Hq=16, Hkv=8, HD=128, S=2048
(max_seq -- attention always runs the full padded window regardless of n_past), gqa_group =
Hq//Hkv = 2, COLS = 8 (== Hkv, which TMatVec's "one matrix per column" placement requires).

On-chip residency (the second prize named in the task -- objectFIFO MemTile buffers are allocated
per DESIGN, so fusing into one is what lets these skip DDR): NONE of sc/sw/cx make it. All three
round-trip a DRAM scratch buffer, exactly like the unfused designs' own scratch buffers -- just now
inside this one device instead of three. See "Why nothing stays on-chip" below: this is a real
primitive limitation hit twice, in two different ways, not something left undone for lack of time.

Why nothing stays on-chip:
  1. sc/softmax is a genuine crossbar. GEMV-scores splits its OUTPUT by SEQUENCE POSITION (every
     column contributes a slice to every head), while softmax needs one column per HEAD (every full
     row from one head). Reshuffling between those two partitions on-chip is an 8-source/
     8-destination exchange, and `aie.objectfifo_link` refuses that outright: "An ObjectFifoLink may
     only have > 1 of either sources or destinations, but not both" (iron/dataflow/objectfifo.py).
     Confirmed empirically, not just read off the API: an isolated repro of JUST "8 cores join into
     one buffer, then that buffer forwards into a fresh one for a later split" (no split even
     attempted yet) already fails mlir-aie's verifier with "objectfifo cannot be in more than one
     ObjectFifoLinkOp" -- the join already spends the one link a buffer is allowed, so it can never
     also feed a split (or even a forward) downstream.
  2. cx looked simpler (an 8-to-1 join then a plain broadcast, no split -- broadcast is bare
     `.cons()`, which creates no link at all, so cx's one link is never contended) and it DOES place
     and verify in isolation. But aiecc's place-tiles pass rejects it in the full design with "no
     MemTile on the device has 8 input/1 output DMA channel(s) free" -- confirmed against
     AIE2TargetModel.cpp: a MemTile has exactly 6 DMA source/dest switchbox connections
     (getNum{Source,Dest}SwitchboxConnections, WireBundle::DMA -> 6), not 8. All 8 join sources
     must land on ONE MemTile, so an 8-source join is flatly oversized for any single MemTile on
     this device, independent of what else is or isn't using it (the message's "0/48 ... used"
     is the 8-MemTile device TOTAL headroom -- plenty in aggregate, useless when one link needs 8 on
     ONE tile). A hierarchical join (e.g. two 4-source joins into two MemTiles, then a 2-source join
     of those) hits the SAME one-link-per-fifo rule as (1): the first-level destination fifo cannot
     also be a second-level join's source without a relay CORE re-emitting it, and this design
     already spends all 32 of NPU2's core tiles on op_scores/op_softmax/op_ctx/op_o with none spare.

Two engineering simplifications from the shipped (non-fused) design, both chosen for correctness
confidence over peak throughput, given this gate is device-free (no hardware/ISS to check a subtler
version against):

  1. attn_scale (a trace-time-known float, head_dim**-0.5) is applied inside softmax's own per-tile
     loop (one extra `scale_tile_bf16` kernel call on a tile softmax already holds) instead of as a
     separate ElementwiseMul stage. This is a pure win, not a tradeoff: it deletes a whole stage
     (own ObjectFifos, own aiex.configure) for the price of one kernel call, exactly like GEMV's
     existing `epilogue` mechanism folds an activation into a producing core.

  2. GEMV-scores reads kc in NATURAL per-head order (head 0, 1, 2, ..., 15) rather than gemv's own
     "coalesced" interleaved batch_group order (member-major: 0,2,4,...,14,1,3,5,...,15 -- see
     iron/operators/gemv/design.py's `coalesced_tap`). Two consecutive heads share one kv-head
     matrix; natural order reads that matrix's 256-row column-slice TWICE in a row (an ObjectFifo
     `repeat_count=2` on the A port) rather than once via gemv's cleverer interleaved addressing.
     Same math, ~2x the intended A bandwidth for this one read. Chosen because re-deriving gemv's
     interleaved-order arithmetic by hand, for output whose correctness cannot be checked on this
     device-free gate, is a real error-injection risk; the natural-order form makes every DRAM tap
     a single contiguous span with no permutation to get wrong.

Known gaps, stated plainly:
  - sc, sw AND cx are all DRAM scratch, not resident -- the byte-savings prize named in the task is
    NOT captured at all. What this operator delivers is purely the configure-count win (7 -> 1).
  - This design uses ALL 32 AIE2P core tiles (8 columns x 4 rows: NPU2's whole array) with zero
    spare -- 8 for op_scores, 8 for op_softmax, 8 for op_ctx, 8 for op_o (op_sck/op_scv are pure
    DMA, no cores, like strided_copy/design.py's own Program with no `workers=`). Any additional
    core requirement anywhere breaks placement, and it is also why neither reshuffle problem above
    can be worked around with a relay core.
  - op_ctx (TMatVec) inherits its A port's known, diagnosed, UNFIXED race (see
    iron/operators/tmatvec/design.py's "KNOWN DEFECT" comment): a single BD streams the whole
    K*M-element vc slice into a small ring with no per-object lock, correct standalone but racy
    across repeated invocations of the same design within one runtime sequence -- which is exactly
    what N decoder layers are. This is not introduced by fusion; the shipped TMV_CTX=1 default
    already invokes TMatVec once per layer today.
"""

BF16 = bfloat16


def attn_core(
    dev,
    cols=8,
    D=1024,
    Hq=16,
    Hkv=8,
    HD=128,
    S=2048,
    rows_per_chunk=64,
    attn_scale=None,
    scores_m_input=64,
    o_m_input=4,
    func_prefix="",
    kernel_object_mv_scores="attn_scores_mv.o",
    kernel_object_mv_o="attn_o_mv.o",
    # mv.cc's function name is fixed regardless of DIM_K, so op_scores' and op_o's compiled
    # objects both define `matvec_vectorized_bf16_bf16` -- fine as separate designs, but this is
    # ONE aie.device and mlir-aie's symbol table is module-global. op.py's KernelObjectArtifact for
    # attn_o_mv.o renames it (objcopy --redefine-sym) to the value below; this parameter is how the
    # two sides (kernel object + Kernel() declaration) stay in sync without a second hard-coded copy.
    kernel_name_mv_o="o_matvec_vectorized_bf16_bf16",
    kernel_object_scale="attn_scale_tile.o",
    kernel_object_softmax="softmax.o",
    kernel_object_taccum=None,
    verbose=False,
):
    group = Hq // Hkv
    assert group * Hkv == Hq, f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})"
    assert Hkv == cols, (
        f"TMatVec places one matrix (kv head) per column: Hkv ({Hkv}) must equal cols ({cols})"
    )
    QD = Hq * HD
    KVD = Hkv * HD
    S_per_col = S // cols
    assert S_per_col * cols == S, f"S ({S}) must be a multiple of cols ({cols})"
    if attn_scale is None:
        attn_scale = HD ** -0.5  # Qwen3: head_dim**-0.5 (llm_decode_spec.py QWEN3_0_6B)
    if kernel_object_taccum is None:
        kernel_object_taccum = f"{func_prefix}tmv_{HD}n.o"

    if verbose:
        print(f"attn_core: D={D} Hq={Hq} Hkv={Hkv} HD={HD} S={S} cols={cols} group={group} "
              f"attn_scale={attn_scale}")

    # A single scratchpad value shared by op_sck and op_scv (n_past*HD, element units) and one
    # shared by softmax's causal-length mask -- same mechanism and names as the unfused designs
    # (strided_copy's output_offset_parameter="kv_off", softmax's vector_size_parameter="sm_mask"),
    # so the host-side per-token write protocol is unchanged.
    kv_off = ScratchpadParameter(f"{func_prefix}kv_off", np.int32)
    sm_mask = ScratchpadParameter(f"{func_prefix}sm_mask", np.int32)

    # ================================================================== op_sck / op_scv
    # Pure DMA (no AIE core), faithfully reproducing the `sc` dict gen_llm_decode.py builds for
    # both: input (Hkv, HD) contiguous -> output row `kv_off` of every kv head in a (Hkv, S, HD)
    # cache. Two instances (k->kc, v->vc) sharing the one `kv_off` parameter.
    kv_tile_ty = np.ndarray[(KVD,), np.dtype[BF16]]
    cache_ty = np.ndarray[(Hkv * S * HD,), np.dtype[BF16]]

    def _kv_append_taps():
        in_tap = TensorAccessPattern(
            tensor_dims=(KVD,), offset=0, sizes=[1, 1, Hkv, HD], strides=[0, 0, HD, 1]
        )
        out_tap = TensorAccessPattern(
            tensor_dims=(Hkv * S * HD,), offset=0, sizes=[1, 1, Hkv, HD],
            strides=[0, 0, S * HD, 1],
        )
        return in_tap, out_tap

    k_in_of = ObjectFifo(kv_tile_ty, name=f"{func_prefix}k_in", depth=1)
    k_out_of = k_in_of.cons().forward(name=f"{func_prefix}k_out", depth=1)
    v_in_of = ObjectFifo(kv_tile_ty, name=f"{func_prefix}v_in", depth=1)
    v_out_of = v_in_of.cons().forward(name=f"{func_prefix}v_out", depth=1)
    k_in_tap, k_out_tap = _kv_append_taps()
    v_in_tap, v_out_tap = _kv_append_taps()

    # ================================================================== op_scores (GEMV)
    # Column c computes scores[h, c*S_per_col:(c+1)*S_per_col] for h = 0..Hq-1 in NATURAL head
    # order (see module docstring point 2). scores_m_output == S_per_col: one C-tile (one head's
    # slice) per acquire, which is exactly the join granularity below.
    scores_m_output = S_per_col
    assert scores_m_output % scores_m_input == 0, (
        f"scores_m_input ({scores_m_input}) must divide S_per_col ({scores_m_output})"
    )

    L1_scA_ty = np.ndarray[(scores_m_input, HD), np.dtype[BF16]]
    L1_scB_ty = np.ndarray[(HD,), np.dtype[BF16]]
    L1_scC_ty = np.ndarray[(scores_m_output,), np.dtype[BF16]]

    scores_mv = Kernel(
        f"{func_prefix}matvec_vectorized_bf16_bf16",
        kernel_object_mv_scores,
        [np.int32, np.int32, L1_scA_ty, L1_scB_ty, L1_scC_ty],
    )

    # repeat_count=group: the sending end (kc's per-column fill, 8 matrices per column) replicates
    # each matrix object `group` times before advancing -- so the CONSUMER's 16 natural-order
    # acquires see matrix0,matrix0,matrix1,matrix1,...,matrix7,matrix7, i.e. head h reads kv-head
    # h//group, exactly matching gemv's own (batch//batch_group) addressing but via a
    # single-level DMA repeat instead of an interleaved tap.
    # repeat_count is rejected on a shim-fed ObjectFifo ("unavailable for shim tiles"), so the
    # shim fill lands in a plain (un-repeated) fifo first, and a MemTile .forward() hop applies
    # the repeat on its way into L1.
    scA_l3_fifos = [
        ObjectFifo(L1_scA_ty, name=f"{func_prefix}scA_l3_{c}", depth=2) for c in range(cols)
    ]
    scA_fifos = [
        scA_l3_fifos[c].cons().forward(
            name=f"{func_prefix}scA_{c}", depth=2, repeat_count=group
        )
        for c in range(cols)
    ]
    scB_fifos = [ObjectFifo(L1_scB_ty, name=f"{func_prefix}scB_{c}", depth=2) for c in range(cols)]

    def scores_core(a_cons, b_cons, c_prod, mv):
        for _ in range_(0xFFFFFFFF):  # once per head; 16 iterations/invocation
            b = b_cons.acquire(1)
            c = c_prod.acquire(1)
            for j_idx in range_(scores_m_output // scores_m_input):
                j_i32 = index.casts(T.i32(), j_idx)
                row_off = j_i32 * scores_m_input
                a = a_cons.acquire(1)
                mv(scores_m_input, row_off, a, b, c)
                a_cons.release(1)
            c_prod.release(1)
            b_cons.release(1)

    # A: column c, all 8 kv-head matrices' 256-row column-slice (repeat_count above doubles each).
    scA_taps = [
        TensorAccessPattern(
            tensor_dims=(Hkv * S * HD,),
            offset=c * S_per_col * HD,
            sizes=[1, 1, Hkv, S_per_col * HD],
            strides=[0, 0, S * HD, 1],
        )
        for c in range(cols)
    ]
    # B: every column reads the whole (unscaled) query vector, natural head order -- same flat
    # tap gemv/design.py uses at batch_group==1.
    q_ty = np.ndarray[(QD,), np.dtype[BF16]]
    scB_tap = TensorAccessPattern(
        tensor_dims=(QD,), offset=0, sizes=[1, 1, 1, QD], strides=[0, 0, 0, 1]
    )

    # sc is NOT resident -- see module docstring "Known gaps". GEMV-scores splits by SEQUENCE
    # POSITION (every column contributes to every head) while softmax needs a full row PER HEAD;
    # reshuffling that on-chip is a genuine 8-source/8-destination crossbar, which
    # `aie.objectfifo_link` refuses outright ("may only have > 1 of either sources or
    # destinations, but not both") -- confirmed empirically: even a plain join-then-forward on one
    # intermediate fifo (no split at all yet) fails verification with "objectfifo cannot be in
    # more than one ObjectFifoLinkOp", because the join already consumes that fifo's one allowed
    # link. Bridging it would need a relay CORE (acquire from the joined buffer, re-emit into a
    # split), and this design already spends all 32 of NPU2's core tiles with none spare. So sc
    # drains to a DRAM scratch buffer, column c writing all Hq heads' S_per_col-wide slices in one
    # tap (sizes=[1,1,Hq,S_per_col], stride S per head) -- the same natural order the join above
    # would have used, just landing in DRAM instead of a MemTile.
    sc_out_fifos = [
        ObjectFifo(L1_scC_ty, name=f"{func_prefix}sc_out_{c}", depth=2) for c in range(cols)
    ]

    scores_workers = [
        Worker(
            scores_core,
            [scA_fifos[c].cons(), scB_fifos[c].cons(), sc_out_fifos[c].prod(), scores_mv],
        )
        for c in range(cols)
    ]

    SC_ty = np.ndarray[(Hq * S,), np.dtype[BF16]]
    sc_out_taps = [
        TensorAccessPattern(
            tensor_dims=(Hq * S,), offset=c * S_per_col,
            sizes=[1, 1, Hq, S_per_col], strides=[0, 0, S, 1],
        )
        for c in range(cols)
    ]

    # ================================================================== op_softmax (+ folded scale)
    # heads_per_col heads per column (2 here), matching Softmax's own shipped partition exactly
    # (num_aie_columns=cols, tile_size=S -> chunk = Hq*S/cols = heads_per_col*S per column).
    heads_per_col = Hq // cols
    assert heads_per_col * cols == Hq

    softmax_tile_ty = np.ndarray[(S,), np.dtype[BF16]]

    scale_kernel = Kernel(
        f"{func_prefix}scale_tile_bf16", kernel_object_scale, [softmax_tile_ty, np.int32]
    )
    mask_kernel = Kernel(
        f"{func_prefix}mask_bf16", kernel_object_softmax,
        [softmax_tile_ty, np.int32, np.int32],
    )
    softmax_kernel = Kernel(
        f"{func_prefix}softmax_bf16", kernel_object_softmax,
        [softmax_tile_ty, softmax_tile_ty, np.int32],
    )

    # sc input: column c reads heads [c*heads_per_col : (c+1)*heads_per_col) from the sc DRAM
    # scratch, one contiguous heads_per_col*S-wide span -- a plain fill, matching the shipped
    # Softmax operator's own chunking exactly (chunk = num_elements/cols = heads_per_col*S).
    sc_in_fifos = [
        ObjectFifo(softmax_tile_ty, name=f"{func_prefix}sc_in_{c}", depth=2) for c in range(cols)
    ]
    sc_in_taps = [
        TensorAccessPattern(
            tensor_dims=(Hq * S,), offset=c * heads_per_col * S,
            sizes=[1, 1, 1, heads_per_col * S], strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]
    sw_out_fifos = [
        ObjectFifo(softmax_tile_ty, name=f"{func_prefix}sw_{c}", depth=2) for c in range(cols)
    ]

    def softmax_core(in_cons, out_prod, mask_len_param, scale_k, mask_k, softmax_k):
        mask_len = mask_len_param.read()
        for _ in range_(heads_per_col):
            x = in_cons.acquire(1)
            y = out_prod.acquire(1)
            scale_k(x, S)
            mask_k(x, mask_len, S)
            softmax_k(x, y, S)
            in_cons.release(1)
            out_prod.release(1)

    softmax_workers = [
        Worker(
            softmax_core,
            [
                sc_in_fifos[c].cons(),
                sw_out_fifos[c].prod(),
                sm_mask,
                scale_kernel,
                mask_kernel,
                softmax_kernel,
            ],
        )
        for c in range(cols)
    ]

    # sw: NOT resident -- see module docstring "Known gaps". Column c's span is identical on both
    # ends (softmax's drain and TMatVec's fill both address [c*heads_per_col*S, +heads_per_col*S)),
    # so the DRAM round trip is a flat copy, not a second reshuffle.
    SW_ty = np.ndarray[(Hq * S,), np.dtype[BF16]]
    sw_taps = [
        TensorAccessPattern(
            tensor_dims=(Hq * S,), offset=c * heads_per_col * S,
            sizes=[1, 1, 1, heads_per_col * S], strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]

    # ================================================================== op_ctx (TMatVec)
    # Column c == kv-head c. W (this group's two softmax rows) now comes from the sw DRAM scratch
    # instead of an on-chip source, but the per-column (offset, size) is unchanged from
    # tmatvec/design.py's own W_taps at batch_group=group. A (vc) is untouched: it must stay in
    # DRAM, so this is tmatvec/design.py's own A_taps verbatim (alloc_K=None, _AK=K=S).
    n_chunks = S // rows_per_chunk
    assert n_chunks * rows_per_chunk == S

    ACC_ty = np.ndarray[(group * HD,), np.dtype[np.float32]]
    L1_tA_ty = np.ndarray[(rows_per_chunk * HD,), np.dtype[BF16]]
    L1_tW_ty = np.ndarray[(group * S,), np.dtype[BF16]]
    L1_tC_ty = np.ndarray[(group * HD,), np.dtype[BF16]]

    k_zero = Kernel(f"{func_prefix}taccum_zero_f32", kernel_object_taccum, [np.int32, ACC_ty])
    k_rows = Kernel(
        f"{func_prefix}taccum_rows_bf16_f32", kernel_object_taccum,
        [np.int32, np.int32, np.int32, np.int32, L1_tA_ty, L1_tW_ty, ACC_ty],
    )
    k_finish = Kernel(
        f"{func_prefix}taccum_finish_bf16", kernel_object_taccum, [np.int32, ACC_ty, L1_tC_ty]
    )

    tA_fifos = [ObjectFifo(L1_tA_ty, name=f"{func_prefix}tA_{c}", depth=2) for c in range(cols)]
    tW_fifos = [ObjectFifo(L1_tW_ty, name=f"{func_prefix}tW_{c}", depth=1) for c in range(cols)]

    def tmatvec_core(a_cons, w_cons, c_prod, acc, zero_k, rows_k, finish_k):
        for _ in range_(0xFFFFFFFF):
            w = w_cons.acquire(1)
            zero_k(group, acc)
            for i in range_(n_chunks):
                a = a_cons.acquire(1)
                rows_k(rows_per_chunk, group, S, i * rows_per_chunk, a, w, acc)
                a_cons.release(1)
            c = c_prod.acquire(1)
            finish_k(group, acc, c)
            c_prod.release(1)
            w_cons.release(1)

    tA_taps = [
        TensorAccessPattern(
            tensor_dims=(Hkv * S * HD,), offset=c * S * HD,
            sizes=[1, 1, 1, S * HD], strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]
    tW_taps = [
        TensorAccessPattern(
            tensor_dims=(Hq * S,), offset=c * group * S,
            sizes=[1, 1, 1, group * S], strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]

    # cx is NOT resident either -- see module docstring "Why nothing stays on-chip". Column c's
    # release is `group*HD` contiguous elements at c*group*HD (natural order, TMatVec's own
    # C_taps shape); it drains to a DRAM scratch buffer the same way sc/sw do.
    cx_out_fifos = [
        ObjectFifo(L1_tC_ty, name=f"{func_prefix}cx_out_{c}", depth=2) for c in range(cols)
    ]

    tmatvec_workers = [
        Worker(
            tmatvec_core,
            [
                tA_fifos[c].cons(), tW_fifos[c].cons(), cx_out_fifos[c].prod(),
                Buffer(type=ACC_ty, name=f"{func_prefix}tacc_{c}"),
                k_zero, k_rows, k_finish,
            ],
        )
        for c in range(cols)
    ]

    CX_ty = np.ndarray[(Hq * HD,), np.dtype[BF16]]
    cx_out_taps = [
        TensorAccessPattern(
            tensor_dims=(Hq * HD,), offset=c * group * HD,
            sizes=[1, 1, 1, group * HD], strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]

    # ================================================================== op_o (GEMV)
    # Plain (num_batches=1) GEMV, M=D K=QD. B (cx) is read fresh from DRAM by every column --
    # 8 independent small fills of the same QD-wide vector, exactly gemv/design.py's own
    # batch_group==1 B_tap ("every column gets the entirety of the vector").
    o_m_output = D // cols
    assert o_m_output % o_m_input == 0, f"o_m_input ({o_m_input}) must divide D//cols ({o_m_output})"

    L1_oA_ty = np.ndarray[(o_m_input, QD), np.dtype[BF16]]
    L1_oB_ty = np.ndarray[(QD,), np.dtype[BF16]]
    L1_oC_ty = np.ndarray[(o_m_output,), np.dtype[BF16]]

    o_mv = Kernel(
        f"{func_prefix}{kernel_name_mv_o}",
        kernel_object_mv_o,
        [np.int32, np.int32, L1_oA_ty, L1_oB_ty, L1_oC_ty],
    )

    oA_fifos = [ObjectFifo(L1_oA_ty, name=f"{func_prefix}oA_{c}", depth=2) for c in range(cols)]
    oB_fifos = [ObjectFifo(L1_oB_ty, name=f"{func_prefix}oB_{c}", depth=2) for c in range(cols)]
    oC_fifos = [ObjectFifo(L1_oC_ty, name=f"{func_prefix}oC_{c}", depth=2) for c in range(cols)]

    def o_core(a_cons, b_cons, c_prod, mv):
        for _ in range_(0xFFFFFFFF):
            b = b_cons.acquire(1)
            c = c_prod.acquire(1)
            for j_idx in range_(o_m_output // o_m_input):
                j_i32 = index.casts(T.i32(), j_idx)
                row_off = j_i32 * o_m_input
                a = a_cons.acquire(1)
                mv(o_m_input, row_off, a, b, c)
                a_cons.release(1)
            c_prod.release(1)
            b_cons.release(1)

    o_workers = [
        Worker(o_core, [oA_fifos[c].cons(), oB_fifos[c].cons(), oC_fifos[c].prod(), o_mv])
        for c in range(cols)
    ]

    Wo_ty = np.ndarray[(D * QD,), np.dtype[BF16]]
    a_ty = np.ndarray[(D,), np.dtype[BF16]]
    oA_taps = [
        TensorAccessPattern(
            tensor_dims=(D * QD,), offset=c * o_m_output * QD,
            sizes=[1, 1, 1, o_m_output * QD], strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]
    oB_tap = TensorAccessPattern(
        tensor_dims=(QD,), offset=0, sizes=[1, 1, 1, QD], strides=[0, 0, 0, 1]
    )
    oC_taps = [
        TensorAccessPattern(
            tensor_dims=(D,), offset=c * o_m_output,
            sizes=[1, 1, 1, o_m_output], strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]

    # ================================================================== runtime sequence
    def sequence(
        k, v, kc, vc, q, Wo, a, sc_scr, sw_scr, cx_scr,
        k_in_p, k_out_c, v_in_p, v_out_c,
        scA_p, scB_p, sc_c, sc_p, sw_c, tA_p, tW_p, cx_c, oA_p, oB_p, oC_c,
    ):
        sync_parameters()  # delivers kv_off (sck/scv) and sm_mask (softmax) for this dispatch

        # Phase 0: append this token's K/V row. Must complete (wait=True) before anything below
        # reads kc/vc, or op_scores/op_ctx can read a stale cache.
        tg0 = TaskGroup()
        k_in_p.fill(k, k_in_tap, group=tg0)
        v_in_p.fill(v, v_in_tap, group=tg0)
        k_out_c.drain(kc, k_out_tap, group=tg0, wait=True, offset_parameter=kv_off)
        v_out_c.drain(vc, v_out_tap, group=tg0, wait=True, offset_parameter=kv_off)
        tg0.finish()

        # Phase 1: op_scores' A/B fills, and its sc output drain. sc is NOT resident (see "Why
        # nothing stays on-chip"); must complete before op_softmax's fill (phase 2) reads it back.
        tg1 = TaskGroup()
        for c in range(cols):
            scA_p[c].fill(kc, scA_taps[c], group=tg1)
            scB_p[c].fill(q, scB_tap, group=tg1)
        for c in range(cols):
            sc_c[c].drain(sc_scr, sc_out_taps[c], group=tg1, wait=True)
        tg1.finish()

        # Phase 2: softmax's sc input fill and sw output drain. Must complete before op_ctx reads
        # sw back (phase 3).
        tg2 = TaskGroup()
        for c in range(cols):
            sc_p[c].fill(sc_scr, sc_in_taps[c], group=tg2)
        for c in range(cols):
            sw_c[c].drain(sw_scr, sw_taps[c], group=tg2, wait=True)
        tg2.finish()

        # Phase 3: op_ctx's A (vc)/W (sw) fills, and its cx output drain. cx is not resident
        # either (see module docstring); must complete before op_o's fill (phase 4) reads it back.
        tg3 = TaskGroup()
        for c in range(cols):
            tA_p[c].fill(vc, tA_taps[c], group=tg3)
            tW_p[c].fill(sw_scr, tW_taps[c], group=tg3)
        for c in range(cols):
            cx_c[c].drain(cx_scr, cx_out_taps[c], group=tg3, wait=True)
        tg3.finish()

        # Phase 4: op_o's A (Wo)/B (cx) fills and the group's one DRAM output.
        tg4 = TaskGroup()
        for c in range(cols):
            oA_p[c].fill(Wo, oA_taps[c], group=tg4)
            oB_p[c].fill(cx_scr, oB_tap, group=tg4)
        for c in range(cols):
            oC_c[c].drain(a, oC_taps[c], group=tg4, wait=True)
        tg4.finish()

    rt = Runtime(
        sequence,
        [
            kv_tile_ty, kv_tile_ty, cache_ty, cache_ty, q_ty, Wo_ty, a_ty, SC_ty, SW_ty, CX_ty,
            k_in_of.prod(), k_out_of.cons(), v_in_of.prod(), v_out_of.cons(),
            [f.prod() for f in scA_l3_fifos], [f.prod() for f in scB_fifos],
            [f.cons() for f in sc_out_fifos], [f.prod() for f in sc_in_fifos],
            [f.cons() for f in sw_out_fifos],
            [f.prod() for f in tA_fifos], [f.prod() for f in tW_fifos],
            [f.cons() for f in cx_out_fifos],
            [f.prod() for f in oA_fifos], [f.prod() for f in oB_fifos],
            [f.cons() for f in oC_fifos],
        ],
    )

    all_workers = scores_workers + softmax_workers + tmatvec_workers + o_workers
    return Program(dev, rt, workers=all_workers).resolve_program()
