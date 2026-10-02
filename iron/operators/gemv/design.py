# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
import aie.extras.dialects.arith as arith
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.dialects.scf import if_, else_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    WorkerRuntimeBarrier,
    sync_parameters,
)
from iron.common.worker_compat import create_worker

"""
Matrix-vector design

Calls into the mv.cc kernel code. That kernel computes `m_input` output rows per call.


 - cols: Number of AIE columns to split work across
 - M: number of rows in the matrix
 - K: number of columns in the matrix == number of rows in the vector
 - m_input: number of input rows stored on each AIE core == chunk size for data movement of input A
 - m_output: number of output rows stored on each AIE core == chunk size for data movement of output C
 - num_batches: number of iterations of this mat-vec to perform on contiguous matrices and vectors in memory (results concatenated)
"""


# The largest batch_group whose grouped objectFIFO depths are known to place on a core tile.
# MEASURED, not derived -- see the GROUP REUSE block in my_matvec for why the arithmetic cannot be
# done from here, and raise this only by BUILDING the larger value rather than by re-deriving it.
MAX_GROUP_REUSE = 4

# The core's stack and locals, held back from the L1 budget below -- mirrors
# tmatvec/design.py's L1_HEADROOM_BYTES for the same reason: this tree has twice paid for a
# kernel frame that silently overwrote the objectFIFO buffers placed above it.
L1_HEADROOM_BYTES = 4096


def l1_budget_bytes(dev):
    """This core's usable L1, in bytes -- DERIVED from the target model, not a literal.

    tmatvec/design.py's AIE2P_L1_BYTES comment says getLocalMemorySize() is "C++ only" and the
    Python bindings expose no accessor. That is stale for this checkout:
    `aie.dialects.aie.get_target_model(int(dev.resolve())).get_local_memory_size()` IS bound
    (verified 2026-09-09; returns 65536 for npu1, npu2 and every column-count variant of both --
    core-tile local memory is uniform across the family). Deriving it here means a future core
    tile with a different size does not silently inherit this chip's number.
    """
    from aie.dialects.aie import get_target_model

    return get_target_model(int(dev.resolve())).get_local_memory_size()


# --- shim/mem-tile BD field widths, and the coalescing verdict they decide ----------------
# MODULE LEVEL, not local to my_matvec, because the CALLER has to know the verdict before it
# builds: `tile_size_output` is sized against `n_vec` (every objectFIFO depth follows it, see
# l1_footprint_bytes), and a caller choosing a flat or a blocked A layout is choosing between two
# verdicts of `group_reuse_n_vec`. Both were re-derived outside this file until 2026-09-18 -- a
# tiler that never saw batch_group, and prose -- which is the shape a-name-and-a-graph-need-one-
# predicate names: a claim computed by a weaker path than the thing it claims about.
MAX_WRAP = 1023
# The shim NOC tile's BD step field is 20 bits -- not a convention, the hardware width:
# `AIETargetModel.h::AIE2TargetModel::getDmaBdStepBits` returns 20 for ShimNOCTile (17 for a
# MemTile, 13 for a core tile). But the field counts ADDRESS GRANULES, not elements:
# `getAddressGenGranularity()` is 32 bits on AIE2/AIE2P, and `getHardwareStridesWraps` scales an
# element stride by `elemWidth / addressGranularity` before `verifyStridesWraps` compares it.
#
# This bound was written in ELEMENTS and compared against the granule field width, which for
# bf16 is exactly 2x too strict -- and its own FIXME below predicted it ("pull these shim BD
# bounds from the MLIR-AIE target model rather than hard-coding them"). Cost, measured: the
# decode's KV cache has a per-head stride of `alloc * head_dim` elements, so at head_dim=128 the
# element bound caps the allocation at 8191 where the hardware allows 16383 -- one whole
# doubling of the context a wide-allocation decode can address before its reads stop coalescing.
MAX_STRIDE_GRANULES = (1 << 20) - 1
GRAN_ELEMS = 2  # 4-byte shim granularity / 2-byte bf16 element
MAX_STRIDE = MAX_STRIDE_GRANULES * GRAN_ELEMS


def split_run(run, lim=MAX_WRAP, gran=GRAN_ELEMS):
    """Factor a contiguous run into (hi, lo), both <= lim and lo a multiple of gran
    (the address-granularity-aligned inner size), lo maximal. None if no such
    split exists (caller then falls back to the per-batch path)."""
    lo_start = (lim // gran) * gran
    for lo in range(lo_start, 0, -gran):
        if run % lo == 0 and (run // lo) <= lim:
            return (run // lo, lo)
    return None


def coalesce_plan(M, cols, num_batches, batch_group, a_row_width, alloc_M=None, block_size=None):
    """The A/C tap shapes and whether they coalesce, for one (flat or blocked) A layout.

    Returns `(coalesce, A_split, C_split, A_bstride, C_bstride)`; `my_matvec` unpacks all five,
    `group_reuse_n_vec` reads only the first.
    """
    n_matrices = num_batches // batch_group
    _AM = M if alloc_M is None else alloc_M
    _BLK = _AM if block_size is None else block_size
    blocked = _BLK != _AM
    # Under blocking the per-matrix run is no longer one `_AM`-row slab -- A is read as
    # `num_col_blocks` separate `_BLK`-row chunks per column, so A_run/A_split (the flat-run wrap
    # split) do not apply to A at all; only C keeps the flat run/split below. A_bstride here is the
    # LARGEST stride A's blocked tap will actually carry -- block-to-block (`n_matrices*_BLK*
    # a_row_width`), which bounds head-to-head (`_BLK*a_row_width`) too since n_matrices >= 1 -- so
    # the same coalesce/MAX_STRIDE gate below still means "does this tap's widest stride fit the
    # hardware field", just against the blocked stride instead of the flat `_AM*a_row_width` one
    # (which is exactly the quantity blocking exists to avoid bounding by).
    A_run = (M // cols) * a_row_width
    A_bstride = (n_matrices * _BLK * a_row_width) if blocked else (_AM * a_row_width)
    C_run, C_bstride = (M // cols), M
    A_split, C_split = (None if blocked else split_run(A_run)), split_run(C_run)
    # A_bstride steps the dim whose SIZE is `n_matrices` (coalesced_tap's A call), so at one matrix
    # the shim never walks it and this bound refuses a tap the hardware accepts. Measured on
    # Gemma-4's global attention (M=6912 K=512 hkv=1): 3,538,944 over the 2,097,150 bound forced
    # the per-batch fallback, 28.31 MB of KV read per invocation against 7.08 MB unique. Blocked
    # taps keep the check -- there A_bstride is the BLOCK step, live at any n_matrices.
    a_bstride_live = blocked or n_matrices > 1
    coalesce = (
        num_batches > 1
        and num_batches % batch_group == 0
        and (not a_bstride_live
             or (A_bstride <= MAX_STRIDE and A_bstride % GRAN_ELEMS == 0))
        and C_bstride <= MAX_STRIDE
        and C_bstride % GRAN_ELEMS == 0
        and (blocked or A_split is not None)
        and C_split is not None
    )
    return coalesce, A_split, C_split, A_bstride, C_bstride


def group_reuse_n_vec(M, cols, num_batches, batch_group, a_row_width,
                      alloc_M=None, block_size=None):
    """`n_vec`: how many vectors one A delivery is reused for. 1 means no reuse.

    A delivers `n_matrices` times per invocation at n_vec > 1 and `num_batches` times at 1, so
    this is also the A-side delivery ratio between a flat and a blocked layout of the same cache.
    """
    if not (batch_group > 1 and batch_group <= MAX_GROUP_REUSE):
        return 1
    return batch_group if coalesce_plan(M, cols, num_batches, batch_group, a_row_width,
                                        alloc_M, block_size)[0] else 1


def l1_footprint_bytes(m_input, m_output, K, a_row_width, itemsize_in, n_vec):
    """Bytes this design places in one core's L1, by term.

    A, B and C ALL scale with n_vec -- unlike TMatVec's rows_per_chunk, which sizes only its A
    term, GROUP REUSE (see the block below) holds n_vec vectors/tiles on-core at once, so every
    FIFO's depth follows it: A at 2*n_vec, B at n_vec, C at 2*n_vec, matching the ObjectFifo()
    calls below exactly. B and C are always bf16 (2 B/elem); A's itemsize follows weight_dtype --
    a_row_width is already BYTES for a quantized (packed) row and ELEMENTS for bf16, so
    itemsize_in (1 or 2, set where a_row_width itself is) makes both cases the same formula.
    """
    return (
        2 * n_vec * m_input * a_row_width * itemsize_in  # A objectfifo
        + n_vec * K * 2  # B objectfifo, always bf16
        + 2 * n_vec * m_output * 2  # C objectfifo, always bf16
    )


def largest_fitting_n_vec(m_input, m_output, K, a_row_width, itemsize_in, l1_bytes):
    """The largest n_vec in [1, MAX_GROUP_REUSE] that FITS, or 0 if even n_vec=1 does not."""
    ok = [
        n for n in range(1, MAX_GROUP_REUSE + 1)
        if l1_footprint_bytes(m_input, m_output, K, a_row_width, itemsize_in, n)
        + L1_HEADROOM_BYTES <= l1_bytes
    ]
    return max(ok) if ok else 0


def check_l1_fits(dev, m_input, m_output, K, a_row_width, itemsize_in, n_vec, l1_bytes=None):
    """Raise-worthy message if the tiling does not FIT, else None.

    K008: the tiling must FIT, not merely divide -- nothing downstream checks it, and a miss
    surfaces at aiecc as "'aie.tile' op Basic sequential allocation also failed", naming a tile
    and not a size (ports tmatvec/design.py's check_l1_fits; see that file for the first
    measurement of this failure mode). Called with the FINAL n_vec -- already declined to 1 by
    the group_fits_bds gate above if it was going to be -- so unlike that gate's BD-count ceiling,
    there is no further fallback here: a miss is unconditional and must raise.

    Because A, B and C all scale with n_vec (see l1_footprint_bytes), n_vec=1 is always the best
    this knob can do -- there is no "blocking term the knob cannot reach" case the way TMatVec's W
    is independent of rows_per_chunk. So the advice is either the largest n_vec that fits, or,
    once n_vec=1 itself does not fit, that m_input/m_output/K -- not batch_group -- has to shrink.
    """
    budget = l1_bytes if l1_bytes is not None else l1_budget_bytes(dev)
    used = l1_footprint_bytes(m_input, m_output, K, a_row_width, itemsize_in, n_vec)
    if used + L1_HEADROOM_BYTES <= budget:
        return None
    a = 2 * n_vec * m_input * a_row_width * itemsize_in
    b = n_vec * K * 2
    c = 2 * n_vec * m_output * 2
    terms = f"A {a} + B {b} + C {c}"
    fits = largest_fitting_n_vec(m_input, m_output, K, a_row_width, itemsize_in, budget)
    if fits:
        advice = (
            f"largest n_vec (batch_group, capped at MAX_GROUP_REUSE={MAX_GROUP_REUSE}) that fits "
            f"here is {fits}."
        )
    else:
        floor = l1_footprint_bytes(m_input, m_output, K, a_row_width, itemsize_in, 1)
        advice = (
            f"NO n_vec fits, not even 1 (no reuse): A+B+C alone total {floor} B. batch_group "
            f"cannot help, it is already at its floor -- shrink m_input, m_output or K instead."
        )
    return (
        f"GEMV does not fit L1: {used} B ({terms}) + {L1_HEADROOM_BYTES} B headroom exceeds "
        f"{budget} B at m_input={m_input} m_output={m_output} K={K} a_row_width={a_row_width} "
        f"n_vec={n_vec}. " + advice
    )


def my_matvec(
    dev,
    cols,
    M,
    K,
    m_input,
    m_output=None,
    num_batches=1,
    batch_group=1,
    kernel_object="mv.o",
    func_prefix="",
    verbose=False,
    epilogue="none",
    weight_dtype="bf16",
    group_size=0,
    scale_dtype="f32",
    alloc_M=None,
    barrier_chunk=1,
    block_size=None,
    allocation_scheme=None,
    vector_size_parameter=None,
    tiles_rtp=False,
    runtime_k=False,
    k_max=None,
    prologue="none",
    prologue_capability="auto",
    prologue_epsilon=1e-6,
    norm_kernel_object="rms_norm.o",
):
    if m_output is None:
        m_output = m_input

    if verbose:
        print(f"Device: {dev}")
        print(f"Matrix dimensions: M={M}, K={K}")
        print(f"Tiling: m_input={m_input}, m_output={m_output}")
        print(f"Columns: {cols}")

    # RUNTIME K: the core reads K from a per-sequence RTP write instead of a compile-time
    # -DDIM_K, so ONE kernel object (and one device body) serves several K's -- see
    # gemma4-w-device-runtime-k-unsplit. The L1 buffers below are sized for `k_max` (the widest K
    # any merged sequence will use); THIS instance's real K only sizes the L3 taps (the DMA moves
    # exactly K's bytes, never k_max's) and the RTP value the core is told to use. Excludes every
    # other axis that already has its own runtime mechanism (tiles_rtp's M, runtime_m's window,
    # group reuse's batching) -- composing them is unbuilt, not needed by this family (down and
    # both o_proj geometries share one M=D and one m_output/m_input already).
    if runtime_k:
        assert weight_dtype != "bf16", "runtime_k is for a quantized A only"
        assert num_batches == 1 and batch_group == 1, "runtime_k is single-batch, no group reuse"
        assert not tiles_rtp and vector_size_parameter is None, (
            "runtime_k does not compose with tiles_rtp or runtime_m yet"
        )
        assert k_max is not None and k_max >= K, "runtime_k needs k_max >= K"

    # The reason for the following requirement is because we first acquire output rows from the C FIFO, then fill those acquiring rows of the A input.
    assert (
        m_output % m_input == 0 and m_output >= m_input
    ), "m_output must be a multiple of m_input"
    assert m_output <= M // cols, "m_output must be less than or equal to M/cols"
    assert (M // cols) % m_output == 0, "m_output must evenly divide M/cols"
    assert m_input <= M // cols, "m_input must be less than or equal to M/cols"
    assert (M // cols) % m_input == 0, "m_input must evenly divide M/cols"

    vectorized = True
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"
    # B (the vector) and C (the output) are always bf16 -- weight_dtype is an axis on A (the MxK
    # weight matrix) only, never on the activation/output path.
    dtype_b = np.dtype[bfloat16]

    assert M % cols == 0

    # itemsize_in tracks dtype_in's byte width by hand: np.dtype[bfloat16]/np.dtype[np.int8] are
    # typing GenericAliases at runtime (numpy's __class_getitem__), not real dtype instances, so
    # neither has a usable .itemsize -- the L1 check below needs the width as a plain int.
    if weight_dtype == "bf16":
        dtype_in = np.dtype[bfloat16]
        dtype_in_str = "bf16"
        a_row_width = K  # elements/row, dtype_in-sized
        itemsize_in = 2
    else:
        # Group-quantized A: see iron/common/quant.py for the exact byte layout
        # (`[n_groups x f32 scale][payload]` per row) and aie_kernels/generic/mv_quant.cc for the
        # device-side dequant. Packing the scale into A's own buffer (rather than a 3rd FIFO) is
        # forced by the 2-input-DMA-channel budget: A and B already spend both.
        from iron.common.quant import row_stride_bytes

        assert weight_dtype in ("int4", "int8", "int4a", "int8a"), \
            f"unknown weight_dtype {weight_dtype!r}"
        assert num_batches == 1, (
            "GEMV weight_dtype != 'bf16' does not support num_batches>1 yet -- the per-column "
            "A layout assumes a single contiguous [row_stride-byte rows] slab per column "
            "(YAGNI: no quantized decode caller needs batching today)"
        )
        assert group_size > 0, "weight_dtype != 'bf16' needs an explicit group_size"
        dtype_in = np.dtype[np.int8]
        dtype_in_str = weight_dtype
        # bytes/row, int8-sized. scale_dtype moves this where the layout axis does not: layout
        # reorders the same bytes, a narrower scale shrinks the row, so it reaches the L1 fit.
        a_row_width = row_stride_bytes(K, group_size, weight_dtype, scale_dtype)
        itemsize_in = 1

    # runtime_k: the L1 buffers below are ONE physical ObjectFifo shared by every merged
    # sequence's device body (merge_devices needs the declared buffer TYPE identical across
    # them), so they are sized for k_max -- the widest K any sequence uses -- never this
    # instance's own K. `a_row_width`/`K` stay THIS instance's real values and size every L3
    # (DDR) tap below unchanged: the DMA moves exactly this instance's bytes, never k_max's.
    a_row_width_l1, K_l1 = (
        (row_stride_bytes(k_max, group_size, weight_dtype, scale_dtype), k_max)
        if runtime_k else (a_row_width, K)
    )

    L1_A_ty = np.ndarray[
        (
            m_input,
            a_row_width_l1,
        ),
        dtype_in,
    ]
    L1_B_ty = np.ndarray[(K_l1,), dtype_b]
    L1_C_ty = np.ndarray[(m_output,), dtype_out]
    # `batch_group` consecutive batches SHARE one matrix, so A holds num_batches//batch_group of
    # them, not num_batches. This is the GQA case: gqa_group query heads attend to one kv head, and
    # today that sharing is expressed by materialising a duplicate (Repeat) instead of by an access
    # pattern. batch_group=1 is the old behaviour exactly.
    assert num_batches % batch_group == 0, (
        f"num_batches ({num_batches}) must be a multiple of batch_group ({batch_group})"
    )
    n_matrices = num_batches // batch_group
    # A's rows ALLOCATED per matrix, which is not always the rows COMPUTED. The two differ whenever
    # a narrow window is read out of a buffer sized for a wider one -- decode attention reads
    # n_past rows of a KV cache allocated at max_seq. `M` stays the compute extent (it sizes the
    # run, the C tile and the core loop); `alloc_M` sizes the buffer and the per-matrix stride, so
    # a narrow read addresses the wide buffer correctly instead of walking off its own smaller one.
    assert alloc_M is None or alloc_M >= M, (
        f"alloc_M ({alloc_M}) must be >= M ({M}): it is the ALLOCATED row count, not a second window"
    )
    _AM = M if alloc_M is None else alloc_M
    # BLOCKED matrix storage: instead of each matrix's `_AM` rows sitting contiguous (per-matrix
    # stride `_AM*a_row_width`, which SCALES WITH `_AM` and is what overflows a narrow hardware
    # stride field at a wide allocation -- see kv-cache-layout-for-full-context), the `_AM` rows are
    # stored in `_AM//block_size` BLOCKS of `block_size` rows, `n_matrices` interleaved every block:
    # block-major, matrix-minor, row innermost. `block_size is None` (default) is ONE block == the
    # whole allocation -- byte-identical to the pre-blocking layout; every existing caller is
    # unaffected. This is a GENERIC access-pattern capability (not KV-cache-specific -- the KV
    # cache's own addressing, T-derivation and runtime offset live in iron.common.kv_layout, which
    # gen_llm_decode.py asks to compute the `alloc_M`/`block_size` this design is handed).
    assert block_size is None or (block_size > 0 and _AM % block_size == 0), (
        f"block_size ({block_size}) must be a positive divisor of alloc_M ({_AM})"
    )
    _BLK = _AM if block_size is None else block_size
    blocked = _BLK != _AM
    L3_A_ty = np.ndarray[
        (n_matrices * _AM * a_row_width,),
        dtype_in,
    ]
    L3_B_ty = np.ndarray[(num_batches * K,), dtype_b]
    L3_C_ty = np.ndarray[(num_batches * M,), dtype_out]
    # Only "on" ever reads this: an "off" instance's sequence dummy-refills B's own tap instead
    # (see sequence() below), so it never binds a Gain argument.
    L3_Gain_ty = np.ndarray[(K,), dtype_b] if prologue == "on" else None
    # "residual" needs three more real L3 arguments beyond (A, B=o, C): the residual stream x and
    # both gains. x1 (the fourth) is an OUTPUT, typed further down by C's own convention.
    L3_X_ty = np.ndarray[(K,), dtype_b] if prologue == "residual" else None
    L3_GainA_ty = np.ndarray[(K,), dtype_b] if prologue == "residual" else None
    L3_GainB_ty = np.ndarray[(K,), dtype_b] if prologue == "residual" else None
    L3_X1_ty = np.ndarray[(K,), dtype_out] if prologue == "residual" else None

    func_type = "vectorized" if vectorized else "scalar"
    matvec = Kernel(
        f"{func_prefix}matvec_{func_type}_{dtype_in_str}_{dtype_out_str}"
        + ("_rtk" if runtime_k else ""),
        f"{func_prefix}{kernel_object}",
        ([np.int32, np.int32, np.int32] if runtime_k else [np.int32, np.int32])
        + [L1_A_ty, L1_B_ty, L1_C_ty],
    )
    # Optional fused activation over the full m_output C-tile, applied once per tile in core_body
    # (after the matvec inner-loop has filled all rows) rather than per matvec call, whose m_input
    # tile can be smaller than the 16-wide activation vector.
    assert epilogue in ("none", "gelu", "silu")
    gelu_kernel = None
    if epilogue != "none":
        # 32, not 16: both tile epilogues walk the C tile with a 32-lane iterator, so a tile that is
        # only 16-aligned makes the last iteration read and write past its end.
        assert (
            m_output % 32 == 0
        ), f"{epilogue} epilogue needs m_output % 32 == 0 (got {m_output})"
        gelu_kernel = Kernel(
            f"{func_prefix}{epilogue}_tile_bf16",
            f"{func_prefix}{kernel_object}",
            [np.int32, L1_C_ty],
        )

    # PROLOGUE: an optional norm/residual-fusion on B, redundant per core (each already holds the
    # whole B vector in L1 -- gemma4-12b-decode-map.md S6). "off"/"on"/"residual" are RUNTIME modes
    # (an RTP word) of ONE compiled body; design_key() keys on CAPABILITY, not mode, so e.g. Wqkv
    # ("on") and gate/up ("residual") can share a merged device once the family's capability is
    # uniformly "residual". CAPABILITY sizes the body ("prenorm": 2 objects, Stage 1; "residual": 4
    # objects + an x1 output, Stage 2 -- formula and layout at rms_norm_residual_add's own
    # docstring) and must be the SAME across a merged family; a mode that needs fewer objects
    # dummy-fills the rest from B.
    assert prologue in ("none", "off", "on", "residual")
    _CAP_OBJECTS = {"none": None, "prenorm": 2, "residual": 4}
    if prologue_capability == "auto":
        prologue_capability = {"none": "none", "off": "prenorm", "on": "prenorm",
                               "residual": "residual"}[prologue]
    assert prologue_capability in _CAP_OBJECTS, f"unknown prologue_capability {prologue_capability!r}"
    assert (prologue_capability == "none") == (prologue == "none")
    assert prologue != "residual" or prologue_capability == "residual"

    norm_kernel = copy_kernel = residual_kernel = None
    if prologue_capability != "none":
        assert tiles_rtp, "prologue rides tiles_rtp's barrier+RTP channel for its mode word"
        assert batch_group == 1 and num_batches == 1, (
            "prologue is unbuilt under group reuse/batching -- gemma4-12b's merged W device is "
            "single-vector (num_batches=1, batch_group=1), the only case this covers"
        )
        norm_kernel = Kernel(
            f"{func_prefix}weighted_rms_norm", f"{func_prefix}{norm_kernel_object}",
            [L1_B_ty, L1_B_ty, L1_B_ty, np.int32, np.float32],
        )
        copy_kernel = Kernel(
            f"{func_prefix}bf16_copy_vector", f"{func_prefix}{norm_kernel_object}",
            [L1_B_ty, L1_B_ty, np.int32],
        )
    if prologue_capability == "residual":
        residual_kernel = Kernel(
            f"{func_prefix}rms_norm_residual_add", f"{func_prefix}{norm_kernel_object}",
            [L1_B_ty, L1_B_ty, L1_B_ty, L1_B_ty, np.int32, np.float32],
        )

    # Elements per granule is a property of the ELEMENT TYPE, and this file hardcodes the bf16 ratio.
    # A quantized A is i8-typed (4 elements per granule), so the conversion would differ -- but
    # `num_batches` is asserted ==1 for a non-bf16 weight_dtype, which makes `coalesce` False and
    # this arithmetic unreachable on that path. Asserted rather than left to a comment.
    assert dtype_in_str == "bf16" or num_batches == 1, (
        f"coalescing arithmetic assumes {GRAN_ELEMS} elements/granule (bf16); "
        f"dtype_in={dtype_in_str} with num_batches={num_batches} would need its own ratio"
    )
    coalesce, A_split, C_split, A_bstride, C_bstride = coalesce_plan(
        M, cols, num_batches, batch_group, a_row_width, alloc_M, block_size)
    if blocked:
        assert (M // cols) % _BLK == 0, (
            f"blocked GEMV needs each column's share of M ({M // cols}) to be a whole number of "
            f"blocks (block_size={_BLK}) -- got M={M} cols={cols} block_size={_BLK}"
        )

    # GROUP REUSE: consume the shared matrix ONCE and run `batch_group` vectors over it, instead of
    # re-streaming it per group member.
    #
    # `batch_group` exists so GQA does not need a `Repeat` op materialising a duplicate KV cache in
    # DDR. It did save that copy -- and it did NOT save the read: the matrix operand carried an
    # outer BD dim of `batch_group` at STRIDE 0, and the shim DMA has no cache, so the cache was
    # physically streamed once per group member. Measured on device: at equal DDR bytes, one matrix
    # read sixteen times costs the same as sixteen distinct matrices read once (269.3 vs 268.0 us),
    # so the repeat was paid in full. On the Qwen3-0.6B decode that is 117.4 MB/token.
    #
    # Inverting the loop nesting -- A outer, the group inner -- removes it, and removes the
    # permutation with it: the [group, matrix] ordering was FORCED by the repeat (only the
    # outermost BD dim may carry a zero stride), and it is what made B need a 4-D wrap-capped tap.
    # With the matrix held, iteration order is plain matrix-major, output lands at
    # `q = batch_group*matrix + member` by construction, and A, B and C are all flat again.
    #
    # Gated on `coalesce` only to keep the change to one path: without it the per-batch fallback
    # would need its own restructuring for A and C to iterate different counts. That combination
    # keeps the old repeat, which is correct and slower, and shows up as unchanged DDR bytes in
    # `decode_ddr_bytes.py` rather than as silence.
    #
    # AND gated on a MEASURED batch_group ceiling, because the three FIFO depths below all follow
    # n_vec (A at 2*n_vec, B at n_vec, C at 2*n_vec) and they share ONE budget: the objectFIFO
    # lowering places one MemOp per core tile and every endpoint targeting that tile appends BD
    # blocks into the same region, which HasValidBDs then counts cumulatively.
    #
    # The budget is the CORE tile's, and say so, because the number is ambiguous on this chip:
    # `AIE2TargetModel::getNumBDs` returns `MemTile ? 48 : 16`, so a core tile and a shim NOC tile
    # both have 16 and a MemTile has 48. A shim-side bound of 16 elsewhere in this file is a
    # DIFFERENT resource that happens to share the number, not the same pool.
    #
    # The ceiling is MEASURED, not derived, and that distinction is the point. The obvious model --
    # 5*batch_group BDs against 16 -- is REFUTED: it predicts batch_group 4 needs 20 and fails, and
    # batch_group 4 builds. Counting `^bb` blocks per `aie.mem` in a lowered design shows why the
    # arithmetic cannot be done this way from here: at n_vec 1 a core tile already carries 8-9
    # blocks, so there is baseline traffic this operator does not know about, and an objectFIFO's
    # depth is not one BD per unit of depth either. Deriving the real bound means asking the target
    # model what the tile has and what else is already placed on it -- which is exactly what this
    # file's own FIXME asks for ("pull these shim BD bounds from the MLIR-AIE target model rather
    # than hard-coding them") and what the sibling MAX_STRIDE constant was just fixed for getting
    # wrong (6f48e87: the bound was in elements where the field counts address granules).
    #
    # So this records the measurements and nothing more. Measured 2026-09-09 on the aie2p decode
    # rail: batch_group 4 (gemma3-270m, 4 q heads over 1 kv head) BUILDS with reuse on; batch_group
    # 16 (gemma4-12b's global layers, 16 over 1) does NOT -- aiecc dies with "'aie.mem' op has more
    # than 16 blocks" naming `B_L3L1_0` at depth 16, a diagnostic that never mentions batch_group.
    # 5 through 15 are UNTESTED; there is no shipped geometry in that range to test with, so the
    # ceiling sits at the largest value known to work rather than at a guess about where it breaks.
    #
    # Capping the depths instead is not available: the core body does `acquire(n_vec)` on B and C,
    # so depth >= n_vec is a precondition of the loop, and a narrower group means sub-tiling the
    # group loop rather than turning a knob.
    group_fits_bds = batch_group <= MAX_GROUP_REUSE
    # Read from the shared predicate rather than recomputed from `coalesce` here: the caller sized
    # its tile_size_output against the same call, and a second spelling of this expression is how
    # the two would disagree about an L1 footprint that only one of them checks.
    n_vec = group_reuse_n_vec(M, cols, num_batches, batch_group, a_row_width, alloc_M, block_size)
    group_reuse = n_vec > 1
    # Say so when the reuse is declined. A silent downgrade here is a per-token DDR regression that
    # nobody can attribute later without re-deriving this by hand.
    #
    # Print-and-continue rather than raise, which is the OPPOSITE of what the sibling TMatVec
    # operator does on its analogous L1 check (`tmatvec/op.py` raises ValueError; it has no
    # fallback and prints nothing). The two differ because the outcomes differ: declining the reuse
    # is safe -- the ungrouped path is the pre-existing mechanism and was verified byte-identical on
    # device -- whereas TMatVec's check guards a shape that would not fit at all.
    if batch_group > 1 and coalesce and not group_fits_bds:
        print(f"[gemv] group_reuse DECLINED: batch_group={batch_group} > the measured-good "
              f"ceiling {MAX_GROUP_REUSE}; falling back to the ungrouped read (M={M} K={K})")

    # K008, the sibling gap this operator had until now: nothing checked that the tiling FITS L1,
    # only that shapes divide evenly (the asserts at the top of this function). n_vec is FINAL
    # here -- already declined to 1 above if the BD ceiling was going to decline it -- so raise
    # rather than print: unlike that decline, an L1 miss has nothing smaller left to fall back to.
    _msg = check_l1_fits(dev, m_input, m_output, K_l1, a_row_width_l1, itemsize_in, n_vec)
    if _msg is not None:
        raise ValueError(_msg)

    # PROLOGUE L1: (n_obj-1) extra B objects beyond check_l1_fits' own n_vec=1 B term, plus the
    # b_norm scratch every mode writes through, plus (residual only) the x1 output object column 0
    # drains. check_l1_fits above priced none of these.
    if prologue_capability != "none":
        _n_obj = _CAP_OBJECTS[prologue_capability]
        _prologue_extra = (_n_obj - 1) * K * 2 + K * 2
        if prologue_capability == "residual":
            _prologue_extra += K * 2  # x1 output, depth 1 -- one shot per dispatch, no pipelining
        _budget = l1_budget_bytes(dev)
        _base = l1_footprint_bytes(m_input, m_output, K, a_row_width, itemsize_in, n_vec)
        if _base + _prologue_extra + L1_HEADROOM_BYTES > _budget:
            raise ValueError(
                f"GEMV+prologue does not fit L1: base {_base} B + prologue {_prologue_extra} B + "
                f"{L1_HEADROOM_BYTES} B headroom exceeds {_budget} B at K={K} "
                f"capability={prologue_capability!r}"
            )

    if blocked and not group_reuse:
        # Scope boundary, not a hardware limit: the blocked tap below reuses d3 (the iteration
        # dim), which group_reuse's A already leaves as a dead [outer=1] placeholder -- see
        # coalesced_tap's call for A. The non-group_reuse coalesced path uses d3 for a REAL
        # batch_group repeat (stride 0, only ever read on A's old defect path) and the per-batch
        # fallback (coalesce=False) has no iterated BD to extend at all; both would need their own
        # dimension budget worked out, unexercised by the shipped decode graph, so this refuses
        # loud instead of silently addressing A wrong.
        raise NotImplementedError(
            "blocked GEMV (block_size != alloc_M) is only implemented for the group_reuse path "
            f"(batch_group={batch_group} > 1 and coalesce=True); got group_reuse={group_reuse}"
        )

    # A's depth follows n_vec too, and it is NOT cosmetic. Reusing the tile means the core spends
    # n_vec matvec calls on each one, so a depth-2 fifo lets the DMA run only one tile ahead of a
    # core that now takes n_vec times as long to drain it -- and the stream stalls. Measured on
    # device at depth 2: the fix halved the DDR bytes (8.487 -> 4.293 MB on the scores arm) and
    # halved the achieved bandwidth with them (44.3 -> 23.1 GB/s), for a net ~zero. The prefetch
    # window has to grow with the work per tile. Costs 1 KB of L1 per extra slot on that arm.
    A_L3L1_fifos = [
        ObjectFifo(L1_A_ty, name=f"A_L3L1_{i}", depth=2 * n_vec) for i in range(cols)
    ]
    # B and C likewise: the core holds n_vec vectors and output tiles at once. At n_vec == 1 all
    # three are 2, 1 and 2 -- exactly as before. A prologue-capable B holds _CAP_OBJECTS[capability]
    # objects regardless of MODE (n_vec is 1 there, asserted above), because the depth is a per-FIFO
    # compile-time property and every merged-family member shares the same compiled core; modes
    # that need fewer objects than the family's capability dummy-fill the rest from B itself (see
    # the sequence below).
    b_acquire_n = _CAP_OBJECTS[prologue_capability] if prologue_capability != "none" else n_vec
    B_L3L1_fifos = [
        ObjectFifo(L1_B_ty, name=f"B_L3L1_{i}", depth=b_acquire_n) for i in range(cols)
    ]
    C_L1L3_fifos = [
        ObjectFifo(L1_C_ty, name=f"C_L1L3_{i}", depth=2 * n_vec) for i in range(cols)
    ]

    # RUNTIME ROW EXTENT -- the one explanation; tmatvec/design.py and both op.py files point here.
    # `M` stays the ALLOCATED and streamed extent, because a shim BD length is a static field on
    # the binary TXN target; the core computes only the rows below `vector_size` and drains the
    # rest untouched. So this buys CORE TIME, never bytes.
    #
    # Skipping is EXACT where the consumer overwrites the skipped rows -- decode attention's
    # softmax writes -inf over everything past its own mask before any exp2. A caller whose
    # consumer READS those rows must not use this.
    #
    # Rows split across columns CONTIGUOUSLY, so a short dispatch idles the later columns rather
    # than sharing the work: cost is min(vector_size, M/cols), not vector_size/cols.
    runtime_m = vector_size_parameter is not None
    # TILE COUNT FROM THE SEQUENCE: each core computes as many output tiles as its own RTP says, and
    # the sequence streams exactly that many. The core program then carries no M, so designs that
    # differ only in M have identical bodies and can share one device with one named sequence each.
    assert not (tiles_rtp and runtime_m), "tiles_rtp and vector_size_parameter are exclusive"
    assert not tiles_rtp or num_batches == 1, "tiles_rtp is single-batch"
    # PROLOGUE MODE rides in tiles_rtp[1] -- a second RTP word under the SAME barrier, rather than
    # a barrier of its own, because the two are always set together by this design's own sequence
    # (see below) and a second WorkerRuntimeBarrier would be a second lock/BD for no new capability.
    _rtp_words = 2 if prologue_capability != "none" else 1
    tiles_rtps = ([Buffer(np.ndarray[(_rtp_words,), np.dtype[np.int32]], name=f"tiles_rtp_{i}",
                          use_write_rtp=True) for i in range(cols)] if tiles_rtp else [])
    b_norm_bufs = ([Buffer(type=L1_B_ty, name=f"B_norm_{i}") for i in range(cols)]
                  if prologue_capability != "none" else [])
    # Column 0 only (see the PROLOGUE comment above): one output object, no pipelining needed --
    # this drains once per dispatch, not once per output tile the way C does.
    X1_L1L3_fifo = (ObjectFifo(L1_B_ty, name="X1_L1L3", depth=1)
                    if prologue_capability == "residual" else None)
    vs_param = (
        ScratchpadParameter(vector_size_parameter, np.int32) if runtime_m else None
    )
    # K FROM THE SEQUENCE, same idiom as tiles_rtp's tile count: each core reads which K this
    # dispatch is for from a write-RTP the sequence sets before it runs, so the core program
    # carries no K and can share a device across several K's -- see the runtime_k assertions above.
    k_rtps = ([Buffer(np.ndarray[(1,), np.dtype[np.int32]], name=f"k_rtp_{i}",
                      use_write_rtp=True) for i in range(cols)] if runtime_k else [])
    n_tiles = M // m_output // cols

    n_j = m_output // m_input

    def _ceil_clamp(rem, gran, hi):
        up = arith.addi(rem, arith.constant(gran - 1, T.i32()))
        return arith.maxsi(
            arith.minsi(arith.divsi(up, arith.constant(gran, T.i32())),
                        arith.constant(hi, T.i32())),
            arith.constant(0, T.i32()),
        )

    def active_tiles(width, col):
        """Output tiles this column must compute for `width` rows, CEIL, clamped to [0, n_tiles]."""
        return _ceil_clamp(
            arith.subi(width, arith.constant(col * (M // cols), T.i32())), m_output, n_tiles
        )

    def active_calls(width, col, i_i32):
        """Kernel calls inside tile `i`. Narrowing ONLY the tile loop is not enough: gemv_tile_output
        picks the largest legal C tile, and at the decode scores shape that is the whole per-column
        slab (n_tiles == 1), so the outer bound can only ever be 0 or everything."""
        base = arith.addi(arith.constant(col * (M // cols), T.i32()),
                          arith.muli(i_i32, arith.constant(m_output, T.i32())))
        return _ceil_clamp(arith.subi(width, base), m_input, n_j)

    def make_core_body(col):
        # `*rest` rather than defaulted names: both tails are optional and independent, so a fixed
        # positional order binds the wrong one for one of the four combinations. Unpacked in the
        # SAME order worker_args builds them.
        def core_body(A_L3L1_fifo, B_L3L1_fifo, C_L1L3_fifo, matvec, *rest):
            rest = list(rest)
            gelu_kernel = rest.pop(0) if epilogue != "none" else None
            vs_src, barrier = ((rest.pop(0), rest.pop(0)) if (runtime_m or tiles_rtp or runtime_k)
                               else (None, None))
            norm_k = copy_k = res_k = b_norm = x1_fifo = None
            if prologue_capability != "none":
                norm_k, copy_k, b_norm = rest.pop(0), rest.pop(0), rest.pop(0)
            if prologue_capability == "residual":
                res_k = rest.pop(0)
                if col == 0:
                    x1_fifo = rest.pop(0)
            one_idx = index.constant(1)
            k_val = None
            for _ in range_(0xFFFFFFFF):  # batch dim handled as part of this loop
                if runtime_m:
                    # Read AFTER the barrier, never before: the sequence syncs the scratchpad and only
                    # then sets it, so a read here sees THIS dispatch's value.
                    barrier.wait_for_value(1)
                    vs = vs_src.read()
                    n_active = active_tiles(vs, col)
                elif tiles_rtp:
                    barrier.wait_for_value(1)
                    n_active = vs_src[0]
                    if prologue_capability != "none":
                        mode = vs_src[1]
                    # Acquire does not consume the lock: move it off 1 now, before any output, so the
                    # next run waits for its own sequence's count instead of reusing this one.
                    barrier.release_with_value(1)
                elif runtime_k:
                    barrier.wait_for_value(1)
                    k_val = vs_src[0]
                    barrier.release_with_value(1)
                b = B_L3L1_fifo.acquire(b_acquire_n)
                if prologue_capability == "residual":
                    # b[0]=o, b[1]=x, b[2]=gain_a, b[3]=gain_b under mode 2 (or the sequence's own
                    # dummy re-fills under mode 0/1 -- see my_matvec's sequence()). Every branch
                    # leaves the shared row-tile loop below reading ONE buffer, b_norm.
                    with if_(mode == 2) as is_res:
                        res_k(b[0], b[2], b[1], b[0], K, prologue_epsilon)  # x1 -> b[0], in place
                        norm_k(b[0], b[3], b_norm, K, prologue_epsilon)     # h -> b_norm
                        if col == 0:
                            x1o = x1_fifo.acquire(1)
                            copy_k(b[0], x1o, K)
                            x1_fifo.release(1)
                    with else_(is_res):
                        with if_(mode == 1) as is_rms:
                            norm_k(b[0], b[1], b_norm, K, prologue_epsilon)
                        with else_(is_rms):
                            copy_k(b[0], b_norm, K)
                    b = b_norm
                elif prologue_capability != "none":
                    # b[0] is the activation, b[1] the gain (or, in "off" mode, the sequence's own
                    # dummy re-fill of b[0] -- see my_matvec's sequence()). Either branch leaves the
                    # shared row-tile loop below reading ONE buffer, b_norm, regardless of mode.
                    with if_(mode == 1) as is_rms:
                        norm_k(b[0], b[1], b_norm, K, prologue_epsilon)
                    with else_(is_rms):
                        copy_k(b[0], b_norm, K)
                    b = b_norm
                # The kernel function computes m output rows; each core is responsible for (M/cols) output rows, so we need to call the kernel (M/cols)/m times.
                for i_idx in range_(n_active if (runtime_m or tiles_rtp) else n_tiles):
                    c = C_L1L3_fifo.acquire(n_vec)
                    i_i32 = index.casts(T.i32(), i_idx)
                    if runtime_m:
                        n_j_active = active_calls(vs, col, i_i32)
                    for j_idx in range_(n_j_active if runtime_m else n_j):
                        j_i32 = index.casts(T.i32(), j_idx)
                        output_row_offset = j_i32 * m_input
                        a = A_L3L1_fifo.acquire(1)
                        # The A tile is acquired ONCE and every vector in the group runs over it. The
                        # group is a python-level unroll because batch_group is a build constant, and
                        # because `b`/`c` are indexable views only when more than one was acquired.
                        if n_vec == 1:
                            if runtime_k:
                                matvec(m_input, output_row_offset, k_val, a, b, c)
                            else:
                                matvec(m_input, output_row_offset, a, b, c)
                        else:
                            for g in range(n_vec):
                                matvec(m_input, output_row_offset, a, b[g], c[g])
                        A_L3L1_fifo.release(1)
                    if runtime_m:
                        for _ in range_(arith.subi(arith.constant(n_j, T.i32()), n_j_active)):
                            A_L3L1_fifo.acquire(1)
                            A_L3L1_fifo.release(1)
                    if gelu_kernel is not None:
                        if n_vec == 1:
                            gelu_kernel(m_output, c)
                        else:
                            for g in range(n_vec):
                                gelu_kernel(m_output, c[g])
                    C_L1L3_fifo.release(n_vec)
                if runtime_m:
                    # The fill and the drain are both sized to n_tiles, so the surplus has to move
                    # through the fifos anyway; what is saved is the MACs, not the bytes.
                    for _ in range_(arith.subi(arith.constant(n_tiles, T.i32()), n_active)):
                        C_L1L3_fifo.acquire(n_vec)
                        for _j in range_(n_j):
                            A_L3L1_fifo.acquire(1)
                            A_L3L1_fifo.release(1)
                        C_L1L3_fifo.release(n_vec)
                B_L3L1_fifo.release(b_acquire_n)

        return core_body

    # STACK. The bf16 and symmetric paths take the device default (1024 B) and are left alone --
    # raising it would move every existing build's L1 layout. The AFFINE kernels keep one float
    # per quant group on the stack (mv_quant.cc's `float bsum[n_groups]`, the per-group sums of
    # B), so they need the default plus that array. Sized here rather than guessed: aiecc measures
    # each core's real requirement and fails the build when stack_size is short, and it DID --
    # K=1024 g=32 needs 1088 against the 1024 default, which is bsum's 128 B minus the slack the
    # frame already had. The rounding to 64 B keeps the L1 allocator's granule.
    _stack_kw = {}
    if weight_dtype in ("int4a", "int8a"):
        _bsum = 4 * (K // group_size)
        _stack_kw["stack_size"] = 1024 + -(-_bsum // 64) * 64

    barriers = ([WorkerRuntimeBarrier() for _ in range(cols)]
                if (runtime_m or tiles_rtp or runtime_k) else [])

    workers = [
        create_worker(
            make_core_body(i),
            [
                A_L3L1_fifos[i].cons(),
                B_L3L1_fifos[i].cons(),
                C_L1L3_fifos[i].prod(),
                matvec,
            ]
            + ([gelu_kernel] if epilogue != "none" else [])
            + ([vs_param, barriers[i]] if runtime_m else [])
            + ([tiles_rtps[i], barriers[i]] if tiles_rtp else [])
            + ([k_rtps[i], barriers[i]] if runtime_k else [])
            + ([norm_kernel, copy_kernel, b_norm_bufs[i]] if prologue_capability != "none" else [])
            + ([residual_kernel] if prologue_capability == "residual" else [])
            + ([X1_L1L3_fifo.prod()] if (prologue_capability == "residual" and i == 0) else []),
            allocation_scheme=allocation_scheme,
            **_stack_kw,
        )
        for i in range(cols)
    ]

    # Distribution pattern for the input matrix A: each AIE core gets a contiguous chunk of rows.
    # The input matrix in DDR is MxK-sized (row-major); each core processes (M/cols)xK-sized matrices in chunks of mxK-sized tiles.
    # The chunking into mxK-sized tiles happens in the ObjectFIFO; the shim puts all data on the stream in sequence.
    # One tap per DELIVERY of the matrix. Reusing the group means that is once per MATRIX;
    # otherwise it stays once per batch, which for batch_group>1 is the same matrix twice.
    n_a_deliveries = n_matrices if group_reuse else num_batches
    A_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_A_ty.__args__[0],
                offset=col * (M // cols) * a_row_width
                + (d if group_reuse else d // batch_group) * _AM * a_row_width,
                sizes=[1, 1, 1, (M // cols) * a_row_width],
                strides=[0, 0, 0, 1],
            )
            for d in range(n_a_deliveries)
        ]
        for col in range(cols)
    ]

    # Every column gets the entirety of the vector B.
    # This design assumes that all of B fits on the cores.
    # B must follow the SAME permutation as C. The core consumes B from its FIFO in ITERATION
    # order, so once the A/C dims are [group, matrix] (forced: only the outermost dim may carry a
    # zero stride) a flat linear B hands step i the vector of head i while A/C are addressing head
    # group*matrix + member. That mismatch is silent -- every head simply gets the wrong query
    # vector -- and it reads as 0/8 parity, not as a near miss.
    #
    # At batch_group=1 this is the old flat read: sizes=[1, num_batches, 1, K] with offset m*K.
    # At batch_group=1 keep the ORIGINAL flat one-dimensional read byte for byte: leading 1s make
    # it a plain contiguous transfer whose length is not subject to the 10-bit wrap cap.
    #
    # When grouping, B must follow the SAME permutation as C -- the core consumes B from its FIFO in
    # ITERATION order, and A/C are forced into [group, matrix] because only the outermost dim may
    # carry a zero stride. A flat B then hands step i the vector of head i while A/C address head
    # group*matrix + member, which is silent and reads as 0/8 parity.
    #
    # That makes it genuinely 4-D, so its innermost size IS wrap-capped and K must be split exactly
    # like a run: op_ctx is gemv(M=head_dim, K=S), and at S=2048 an unsplit K trips
    # "Size 0 exceeds the [0:1023] range". op_scores (K=head_dim=128) never would, which is why the
    # k-only arm built and this one did not.
    # The permutation exists ONLY to satisfy the coalesced tap, whose [group, matrix] dim order is
    # forced (only the outermost dim may carry a zero stride). The per-batch FALLBACK has no such
    # constraint: its A taps address matrix `w // batch_group` in plain batch order, so a permuted B
    # hands delivery i the vector of head `member + batch_group*matrix` while A is on matrix
    # `i // batch_group` -- 14 of 16 deliveries pair the wrong operands.
    #
    # This combination was UNREACHABLE until a wide allocation made `coalesce` false at
    # batch_group>1: every shipped multi-batch GEMV coalesces. It is a defect this file's own
    # comment predicts one paragraph down ("that mismatch is silent -- every head simply gets the
    # wrong query vector -- and it reads as 0/8 parity, not as a near miss") and it went unexercised
    # because nothing could reach the path.
    if batch_group == 1 or group_reuse or not coalesce:
        # Flat, and for group_reuse that is the POINT: the core now consumes vectors in plain batch
        # order (matrix-major, member-inner), which is the order they already sit in, so the
        # permuted 4-D tap below -- and its wrap cap on K -- is not needed.
        B_tap = TensorAccessPattern(
            tensor_dims=L3_B_ty.__args__[0], offset=0,
            sizes=[1, 1, 1, num_batches * K], strides=[0, 0, 0, 1],
        )
    else:
        B_split = split_run(K)
        assert B_split is not None, (
            f"K ({K}) has no wrap-legal split; batch_group>1 needs a 4-D B tap"
        )
        k_hi, k_lo = B_split
        B_tap = TensorAccessPattern(
            tensor_dims=L3_B_ty.__args__[0], offset=0,
            sizes=[batch_group, n_matrices, k_hi, k_lo],
            strides=[K, batch_group * K, k_lo, 1],
        )

    # Gain tap for prologue="on": one flat K-element read, the same shape B_tap takes at
    # batch_group==1 (asserted above, so B_tap is always that flat form when this is built).
    Gain_tap = (
        TensorAccessPattern(
            tensor_dims=L3_Gain_ty.__args__[0], offset=0,
            sizes=[1, 1, 1, K], strides=[0, 0, 0, 1],
        )
        if prologue == "on" else None
    )

    # Same flat K-element shape for "residual"'s three extra real inputs and its x1 output --
    # batch_group==1 here too (asserted in the residual capability check above).
    def _flat_k_tap(ty):
        return TensorAccessPattern(
            tensor_dims=ty.__args__[0], offset=0, sizes=[1, 1, 1, K], strides=[0, 0, 0, 1],
        )
    X_tap = _flat_k_tap(L3_X_ty) if prologue == "residual" else None
    GainA_tap = _flat_k_tap(L3_GainA_ty) if prologue == "residual" else None
    GainB_tap = _flat_k_tap(L3_GainB_ty) if prologue == "residual" else None
    X1_tap = _flat_k_tap(L3_X1_ty) if prologue == "residual" else None

    # Collection pattern for the output vector C: each AIE core writes back its contiguous chunk of rows.
    C_taps = [
        [
            TensorAccessPattern(
                tensor_dims=L3_C_ty.__args__[0],
                offset=col * (M // cols) + batch * M,
                sizes=[1, 1, 1, (M // cols)],
                strides=[0, 0, 0, 1],
            )
            for batch in range(num_batches)
        ]
        for col in range(cols)
    ]

    # Batch coalescing replaces the per-batch unroll with a single iterated BD. The predicate and
    # the run splits are computed above, because `group_reuse` depends on them.
    #
    # Within one delivery the run is contiguous (A_run = (M//cols)*K elements). The stride between
    # deliveries is the full matrix (A_bstride = M*K), so for cols>1 each column gathers its own
    # slice out of every one with a gap in between. The contiguous run is split into two wrap dims
    # [run_hi, run_lo] ONLY to fit the AIE shim's 10-bit (1023) wrap-size cap.
    #
    # FIXME: pull these shim BD bounds from the MLIR-AIE target model rather than
    # hard-coding them; they live in verifyStridesWraps in
    # https://github.com/Xilinx/mlir-aie/blob/main/lib/Dialect/AIEX/IR/AIEXDialect.cpp

    # The outer dim used to be a dead placeholder (size 1, stride 0). It carries the GROUP now:
    # [group_member, matrix, run_hi, run_lo].
    #
    # The group must sit OUTERMOST because only the outer dim may have stride 0 -- aie.dma_bd
    # rejects a zero stride further in ("Stride 2 must be a positive integer"), which is what a
    # [matrix, group] ordering hits. So A repeats the whole matrix sweep per group member (outer
    # stride 0, inner A_bstride), and C compensates: its outer advances ONE batch and its inner
    # skips a whole group, so the output still lands at q = batch_group*matrix + member -- the
    # natural query-head numbering, needing no reordering downstream.
    #
    # At batch_group=1 this is byte-for-byte the original tap: sizes=[1, num_batches, ...],
    # strides=[0, bstride, ...]. A shim BD has exactly four dims and this uses all of them, so a
    # shape whose run needs a third dim cannot coalesce.
    def coalesced_tap(L3_ty, col_off, split, outer_stride, inner_stride, outer=None, inner=None):
        run_hi, run_lo = split
        return TensorAccessPattern(
            tensor_dims=L3_ty.__args__[0],
            offset=col_off,
            sizes=[batch_group if outer is None else outer,
                   n_matrices if inner is None else inner, run_hi, run_lo],
            strides=[outer_stride, inner_stride, run_lo, 1],
        )

    if coalesce:
        # Dropping the per-batch drain wait lets the single iterated fill BD run ahead of
        # the core. ObjectFifo lock backpressure keeps that safe: a producer that gets
        # ahead BLOCKS on the buffer lock (worst case a stall, never a corrupting
        # overrun). depth>=2 only buys OVERLAP of fill with compute, so it is a
        # performance guard here, not a correctness requirement (depth==1 is correct but
        # fully serial).
        assert all(f.depth >= 2 for f in A_L3L1_fifos) and all(
            f.depth >= 2 for f in C_L1L3_fifos
        ), "coalesced GEMV wants A/C ObjectFifo depth>=2 for fill/compute overlap"
        # With the group reused there is no repeat: A walks its matrices with a dead outer dim.
        if blocked:
            # BLOCKED A: the [outer=1, inner=n_matrices] shape above repurposes its dead outer dim
            # (group_reuse's A never uses it -- see coalesced_tap's call: outer_stride is always 0
            # for A) to carry MATRIX instead, freeing the inner dim for BLOCK. Per column, this
            # column's (M//cols) rows are `blocks_per_col` CONSECUTIVE blocks starting at
            # `block_start`; for a fixed matrix the run within one block (`_BLK*a_row_width`
            # elements) still needs the same hi/lo wrap split A_run did before blocking.
            #   sizes   = [n_matrices,        blocks_per_col,         run_hi, run_lo]
            #   strides = [_BLK*a_row_width,  n_matrices*_BLK*a_row_width, run_lo, 1]
            # matches iron.common.kv_layout.KVLayout's head_stride/block_stride one-to-one, with
            # `n_matrices` standing in for that module's `Hkv` -- this design stays KV-agnostic,
            # the caller (gen_llm_decode.py) is what knows this A happens to be a KV cache.
            blocks_per_col = (M // cols) // _BLK
            blk_run_split = split_run(_BLK * a_row_width)
            assert blk_run_split is not None, (
                f"blocked GEMV: no wrap-legal split for one block's run "
                f"({_BLK * a_row_width} elements, block_size={_BLK})"
            )
            blk_run_hi, blk_run_lo = blk_run_split
            head_stride = _BLK * a_row_width
            block_stride = n_matrices * _BLK * a_row_width
            A_taps_coalesced = [
                TensorAccessPattern(
                    tensor_dims=L3_A_ty.__args__[0],
                    offset=(col * blocks_per_col) * block_stride,
                    sizes=[n_matrices, blocks_per_col, blk_run_hi, blk_run_lo],
                    strides=[head_stride, block_stride, blk_run_lo, 1],
                )
                for col in range(cols)
            ]
        else:
            # 0, not a dead A_bstride: the step still reaches the BD's 20-bit granule field.
            A_taps_coalesced = [
                coalesced_tap(L3_A_ty, col * (M // cols) * K, A_split, 0,
                              A_bstride if n_matrices > 1 else 0,
                              *((1, n_matrices) if group_reuse else (None, None)))
                for col in range(cols)
            ]
        if group_reuse:
            # C must follow the CORE's emission order, which is tile-major and batch-MINOR: the
            # core acquires n_vec C objects INSIDE the tile loop and fills one per batch before
            # advancing the tile. A batch-major tap transposes that -- object (tile t, batch b)
            # lands where (b, t) belongs, leaving only the b == t diagonal correct.
            # Only expressible while the GROUP dim is dead. The core loops group -> tile ->
            # batch-in-group, so the general tap needs [n_matrices, n_tiles, n_vec, <tile>],
            # which overflows the 4-D descriptor once a C tile itself needs a wrap split.
            # n_matrices == 1 collapses the group dim; n_tiles == 1 makes the order moot.
            assert n_matrices == 1 or n_tiles == 1, (
                f"coalesced group-reuse GEMV needs batch_group == num_batches (one matrix) or a "
                f"single C tile: got num_batches={num_batches} batch_group={batch_group} "
                f"(n_matrices={n_matrices}) with n_tiles={n_tiles}"
            )
            m_split = split_run(m_output)
            assert m_split is not None, (
                f"coalesced group-reuse GEMV: no wrap-legal split for one C tile "
                f"({m_output} elements)"
            )
            C_taps_coalesced = [
                coalesced_tap(L3_C_ty, col * (M // cols), m_split,
                              m_output, C_bstride, n_tiles, num_batches)
                for col in range(cols)
            ]
        else:
            C_taps_coalesced = [
                coalesced_tap(L3_C_ty, col * (M // cols), C_split,
                              C_bstride, batch_group * C_bstride, None, None)
                for col in range(cols)
            ]

    _PROLOGUE_MODE = {"off": 0, "on": 1, "residual": 2}

    def _sequence_body(A, B, X, GainA, GainB, C, X1, B_L3L1_fifos_prods, A_L3L1_fifos_prods,
                       C_L1L3_fifos_conss, X1_L1L3_fifo_cons):
        if runtime_m:
            sync_parameters()
            for i in range(cols):
                barriers[i].set(1)
        if tiles_rtp:
            for i in range(cols):
                tiles_rtps[i][0] = n_tiles
                if prologue_capability != "none":
                    tiles_rtps[i][1] = _PROLOGUE_MODE[prologue]
                barriers[i].set(1)
        if runtime_k:
            for i in range(cols):
                k_rtps[i][0] = K
                barriers[i].set(1)
        tg_b = TaskGroup()
        for col in range(cols):
            # Simple linear transfer of B, includes all batches in sequence. Under "residual" this
            # slot is the ATTENTION OUTPUT (o), not the residual stream (x) -- see the object-order
            # comment in my_matvec's PROLOGUE block.
            B_L3L1_fifos_prods[col].fill(B, B_tap, group=tg_b)
            if prologue == "on":
                # Second object: the gain, over the SAME channel -- a core has only 2 input DMA
                # channels and both are already A and B (mv_quant.cc's own comment on why the
                # quant scale rides in A rather than a 3rd fifo; this is the same budget). Any
                # further objects up to the family's capability are dummy-refilled from B_tap.
                B_L3L1_fifos_prods[col].fill(GainA, Gain_tap, group=tg_b)
                for _ in range(b_acquire_n - 2):
                    B_L3L1_fifos_prods[col].fill(B, B_tap, group=tg_b)
            elif prologue == "off":
                # The core still acquires the family's full object count (one compiled body,
                # shared with "on"/"residual" instances) but its "off" branch never reads past
                # object 0 -- refill B's own tap for every remaining object rather than add real
                # arguments this instance has no use for.
                for _ in range(b_acquire_n - 1):
                    B_L3L1_fifos_prods[col].fill(B, B_tap, group=tg_b)
            elif prologue == "residual":
                B_L3L1_fifos_prods[col].fill(X, X_tap, group=tg_b)
                B_L3L1_fifos_prods[col].fill(GainA, GainA_tap, group=tg_b)
                B_L3L1_fifos_prods[col].fill(GainB, GainB_tap, group=tg_b)
        # Coalesced: one iterated BD per column covers all batches (num_waits==1, a
        # single drain wait for the whole column). Fallback (incl. num_batches==1): the
        # stock per-batch unroll (num_waits==num_batches, one wait per batch). The fills
        # and drains are otherwise identical; only the TAP and the wait count differ.
        num_waits = 1 if coalesce else num_batches
        # BARRIER CHUNK -- how many batches share one TaskGroup, i.e. one device-side drain wait.
        #
        # The fallback issues num_batches fills+drains per column AND num_batches barriers. Those
        # are two different costs and the wait count is the one nothing forced: the coalesced path
        # already argues, in this file, that dropping the per-batch wait is safe because ObjectFifo
        # lock backpressure blocks a runaway producer rather than corrupting it ("worst case a
        # stall, never a corrupting overrun"). The same fifos and the same locks are in play here.
        #
        # What DOES bound it is the shim's BD budget: batches in flight per column cannot exceed it,
        # and an earlier attempt to hold one fill per object across a whole design exceeded 16 BDs
        # and deadlocked (see tmatvec/design.py's KNOWN DEFECT note). So this is a chunk, not a
        # hoist -- 1 reproduces today's behaviour exactly, and the useful range is bounded above by
        # the BD budget rather than by taste.
        chunk = max(1, min(barrier_chunk, num_waits))
        for w0 in range(0, num_waits, chunk):
            tg_ac = TaskGroup()
            for w in range(w0, min(w0 + chunk, num_waits)):
                for col in range(cols):
                    a_tap = A_taps_coalesced[col] if coalesce else A_taps[col][w]
                    A_L3L1_fifos_prods[col].fill(A, a_tap, group=tg_ac)
                for col in range(cols):
                    c_tap = C_taps_coalesced[col] if coalesce else C_taps[col][w]
                    C_L1L3_fifos_conss[col].drain(
                        C,
                        c_tap,
                        group=tg_ac,
                        wait=True,
                    )
            tg_ac.finish()
        tg_b.finish()
        # x1 out: one object, column 0 only, one drain per dispatch (not per output tile -- this
        # is the residual stream, not a matvec row range).
        if prologue == "residual":
            tg_x1 = TaskGroup()
            X1_L1L3_fifo_cons.drain(X1, X1_tap, group=tg_x1, wait=True)
            tg_x1.finish()

    # Extra real Program arguments are per-mode: "on" takes one gain, "residual" takes the
    # residual stream and both its gains plus the x1 output; "off"/"none" take none (their
    # dummy-refills need no argument at all). Three thin wrappers rather than default parameters:
    # the Runtime's arg-type list below must match each sequence's arity exactly, and that list
    # differs between the three cases.
    # X1_L1L3's CONS endpoint must be bound (i.e. `.cons()` called) whenever the FIFO exists at
    # all -- device_body's tile discovery walks every declared ObjectFifo and requires both
    # endpoints regardless of whether a given instance's sequence ever fills/drains it -- so an
    # "on"/"off" instance sharing "residual" capability takes the handle as a trailing arg it never
    # touches, rather than skipping the call and leaving the fifo half-wired.
    _x1_cons_arg = [[X1_L1L3_fifo.cons()]] if prologue_capability == "residual" else []
    if prologue == "on":
        def sequence(A, B, Gain, C, B_L3L1_fifos_prods, A_L3L1_fifos_prods, C_L1L3_fifos_conss,
                    *x1_unused):
            _sequence_body(A, B, None, Gain, None, C, None, B_L3L1_fifos_prods,
                          A_L3L1_fifos_prods, C_L1L3_fifos_conss, None)
    elif prologue == "residual":
        def sequence(A, B, X, GainA, GainB, C, X1, B_L3L1_fifos_prods, A_L3L1_fifos_prods,
                    C_L1L3_fifos_conss, X1_L1L3_fifos_conss):
            _sequence_body(A, B, X, GainA, GainB, C, X1, B_L3L1_fifos_prods,
                          A_L3L1_fifos_prods, C_L1L3_fifos_conss, X1_L1L3_fifos_conss[0])
    else:
        def sequence(A, B, C, B_L3L1_fifos_prods, A_L3L1_fifos_prods, C_L1L3_fifos_conss,
                    *x1_unused):
            _sequence_body(A, B, None, None, None, C, None, B_L3L1_fifos_prods,
                          A_L3L1_fifos_prods, C_L1L3_fifos_conss, None)

    rt = Runtime(
        sequence,
        [
            L3_A_ty,
            L3_B_ty,
        ]
        + ([L3_Gain_ty] if prologue == "on" else [])
        + ([L3_X_ty, L3_GainA_ty, L3_GainB_ty] if prologue == "residual" else [])
        + [L3_C_ty]
        + ([L3_X1_ty] if prologue == "residual" else [])
        + [
            [of.prod() for of in B_L3L1_fifos],
            [of.prod() for of in A_L3L1_fifos],
            [of.cons() for of in C_L1L3_fifos],
        ]
        + _x1_cons_arg,
    )
    return Program(dev, rt, workers=workers).resolve_program()
