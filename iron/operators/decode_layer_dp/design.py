# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A WHOLE decoder layer as ONE `aie.device`: the fused attention block and the fused SwiGLU MLP,
on DIFFERENT CORES of the same array.

    8 cores   attn_block_dp    norm, QKV, qk-norm, RoPE, KV append, scores, softmax, ctx
    4 cores   swiglu_mlp_dp    Wo, norm, gate/up, SiLU-mul, down, residual   (fuse_o=1)

MEASURED placement (aiecc, this design, 2026-09-09): columns 0-2 x rows 2-5. Attention takes
columns 0-1 and the MLP column 2 -- the placer column-major-fills four rows before moving on, so
the split lands on COLUMNS, not rows. That is worth stating because the layout this was designed
against assumed a row split with idle donor rows between; the requirement was never "rows", it was
only that a core runs ONE phase, and the placer picks how. No donor row exists in the result and
none is needed.

This is the last configure collapse available on the decode rail. With one design per layer the 28
layers are a CONTIGUOUS run of the same design, so they cost **one** `aiex.configure` between them:
3 configures per token (1 + 2 tail) against today's 142, where the two-design split
(attn_block_dp + swiglu_mlp_dp as separate runlist entries) alternates A,B,A,B and costs 58.

WHY IT IS ROWS AND NOT ONE CORE. Fusing both phases onto ONE core is foreclosed by PROGRAM MEMORY,
and that wall does not move: `getProgramMemorySize()` is 0x4000 = 16 KB and the two halves measure
11,760 B and 8,048 B of `.text`, 120.9% of the region together (108.2% after the 89.5% inter-design
code sharing measured on attn_block_dp itself). Unlike L1, program memory cannot be borrowed --
there is no program-memory base address anywhere in the target model, so PM is outside the core's
data address space and a neighbour's cannot be addressed. Splitting the phases across DIFFERENT
CORES sidesteps it entirely: each core carries only its own phase's `.text`, measured at 71.8% and
49.1% of the region -- which is exactly the standalone figure for each half, unchanged by merging.

NO DONOR ROWS. The idle-row layout that would let a compute row borrow a neighbour's 64 KB data
module (`isLegalMemAffinity`: own module plus South-non-memtile, North, West, East, up to 320 KB)
is NOT used, because nothing here overruns: attention needs 45,568 B of L1 and the MLP 50,180 B,
69.5% and 76.6% of one tile's own 64 KB. Neighbour memory is insurance against a budget that does
not exist, and it would additionally need IRON to express a buffer on another tile's module, which
it does not today. Left unbuilt deliberately -- reach for it only if a budget actually overruns.

COLUMN COUNTS DELIBERATELY DIFFER, 8 and 4. The MLP is at 4 columns because 8 was measured slower
twice; attention requires 8 because `Hkv == n_aie_cols` is TMatVec's "one matrix per column" rule
binding on every stage. One `aie.device` does not require one column count, so they are not forced
equal -- 12 of the array's 32 cores are used, and 3 of its 8 columns.

SHIM CHANNELS, the constraint that killed the naive fusion. Separate cores means the two halves do
NOT share their misc/weight/output fifos, so the budget is the SUM, not
the max: attention 9 in / 8 out plus MLP 5 in / 4 out = **14 input and 12 output** of the
device-wide 16 each (`get_shim_dma_limit`). Two input channels spare. This is tighter than either
half alone and is the first thing to re-check if a shape changes.

`cx` STAYS IN DDR, and that is not a defect to fix here. Attention produces it on 8 cores of one
row and the MLP wants it broadcast to 4 cores of another: an 8-source join is oversized for a
single MemTile (6 DMA source/dest connections) and a joined fifo cannot also feed a forward (one
link per fifo). It is 4 KB per layer, already the traffic the shipped graph moves, and the
attention half's own closing barrier already orders it -- so the phase crossing adds NO barrier.

WHAT ACTUALLY BOUNDS THIS DESIGN is none of the above: it is aiecc's `kMaxHostBOs = 16` cap on host
buffer arguments (tools/aiecc/SidecarFiles.h, "conservative, hardware-verified ceiling ... counts
above this are unvalidated and rejected" -- policy, not silicon, and not overridable). The natural
argument list is SEVENTEEN and is rejected outright, after placement, routing and every core ELF
has already been generated. Packing the three STATIC norm gains into one blob takes it to 15. Any
future addition to this device's argument list has one slot of headroom.
"""

from iron.operators.attn_block_dp.design import attn_block_dp
from iron.operators.swiglu_mlp_dp.design import my_swiglu_mlp_dp

from aie.iron import Program, Runtime
from iron.operators._trace import maybe_enable_trace


def decode_layer_dp(
    dev,
    D,
    FF,
    HD,
    Hq,
    Hkv,
    max_seq,
    eps_attn=1e-6,
    eps_mlp=1e-5,
    attn_cols=8,
    mlp_cols=4,
    tile_size_input=4,
    attn_stack_size=0xD00,
    mlp_stack_size=0x800,
    func_prefix="",
    trace_size=0,
    weight_depth=2,
    tile_rows_gu=None,
    wqkv_head_major=False,
    kv_alloc=None,
    kv_block_size=None,
):
    QD = Hq * HD
    # The two halves land in ONE device-wide symbol table and ONE fifo namespace, so each gets its
    # own prefix on BOTH. Without the symbol prefix they collide immediately -- both declare
    # `copy_offset_bf16_vector` and `matvec_vectorized_bf16_bf16` at different memref shapes, and
    # a func.func symbol is keyed by NAME only, which MLIR's verifier rejects as "redefinition of
    # symbol" rather than overloading. Without the fifo prefix both declare `misc` and `out_0..3`.
    a = attn_block_dp(
        dev, D, HD, Hq, Hkv, max_seq, epsilon=eps_attn, tile_size_input=tile_size_input,
        stack_size=attn_stack_size, func_prefix=f"{func_prefix}attn_", n_aie_cols=attn_cols,
        weight_depth=weight_depth, wqkv_head_major=wqkv_head_major,
        kv_alloc=kv_alloc, kv_block_size=kv_block_size,
        fifo_prefix=f"{func_prefix}a_", parts_only=True, norms_packed=True,
    )
    m = my_swiglu_mlp_dp(
        dev, D, FF, epsilon=eps_mlp, stack_size=mlp_stack_size,
        func_prefix=f"{func_prefix}mlp_", n_aie_cols=mlp_cols, n_aie_rows=1,
        QD=QD, fuse_o=True, weight_depth=weight_depth, tile_rows_gu=tile_rows_gu,
        fifo_prefix=f"{func_prefix}m_", parts_only=True,
    )

    # L3 arguments, with the two the halves SHARE folded together rather than duplicated:
    #   attn: cur norms Wqkv ang kc vc cx      (norms = n_in | n_qn | n_kn, packed)
    #   mlp : cur cx   npf  Wo   Wg   Wu Wd gh_scratch a_scratch nxt
    # `cur` is the layer input both read; `cx` is attention's output and the MLP's input.
    A_N = len(a["l3_types"])                       # 7 with norms packed
    l3_types = a["l3_types"] + m["l3_types"][2:]   # drop the mlp's own cur and cx slots
    n_l3 = len(l3_types)

    def sequence(*args):
        l3, h = args[:n_l3], args[n_l3:]
        cur, cx = l3[0], l3[A_N - 1]
        # Attention first, MLP second, and the order is the dependency: the MLP's first fill reads
        # `cx`, which attention's own closing TaskGroup has already waited on. That is the
        # invariant both halves already state -- every task in group k reachable using only groups
        # <= k -- and it is why the row crossing needs no barrier of its own.
        a["seq"](*l3[:A_N], *h[:3])
        m["seq"](cur, cx, *l3[A_N:], *h[3:])

    rt = Runtime(sequence, l3_types + a["handles"] + m["handles"])
    workers = a["workers"] + m["workers"]
    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
