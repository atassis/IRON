# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single owner of the KV-cache element-offset arithmetic.

Until 2026-09-09 "where does (head, position) live in the KV cache" was hand-written in five
places (the generator's buffer sizing, the gemv and tmatvec operator taps, qkv_head_dp's append,
the Rust host's `kv_off`) with nothing checking they agreed -- see the `kv-cache-layout-for-full-
context` task. This module is the one place that formula lives; every other site ASKS it.

Physical layout: **blocked**, `[S/T blocks, Hkv heads, T positions, HD dims]`, block-major.
`T == S` (one block covering the whole capacity) is the degenerate case, and is BYTE-IDENTICAL to
the historical flat `[Hkv, S, HD]` layout this replaces -- `head_stride`/`block_stride` both reduce
to the old `S*HD`/`Hkv*S*HD` formulas. That degeneracy is what makes the T=S state a pure refactor:
switching a call site from a hand-written `S*HD` to `KVLayout(Hkv, S, HD, T=S).head_stride` changes
no number.

Blocking exists because the FLAT layout's per-head stride (`S*HD` elements) scales with capacity,
and that stride lands in a 20-bit shim BD step field counting 32-bit address granules -- so it caps
addressable capacity at a few thousand positions regardless of anything else in the design. Under
blocking, `head_stride` and `block_stride` are both independent of `S`: the field the layout used to
overflow simply never sees the window again.

Scope: this module owns ORDERING (which element index a given (head, position) maps to) in
ELEMENTS of the cache's own dtype. It does not know the dtype's byte width, does not touch
hardware DMA field widths beyond what `derive_block_size` needs to pick T, and is not the tensor
dtype-arithmetic owner proposed in
`docs/superpowers/specs/2026-09-06-tensor-layout-abstraction-design.md` (that design owns "how many
BYTES does a tensor occupy" across dtypes; this module owns "which element index", one axis over).
The two compose: multiply this module's element offsets by that design's `size_bytes`-style
per-element width when one lands.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVLayout:
    """Where a (head, position) slice of a bf16-granularity KV cache lives, in ELEMENTS.

    Hkv: number of kv heads.
    S:   capacity, in positions (today, capacity == the attention window -- see
         [[the-kv-window-and-the-kv-capacity-are-separable]] for the DIFFERENT axis that
         decouples them, which this class does not attempt).
    HD:  head_dim, in elements.
    T:   block size, in positions. `T == S` means "one block" -- today's flat layout.
    """

    Hkv: int
    S: int
    HD: int
    T: int

    def __post_init__(self):
        if self.T <= 0:
            raise ValueError(f"T ({self.T}) must be positive")
        if self.S % self.T:
            raise ValueError(f"S ({self.S}) must be a whole number of blocks of T ({self.T})")

    @property
    def num_blocks(self) -> int:
        return self.S // self.T

    @property
    def head_stride(self) -> int:
        """Elements between head h and head h+1, within one block. A BUILD-TIME constant baked
        into a tap's own stride -- independent of S, which is the entire point of blocking."""
        return self.T * self.HD

    @property
    def block_stride(self) -> int:
        """Elements between block b and block b+1, for one fixed head. Also independent of S."""
        return self.Hkv * self.T * self.HD

    @property
    def total_elems(self) -> int:
        """Total buffer size. Layout-independent: blocking rearranges the same Hkv*S*HD
        elements, it does not change how many there are."""
        return self.Hkv * self.S * self.HD

    def _check_head(self, head: int) -> None:
        if not (0 <= head < self.Hkv):
            raise ValueError(f"head {head} out of range [0, {self.Hkv})")

    def _check_pos(self, pos: int) -> None:
        if not (0 <= pos < self.S):
            raise ValueError(f"pos {pos} out of range [0, {self.S})")

    def head_base(self, head: int) -> int:
        """The per-head BUILD-TIME base offset. Independent of position -- this is what a tap's
        own (constant) stride encodes, e.g. qkv_head_dp's per-head append offset or a column's
        per-matrix base in gemv/tmatvec's access pattern."""
        self._check_head(head)
        return head * self.head_stride

    def block_of(self, pos: int) -> int:
        self._check_pos(pos)
        return pos // self.T

    def within_block(self, pos: int) -> int:
        self._check_pos(pos)
        return pos % self.T

    def kv_off(self, pos: int) -> int:
        """The RUNTIME per-token scratchpad value: everything that depends on `pos` alone, common
        to every head (the head term is `head_base`, folded in separately -- see the module
        docstring). At T == S this is exactly `pos * HD`, the pre-blocking formula, unchanged.
        """
        return self.block_of(pos) * self.block_stride + self.within_block(pos) * self.HD

    def offset(self, head: int, pos: int) -> int:
        """Full element offset of (head, pos, 0). Spelled out for tests/documentation; every real
        call site uses the two halves separately (`head_base` baked in at build time, `kv_off`
        written by the host once per token) because that split is what lets one compiled ELF serve
        every position without a rebuild."""
        return self.head_base(head) + self.kv_off(pos)


def split_run(run: int, lim: int = 1023, gran: int = 2):
    """Factor a contiguous element run into `(hi, lo)`, `hi*lo == run`, both `<= lim`, `lo` a
    multiple of `gran` (the address-generation-granularity-aligned inner size), `lo` maximal.
    `None` if no such split exists.

    General DMA-tiling utility, not KV-specific -- a shim/mem-tile BD's wrap-size (`sizes[]`)
    field is `lim`-bits limited (1023 at the AIE2/AIE2P 10-bit shim/mem-tile width) while its
    step/stride field is wider, so a big contiguous run must be split across two dimensions to
    fit. Lives here (rather than duplicated per operator) because commit 2 of the KV-blocked-
    layout task needs the SAME split in both gemv/design.py (which had its own private copy) and
    tmatvec/design.py (which had none) -- exactly the duplication class this module exists to end.
    """
    lo_start = (lim // gran) * gran
    for lo in range(lo_start, 0, -gran):
        if run % lo == 0 and (run // lo) <= lim:
            return (run // lo, lo)
    return None


def derive_block_size(HD: int, Hkv: int, addr_gran_elems: int = 2,
                      shim_step_bits: int = 20, memtile_step_bits: int = 17) -> int:
    """Pick T: the largest power-of-two block size whose `block_stride` (in 4-byte address-
    generation granules) fits the NARROWEST DMA step field this cache stream may ever cross.

    Derived, not chosen -- three field limits were missed by exactly one elsewhere on this task
    (shim stride at S=16384, mem-tile stride at a T=256 trial, both one granule over), so this
    function is the checked replacement for picking T by hand. The bit widths default to AIE2P's
    (verified against `AIE2TargetModel::getDmaBdStepBits` in
    lib/Dialect/AIE/IR/AIETargetModel.h: shim NOC/PL 20-bit, mem tile 17-bit, core tile 13-bit);
    callers on a different target model pass the real numbers rather than trusting the default.

    Binds against the MEM-TILE field (the narrower of the two) even though the design this
    unblocks does not yet stage the cache through a MemTile -- that staging is a later, explicit
    step in this task's own ordering, and deriving against the wider shim-only bound now would
    produce a T that has to be re-derived (and every stride re-checked) when that step lands.
    `addr_gran_elems` is dtype-dependent (2 for bf16: 4-byte granule / 2-byte element) and is a
    caller-supplied constant, not discovered here -- this module does not own dtype width (see
    the module docstring).
    """
    step_bits = min(shim_step_bits, memtile_step_bits)
    max_stride_granules = (1 << step_bits) - 1
    T = 1
    while True:
        candidate = T * 2
        block_stride_elems = Hkv * candidate * HD
        if block_stride_elems % addr_gran_elems:
            break
        block_stride_granules = block_stride_elems // addr_gran_elems
        if block_stride_granules > max_stride_granules:
            break
        T = candidate
    return T


def validate_block_size(T: int, HD: int, Hkv: int, addr_gran_elems: int = 2,
                        shim_step_bits: int = 20, memtile_step_bits: int = 17) -> None:
    """Raise if T's strides do not fit the narrowest DMA step field. `derive_block_size` and every
    call site share THIS check rather than each re-deriving the bound, so a hand-overridden T
    (e.g. via an env var, for an A/B) fails loud instead of silently overflowing a BD field."""
    step_bits = min(shim_step_bits, memtile_step_bits)
    max_stride_granules = (1 << step_bits) - 1
    layout = KVLayout(Hkv=Hkv, S=T, HD=HD, T=T)  # S=T: only block_stride/head_stride matter here
    if layout.block_stride % addr_gran_elems:
        raise ValueError(
            f"block_stride ({layout.block_stride} elements) is not a whole number of "
            f"{addr_gran_elems}-element address-generation granules"
        )
    granules = layout.block_stride // addr_gran_elems
    if granules > max_stride_granules:
        raise ValueError(
            f"T={T} (Hkv={Hkv}, HD={HD}): block_stride is {granules} granules, over the "
            f"{step_bits}-bit step field's {max_stride_granules} -- pick a smaller T "
            f"(derive_block_size({HD}, {Hkv}) = {derive_block_size(HD, Hkv, addr_gran_elems, shim_step_bits, memtile_step_bits)})"
        )
