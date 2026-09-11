# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass, field
from typing import ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils
from iron.common.device_utils import get_kernel_dir
from iron.common.operator_bases import lut_based_ops_artifacts


def _mlp_l1_footprint_bytes(D, FF, mlp_cols, QD, tile_rows_gu, weight_depth, stack_size):
    """The MLP half's per-core L1 use, bf16 + fuse_o=True (decode_layer_dp's only call shape).
    Mirrors swiglu_mlp_dp/design.py's L1 budget block and this file's own get_arg_spec overlap
    arithmetic -- duplicated, not imported, because both are computed inline there rather than
    exported; the same trade SwiGLUMLPDataParallel._wo_rows_padded already makes. No term here
    depends on max_seq."""
    d_per_core, ff_per_core = D // mlp_cols, FF // mlp_cols
    wtile_units = (tile_rows_gu or 6) * D
    o_window = -(-(D // mlp_cols) // (wtile_units // QD)) * (wtile_units // QD)
    misc = 2 * (D * 2)
    weight = weight_depth * (wtile_units * 2)
    out = 2 * (d_per_core * 2)
    persistent = 2 * (D * 2) + (FF * 2) + 2 * (ff_per_core * 2) + (d_per_core * 2)
    persistent += (QD * 2) + (o_window * 2)   # fuse_o: cx_buf + a_slice_buf
    return misc + weight + out + persistent + stack_size


@dataclass
class DecodeLayerDataParallel(MLIROperator):
    """A whole decoder layer as ONE `aie.device`: attention on 8 cores, the MLP on 4 others.

    See design.py for why the phases are on different CORES and not one (program memory, which
    unlike L1 cannot be borrowed from a neighbour), why no donor row is used (neither half
    overruns its own 64 KB), why the column counts differ (8 and 4 -- one device does not require
    one column count), and what actually bounds the design (aiecc's 16 host-buffer cap).

    Runtime interface is the two halves' concatenated, with `cur` and `cx` shared:
      cur, norms, Wqkv, ang, kc, vc, cx, n_pf, Wo, Wg, Wu, Wd, gh, a_scr -> nxt
    where `norms` packs n_in | n_qn | n_kn -- see get_arg_spec for why the count matters.
    """

    D: int
    FF: int
    HD: int
    Hq: int
    Hkv: int
    max_seq: int
    attn_cols: int = 8
    mlp_cols: int = 4
    eps_attn: float = 1e-6
    eps_mlp: float = 1e-5
    tile_size_input: int = 4
    attn_stack_size: int = 0xD00
    mlp_stack_size: int = 0x800
    weight_depth: int = field(default=2, repr=False)
    tile_rows_gu: int | None = field(default=None, repr=False)
    wqkv_head_major: bool = False
    # Cache CAPACITY, when it differs from the attention WINDOW (`max_seq`). None (default) keeps
    # them equal -- today's behaviour, byte for byte. `max_seq` stays what sizes the compute (sc/sw,
    # the KV-chunk loop, the mask); `kv_alloc` sizes the KV cache buffers and the per-head stride,
    # so a wide RESIDENT cache can be read through a narrow window -- one level up from gemv's
    # alloc_M / tmatvec's alloc_K. repr=False + the `name` override below, matching that convention.
    kv_alloc: int | None = field(default=None, repr=False)
    # BLOCKED KV-cache storage: `kv_alloc` positions stored as `kv_alloc // kv_block_size` BLOCKS of
    # `kv_block_size` positions, Hkv heads interleaved every block, instead of one `kv_alloc`-
    # position slab per head. None (default) is one block -- byte-identical to the flat layout.
    # Same meaning as gemv/tmatvec's `block_size`, one level up. See attn_block_dp/design.py and
    # iron.common.kv_layout (KVLayout), the single owner of the offset/stride formulas this uses.
    kv_block_size: int | None = field(default=None, repr=False)
    # Runtime attention window: None (default) keeps N_KV_CHUNKS a build constant -- today's
    # behaviour, byte for byte. A name makes it a per-dispatch ScratchpadParameter; delegated
    # straight to attn_block_dp, which owns every window term (mask/softmax row length, the two
    # KV-chunk loops, the drain). repr=False + the `name` override below, matching kv_alloc/
    # kv_block_size's convention above rather than attn_block_dp/op.py's own window_parameter
    # field: that one rides the automatic name aggregation because attn_block_dp's `name` has no
    # override to extend, while this class already has one for kv_alloc/kv_block_size.
    window_parameter: str | None = field(default=None, repr=False)
    # INSTRUMENT (see swiglu_mlp_dp/design.py): chop the gh drain group into k groups. Moves ONLY
    # the sync-point count, +(k-1) per layer, at identical bytes/tasks/BDs/configures/designs.
    split_gh: int = field(default=1, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "attn_cols": "acol", "mlp_cols": "mcol", "max_seq": "S",
        "eps_attn": "ea", "eps_mlp": "em", "tile_size_input": "tsi",
        "attn_stack_size": "ass", "mlp_stack_size": "mss",
        "weight_depth": "wd", "tile_rows_gu": "trg", "wqkv_head_major": "hm",
    }

    def __post_init__(self):
        # Delegate every per-half rule to the half that owns it rather than restating it here --
        # the two operators already raise at construction with their own messages. What is checked
        # HERE is only what is true of the MERGED device and of neither half alone.
        if self.Hkv != self.attn_cols:
            raise ValueError(
                f"attention places one KV head per core: Hkv ({self.Hkv}) must equal attn_cols "
                f"({self.attn_cols})"
            )
        # kv_alloc/kv_block_size are the two fields this class does NOT delegate to a half's own
        # operator (attn_block_dp is called as a bare function here, not through
        # AttnBlockDataParallel), so -- like GEMV's alloc_M/block_size -- they are checked at
        # construction AND, redundantly, inside attn_block_dp/design.py for a caller that drives
        # the generator directly.
        if self.kv_alloc is not None and self.kv_alloc < self.max_seq:
            raise ValueError(
                f"kv_alloc ({self.kv_alloc}) must be >= max_seq ({self.max_seq}): it is the cache "
                f"CAPACITY, not a second window"
            )
        _kva = self.max_seq if self.kv_alloc is None else self.kv_alloc
        if self.kv_block_size is not None and (
            self.kv_block_size <= 0 or _kva % self.kv_block_size != 0
        ):
            raise ValueError(
                f"kv_block_size ({self.kv_block_size}) must be a positive divisor of kv_alloc "
                f"({_kva})"
            )
        # GRANULE (K007: divisibility asserted where the shape is picked). Two constraints on
        # max_seq already exist, but only inside attn_block_dp/design.py, at MLIR-generation time,
        # and never combined: `assert S % rpc == 0` (unconditional -- rpc = tsi*D/HD is the stream
        # tile in cache rows, and a windowed dispatch divides its runtime win_len by rpc via
        # arith.divsi to get the KV-chunk trip count) and `assert S % _KVT == 0`, gated on
        # `kv_blocked = _KVT != KV_ALLOC` -- i.e. ONLY when kv_block_size actually carves the cache
        # into more than one block. A wide, UNBLOCKED kv_alloc (kv_block_size=None) reads its
        # window as one flat run (`_kv_read_tap`'s non-blocked branch) and carries no relationship
        # to max_seq at all -- design.py never asserts one, and test_kv_alloc_sizes_the_cache_not_
        # the_window (max_seq=256, kv_alloc=4096) is exactly that legitimate shape. So the second
        # term here mirrors kv_blocked exactly: kv_block_size when it is set AND differs from the
        # alloc (real blocking), else max_seq itself (unblocked -- the whole window is its own one
        # block, which folds this check down to rpc alone, byte for byte with today's kv_alloc
        # tests). lcm(rpc, kv_block) is the smallest max_seq granularity that satisfies both real
        # constraints at once; checked here, redundantly, for the same reason kv_alloc/
        # kv_block_size are above -- so a bad shape fails at construction, not three frames down.
        rpc = self.tile_size_input * self.D // self.HD      # stream tile in cache rows
        kv_blocked = self.kv_block_size is not None and self.kv_block_size != _kva
        kv_block = self.kv_block_size if kv_blocked else self.max_seq
        granule = math.lcm(rpc, kv_block)
        if self.max_seq % granule != 0:
            raise ValueError(
                f"max_seq ({self.max_seq}) must be a multiple of the granule ({granule}) = "
                f"lcm(stream-tile rows={rpc}, kv block={kv_block})"
            )
        # Exposed so a caller building a runtime attn_window protocol (gen_llm_decode.py's
        # DYNAMIC_WINDOW) can put this in meta.json without restating the formula -- the host must
        # not re-derive it from dims.kv_block, which coincides with this value only at this model's
        # shape and would be silently wrong at another head_dim/tile_size_input.
        self.window_granule = granule
        # L1 IS THE EXCEPTION TO "delegate to the half" above: attn_block_dp's guard lives inside
        # its design.py FUNCTION (an assert), and this file calls that function directly rather
        # than through AttnBlockDataParallel -- so it fires at MLIR generation, not here, unless
        # repeated. sc/sw are the only terms max_seq drives (attn_block_dp/design.py's
        # l1_footprint_bytes); derive the per-token rate from the function itself rather than
        # re-deriving its algebra, so this stays correct if that formula grows a term.
        from iron.operators.attn_block_dp.design import l1_footprint_bytes, L1_BYTES

        gqa = self.Hq // self.Hkv
        tile_elems = self.tile_size_input * self.D
        attn_args = (self.D, self.HD, gqa, self.max_seq, tile_elems, self.weight_depth,
                     self.attn_stack_size)
        attn_used = l1_footprint_bytes(*attn_args)
        if attn_used > L1_BYTES:
            fixed = l1_footprint_bytes(*(attn_args[:3] + (0,) + attn_args[4:]))
            per_seq = l1_footprint_bytes(*(attn_args[:3] + (1,) + attn_args[4:])) - fixed
            raise ValueError(
                f"attention L1 use {attn_used} B exceeds {L1_BYTES} B at max_seq={self.max_seq}: "
                f"sc+sw cost {per_seq} B per unit of max_seq and are the only terms it drives; "
                f"largest max_seq that fits is {(L1_BYTES - fixed) // per_seq}"
            )

        if self.D % self.mlp_cols or self.FF % self.mlp_cols:
            raise ValueError(
                f"D ({self.D}) and FF ({self.FF}) must both divide mlp_cols ({self.mlp_cols})"
            )
        # The MLP half does not depend on max_seq at all (every term in its L1 block is D/FF/
        # mlp_cols/QD) -- checked anyway because it was exactly as unguarded as attention was.
        mlp_used = _mlp_l1_footprint_bytes(self.D, self.FF, self.mlp_cols, self.Hq * self.HD,
                                           self.tile_rows_gu, self.weight_depth,
                                           self.mlp_stack_size)
        if mlp_used > L1_BYTES:
            raise ValueError(
                f"MLP L1 use {mlp_used} B exceeds {L1_BYTES} B at mlp_cols={self.mlp_cols} -- "
                "independent of max_seq, so raising max_seq will not fix this"
            )
        # SHIM CHANNELS. The halves sit on different cores, so they do NOT share their misc /
        # weight / output fifos and the budget is the SUM: attention misc(1)+stream(attn_cols),
        # out(attn_cols); MLP misc(1)+weight(mlp_cols), out(mlp_cols). Checked here because it is
        # exactly the constraint that is invisible to each half and that rejected the naive
        # 4-operator fusion (49 in / 32 out) -- and because aiecc reports it as "no ShimNOCTile on
        # the device has 0 input/1 output DMA channel(s) free", which names no operator.
        shim_in = (1 + self.attn_cols) + (1 + self.mlp_cols)
        shim_out = self.attn_cols + self.mlp_cols
        limit = 16   # get_shim_dma_limit(NPU2): 8 ShimNOCTiles x 2 channels each direction
        if shim_in > limit or shim_out > limit:
            raise ValueError(
                f"merged shim DMA budget exceeded: {shim_in} input / {shim_out} output channels "
                f"against {limit} each (attention {1+self.attn_cols}/{self.attn_cols}, MLP "
                f"{1+self.mlp_cols}/{self.mlp_cols})"
            )
        cores = self.attn_cols + self.mlp_cols
        if cores > 32:
            raise ValueError(f"{cores} workers exceeds NPU2's 32 core tiles (4 rows x 8 columns)")
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # kv_alloc/kv_block_size are repr=False so the default path's name is unchanged, but a
        # wide-cache or blocked design must not share an artifact name with the plain one at the
        # same max_seq: both would emit the same .mlir/.xclbin, and in a shared build dir a cached
        # plain build could then silently satisfy the windowed/blocked op. Mirrors gemv/op.py's
        # alloc_M/block_size suffixes.
        base = super().name
        if self.kv_alloc is not None and self.kv_alloc != self.max_seq:
            base = f"{base}_kva{self.kv_alloc}"
        if self.kv_block_size is not None and self.kv_block_size != (self.kv_alloc or self.max_seq):
            base = f"{base}_kvblk{self.kv_block_size}"
        if self.window_parameter is not None:
            base = f"{base}_win{self.window_parameter}"
        if self.split_gh != 1:
            base = f"{base}_sgh{self.split_gh}"
        return base

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "decode_layer_dp",
                (aie_utils.get_current_device(), self.D, self.FF, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                {
                    "eps_attn": self.eps_attn, "eps_mlp": self.eps_mlp,
                    "attn_cols": self.attn_cols, "mlp_cols": self.mlp_cols,
                    "tile_size_input": self.tile_size_input,
                    "attn_stack_size": self.attn_stack_size,
                    "mlp_stack_size": self.mlp_stack_size,
                    "weight_depth": self.weight_depth,
                    "tile_rows_gu": self.tile_rows_gu,
                    "wqkv_head_major": self.wqkv_head_major,
                    "kv_alloc": self.kv_alloc,
                    "kv_block_size": self.kv_block_size,
                    "window_parameter": self.window_parameter,
                    "split_gh": self.split_gh,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch, kdir = get_kernel_dir(), self.context.base_dir / "aie_kernels"
        QD = self.Hq * self.HD
        gen, a2 = kdir / "generic", kdir / arch

        _rb = int(os.environ.get("SCORES_ROWBATCH", "1"))
        _rb_tag = f"_rb{_rb}" if _rb > 1 else ""

        def obj(name, src, flags=(), prefix=None):
            return KernelObjectArtifact(name, dependencies=[SourceArtifact(src)],
                                        extra_flags=list(flags), prefix_symbols=prefix)

        # EVERY symbol in this device carries its half's prefix. Both halves declare
        # `copy_offset_bf16_vector` and `matvec_vectorized_bf16_bf16` at different memref shapes,
        # and a func.func symbol is keyed by NAME only -- MLIR rejects that as "redefinition of
        # symbol", not as an overload. Object FILENAMES are prefixed too: the archive is keyed by
        # name, and an unprefixed `add.o` built for one half would be reused for the other.
        attn = [
            obj("attn_add.o", gen / "add.cc", prefix="attn_"),
            obj(f"attn_rms_{self.D}.o", a2 / "rms_norm.cc",
                [f"-DRMS_COLS={self.D}"], "attn_"),
            obj(f"attn_hd_rms_{self.HD}.o", a2 / "rms_norm.cc",
                [f"-DRMS_COLS={self.HD}"], "attn_hd_"),
            obj(f"attn_gemv_{self.D}k.o", gen / "mv.cc",
                [f"-DDIM_K={self.D}", "-DVEC_SIZE=64"], "attn_"),
            # The SCORES gemv is the only core-bound op in this graph, so it is the one where a
            # row-batched reduce can convert. GEMV_ROWBATCH rides in the NAME for the same reason
            # the prefixes above do: the archive is keyed by name, so a -D-only difference would
            # silently reuse the other variant's object. ROWBATCH=1 keeps the historical name.
            obj(f"attn_sc_gemv_{self.HD}k{_rb_tag}.o", gen / "mv.cc",
                [f"-DDIM_K={self.HD}", "-DVEC_SIZE=64", f"-DGEMV_ROWBATCH={_rb}"], "attn_sc_"),
            obj("attn_rope.o", gen / "rope.cc", ["-DTWO_HALVES"], "attn_"),
            obj("attn_softmax.o", a2 / "softmax.cc", (), "attn_"),
            obj(f"attn_tmv_{self.HD}n.o", gen / "mv_taccum.cc",
                [f"-DDIM_N={self.HD}"], "attn_"),
        ] + lut_based_ops_artifacts(arch)
        mlp = [
            obj("mlp_add.o", gen / "add.cc", (), "mlp_"),
            obj("mlp_mul.o", gen / "mul.cc", (), "mlp_"),
            obj(f"mlp_rms_{self.D}.o", a2 / "rms_norm.cc",
                [f"-DRMS_COLS={self.D}"], "mlp_"),
            obj("mlp_silu.o", a2 / "silu.cc", (), "mlp_"),
            obj(f"mlp_gemv_{self.D}k.o", gen / "mv.cc",
                [f"-DDIM_K={self.D}", "-DVEC_SIZE=64"], "mlp_"),
            obj(f"mlp_down_gemv_{self.FF}k.o", gen / "mv.cc",
                [f"-DDIM_K={self.FF}", "-DVEC_SIZE=64"], "mlp_down_"),
            obj(f"mlp_o_gemv_{QD}k.o", gen / "mv.cc",
                [f"-DDIM_K={QD}", "-DVEC_SIZE=64"], "mlp_o_"),
            obj("mlp_add_cxcopy.o", gen / "add.cc", (), "mlp_cx_"),
            obj("mlp_add_oacopy.o", gen / "add.cc", (), "mlp_oa_"),
        ]
        # Archive names must match what each design's own CORE_ARCHIVE builds from its
        # func_prefix -- "attn_" and "mlp_", set by design.py's decode_layer_dp.
        return [
            KernelArchiveArtifact("attn_attn_block_dp_core.a", dependencies=attn),
            KernelArchiveArtifact("mlp_swiglu_mlp_dp_core.a", dependencies=mlp),
        ]

    def get_arg_spec(self):
        D, FF, HD, Hq, Hkv, S = self.D, self.FF, self.HD, self.Hq, self.Hkv, self.max_seq
        QD, KVD = Hq * HD, Hkv * HD
        # kc/vc are sized by the cache CAPACITY, not the attention window: None (default) keeps
        # them equal, byte for byte -- see attn_block_dp/design.py's KV_ALLOC.
        KV_ALLOC = S if self.kv_alloc is None else self.kv_alloc
        cache = Hkv * KV_ALLOC * HD
        # Wo carries the fused-o overlap padding: swiglu_mlp_dp reads ceil(D_PER_CORE/TSI_O) full
        # TSI_O-row tiles per core, so its last core's window runs past Wo's D real rows.
        tsi_gu = self.tile_rows_gu or 6
        TSI_O = (tsi_gu * D) // QD
        O_WINDOW = -(-(D // self.mlp_cols) // TSI_O) * TSI_O
        wo_rows = D + (O_WINDOW - D // self.mlp_cols)
        # FIFTEEN host buffers, and the count is load-bearing: aiecc caps a device at 16
        # (kMaxHostBOs, tools/aiecc/SidecarFiles.h -- "conservative, hardware-verified ceiling",
        # policy rather than silicon, and not overridable). The unpacked form is 17 and is
        # REJECTED; packing the three static norm gains into one blob is what buys the margin.
        return [
            AIERuntimeArgSpec("in", (D,)),                    # cur   (shared)
            AIERuntimeArgSpec("in", (D + 2 * HD,)),           # norms = n_in | n_qn | n_kn
            AIERuntimeArgSpec("in", ((QD + 2 * KVD) * D,)),   # Wqkv
            AIERuntimeArgSpec("in", (HD,)),                   # ang   (per token, so not packed)
            AIERuntimeArgSpec("inout", (cache,)),             # kc
            AIERuntimeArgSpec("inout", (cache,)),             # vc
            AIERuntimeArgSpec("inout", (QD,)),                # cx    (shared: attn out, mlp in)
            AIERuntimeArgSpec("in", (D,)),                    # n_pf
            AIERuntimeArgSpec("in", (wo_rows * QD,)),         # Wo
            AIERuntimeArgSpec("in", (FF * D,)),               # Wg
            AIERuntimeArgSpec("in", (FF * D,)),               # Wu
            AIERuntimeArgSpec("in", (D * FF,)),               # Wd
            AIERuntimeArgSpec("inout", (FF,)),                # gh all-gather scratch
            AIERuntimeArgSpec("inout", (D,)),                 # a all-gather scratch
            AIERuntimeArgSpec("out", (D,)),                   # nxt
        ]
