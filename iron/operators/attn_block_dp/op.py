# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from iron.operators.attn_block_dp.design import GEMV_VEC_SIZE
from iron.common.operator_bases import lut_based_ops_artifacts


@dataclass
class AttnBlockDataParallel(MLIROperator):
    """Decode attention block as ONE `aie.device`, data-parallel with one KV HEAD per core.

    Fuses RMSNorm(in) -> concatenated QKV GEMV -> per-head qk-RMSNorm -> RoPE -> KV append ->
    scores -> softmax -> context. See design.py for why the head->core mapping is the design and
    why the four operators this replaces cannot be fused as they stand.

    Runtime interface: cur, n_in, Wqkv, n_qn, n_kn, ang, kc, vc -> cx, with kc/vc `inout` (this
    token's row is appended at `kv_off`, then the whole window is read back).

    `wqkv_head_major=False` (the default) keeps Wqkv's stock [Wq | Wk | Wv] row-major layout: the
    head re-mapping is expressed in the FILLS, not in the blob, so it is a drop-in against the
    existing weight artifact at the cost of three fills per core instead of one. True expects the
    blob pre-permuted per core and spends one. Same bytes; see design.py for the await-count trade.
    """

    D: int
    HD: int
    Hq: int
    Hkv: int
    max_seq: int
    num_aie_columns: int = 8
    epsilon: float = 1e-6
    tile_size_input: int = 4
    stack_size: int = 0xD00
    kv_offset_parameter: str | None = "kv_off"
    mask_parameter: str = "sm_mask"
    # None (default) keeps the window a BUILD constant -- today's behaviour, byte for byte. A name
    # makes it a per-dispatch ScratchpadParameter. NOT repr=False: like kv_offset_parameter and
    # mask_parameter above, this field rides MLIROperator.name's automatic field aggregation, so a
    # windowed build's name diverges from a plain one for free -- no hand-written suffix needed
    # (contrast decode_layer_dp/op.py's kv_alloc/kv_block_size, which ARE repr=False and own a
    # `name` property override instead).
    window_parameter: str | None = None
    wqkv_head_major: bool = False
    # SPLIT-K. The attention window is processed in segments of `attn_split` positions, with the
    # softmax carrying a running max/sum across them, so sc/sw are sized to a SEGMENT and L1 stops
    # scaling with `max_seq`. None (default) is one segment -- byte for byte the pre-split design.
    # This is what lifts the 4544-position window cap; see the L1 check below.
    attn_split: int | None = field(default=None, repr=False)
    # Cache CAPACITY, when it differs from the attention WINDOW (`max_seq`). None (default) keeps
    # them equal -- today's behaviour, byte for byte. `max_seq` stays what sizes the compute (sc/sw,
    # the KV-chunk loop, the mask); `kv_alloc` sizes the KV cache buffers and the per-head stride, so
    # a wide RESIDENT cache can be read through a narrow window. repr=False + the `name` override
    # below -- mirrors decode_layer_dp/op.py's identical fields onto the operator whose design.py
    # actually owns kc/vc's addressing (attn_block_dp itself was never given these two).
    kv_alloc: int | None = field(default=None, repr=False)
    # BLOCKED KV-cache storage: `kv_alloc` positions stored as `kv_alloc // kv_block_size` BLOCKS of
    # `kv_block_size` positions, Hkv heads interleaved every block, instead of one `kv_alloc`-
    # position slab per head. None (default) is one block -- byte-identical to the flat layout.
    kv_block_size: int | None = field(default=None, repr=False)
    weight_depth: int = field(default=2, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
        "tile_size_input": "tsi",
        "stack_size": "ss",
        "max_seq": "S",
        "attn_split": "sp",
        "kv_offset_parameter": "kvpar",
        "mask_parameter": "mpar",
        "window_parameter": "winpar",
        "weight_depth": "wd",
        "wqkv_head_major": "hm",
    }

    @property
    def name(self) -> str:
        # kv_alloc/kv_block_size are repr=False so the default path's name is unchanged, but a
        # wide-cache or blocked design must not share an artifact name with the plain one at the
        # same max_seq -- both would emit the same .mlir/.xclbin. Mirrors decode_layer_dp/op.py.
        base = super().name
        if self.kv_alloc is not None and self.kv_alloc != self.max_seq:
            base = f"{base}_kva{self.kv_alloc}"
        if self.kv_block_size is not None and self.kv_block_size != (self.kv_alloc or self.max_seq):
            base = f"{base}_kvblk{self.kv_block_size}"
        return base

    def __post_init__(self):
        # Every rule here is design.py's, checked at construction so a graph fails where it is
        # built rather than three frames down in taplib.
        if self.Hkv != self.num_aie_columns:
            raise ValueError(
                f"one KV head per core: Hkv ({self.Hkv}) must equal num_aie_columns "
                f"({self.num_aie_columns})"
            )
        if self.Hq % self.Hkv:
            raise ValueError(f"Hq ({self.Hq}) must be a multiple of Hkv ({self.Hkv})")
        if self.D % self.HD:
            raise ValueError(
                f"d_model ({self.D}) must be a whole number of head_dim ({self.HD}) chunks -- "
                "`cur` and `n_in` ride the HD-wide misc channel"
            )
        if self.HD % self.tile_size_input:
            raise ValueError(
                f"head_dim ({self.HD}) must be divisible by tile_size_input "
                f"({self.tile_size_input})"
            )
        tile = self.tile_size_input * self.D
        if tile % self.HD:
            raise ValueError(
                f"the shared stream tile (tsi*D = {tile}) must be a whole number of cache rows "
                f"(head_dim {self.HD})"
            )
        rpc = tile // self.HD
        if self.max_seq % rpc:
            raise ValueError(
                f"max_seq ({self.max_seq}) must divide by the cache rows per stream tile ({rpc})"
            )
        # kv_alloc/kv_block_size: checked here too, redundantly with design.py's own asserts --
        # decode_layer_dp's identical fields make the same trade, for a caller that wants to fail
        # at construction rather than three frames down at MLIR generation.
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
        # K008: the tiling must FIT, not merely divide -- and sc/sw are new L1 terms that scale
        # with max_seq, so a window this design cannot hold has to fail here and not as
        # "'aie.tile' op Basic sequential allocation also failed".
        from iron.operators.attn_block_dp.design import l1_footprint_bytes, L1_BYTES

        split = self.max_seq if self.attn_split is None else self.attn_split
        used = l1_footprint_bytes(self.D, self.HD, self.Hq // self.Hkv, split, tile,
                                 self.weight_depth, self.stack_size)
        if used > L1_BYTES:
            gqa = self.Hq // self.Hkv
            raise ValueError(
                f"L1 use {used} B exceeds {L1_BYTES} B at attn_split={split}: sc+sw alone are "
                f"{2 * gqa * split * 2} B and are the terms the SPLIT drives (max_seq="
                f"{self.max_seq} no longer enters this budget)"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "attn_block_dp",
                (aie_utils.get_current_device(), self.D, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                {
                    "epsilon": self.epsilon,
                    "kv_offset_parameter": self.kv_offset_parameter,
                    "mask_parameter": self.mask_parameter,
                    "window_parameter": self.window_parameter,
                    "weight_depth": self.weight_depth,
                    "tile_size_input": self.tile_size_input,
                    "stack_size": self.stack_size,
                    "n_aie_cols": self.num_aie_columns,
                    "wqkv_head_major": self.wqkv_head_major,
                    "kv_alloc": self.kv_alloc,
                    "kv_block_size": self.kv_block_size,
                    "attn_split": self.attn_split,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        copy_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        # One object for both norm lengths: rms_norm.cc takes the length as an argument and aliases
        # the hd_ name onto the same body, so the D and head_dim call sites share one copy.
        rms_obj = KernelObjectArtifact(
            "rms_norm.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
        )
        # One object for both reduction lengths: the projection reduces over d_model and the scores
        # over head_dim, and mv.cc's runtime-K body carries the scores' second name as an alias.
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_{GEMV_VEC_SIZE}vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.D}", f"-DVEC_SIZE={GEMV_VEC_SIZE}", "-DGEMV_ALIAS_SC"],
        )
        rope_obj = KernelObjectArtifact(
            "rope_0.o",
            dependencies=[SourceArtifact(kdir / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        softmax_obj = KernelObjectArtifact(
            "softmax.o", dependencies=[SourceArtifact(kdir / arch_dir / "softmax.cc")]
        )
        tmv_obj = KernelObjectArtifact(
            f"tmv_{self.HD}n.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv_taccum.cc")],
            extra_flags=[f"-DDIM_N={self.HD}"],
        )
        deps = [copy_obj, rms_obj, mv_obj, rope_obj, softmax_obj, tmv_obj]
        deps += lut_based_ops_artifacts(arch_dir)   # softmax's exp2 LUT, when the arch needs one
        return [KernelArchiveArtifact("attn_block_dp_core.a", dependencies=deps)]

    def get_arg_spec(self):
        QD, KVD = self.Hq * self.HD, self.Hkv * self.HD
        # kc/vc are sized by the cache CAPACITY, not the attention window: None (default) keeps
        # them equal, byte for byte -- see design.py's KV_ALLOC.
        KV_ALLOC = self.max_seq if self.kv_alloc is None else self.kv_alloc
        cache = self.Hkv * KV_ALLOC * self.HD
        return [
            AIERuntimeArgSpec("in", (self.D,)),                    # cur
            AIERuntimeArgSpec("in", (self.D,)),                    # n_in
            AIERuntimeArgSpec("in", ((QD + 2 * KVD) * self.D,)),   # Wqkv, flat [QD+2KVD, D]
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_qn
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_kn
            AIERuntimeArgSpec("in", (self.HD,)),                   # ang
            AIERuntimeArgSpec("inout", (cache,)),                  # kc: appended, then read back
            AIERuntimeArgSpec("inout", (cache,)),                  # vc: appended, then read back
            AIERuntimeArgSpec("out", (QD,)),                       # cx
        ]
