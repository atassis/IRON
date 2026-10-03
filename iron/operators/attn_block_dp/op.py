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
    # Gemma-4's gainless value-norm: `v = weighted_RMSNorm(Wv[head c] @ hn, ones)`. False (default)
    # keeps step 5 exactly what it was -- v written straight into the drain tile, no norm, no L1
    # or host-argument cost. See design.py's v_norm docstring for the gainless-constant rationale.
    v_norm: bool = False
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
    # Quantized Wqkv needs its own FIFO because a packed row does not fit the KV tile.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
    layout: str = field(default="header_first", repr=False)
    row_group: int | None = field(default=None, repr=False)
    scale_dtype: str = field(default="f32", repr=False)
    quant_vec_size: int = field(default=64, repr=False)
    # A_s/A_g (gemma4-weightless-attention-block): drops the input RMSNorm and Wqkv matvec, so the
    # runtime interface is `qkv, n_qn, n_kn, ang, kc, vc -> cx` -- the qkv vector produced by a
    # separate W device replaces `cur`/`n_in`/`Wqkv`. Every weight-stream axis below (tile_size_
    # input, wqkv_head_major, weight_dtype and its quant siblings) is meaningless here and checked
    # against its default in __post_init__ rather than silently ignored.
    weightless: bool = False
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
        "v_norm": "vn",
        "weight_dtype": "wdt",
        "group_size": "g",
        "layout": "lay",
        "row_group": "rg",
        "scale_dtype": "sdt",
        "weightless": "wless",
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
        # repr=False like its siblings above; same DYNAMIC_WINDOW-class collision risk a shared
        # build dir would otherwise hit -- a cached attn_split=None (or different split) ELF could
        # silently satisfy a request for another. attn_split=None keeps the pre-split name unchanged.
        if self.attn_split is not None:
            base = f"{base}_sp{self.attn_split}"
        if self.weight_dtype != "bf16":
            base = f"{base}_wdt{self.weight_dtype}g{self.group_size}"
            if self.layout == "row_group_planar":
                base = f"{base}_planar{self.row_group}"
            if self.scale_dtype != "f32":
                base = f"{base}_s{self.scale_dtype}"
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
        if self.weightless:
            # Every weight-stream axis is meaningless with no Wqkv to stream; checked against its
            # default rather than silently ignored (K009). tile_size_input is exempt -- callers
            # construct the weighted and weightless arms from the same TSI for one geometry, and
            # this design never reads it, so forbidding a non-default value would reject a caller
            # that changed nothing this design cares about.
            _bad = [n for n, v in (
                ("weight_dtype", self.weight_dtype != "bf16"),
                ("group_size", self.group_size),
                ("row_group", self.row_group is not None),
                ("layout", self.layout != "header_first"),
                ("scale_dtype", self.scale_dtype != "f32"),
                ("wqkv_head_major", self.wqkv_head_major),
            ) if v]
            if _bad:
                raise ValueError(
                    f"weightless=True carries no Wqkv, so {', '.join(_bad)} (weight-stream axes) "
                    f"must stay at their default"
                )
            if not self.v_norm:
                raise ValueError(
                    "weightless=True needs v_norm=True: v's raw row arrives on the same per-core "
                    "stream as q/k, and this design has no un-normed passthrough for it"
                )
            if self.kv_alloc is not None and self.kv_alloc < self.max_seq:
                raise ValueError(
                    f"kv_alloc ({self.kv_alloc}) must be >= max_seq ({self.max_seq}): it is the "
                    f"cache CAPACITY, not a second window"
                )
            _kva = self.max_seq if self.kv_alloc is None else self.kv_alloc
            if self.kv_block_size is not None and (
                self.kv_block_size <= 0 or _kva % self.kv_block_size != 0
            ):
                raise ValueError(
                    f"kv_block_size ({self.kv_block_size}) must be a positive divisor of "
                    f"kv_alloc ({_kva})"
                )
            from iron.operators.attn_block_dp.design import l1_footprint_bytes, L1_BYTES
            gqa = self.Hq // self.Hkv
            split = self.max_seq if self.attn_split is None else self.attn_split
            used = l1_footprint_bytes(self.D, self.HD, gqa, split, self.HD, self.weight_depth,
                                     self.stack_size, self.v_norm, weightless=True)
            if used > L1_BYTES:
                raise ValueError(
                    f"L1 use {used} B exceeds {L1_BYTES} B at attn_split={split}: sc+sw alone "
                    f"are {2 * gqa * split * 2} B and are the terms the SPLIT drives"
                )
            MLIROperator.__init__(self, context=self.context)
            return
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
        if self.weight_dtype not in ("bf16", "int4", "int8", "int4a", "int8a"):
            raise ValueError(
                f"unknown weight_dtype {self.weight_dtype!r} (expected 'bf16', 'int4', 'int8', "
                f"'int4a' or 'int8a')"
            )
        if self.layout not in ("header_first", "row_group_planar"):
            raise ValueError(
                f"unknown layout {self.layout!r} (expected 'header_first' or 'row_group_planar')"
            )
        if self.layout == "row_group_planar" and self.weight_dtype == "bf16":
            raise ValueError("layout='row_group_planar' needs weight_dtype != 'bf16'")
        if self.layout == "header_first" and self.row_group is not None:
            raise ValueError("row_group is only meaningful under layout='row_group_planar'")
        if self.row_group is not None and self.row_group < 1:
            raise ValueError(f"row_group ({self.row_group}) must be >= 1")
        if self.scale_dtype not in ("f32", "bf16"):
            raise ValueError(
                f"unknown scale_dtype {self.scale_dtype!r} (expected 'f32' or 'bf16')"
            )
        if self.scale_dtype != "f32" and self.weight_dtype == "bf16":
            raise ValueError("scale_dtype is only meaningful for a quantized weight_dtype")
        if self.weight_dtype != "bf16":
            if self.group_size <= 0:
                raise ValueError("weight_dtype != 'bf16' needs an explicit group_size > 0")
            if self.D % self.group_size != 0:
                raise ValueError(
                    f"D={self.D} must be a whole number of groups (group_size={self.group_size})"
                )
            from iron.common.quant import max_legal_vec_size
            if self.layout == "row_group_planar":
                from iron.common.quant import derive_row_group, widest_chunk
                if self.row_group is None:
                    provisional_vec = widest_chunk(self.group_size, self.weight_dtype, cap=64)
                    object.__setattr__(self, "row_group", derive_row_group(
                        [self.D], self.group_size, self.weight_dtype,
                        vec_size=provisional_vec, max_rows=self.tile_size_input,
                        scale_dtype=self.scale_dtype))
                # Complete-block requirement; see iron/common/quant.py::derive_row_group.
                if self.tile_size_input % self.row_group != 0:
                    raise ValueError(
                        f"row_group_planar needs tile_size_input ({self.tile_size_input}) to be "
                        f"a multiple of row_group ({self.row_group})"
                    )
                legal = max_legal_vec_size([self.D], self.group_size, self.weight_dtype,
                                            scale_dtype=self.scale_dtype, layout=self.layout,
                                            row_group=self.row_group)
            else:
                legal = max_legal_vec_size([self.D], self.group_size, self.weight_dtype,
                                            scale_dtype=self.scale_dtype)
            if self.quant_vec_size > legal:
                object.__setattr__(self, "quant_vec_size", legal)
            if not (self.group_size % self.quant_vec_size == 0
                    or self.quant_vec_size == 2 * self.group_size):
                raise ValueError(
                    f"quant_vec_size={self.quant_vec_size} must divide group_size="
                    f"{self.group_size} or be exactly twice it"
                )
        # THE SHARED-TILE INVARIANT's `rpc` (cache rows per stream tile): tsi-derived at bf16,
        # WB-derived (padded) once Wqkv is quantized -- see design.py's quant_tile_bytes.
        from iron.operators.attn_block_dp.design import quant_tile_bytes
        if self.weight_dtype == "bf16":
            rpc, stream_tile_bytes = tile // self.HD, None
        else:
            _, stream_tile_bytes, rpc = quant_tile_bytes(
                self.D, self.HD, self.group_size, self.weight_dtype, self.scale_dtype,
                self.max_seq)
        if self.max_seq % rpc:
            raise ValueError(
                f"max_seq ({self.max_seq}) must divide by the cache rows per stream tile ({rpc})"
            )
        # K008: the tiling must FIT, not merely divide -- and sc/sw are new L1 terms that scale
        # with max_seq, so a window this design cannot hold has to fail here and not as
        # "'aie.tile' op Basic sequential allocation also failed".
        from iron.operators.attn_block_dp.design import l1_footprint_bytes, L1_BYTES

        split = self.max_seq if self.attn_split is None else self.attn_split
        used = l1_footprint_bytes(self.D, self.HD, self.Hq // self.Hkv, split, tile,
                                 self.weight_depth, self.stack_size, self.v_norm,
                                 stream_tile_bytes=stream_tile_bytes)
        if used > L1_BYTES:
            gqa = self.Hq // self.Hkv
            raise ValueError(
                f"L1 use {used} B exceeds {L1_BYTES} B at attn_split={split}: sc+sw alone are "
                f"{2 * gqa * split * 2} B and are the terms the SPLIT drives (max_seq="
                f"{self.max_seq} no longer enters this budget)"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        if self.weightless:
            design_fn, kwargs = "attn_block_dp_weightless", {
                "epsilon": self.epsilon,
                "kv_offset_parameter": self.kv_offset_parameter,
                "mask_parameter": self.mask_parameter,
                "window_parameter": self.window_parameter,
                "weight_depth": self.weight_depth,
                "stack_size": self.stack_size,
                "n_aie_cols": self.num_aie_columns,
                "kv_alloc": self.kv_alloc,
                "kv_block_size": self.kv_block_size,
                "attn_split": self.attn_split,
            }
        else:
            design_fn, kwargs = "attn_block_dp", {
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
                "v_norm": self.v_norm,
                "weight_dtype": self.weight_dtype,
                "group_size": self.group_size,
                "layout": self.layout,
                "row_group": self.row_group,
                "scale_dtype": self.scale_dtype,
            }
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                design_fn,
                (aie_utils.get_current_device(), self.D, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                kwargs,
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
            "rms_norm_hdalias.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=["-DRMS_ALIAS_HD"],
        )
        # One object for both reduction lengths: the projection reduces over d_model and the scores
        # over head_dim, and mv.cc's runtime-K body carries the scores' second name as an alias.
        # `_sc` rides in the NAME: swiglu_mlp_dp and qkv_head_dp build `gemv_{D}k_64vs.o` from this
        # source without the alias -- see decode_layer_dp/op.py's `_rb_tag` for the rule.
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_{GEMV_VEC_SIZE}vs_sc.o",
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
        if self.weight_dtype != "bf16":
            # ADDED alongside mv_obj, not instead of it: mv_obj still supplies sc_mv_kernel's
            # bf16 runtime-K symbol (scores/context stay bf16 regardless of Wqkv's format). Same
            # collision class as swiglu_mlp_dp's -DPLANAR/-DSCALE_BF16 comment -- the exported
            # symbol stays fixed per dtype while the flags change what it computes, so all three
            # ride in the object NAME.
            _qtag = f"_{self.weight_dtype}g{self.group_size}"
            if self.layout == "row_group_planar":
                _qtag += f"_planar{self.row_group}"
            if self.scale_dtype != "f32":
                _qtag += f"_s{self.scale_dtype}"
            _qflags = [f"-DDIM_K={self.D}", f"-DVEC_SIZE={self.quant_vec_size}",
                       f"-DGROUP_SIZE={self.group_size}",
                       f"-DQUANT_EMIT_{self.weight_dtype.upper()}=1"]
            if self.layout == "row_group_planar":
                _qflags += ["-DPLANAR=1", f"-DROW_GROUP={self.row_group}"]
            if self.scale_dtype == "bf16":
                _qflags.append("-DSCALE_BF16=1")
            wmv_obj = KernelObjectArtifact(
                f"gemv_{self.D}k_{self.quant_vec_size}vs{_qtag}.o",
                dependencies=[SourceArtifact(kdir / "generic" / "mv_quant.cc")],
                extra_flags=_qflags,
            )
            deps.append(wmv_obj)
        deps += lut_based_ops_artifacts(arch_dir)   # softmax's exp2 LUT, when the arch needs one
        # weight_dtype rides in the archive name too -- a bf16 build must never silently link a
        # cached quantized archive out of the build dir (the bug d7e9c0a's history records).
        archive = "attn_block_dp_core.a" if self.weight_dtype == "bf16" else f"attn_block_dp_core{_qtag}.a"
        return [KernelArchiveArtifact(archive, dependencies=deps)]

    def get_arg_spec(self):
        QD, KVD = self.Hq * self.HD, self.Hkv * self.HD
        TOT = QD + 2 * KVD
        # kc/vc are sized by the cache CAPACITY, not the attention window: None (default) keeps
        # them equal, byte for byte -- see design.py's KV_ALLOC.
        KV_ALLOC = self.max_seq if self.kv_alloc is None else self.kv_alloc
        cache = self.Hkv * KV_ALLOC * self.HD
        if self.weightless:
            return [
                AIERuntimeArgSpec("in", (TOT,)),                  # qkv, from the W device
                AIERuntimeArgSpec("in", (self.HD,)),               # n_qn
                AIERuntimeArgSpec("in", (self.HD,)),               # n_kn
                AIERuntimeArgSpec("in", (self.HD,)),               # ang
                AIERuntimeArgSpec("inout", (cache,)),              # kc
                AIERuntimeArgSpec("inout", (cache,)),              # vc
                AIERuntimeArgSpec("out", (QD,)),                   # cx
            ]
        if self.weight_dtype == "bf16":
            wqkv_spec = AIERuntimeArgSpec("in", (TOT * self.D,))
        else:
            import numpy as np
            from iron.operators.attn_block_dp.design import quant_tile_bytes

            # The L3 stride is TB (padded), not WB: THE SHARED-TILE INVARIANT (design.py) needs
            # one whole shared tile per row, so the wire format pads each row to it.
            _, stride, _ = quant_tile_bytes(self.D, self.HD, self.group_size, self.weight_dtype,
                                            self.scale_dtype, self.max_seq)
            wqkv_spec = AIERuntimeArgSpec("in", (TOT * stride,), dtype=np.int8)
        return [
            AIERuntimeArgSpec("in", (self.D,)),                    # cur
            AIERuntimeArgSpec("in", (self.D,)),                    # n_in
            wqkv_spec,                                             # Wqkv, flat [TOT, D|stride]
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_qn
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_kn
            AIERuntimeArgSpec("in", (self.HD,)),                   # ang
            AIERuntimeArgSpec("inout", (cache,)),                  # kc: appended, then read back
            AIERuntimeArgSpec("inout", (cache,)),                  # vc: appended, then read back
            AIERuntimeArgSpec("out", (QD,)),                       # cx
        ]
