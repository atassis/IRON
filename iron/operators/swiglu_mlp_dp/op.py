# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np
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


@dataclass
class SwiGLUMLPDataParallel(MLIROperator):
    """Decode SwiGLU MLP block as ONE `aie.device`, data-parallel across `num_aie_columns` cores
    (one core per column, n_aie_rows fixed at 1 -- see design.py's module docstring). Every core
    runs every stage on its own 1/N slice instead of fuse/mlp-block's 5-core spatial pipeline.

    Runtime interface: cur, a, n_pf, Wg, Wu, Wd -> nxt, plus one internal `gh_scratch` (FF
    elements) DRAM round-trip buffer for the cross-core all-gather -- there is no internal-DRAM-
    scratch primitive in plain Runtime/Program (every DMA endpoint must be a formal Runtime
    argument), so it is exposed as a genuine 7th argument here. Callers that want the operator's
    logical 6-in/1-out surface should allocate it once and never touch its contents (matches how
    OperatorSequence's own `buffer_sizes=` auto-allocates unnamed intermediates for the unfused
    arm in the A/B harness this operator was built for).
    """

    D: int
    FF: int
    num_aie_columns: int = 8
    num_aie_rows: int = 1
    epsilon: float = 1e-5
    QD: int = None
    fuse_o: bool = False
    # Gated-FFN activation: "silu" (default, byte-identical to every prior build) or "gelu_tanh"
    # (Gemma) -- matches llm_decode_spec.py's LlmSpec.act domain so a caller can pass spec.act
    # straight through with no translation. repr=True (unlike the fields below): it changes the
    # ARCHIVE and is worth distinguishing two operators by, not a packing convenience.
    act: str = "silu"
    # Sandwich post-block norm slot (Gemma): see design.py's docstring. Default False keeps the
    # arg list, archive and emitted MLIR byte-identical.
    post_norm: bool = False
    # Group-quantized weight stream (Wg/Wu/Wd, and Wo under fuse_o). bf16 is the
    # byte-for-byte pre-existing path; see design.py's WEIGHT WIRE UNITS block.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
    # Byte layout of a quantized row, and the scale width in its header -- both as GEMV takes
    # them (iron/operators/gemv/op.py). Neither is host-side-only: mv_quant.cc selects the
    # row-group form with -DPLANAR=1 and the narrow header with -DSCALE_BF16=1, so both are
    # design-key axes and both ride in the kernel object's name below.
    layout: str = field(default="header_first", repr=False)
    row_group: int | None = field(default=None, repr=False)
    scale_dtype: str = field(default="f32", repr=False)
    # Weight ObjectFifo depth. 2 is plain double-buffering; deeper hides more of the
    # shim->L1 latency at the cost of L1 (the budget check above follows it).
    weight_depth: int = field(default=2, repr=False)
    tile_rows_gu: int = field(default=0, repr=False)
    # Chunks the down-projection's K (=FF) reduction so gh and Wd's tile never hold more than one
    # chunk of L1; see design.py's module docstring. 1 is today's byte-identical path.
    #
    # At gh_chunks>1 `Wd` must come from quantize_weight_CHUNKED, not quantize_weight: same total
    # bytes, so the arg-spec size check passes and it reads garbage on device instead. Cost of
    # getting it wrong: 75% NaN, see gh-chunks-device-gate-fails-with-a-1-in-4-nan-stride.
    gh_chunks: int = field(default=1, repr=False)
    # Row-parallel down projection (design.py's ROW_PARALLEL_DOWN): no gh all-gather, a per-core
    # column-shard of Wd, cascade-summed across all N cores. d_chunks splits the sum's D axis
    # into rounds; this build requires d_chunks == num_aie_columns*num_aie_rows. 1 is today's
    # byte-identical path.
    row_parallel_down: bool = field(default=False, repr=False)
    d_chunks: int = field(default=1, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "epsilon": "eps",
        "num_aie_columns": "cols",
        "num_aie_rows": "rows",
        "fuse_o": "fo",
        "post_norm": "pn",
        "weight_dtype": "wdt",
        "group_size": "g",
        "layout": "lay",
        "row_group": "rg",
        "scale_dtype": "sdt",
        "weight_depth": "wd",
        "tile_rows_gu": "tr",
        "gh_chunks": "ghc",
        "row_parallel_down": "rpd",
        "d_chunks": "dc",
    }

    def __post_init__(self):
        if self.D <= 0 or self.FF <= 0:
            raise ValueError(f"D ({self.D}) and FF ({self.FF}) must be positive")
        if self.act not in ("silu", "gelu_tanh"):
            raise ValueError(f"unknown act {self.act!r}")
        if self.fuse_o:
            if not self.QD or self.QD <= 0:
                raise ValueError("fuse_o requires QD (the attention context width)")
            if self.num_aie_rows != 1:
                raise ValueError("fuse_o is only derived for num_aie_rows=1 (see design.py)")
        if self.post_norm and self.num_aie_rows != 1:
            raise ValueError("post_norm is only derived for num_aie_rows=1 (see design.py)")
        if self.layout not in ("header_first", "row_group_planar"):
            raise ValueError(
                f"unknown layout {self.layout!r} (expected 'header_first' or 'row_group_planar')")
        if self.layout == "row_group_planar" and self.weight_dtype == "bf16":
            raise ValueError("layout='row_group_planar' needs weight_dtype != 'bf16'")
        if self.layout == "header_first" and self.row_group is not None:
            raise ValueError("row_group is only meaningful under layout='row_group_planar'")
        if self.row_group is not None and self.row_group < 1:
            raise ValueError(f"row_group ({self.row_group}) must be >= 1")
        if self.scale_dtype not in ("f32", "bf16"):
            raise ValueError(
                f"unknown scale_dtype {self.scale_dtype!r} (expected 'f32' or 'bf16')")
        if self.scale_dtype != "f32" and self.weight_dtype == "bf16":
            raise ValueError("scale_dtype is only meaningful for a quantized weight_dtype")
        if self.layout == "row_group_planar" and self.row_group is None:
            from iron.common.quant import derive_row_group, widest_chunk
            _ks = [self.D, self.FF] + ([self.QD] if self.fuse_o else [])
            object.__setattr__(self, "row_group", derive_row_group(
                _ks, self.group_size, self.weight_dtype,
                vec_size=widest_chunk(self.group_size, self.weight_dtype),
                scale_dtype=self.scale_dtype))
        if self.gh_chunks > 1:
            if self.weight_dtype not in ("int4", "int8"):
                raise ValueError(
                    "gh_chunks>1 needs weight_dtype in ('int4','int8') -- a bf16 Wd row alone is "
                    "already most of L1 (see g4-t2-mlp-l1-floor), and the affine dtypes are not "
                    "covered by mv_quant_taccum.cc"
                )
            if self.fuse_o or self.post_norm or self.num_aie_rows != 1:
                raise ValueError(
                    "gh_chunks>1 is only derived for fuse_o=False, post_norm=False, "
                    "num_aie_rows=1")
            if self.FF % self.D != 0 or self.gh_chunks != self.FF // self.D:
                raise ValueError(
                    f"gh_chunks={self.gh_chunks} must equal FF//D={self.FF // self.D if self.FF % self.D == 0 else '?'} "
                    "-- a chunk-row of Wd must pack to the same bytes as a Wg/Wu row (K_CHUNK == D) "
                    "so it can share weight_ofs[c]'s existing tile; see design.py"
                )
        if self.row_parallel_down:
            if self.weight_dtype not in ("int4", "int8"):
                raise ValueError(
                    "row_parallel_down needs weight_dtype in ('int4','int8') -- its local matvec "
                    "reuses mv_quant_taccum.cc, which has no bf16/affine form"
                )
            if self.layout != "header_first":
                raise ValueError(
                    "row_parallel_down needs layout='header_first' -- mv_quant_taccum.cc has no "
                    "row_group_planar form")
            if self.fuse_o or not self.post_norm or self.num_aie_rows != 1 or self.gh_chunks != 1:
                raise ValueError(
                    "row_parallel_down is only derived for fuse_o=False, post_norm=True, "
                    "num_aie_rows=1, gh_chunks=1 -- post_norm=True is REQUIRED here (unlike "
                    "gh_chunks): the cascade-summed result reaches every core by reusing "
                    "post_norm's own raw-d all-gather round trip")
            N = self.num_aie_columns * self.num_aie_rows
            if self.d_chunks != N:
                raise ValueError(
                    f"d_chunks={self.d_chunks} must equal N={N} in this build -- see design.py's "
                    "ROW_PARALLEL_DOWN section")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _wd_rp_units(self):
        """Wd's row_parallel_down arg-spec size, in weight WIRE units: N independently-quantized
        [D, FF/N] column-shard blocks (`quantize_weight_chunked(Wd, ..., n_chunks=N)`), each
        carrying its OWN header -- a genuinely different total than the row-sharded
        `D * row_stride_bytes(FF, ...)` the non-row-parallel arms use. See design.py's Wd_L3_ty."""
        from iron.common.quant import row_stride_bytes
        N = self.num_aie_columns * self.num_aie_rows
        FF_PER_CORE = self.FF // N
        wrow_ffpc = row_stride_bytes(FF_PER_CORE, self.group_size, self.weight_dtype,
                                     self.scale_dtype)
        return N * self.D * wrow_ffpc

    @property
    def _wo_rows_padded(self):
        """Wo's own arg-spec row count once padded for the TSI_O overlap window -- see
        design.py's FUSE_O section. Mirrors design.py's arithmetic exactly (kept in sync by hand;
        design.py itself asserts the same divisibility this depends on)."""
        N = self.num_aie_columns * self.num_aie_rows
        D_PER_CORE = self.D // N
        WTILE_ELEMS = 6 * self.D  # TSI_GU * D, design.py's shared-tile constant
        TSI_O = WTILE_ELEMS // self.QD
        N_O_TILES = -(-D_PER_CORE // TSI_O)
        O_WINDOW = N_O_TILES * TSI_O
        return self.D + (O_WINDOW - D_PER_CORE)

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_swiglu_mlp_dp",
                (aie_utils.get_current_device(), self.D, self.FF, self.epsilon),
                {
                    "stack_size": 0x800,
                    "n_aie_cols": self.num_aie_columns,
                    "n_aie_rows": self.num_aie_rows,
                    "QD": self.QD,
                    "fuse_o": self.fuse_o,
                    "act": self.act,
                    "post_norm": self.post_norm,
                    "weight_dtype": self.weight_dtype,
                    "group_size": self.group_size,
                    "layout": self.layout,
                    "row_group": self.row_group,
                    "scale_dtype": self.scale_dtype,
                    "weight_depth": self.weight_depth,
                    "tile_rows_gu": self.tile_rows_gu or None,
                    "gh_chunks": self.gh_chunks,
                    "row_parallel_down": self.row_parallel_down,
                    "d_chunks": self.d_chunks,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"

        add_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        mul_obj = KernelObjectArtifact(
            "mul.o", dependencies=[SourceArtifact(kdir / "generic" / "mul.cc")]
        )
        rms_norm_obj = KernelObjectArtifact(
            f"rms_norm_{self.D}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.D}"],
        )
        # Activation: silu.cc/gelu.cc export silu_tile_bf16/gelu_tile_bf16 at the identical
        # in-place ABI (design.py's docstring) -- a SWAP, so exactly one is compiled. The object
        # name carries `act`: two objects built under the same name for different activations is
        # the same collision class as the weight-dtype one above.
        _act_src = "silu.cc" if self.act == "silu" else "gelu.cc"
        act_obj = KernelObjectArtifact(
            f"{self.act}.o", dependencies=[SourceArtifact(kdir / arch_dir / _act_src)]
        )
        # One source and one flag set for both weight formats: mv_quant.cc exports
        # matvec_vectorized_{int4,int8}_bf16 with mv.cc's exact signature except that `a_in` is
        # int8, which is the only thing the WTILE_ty change alters. GROUP_SIZE is the extra flag.
        _qsrc = kdir / "generic" / ("mv.cc" if self.weight_dtype == "bf16" else "mv_quant.cc")
        _qtag = "" if self.weight_dtype == "bf16" else f"_{self.weight_dtype}g{self.group_size}"
        # Same collision class as VEC_SIZE below: -DPLANAR/-DSCALE_BF16 change the emitted code
        # while the exported symbol stays identical, so they must ride in the object NAME.
        if self.layout == "row_group_planar":
            _qtag += f"_planar{self.row_group}"
        if self.scale_dtype != "f32":
            _qtag += f"_s{self.scale_dtype}"
        # A vector chunk must not straddle a quant group, so VEC_SIZE is capped by the group
        # width -- and it is ALSO capped by payload alignment, which is what
        # max_legal_vec_size adds. The payload starts header_bytes into a row and load_v needs
        # the pointer on its access width; at K=1024 g=128 the header is 32 B, which int4's
        # 32-byte load clears and int8's 64-byte load does not. That is the whole reason int4
        # was correct on device and int8 computed garbage. It is in the object NAME because two
        # objects compiled at different VEC_SIZE export the same symbol.
        if self.weight_dtype == "bf16":
            _vec = 64
        else:
            from iron.common.quant import max_legal_vec_size
            _ks = [self.D, self.FF] + ([self.QD] if self.fuse_o else [])
            _vec = max_legal_vec_size(_ks, self.group_size, self.weight_dtype)
            # Emit ONLY this dtype's wrapper. All four instantiate otherwise, and a template's
            # static_asserts fire on instantiation -- so one VEC_SIZE would have to be legal for
            # every dtype, dragging int4 to int8's alignment constraint.
        _qflags = ([] if self.weight_dtype == "bf16"
                   else [f"-DGROUP_SIZE={self.group_size}",
                         f"-DQUANT_EMIT_{self.weight_dtype.upper()}=1"])
        if self.weight_dtype != "bf16":
            if self.layout == "row_group_planar":
                _qflags += ["-DPLANAR=1", f"-DROW_GROUP={self.row_group}"]
            if self.scale_dtype == "bf16":
                _qflags.append("-DSCALE_BF16=1")
        mv_gu_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_{_vec}vs{_qtag}.o",
            dependencies=[SourceArtifact(_qsrc)],
            extra_flags=[f"-DDIM_K={self.D}", f"-DVEC_SIZE={_vec}"] + _qflags,
        )
        # Same exported symbol as mv_gu_obj (DIM_K is baked in, not part of the name); this
        # object's own device-wide symbol table entry must be distinct, so it is compiled with a
        # prefix -- see design.py's mv_d_kernel comment and fuse/mlp-block's identical mechanism.
        mv_d_obj = KernelObjectArtifact(
            f"down_gemv_{self.FF}k_{_vec}vs{_qtag}.o",
            dependencies=[SourceArtifact(_qsrc)],
            extra_flags=[f"-DDIM_K={self.FF}", f"-DVEC_SIZE={_vec}"] + _qflags,
            prefix_symbols="down_",
        )
        deps = [add_obj, mul_obj, rms_norm_obj, act_obj, mv_gu_obj, mv_d_obj]
        if self.gh_chunks > 1:
            # T2.1a: mv_quant_taccum.cc's chunk-accumulate matvec, DIM_K = one chunk's width
            # (FF/gh_chunks), never the full FF -- see design.py's module docstring. Symbols
            # (matvec_taccum_*, mv_taccum_zero_f32/finish_bf16) are already unique, no prefix
            # needed. `gc_` is a renamed copy_offset_bf16_vector at gh_chunk_buf's D_ty->D_ty
            # shape, same reasoning as pn_/cx_/oa_ below.
            mv_taccum_obj = KernelObjectArtifact(
                f"down_taccum_{self.FF // self.gh_chunks}k_{_vec}vs{_qtag}.o",
                dependencies=[SourceArtifact(kdir / "generic" / "mv_quant_taccum.cc")],
                extra_flags=[f"-DDIM_K={self.FF // self.gh_chunks}", f"-DVEC_SIZE={_vec}"] + _qflags,
            )
            gc_copy_obj = KernelObjectArtifact(
                "add_gccopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                prefix_symbols="gc_",
            )
            deps += [mv_taccum_obj, gc_copy_obj]
        if self.row_parallel_down:
            # Local zero/accumulate/finish, DIM_K = FF_PER_CORE (a single, unchunked local
            # reduction over this core's own gh slice) -- own VEC_SIZE derivation since
            # FF_PER_CORE is not one of the Ks `_vec` above was picked for. `rp_` prefixed: this
            # object's matvec_taccum_*/mv_taccum_zero_f32/finish_bf16 symbols would otherwise
            # collide with a gh_chunks build at a different DIM_K. cascade_reduce_f32.cc has no
            # dtype/K axis -- one object, aie2p only (see design.py's module docstring).
            N = self.num_aie_columns * self.num_aie_rows
            FF_PER_CORE = self.FF // N
            from iron.common.quant import max_legal_vec_size
            _rp_vec = max_legal_vec_size([FF_PER_CORE], self.group_size, self.weight_dtype,
                                         self.scale_dtype, layout=self.layout)
            rp_taccum_obj = KernelObjectArtifact(
                f"rp_taccum_{FF_PER_CORE}k_{_rp_vec}vs{_qtag}.o",
                dependencies=[SourceArtifact(kdir / "generic" / "mv_quant_taccum.cc")],
                extra_flags=[f"-DDIM_K={FF_PER_CORE}", f"-DVEC_SIZE={_rp_vec}"] + _qflags,
                prefix_symbols="rp_",
            )
            cascade_reduce_obj = KernelObjectArtifact(
                "cascade_reduce_f32.o",
                dependencies=[SourceArtifact(kdir / "aie2p" / "cascade_reduce_f32.cc")],
            )
            deps += [rp_taccum_obj, cascade_reduce_obj]
        if self.post_norm:
            # Renamed recompile, see design.py's gc_copy_kernel. Only referenced under
            # fuse_o=True (pa_gain); harmless, unreferenced archive member otherwise.
            copy_pn_obj = KernelObjectArtifact(
                "add_pncopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                prefix_symbols="pn_",
            )
            # Renamed recompile of mul.cc at pff_gain's D_ty/D_ty/DPC_ty shape -- see
            # design.py's mul_pn_kernel.
            mul_pn_obj = KernelObjectArtifact(
                "mul_pncopy.o", dependencies=[SourceArtifact(kdir / "generic" / "mul.cc")],
                prefix_symbols="pn_",
            )
            deps += [copy_pn_obj, mul_pn_obj]
        if self.fuse_o:
            mv_o_obj = KernelObjectArtifact(
                f"o_gemv_{self.QD}k_{_vec}vs{_qtag}.o",
                dependencies=[SourceArtifact(_qsrc)],
                extra_flags=[f"-DDIM_K={self.QD}", f"-DVEC_SIZE={_vec}"] + _qflags,
                prefix_symbols="o_",
            )
            # copy_offset_bf16_vector (in add.cc) is generic (pointer + runtime size/offset, no
            # compile-time shape), but a func.func declaration's symbol table entry is keyed by
            # NAME only, not by memref type -- two Kernel() Python bindings to the SAME symbol
            # with different declared shapes (QD_ty vs DPC_ty/OWIN_ty here) is a device-wide
            # "redefinition of symbol" verifier error, not a harmless overload (confirmed
            # device-free by aiecc's own MLIR verifier). Same fix as mv_d_kernel's "down_" prefix:
            # compile add.cc again per new call-site shape, under a distinct renamed symbol.
            cx_copy_obj = KernelObjectArtifact(
                "add_cxcopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                prefix_symbols="cx_",
            )
            oa_copy_obj = KernelObjectArtifact(
                "add_oacopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                prefix_symbols="oa_",
            )
            deps += [mv_o_obj, cx_copy_obj, oa_copy_obj]
        _act_tag = "" if self.act == "silu" else f"_{self.act}"
        _pn_tag = "_pn" if self.post_norm else ""
        _ghc_tag = f"_ghc{self.gh_chunks}" if self.gh_chunks > 1 else ""
        _rpd_tag = "_rpd" if self.row_parallel_down else ""
        core_archive = KernelArchiveArtifact(
            f"swiglu_mlp_dp_core{_qtag}{_act_tag}{_pn_tag}{_ghc_tag}{_rpd_tag}.a",
            dependencies=deps)
        return [core_archive]

    def _wrow(self, K):
        """Wire units per weight ROW of width K: bf16 elements, or packed bytes when quantized."""
        if self.weight_dtype == "bf16":
            return K
        from iron.common.quant import row_stride_bytes
        return row_stride_bytes(K, self.group_size, self.weight_dtype)

    def _wspec(self, n_units, comment_unused=None):
        """One weight argument, sized in wire units. dtype is OMITTED for bf16 rather than passed
        as None -- AIERuntimeArgSpec's default is not None, and passing it explicitly sized the
        arena at 8 bytes/element instead of 2 (caught by the sequence's own layout assert)."""
        if self.weight_dtype == "bf16":
            return AIERuntimeArgSpec("in", (n_units,))
        return AIERuntimeArgSpec("in", (n_units,), dtype=np.int8)

    def get_arg_spec(self):
        # POST_NORMS packs the new gain(s) into ONE argument: [pff] at D, or [pa | pff] at 2*D
        # under fuse_o -- mirrors attn_block_dp's norms_packed. Inserted right before `nxt`;
        # omitted entirely at post_norm=False, which is what keeps the default arg list unchanged.
        post_norms_spec = (
            [AIERuntimeArgSpec("in", (2 * self.D if self.fuse_o else self.D,))]
            if self.post_norm else []
        )
        if self.fuse_o:
            return [
                AIERuntimeArgSpec("in", (self.D,)),                       # cur
                AIERuntimeArgSpec("in", (self.QD,)),                      # cx
                AIERuntimeArgSpec("in", (self.D,)),                       # n_pf
                self._wspec(self._wo_rows_padded * self._wrow(self.QD)),   # Wo, flat [D+pad,QD]
                self._wspec(self.FF * self._wrow(self.D)),        # Wg, flat [FF,D]
                self._wspec(self.FF * self._wrow(self.D)),        # Wu, flat [FF,D]
                self._wspec(self.D * self._wrow(self.FF)),        # Wd, flat [D,FF]
                AIERuntimeArgSpec("inout", (self.FF,)),                   # gh_scratch
                AIERuntimeArgSpec("inout", (self.D,)),                    # a_scratch
                *post_norms_spec,                                         # post_norms (optional)
                AIERuntimeArgSpec("out", (self.D,)),                      # nxt
            ]
        # Wd: N column-shard blocks under row_parallel_down (_wd_rp_units, a genuinely different
        # total -- N separate quant headers, see design.py's Wd_L3_ty), else the plain [D,FF]
        # row-sharded total every other arm uses.
        wd_units = self._wd_rp_units if self.row_parallel_down else self.D * self._wrow(self.FF)
        return [
            AIERuntimeArgSpec("in", (self.D,)),                 # cur
            AIERuntimeArgSpec("in", (self.D,)),                 # a
            AIERuntimeArgSpec("in", (self.D,)),                 # n_pf
            self._wspec(self.FF * self._wrow(self.D)),  # Wg, flat [FF,D]
            self._wspec(self.FF * self._wrow(self.D)),  # Wu, flat [FF,D]
            self._wspec(wd_units),                       # Wd, flat [D,FF] or row_parallel_down shards
            AIERuntimeArgSpec("inout", (self.FF,)),             # gh_scratch (internal round-trip)
            *post_norms_spec,                                   # post_norms (optional)
            AIERuntimeArgSpec("out", (self.D,)),                # nxt
        ]

    def reference(self, cur, cx_or_a, n_pf, *rest):
        # NOTE: does not model post_norm (out of scope for this task -- see reference.py). At
        # post_norm=True `rest` carries one extra `post_norms` element beyond what the slices
        # below read, which is simply ignored: this computes the PRE-sandwich-norm math, not what
        # the device does with post_norm=True. Not a check for that arm.
        from iron.operators.swiglu_mlp_dp.reference import reference, reference_fused_o

        if self.fuse_o:
            Wo, Wg, Wu, Wd = rest[:4]
            return reference_fused_o(
                cur, cx_or_a, n_pf, Wo, Wg, Wu, Wd, self.D, self.FF, self.QD, self.epsilon,
                act=self.act,
            )
        Wg, Wu, Wd = rest[:3]
        return reference(cur, cx_or_a, n_pf, Wg, Wu, Wd, self.D, self.FF, self.epsilon,
                          act=self.act)
