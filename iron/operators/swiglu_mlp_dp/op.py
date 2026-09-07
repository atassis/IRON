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
    # Weight-stream FORMAT axis for Wg/Wu/Wd, the same one GEMV carries and packed by the same
    # iron/common/quant.py: "bf16" (default, byte-for-byte the pre-existing path) or "int4"/"int8"
    # group-quantized, dequantized on-core by mv_quant.cc before the same bf16 MAC. Activations
    # stay bf16 throughout. repr=False + the `name` override keep the default path's artifact
    # names stable, matching GEMV's convention for the same field.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
    kernel_vector_size: int = field(default=64, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "epsilon": "eps",
        "num_aie_columns": "cols",
        "num_aie_rows": "rows",
        "fuse_o": "fo",
    }

    def __post_init__(self):
        if self.D <= 0 or self.FF <= 0:
            raise ValueError(f"D ({self.D}) and FF ({self.FF}) must be positive")
        if self.fuse_o:
            if not self.QD or self.QD <= 0:
                raise ValueError("fuse_o requires QD (the attention context width)")
            if self.num_aie_rows != 1:
                raise ValueError("fuse_o is only derived for num_aie_rows=1 (see design.py)")
        if self.weight_dtype not in ("bf16", "int4", "int8"):
            raise ValueError(
                f"unknown weight_dtype {self.weight_dtype!r} (expected 'bf16', 'int4' or 'int8')"
            )
        if self.weight_dtype != "bf16":
            if self.fuse_o:
                # Wo rides the SAME shared weight ObjectFifo as Wg/Wu/Wd, and that tile has ONE
                # byte size. Quantizing three of the four and not the fourth cannot share it.
                raise NotImplementedError(
                    "fuse_o with a quantized weight_dtype is not implemented: Wo shares the "
                    "weight ObjectFifo with Wg/Wu/Wd, so it would have to be quantized too"
                )
            if self.group_size <= 0:
                raise ValueError("weight_dtype != 'bf16' needs an explicit group_size > 0")
            # Same narrowing GEMV applies for the same measured reason: the quantized path
            # amortises the group scale over group_size/kernel_vector_size chunks, so 32 beats
            # the bf16-derived default of 64.
            if self.kernel_vector_size == 64:
                self.kernel_vector_size = 32
            # BOTH matvecs read this format: gate/up at K=D, down at K=FF. A group_size that
            # divides one and not the other would build a kernel whose static_assert fires.
            for label, K in (("D", self.D), ("FF", self.FF)):
                if K % self.group_size != 0:
                    raise ValueError(
                        f"{label}={K} must be a whole number of groups "
                        f"(group_size={self.group_size})"
                    )
            if self.group_size % self.kernel_vector_size != 0:
                # mv_quant.cc's vectorized dequant chunk must never straddle a group boundary.
                raise ValueError(
                    f"group_size={self.group_size} must be a multiple of kernel_vector_size="
                    f"{self.kernel_vector_size}"
                )
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # weight_dtype is repr=False so the bf16 path keeps its existing artifact name, but a
        # quantized variant must not share one with the bf16 op of the same shape: both would
        # emit the same .mlir/.xclbin and a cached bf16 build would silently satisfy it.
        base = super().name
        if self.weight_dtype != "bf16":
            base = f"{base}_wdt{self.weight_dtype}g{self.group_size}"
        return base

    @property
    def _core_archive(self) -> str:
        # Named by format for the same reason as `name`: the archive holds the matvec objects,
        # and the bf16 and quantized ones export DIFFERENT symbols from DIFFERENT sources.
        if self.weight_dtype == "bf16":
            return "swiglu_mlp_dp_core.a"
        return (f"swiglu_mlp_dp_core_{self.weight_dtype}g{self.group_size}"
                f"_{self.kernel_vector_size}vs.a")

    @property
    def _row_widths(self):
        """(gate/up, down) weight row width in arg-spec units -- elements at bf16, packed bytes
        when quantized. design.py derives the same pair; both go through iron/common/quant.py."""
        if self.weight_dtype == "bf16":
            return self.D, self.FF
        from iron.common.quant import row_stride_bytes

        return (row_stride_bytes(self.D, self.group_size, self.weight_dtype),
                row_stride_bytes(self.FF, self.group_size, self.weight_dtype))

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
                    "weight_dtype": self.weight_dtype,
                    "group_size": self.group_size,
                    "core_archive": self._core_archive,
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
            "rms_norm.o", dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")]
        )
        silu_obj = KernelObjectArtifact(
            "silu.o", dependencies=[SourceArtifact(kdir / arch_dir / "silu.cc")]
        )
        vs = self.kernel_vector_size
        if self.weight_dtype == "bf16":
            mv_src, mv_tag, mv_extra = kdir / "generic" / "mv.cc", f"{vs}vs", []
        else:
            # Same signature as mv.cc's, so only the source and the name change here.
            mv_src = kdir / "generic" / "mv_quant.cc"
            mv_tag = f"{self.weight_dtype}g{self.group_size}_{vs}vs"
            mv_extra = [f"-DGROUP_SIZE={self.group_size}"]
        mv_gu_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_{mv_tag}.o",
            dependencies=[SourceArtifact(mv_src)],
            extra_flags=[f"-DDIM_K={self.D}", f"-DVEC_SIZE={vs}", *mv_extra],
        )
        # Same exported symbol as mv_gu_obj (DIM_K is baked in, not part of the name); this
        # object's own device-wide symbol table entry must be distinct, so it is compiled with a
        # prefix -- see design.py's mv_d_kernel comment and fuse/mlp-block's identical mechanism.
        mv_d_obj = KernelObjectArtifact(
            f"down_gemv_{self.FF}k_{mv_tag}.o",
            dependencies=[SourceArtifact(mv_src)],
            extra_flags=[f"-DDIM_K={self.FF}", f"-DVEC_SIZE={vs}", *mv_extra],
            prefix_symbols="down_",
        )
        deps = [add_obj, mul_obj, rms_norm_obj, silu_obj, mv_gu_obj, mv_d_obj]
        if self.fuse_o:
            mv_o_obj = KernelObjectArtifact(
                f"o_gemv_{self.QD}k_64vs.o",
                dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
                extra_flags=[f"-DDIM_K={self.QD}", "-DVEC_SIZE=64"],
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
        core_archive = KernelArchiveArtifact(self._core_archive, dependencies=deps)
        return [core_archive]

    def get_arg_spec(self):
        if self.fuse_o:
            return [
                AIERuntimeArgSpec("in", (self.D,)),                       # cur
                AIERuntimeArgSpec("in", (self.QD,)),                      # cx
                AIERuntimeArgSpec("in", (self.D,)),                       # n_pf
                AIERuntimeArgSpec("in", (self._wo_rows_padded * self.QD,)),  # Wo, flat [D+pad,QD]
                AIERuntimeArgSpec("in", (self.FF * self.D,)),             # Wg, flat [FF,D]
                AIERuntimeArgSpec("in", (self.FF * self.D,)),             # Wu, flat [FF,D]
                AIERuntimeArgSpec("in", (self.D * self.FF,)),             # Wd, flat [D,FF]
                AIERuntimeArgSpec("inout", (self.FF,)),                   # gh_scratch
                AIERuntimeArgSpec("inout", (self.D,)),                    # a_scratch
                AIERuntimeArgSpec("out", (self.D,)),                      # nxt
            ]
        import numpy as np

        gu_row, d_row = self._row_widths
        # Quantized weights are flat byte buffers, int8-typed purely so the emitted shim BDs type
        # as `i8` (matching decode_ddr_bytes.py's parser); the values are opaque packed bytes.
        w_kw = {} if self.weight_dtype == "bf16" else dict(dtype=np.int8)
        return [
            AIERuntimeArgSpec("in", (self.D,)),                 # cur
            AIERuntimeArgSpec("in", (self.D,)),                 # a
            AIERuntimeArgSpec("in", (self.D,)),                 # n_pf
            AIERuntimeArgSpec("in", (self.FF * gu_row,), **w_kw),  # Wg, FF rows of gu_row
            AIERuntimeArgSpec("in", (self.FF * gu_row,), **w_kw),  # Wu, FF rows of gu_row
            AIERuntimeArgSpec("in", (self.D * d_row,), **w_kw),    # Wd, D rows of d_row
            AIERuntimeArgSpec("inout", (self.FF,)),             # gh_scratch (internal round-trip)
            AIERuntimeArgSpec("out", (self.D,)),                # nxt
        ]

    def reference(self, cur, cx_or_a, n_pf, *rest):
        from iron.operators.swiglu_mlp_dp.reference import reference, reference_fused_o

        if self.weight_dtype != "bf16":
            # fuse_o is refused for a quantized weight_dtype, so `rest` is (Wg, Wu, Wd[, scratch]).
            from iron.common.quant import dequantize_weight

            g, u, d = rest[:3]
            rest = (dequantize_weight(g, self.FF, self.D, self.group_size, self.weight_dtype),
                    dequantize_weight(u, self.FF, self.D, self.group_size, self.weight_dtype),
                    dequantize_weight(d, self.D, self.FF, self.group_size, self.weight_dtype),
                    *rest[3:])

        if self.fuse_o:
            Wo, Wg, Wu, Wd = rest[:4]
            return reference_fused_o(
                cur, cx_or_a, n_pf, Wo, Wg, Wu, Wd, self.D, self.FF, self.QD, self.epsilon
            )
        Wg, Wu, Wd = rest[:3]
        return reference(cur, cx_or_a, n_pf, Wg, Wu, Wd, self.D, self.FF, self.epsilon)
