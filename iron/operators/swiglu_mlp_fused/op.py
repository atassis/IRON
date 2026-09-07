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
class SwiGLUMLPFused(MLIROperator):
    """Decode SwiGLU MLP block as ONE `aie.device`: residual-add, weighted RMSNorm, gate/up
    GEMV, SiLU, elementwise-mul, down GEMV, residual-add -- see design.py's module docstring for
    the architecture (single core, no column split, x1/hf/g/u/d never leave L1).

    Runtime interface is exactly cur, a, n_pf, Wg, Wu, Wd -> nxt (6 in, 1 out); every other named
    quantity in the group (x1, hf, g, u, gh, d) is an on-core `Buffer`, not a runtime argument.
    """

    D: int
    FF: int
    epsilon: float = 1e-5
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "epsilon": "eps",
    }

    def __post_init__(self):
        if self.D <= 0 or self.FF <= 0:
            raise ValueError(f"D ({self.D}) and FF ({self.FF}) must be positive")
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_swiglu_mlp_fused",
                (aie_utils.get_current_device(), self.D, self.FF, self.epsilon),
                {"stack_size": 0x800},
            ),
        )

    def get_kernel_artifacts(self):
        # design.py splits the block across 5 cores (P1/P2/A/B/P3 -- see its module docstring
        # for why one core does not place: each AIE2P compute tile has only 2 input/2 output DMA
        # channels, independent of the device-wide 16-channel ShimDMA budget). Only core A calls
        # more than one kernel function (matvec + silu + mul), so only it needs an archive; every
        # other core links a single, solo object.
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"

        add_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        rms_norm_obj = KernelObjectArtifact(
            "rms_norm.o", dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")]
        )
        mv_gu_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_64vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.D}", "-DVEC_SIZE=64"],
        )
        silu_obj = KernelObjectArtifact(
            "silu.o", dependencies=[SourceArtifact(kdir / arch_dir / "silu.cc")]
        )
        mul_obj = KernelObjectArtifact(
            "mul.o", dependencies=[SourceArtifact(kdir / "generic" / "mul.cc")]
        )
        core_a_archive = KernelArchiveArtifact(
            "swiglu_mlp_fused_core_a.a",
            dependencies=[mv_gu_obj, silu_obj, mul_obj],
        )
        # Same exported symbol name as mv_gu_obj (DIM_K is baked in, not part of the name). The two
        # end up on different physical cores, but the MLIR module's symbol table is device-wide
        # (one `aie.device`, one namespace), so aie.core verification rejects the duplicate name
        # regardless of placement -- prefix this object so design.py's Kernel() can bind a
        # distinct name. See ArchiveCompilationRule's own comment on the same mechanism ("two
        # designs [can] carry the same kernel names").
        mv_d_obj = KernelObjectArtifact(
            f"down_gemv_{self.FF}k_64vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.FF}", "-DVEC_SIZE=64"],
            prefix_symbols="down_",
        )
        return [add_obj, rms_norm_obj, core_a_archive, mv_d_obj]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.D,)),          # cur
            AIERuntimeArgSpec("in", (self.D,)),          # a
            AIERuntimeArgSpec("in", (self.D,)),          # n_pf
            AIERuntimeArgSpec("in", (self.FF * self.D,)),  # Wg, flat [FF,D]
            AIERuntimeArgSpec("in", (self.FF * self.D,)),  # Wu, flat [FF,D]
            AIERuntimeArgSpec("in", (self.D * self.FF,)),  # Wd, flat [D,FF]
            AIERuntimeArgSpec("out", (self.D,)),         # nxt
        ]

    def reference(self, cur, a, n_pf, Wg, Wu, Wd):
        from iron.operators.swiglu_mlp_fused.reference import reference

        return reference(cur, a, n_pf, Wg, Wu, Wd, self.D, self.FF, self.epsilon)
