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
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "epsilon": "eps",
        "num_aie_columns": "cols",
        "num_aie_rows": "rows",
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
                "my_swiglu_mlp_dp",
                (aie_utils.get_current_device(), self.D, self.FF, self.epsilon),
                {
                    "stack_size": 0x800,
                    "n_aie_cols": self.num_aie_columns,
                    "n_aie_rows": self.num_aie_rows,
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
        mv_gu_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_64vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.D}", "-DVEC_SIZE=64"],
        )
        # Same exported symbol as mv_gu_obj (DIM_K is baked in, not part of the name); this
        # object's own device-wide symbol table entry must be distinct, so it is compiled with a
        # prefix -- see design.py's mv_d_kernel comment and fuse/mlp-block's identical mechanism.
        mv_d_obj = KernelObjectArtifact(
            f"down_gemv_{self.FF}k_64vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.FF}", "-DVEC_SIZE=64"],
            prefix_symbols="down_",
        )
        core_archive = KernelArchiveArtifact(
            "swiglu_mlp_dp_core.a",
            dependencies=[add_obj, mul_obj, rms_norm_obj, silu_obj, mv_gu_obj, mv_d_obj],
        )
        return [core_archive]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.D,)),                 # cur
            AIERuntimeArgSpec("in", (self.D,)),                 # a
            AIERuntimeArgSpec("in", (self.D,)),                 # n_pf
            AIERuntimeArgSpec("in", (self.FF * self.D,)),       # Wg, flat [FF,D]
            AIERuntimeArgSpec("in", (self.FF * self.D,)),       # Wu, flat [FF,D]
            AIERuntimeArgSpec("in", (self.D * self.FF,)),       # Wd, flat [D,FF]
            AIERuntimeArgSpec("inout", (self.FF,)),             # gh_scratch (internal round-trip)
            AIERuntimeArgSpec("out", (self.D,)),                # nxt
        ]

    def reference(self, cur, a, n_pf, Wg, Wu, Wd, gh_scratch=None):
        from iron.operators.swiglu_mlp_dp.reference import reference

        return reference(cur, a, n_pf, Wg, Wu, Wd, self.D, self.FF, self.epsilon)
