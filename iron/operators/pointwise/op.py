# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import aie.utils as aie_utils

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
from iron.operators.pointwise.design import MODES


@dataclass
class Pointwise(MLIROperator):
    """Elementwise add, multiply or GELU as one core program, the op chosen by `mode` at run time.

    Instances that differ only in `mode` build identical device bodies, so an OperatorSequence
    built with merge_devices runs all of them on one device. Arguments are always (in, in, out);
    GELU ignores the second input.
    """

    size: int
    tile_size: int
    mode: str
    num_aie_columns: int = 8
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r} must be one of {MODES}")
        if self.size % (self.num_aie_columns * self.tile_size):
            raise ValueError(
                f"size ({self.size}) must be a multiple of num_aie_columns * tile_size"
            )
        MLIROperator.__init__(self, context=self.context)

    def design_key(self):
        return "|".join(str(x) for x in (
            "Pointwise", self.size, self.tile_size, self.mode, self.num_aie_columns))

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.size,)),
        ]

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "pointwise_design",
                (aie_utils.get_current_device(), self.size, self.num_aie_columns,
                 self.tile_size, self.mode),
                {},
            ),
        )

    def get_kernel_artifacts(self):
        base = self.context.base_dir / "aie_kernels"
        return [
            KernelObjectArtifact(f"{name}.o",
                                 dependencies=[SourceArtifact(base / subdir / f"{name}.cc")])
            for name, subdir in (("add", "generic"), ("mul", "generic"), ("gelu", "aie2p"))
        ]
