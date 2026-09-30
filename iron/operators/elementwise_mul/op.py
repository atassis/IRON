# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import ClassVar

import aie.utils as aie_utils

from iron.common import (
    AIERuntimeArgSpec,
    BinaryElementwiseOperator,
    DesignGenerator,
    PythonGeneratedMLIRArtifact,
)


@dataclass
class ElementwiseMul(BinaryElementwiseOperator):
    """AIE-accelerated element-wise multiplication.

    `rows`, when set, broadcasts the `[size]` "in2" operand against an `[rows, size]` "in1"/"out"
    operand in one dispatch (one runtime sequence, one gain read per D-chunk instead of one per
    row) -- see design.py. `rows=None` (the default) is the plain `[size] x [size]` operator,
    routed through the shared binary_elementwise_design and untouched by the broadcast path.
    """

    rows: int | None = None

    kernel_name: ClassVar[str] = "mul"
    kernel_fn_name: ClassVar[str] = "eltwise_mul_bf16_vector"
    kernel_subdir: ClassVar[str] = "generic"
    callback_fn: ClassVar[str] = "my_eltwise_mul"

    def __post_init__(self) -> None:
        if self.rows is not None and self.rows < 1:
            raise ValueError(f"rows ({self.rows}) must be >= 1")
        super().__post_init__()

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        if self.rows is None:
            return super().get_arg_spec()
        return [
            AIERuntimeArgSpec("in", (self.rows * self.size,)),
            AIERuntimeArgSpec("in", (self.size,)),
            AIERuntimeArgSpec("out", (self.rows * self.size,)),
        ]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        if self.rows is None:
            return super().get_mlir_artifact()
        callback_args = [
            aie_utils.get_current_device(),
            self.size,
            self.num_aie_columns,
            self.tile_size,
            self.rows,
            0,  # trace_size
            self.kernel_fn_name,
            f"{self.kernel_name}.o",
        ]
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "elementwise_mul_broadcast_design",
                tuple(callback_args),
                {"allocation_scheme": self.allocation_scheme},
            ),
        )

    def reference(self, a, b):
        from iron.operators.elementwise_mul.reference import reference

        return reference(a, b)
