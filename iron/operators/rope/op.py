# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils


@dataclass
class RoPE(MLIROperator):
    """AIE-accelerated RoPE (Rotary Position Embedding) operator"""

    rows: int
    cols: int
    angle_rows: int | None = None
    num_aie_columns: int = 1
    method_type: int = 0
    # Width of a tensor row when only its first `cols` rotate (partial rotary); None = cols.
    row_stride: int | None = None
    # Per-core buffer allocation strategy ('basic-sequential' or 'bank-aware'), forwarded to each
    # Worker -- see GEMM's twin field (iron/operators/gemm/op.py) for the measurement this mirrors.
    allocation_scheme: str | None = field(default=None, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "angle_rows": "arows",
        "method_type": "m",
        "row_stride": "rs",
    }

    def __post_init__(self):
        if self.angle_rows is None:
            self.angle_rows = self.rows

        if not (self.cols % (16 * 2) == 0 and self.cols >= (16 * 2)):
            raise ValueError("cols must be multiple of 32 and >= 32")
        if self.rows % self.num_aie_columns != 0:
            raise ValueError("rows must be divisible by num_aie_columns")
        if not (self.angle_rows <= self.rows and self.rows % self.angle_rows == 0):
            raise ValueError("angle_rows must divide rows")
        if not (
            self.angle_rows >= self.num_aie_columns
            and self.angle_rows % self.num_aie_columns == 0
        ):
            raise ValueError("angle_rows must be divisible by num_aie_columns")
        if self.row_stride is not None and self.row_stride < self.cols:
            raise ValueError(f"row_stride ({self.row_stride}) is narrower than cols ({self.cols})")
        if self.method_type not in {0, 1}:
            raise ValueError(f"method_type must be 0 or 1, got {self.method_type}")

        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "rope",
                (
                    aie_utils.get_current_device(),
                    self.rows,
                    self.cols,
                    self.angle_rows,
                    self.num_aie_columns,
                    0,
                    self.method_type,
                ),
                {"allocation_scheme": self.allocation_scheme,
                 **({"row_stride": self.row_stride} if self.row_stride else {})},
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                f"rope_{self.method_type}.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "rope.cc"
                    )
                ],
                extra_flags=[
                    "-DTWO_HALVES" if 0 == self.method_type else "-DINTERLEAVED"
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.rows, self.row_stride or self.cols)),  # input tensor
            AIERuntimeArgSpec("in", (self.angle_rows, self.cols)),  # angles
            AIERuntimeArgSpec("out", (self.rows, self.row_stride or self.cols)),  # output
        ]

    def reference(self, x, angles):
        """CPU reference for RoPE.

        Assumes ``angles`` holds interleaved [cos, sin, cos, sin, ...] pairs
        along the last dim (length ``cols``).  Only ``method_type == 0``
        (TWO_HALVES) is currently supported.

        ``angles`` may have fewer rows than ``x``; in that case the angles
        are tiled along the row dimension to match ``x``."""
        from iron.operators.rope.reference import reference

        return reference(x, angles, self.method_type, self.rows, self.cols)
