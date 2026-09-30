# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np

import aie.utils as aie_utils

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext
from iron.common.quant import row_stride_bytes


@dataclass
class DequantRows(MLIROperator):
    """GEMV's int4 weight rows (`header_first`, f32 group scales) expanded to bf16.

    Arguments: `a` int8 [rows * row_stride] packed rows, `c` bf16 [rows * out_stride], row r of
    the weight written at c[r * out_stride + out_col : ... + K]. The bf16 values are the ones
    GEMV's mv_quant.cc multiplies, so a GEMM over `c` sees decode's weights exactly.
    """

    rows: int
    K: int
    group_size: int = 32
    num_aie_columns: int = 8
    tile_rows: int | None = None
    out_stride: int | None = None
    out_col: int = 0
    stack_size: int | None = field(default=None, repr=False)
    context: AIEContext | None = field(default=None, repr=False)

    # Double-buffered in and out tiles; what is left of L1 after this is the core's stack.
    L1_BUDGET = 48 * 1024

    def __post_init__(self) -> None:
        self.row_stride = row_stride_bytes(self.K, self.group_size, "int4")
        if self.out_stride is None:
            self.out_stride = self.K
        if self.out_col + self.K > self.out_stride:
            raise ValueError(f"out_col {self.out_col} + K {self.K} exceeds out_stride "
                             f"{self.out_stride}")
        per_col = self.rows // self.num_aie_columns
        if self.tile_rows is None:
            fits = [t for t in range(1, per_col + 1) if per_col % t == 0
                    and 2 * t * (self.row_stride + 2 * self.K) <= self.L1_BUDGET]
            if not fits:
                raise ValueError(f"no row tile of K={self.K} fits {self.L1_BUDGET} B of L1")
            self.tile_rows = max(fits)
        if self.rows % self.num_aie_columns or per_col % self.tile_rows:
            raise ValueError(f"rows ({self.rows}) must split into {self.num_aie_columns} columns "
                             f"of whole {self.tile_rows}-row tiles")
        super().__init__(context=self.context)

    @property
    def _obj(self) -> str:
        return f"dequant_rows_k{self.K}_g{self.group_size}_r{self.tile_rows}.o"

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [AIERuntimeArgSpec("in", (self.rows * self.row_stride,), np.int8),
                AIERuntimeArgSpec("out", (self.rows * self.out_stride,))]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        args = (aie_utils.get_current_device(), self.rows, self.K, self.row_stride,
                self.tile_rows, self.num_aie_columns, self.out_stride, self.out_col, self._obj)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "design.py", "dequant_rows_design", args,
                            {"stack_size": self.stack_size}),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [KernelObjectArtifact(
            self._obj,
            dependencies=[SourceArtifact(self.context.base_dir / "aie_kernels" / "generic" /
                                         "dequant_rows.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={self.group_size}",
                         f"-DDQ_ROWS={self.tile_rows}"],
        )]
