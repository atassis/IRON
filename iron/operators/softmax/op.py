# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import aie.utils as aie_utils

from iron.common.device_utils import get_kernel_dir
from iron.common.operator_bases import lut_based_ops_artifacts
from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelArchiveArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)


@dataclass
class Softmax(MLIROperator):
    """AIE-accelerated Softmax operation"""

    rows: int
    cols: int
    num_aie_columns: int = 1
    num_channels: int = 1
    rtp_vector_size: int | None = None
    vector_size_parameter: str | None = None
    # Elements ALLOCATED per row in the in/out buffers, when that differs from the elements
    # COMPUTED (`cols`). None means they are equal -- the old behaviour, byte for byte. Mirrors
    # gemv's alloc_M: decode's masked softmax runs over a scores row allocated at max_seq while
    # only n_past columns are live, so cols=n_past while the per-row stride must stay max_seq. Only
    # the buffer size and the per-row stride move; the computed row length stays `cols`.
    alloc_cols: int | None = field(default=None, repr=False)
    context: object = field(default=None, repr=False)

    @property
    def size(self):
        return self.rows * self.cols

    def __post_init__(self):
        # `rows % 16` was rejected here with no stated derivation, and design.py does not need it:
        # rows are split across cores and each core then walks `N_div_n = per_core_elements // cols`
        # TILES, so what the design requires of rows is (a) divisibility by the split, checked just
        # below, and (b) at least one tile per core -- rows >= num_aie_columns * num_channels. The
        # %16 rejected Gemma-3's 4 query heads, which run correctly at num_aie_columns=4
        # (N_div_n=1). Checking the real constraint instead, and naming the fix in the message,
        # because a head count under the split otherwise produces N_div_n=0 and an op that
        # silently computes nothing.
        cores = self.num_aie_columns * self.num_channels
        if self.rows < cores:
            raise ValueError(
                f"rows ({self.rows}) < num_aie_columns*num_channels ({cores}): each core would get "
                f"less than one {self.cols}-element tile and the op would compute nothing. "
                f"Lower num_aie_columns to at most {self.rows}."
            )
        if self.cols % 16 != 0:
            raise ValueError(f"cols ({self.cols}) must be a multiple of 16")
        if self.rows % self.num_aie_columns != 0:
            raise ValueError(
                f"rows ({self.rows}) must be a multiple of num_aie_columns ({self.num_aie_columns})"
            )
        if self.alloc_cols is not None and self.alloc_cols < self.cols:
            raise ValueError(
                f"alloc_cols ({self.alloc_cols}) must be >= cols ({self.cols}): it is the "
                f"ALLOCATED per-row stride, not a second window"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # A wide-strided row is a different design from the plain one at the same cols: same
        # compute extent, different buffer size and per-row stride. Without this they collide in
        # the build dir and a cached plain build silently satisfies the windowed op. alloc_cols ==
        # cols is the same design as alloc_cols=None, so it keeps the stable name.
        base = super().name
        if self.alloc_cols is not None and self.alloc_cols != self.cols:
            base = f"{base}_ac{self.alloc_cols}"
        return base

    @property
    def _kernel_link_file(self):
        kernel_dir = get_kernel_dir()
        if kernel_dir == "aie2":
            return f"{self.name}_kernels.a"
        return "softmax.o"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "softmax",
                (),
                {
                    "dev": aie_utils.get_current_device(),
                    "num_elements": self.size,
                    "num_aie_columns": self.num_aie_columns,
                    "num_channels": self.num_channels,
                    "trace_size": 0,
                    "tile_size": self.cols,
                    "rtp_vector_size": self.rtp_vector_size,
                    "vector_size_parameter": self.vector_size_parameter,
                    "kernel_obj_file": self._kernel_link_file,
                    "alloc_cols": self.alloc_cols,
                },
            ),
        )

    def get_kernel_artifacts(self):
        kernel_dir = get_kernel_dir()
        softmax_obj = KernelObjectArtifact(
            "softmax.o",
            dependencies=[
                SourceArtifact(
                    self.context.base_dir / "aie_kernels" / kernel_dir / "softmax.cc"
                )
            ],
        )
        lut_objs = lut_based_ops_artifacts(kernel_dir)
        if lut_objs:
            return [
                KernelArchiveArtifact(
                    f"{self.name}_kernels.a",
                    dependencies=[softmax_obj] + lut_objs,
                )
            ]
        return [softmax_obj]

    def get_arg_spec(self):
        # alloc_cols in (None, cols) is the same design/layout as the old behaviour, so it keeps
        # the flat shape byte for byte. A wide-strided row is sized by the ALLOCATION: a narrow
        # computed row still addresses a buffer whose per-row stride is alloc_cols, so the host
        # operand must be that big or every row after the first lands inside the previous row's
        # tail.
        if self.alloc_cols is None or self.alloc_cols == self.cols:
            return [
                AIERuntimeArgSpec("in", (self.size,)),
                AIERuntimeArgSpec("out", (self.size,)),
            ]
        return [
            AIERuntimeArgSpec("in", (self.rows, self.alloc_cols)),
            AIERuntimeArgSpec("out", (self.rows, self.alloc_cols)),
        ]

    def reference(self, x):
        """CPU reference: row-wise softmax over ``cols``.

        Note: ignores the runtime ``vector_size_parameter`` (if any); the
        reference always softmaxes over the full ``cols``. For decode-style
        usage with a masked tail, the trailing positions will not match the
        NPU output."""
        from iron.operators.softmax.reference import reference

        return reference(x.reshape(self.rows, self.cols))
