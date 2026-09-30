# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

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


@dataclass
class Conv1dStep(MLIROperator):
    """One decode step of a depthwise causal conv (the Mamba / Gated DeltaNet short conv).

    Operates on a `[taps, channels]` window: rows 0..taps-2 are the carried input history, oldest
    first, and row taps-1 is the new input. The output has the same layout with the history
    shifted by one and the conv result in the last row, so passing the window as both input and
    output steps it in place and leaves the last row for the next step's producer to overwrite.
    `w` is tap-major `[taps, channels]`. No activation.

    With `tokens` > 1 the window is `[taps - 1 + tokens, channels]`, the new inputs after the
    history, and the output holds the last `taps - 1` inputs as the next history followed by the
    `tokens` conv results.
    """

    channels: int
    taps: int = 4
    tile_channels: int = 256
    num_aie_columns: int = 8
    # None, not 1: a None field stays out of the artifact name, so decode's op keeps its name.
    tokens: int | None = None
    # ObjectFifo depth; None is the default double buffering. 1 fits a long window tile in L1.
    depth: int | None = None
    allocation_scheme: str | None = field(default=None, repr=False)
    context: AIEContext | None = field(default=None, repr=False)

    kernel_fn_name: ClassVar[str] = "conv1d_step_window"

    def __post_init__(self) -> None:
        self.tokens = self.tokens if self.tokens != 1 else None
        if self.channels % (self.num_aie_columns * self.tile_channels):
            raise ValueError(f"channels ({self.channels}) must be a multiple of num_aie_columns * "
                             f"tile_channels ({self.num_aie_columns * self.tile_channels})")
        if self.tile_channels % 32:
            raise ValueError(f"tile_channels ({self.tile_channels}) must be a multiple of 32")
        if self.taps < 2:
            raise ValueError("a one-tap conv carries no history")
        super().__init__(context=self.context)

    @property
    def _obj(self) -> str:
        t = f"_t{self.tokens}" if self.tokens else ""
        return f"conv1d_step_k{self.taps}_c{self.tile_channels}{t}.o"

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        n = (self.taps - 1 + (self.tokens or 1)) * self.channels
        return [AIERuntimeArgSpec("in", (n,)), AIERuntimeArgSpec("in", (self.taps * self.channels,)),
                AIERuntimeArgSpec("out", (n,))]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        args = (aie_utils.get_current_device(), self.channels, self.taps, self.tile_channels,
                self.num_aie_columns, self.kernel_fn_name, self._obj)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "design.py", "conv1d_step_design", args,
                            {"allocation_scheme": self.allocation_scheme,
                             "tokens": self.tokens or 1, "depth": self.depth or 2}),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [KernelObjectArtifact(
            self._obj,
            dependencies=[SourceArtifact(self.context.base_dir / "aie_kernels" / "aie2p" /
                                         "conv1d_step.cc")],
            extra_flags=[f"-DCS_C={self.tile_channels}", f"-DCS_K={self.taps}",
                         *([f"-DCS_T={self.tokens}"] if self.tokens else [])],
        )]

    def reference(self, window, w):
        from iron.operators.conv1d_step.reference import reference

        return reference(window, w, self.taps, self.channels, self.tokens or 1)
