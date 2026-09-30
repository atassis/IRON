# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar

from iron.common import ChanneledUnaryOperator, DesignGenerator, PythonGeneratedMLIRArtifact


class _PolyAct(ChanneledUnaryOperator):
    """The f32 polynomial chain spills: aiecc measures 3200..3456 B against the 1 KB default and
    refuses anything smaller, so this default is checked, not hoped."""

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        art = super().get_mlir_artifact()
        art.generator.kwargs["stack_size"] = self.stack_size
        return art


@dataclass
class SiLUPoly(_PolyAct):
    """SiLU in f32 polynomial math -- SiLU's drop-in where the tanh LUT's error is visible."""

    num_channels: int = field(default=1, init=False, repr=False)
    stack_size: int = field(default=4096, repr=False)

    kernel_name: ClassVar[str] = "act_poly"
    kernel_fn_name: ClassVar[str] = "silu_poly_bf16"
    callback_fn: ClassVar[str] = "my_silu_poly"


@dataclass
class SigmoidPoly(_PolyAct):
    """sigmoid in f32 polynomial math -- Sigmoid's drop-in where the tanh LUT's error is visible."""

    num_channels: int = field(default=1, init=False, repr=False)
    stack_size: int = field(default=4096, repr=False)

    kernel_name: ClassVar[str] = "act_poly"
    kernel_fn_name: ClassVar[str] = "sigmoid_poly_bf16"
    callback_fn: ClassVar[str] = "my_sigmoid_poly"


@dataclass
class SiLUFast(_PolyAct):
    """SiLUPoly's function at bf16 output accuracy in native bf16 MACs over two-limb operands."""

    num_channels: int = field(default=1, init=False, repr=False)
    stack_size: int = field(default=4096, repr=False)

    kernel_name: ClassVar[str] = "act_poly"
    kernel_fn_name: ClassVar[str] = "silu_fast_bf16"
    callback_fn: ClassVar[str] = "my_silu_fast"


@dataclass
class SigmoidFast(_PolyAct):
    """SigmoidPoly's function at bf16 output accuracy in native bf16 MACs over two-limb operands."""

    num_channels: int = field(default=1, init=False, repr=False)
    stack_size: int = field(default=4096, repr=False)

    kernel_name: ClassVar[str] = "act_poly"
    kernel_fn_name: ClassVar[str] = "sigmoid_fast_bf16"
    callback_fn: ClassVar[str] = "my_sigmoid_fast"
