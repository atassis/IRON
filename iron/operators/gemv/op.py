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
class GEMV(MLIROperator):
    """AIE-accelerated General Matrix-Vector/Vector-Matrix Multiplication layer"""

    M: int
    K: int
    num_aie_columns: int = 1
    tile_size_input: int = 2
    tile_size_output: int | None = None
    num_batches: int = 1
    kernel_vector_size: int = field(default=64, repr=False)
    # coalesce_batch_dma: compat no-op. #127 made batched-DMA coalescing the DEFAULT and auto-derives it
    # in design.py (coalesces only when stride/granularity/splittable constraints hold, else per-batch
    # fallback), so this toggle is no longer plumbed into my_matvec. Kept as an accepted field so the
    # (parked) batched decode callers that still pass coalesce_batch_dma= don't break. repr=False.
    coalesce_batch_dma: bool = field(default=False, repr=False)
    # dtype_a="int8" -> matrix A is int8 (quantized resident K/V cache, halves its LPDDR re-read); vector B
    # + output stay bf16. Default "bf16" = unchanged. repr=False (artifact names stable; opt-in via kwarg).
    dtype_a: str = field(default="bf16", repr=False)
    # epilogue="gelu" -> fused GELU(tanh) over the output C-tile (folds a separate GELU op into the GEMV).
    # Default "none". repr=False keeps artifact names stable for callers.
    epilogue: str = field(default="none", repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
        "num_batches": "batch",
    }

    def __post_init__(self):
        if self.tile_size_output is None:
            self.tile_size_output = self.tile_size_input

        if not (
            self.tile_size_output % self.tile_size_input == 0
            and self.tile_size_output >= self.tile_size_input
        ):
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if not (
            self.K >= self.kernel_vector_size and self.K % self.kernel_vector_size == 0
        ):
            raise ValueError("K must be multiple of kernel_vector_size")

        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)

        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_matvec",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.M,
                    self.K,
                    self.tile_size_input,
                    self.tile_size_output,
                    self.num_batches,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": f"gemv_{self.K}k_{self.kernel_vector_size}vs.o",
                    "dtype_a": self.dtype_a,
                    "epilogue": self.epilogue,
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                f"gemv_{self.K}k_{self.kernel_vector_size}vs.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.K}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
        ]

    def get_arg_spec(self):
        batch_dim = (self.num_batches,) if self.num_batches > 1 else ()
        return [
            AIERuntimeArgSpec("in", batch_dim + (self.M, self.K)),  # matrix
            AIERuntimeArgSpec("in", batch_dim + (self.K,)),  # vector
            AIERuntimeArgSpec("out", batch_dim + (self.M,)),  # output
        ]

    def reference(self, A, B):
        """CPU reference: (optionally batched) matrix-vector product."""
        from iron.operators.gemv.reference import reference

        return reference(A, B)
