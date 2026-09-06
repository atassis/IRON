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
class GEMV(MLIROperator):
    """AIE-accelerated General Matrix-Vector/Vector-Matrix Multiplication layer"""

    M: int
    K: int
    num_aie_columns: int = 1
    tile_size_input: int = 2
    tile_size_output: int | None = None
    num_batches: int = 1
    kernel_vector_size: int = field(default=64, repr=False)
    # Optional fused activation applied to each output tile in the producing core.
    # "none" (default) leaves the output unchanged; "gelu" applies GELU(tanh approx).
    # repr=False keeps operator/artifact names stable for the default path.
    epilogue: str = field(default="none", repr=False)
    # Weight-stream format axis for A (the MxK matrix): "bf16" (default, unchanged), or
    # "int4"/"int8" group-quantized with a per-row f32 scale per `group_size` columns,
    # dequantized on-core right before the same bf16 MAC (see design.py / mv_quant.cc / quant.py).
    # This is a byte-stream lever on the WEIGHT only -- B and C stay bf16 regardless.
    # repr=False + the name/kernel-file overrides below keep the default path's artifact names
    # stable, matching the epilogue field's convention.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
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
        if self.epilogue not in ("none", "gelu"):
            raise ValueError(
                f"unknown epilogue {self.epilogue!r} (expected 'none' or 'gelu')"
            )
        if self.epilogue == "gelu" and self.tile_size_output % 16 != 0:
            raise ValueError(
                f"gelu epilogue needs tile_size_output % 16 == 0 (got {self.tile_size_output})"
            )
        if self.weight_dtype not in ("bf16", "int4", "int8"):
            raise ValueError(
                f"unknown weight_dtype {self.weight_dtype!r} (expected 'bf16', 'int4' or 'int8')"
            )
        if self.weight_dtype != "bf16":
            if self.epilogue != "none":
                # Untested combination, not a hardware conflict -- narrow scope until a caller
                # needs both a quantized weight AND a fused epilogue on the same GEMV.
                raise NotImplementedError(
                    "GEMV weight_dtype != 'bf16' with a fused epilogue is not implemented"
                )
            if self.group_size <= 0:
                raise ValueError("weight_dtype != 'bf16' needs an explicit group_size > 0")
            if self.K % self.group_size != 0:
                raise ValueError(
                    f"K={self.K} must be a whole number of groups (group_size={self.group_size})"
                )
            if self.group_size % self.kernel_vector_size != 0:
                # mv_quant.cc's vectorized dequant chunk (kernel_vector_size wide) must never
                # straddle a quant-group boundary, or a chunk would need two scales.
                raise ValueError(
                    f"group_size={self.group_size} must be a multiple of kernel_vector_size="
                    f"{self.kernel_vector_size}"
                )
            if self.num_batches != 1:
                raise NotImplementedError(
                    "GEMV weight_dtype != 'bf16' does not support num_batches>1 yet"
                )

        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # epilogue/weight_dtype are repr=False so the default path keeps a stable name, but a
        # non-default variant must not share an artifact name with the plain GEMV of the same
        # shape: both would emit the same .mlir/.xclbin, and in a shared build dir a cached
        # default build can then silently satisfy the non-default op.
        base = super().name
        if self.epilogue != "none":
            base = f"{base}_epi{self.epilogue}"
        if self.weight_dtype != "bf16":
            base = f"{base}_wdt{self.weight_dtype}g{self.group_size}"
        return base

    @property
    def _kernel_link_file(self):
        # With the gelu epilogue the core also links the gelu kernel, so the object becomes an
        # archive of (matvec, gelu); the plain matvec stays a single object.
        if self.epilogue == "gelu":
            return f"gemv_{self.K}k_{self.kernel_vector_size}vs_gelu_kernels.a"
        if self.weight_dtype != "bf16":
            return f"gemv_{self.K}k_{self.kernel_vector_size}vs_{self.weight_dtype}g{self.group_size}.o"
        return f"gemv_{self.K}k_{self.kernel_vector_size}vs.o"

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
                    "kernel_object": self._kernel_link_file,
                    "epilogue": self.epilogue,
                    "weight_dtype": self.weight_dtype,
                    "group_size": self.group_size,
                },
            ),
        )

    def get_kernel_artifacts(self):
        if self.weight_dtype != "bf16":
            return [
                KernelObjectArtifact(
                    self._kernel_link_file,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / "generic" / "mv_quant.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_K={self.K}",
                        f"-DVEC_SIZE={self.kernel_vector_size}",
                        f"-DGROUP_SIZE={self.group_size}",
                    ],
                )
            ]
        matvec_obj = KernelObjectArtifact(
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
        )
        if self.epilogue == "gelu":
            # The gelu kernel lives in aie2p/gelu.cc, so the fused epilogue is NPU2-only.
            if get_kernel_dir() != "aie2p":
                raise NotImplementedError(
                    "gemv gelu epilogue is only available on NPU2 (aie2p); "
                    f"current kernel dir is {get_kernel_dir()!r}"
                )
            gelu_obj = KernelObjectArtifact(
                "gelu.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "aie2p" / "gelu.cc"
                    )
                ],
            )
            return [
                KernelArchiveArtifact(
                    self._kernel_link_file, dependencies=[matvec_obj, gelu_obj]
                )
            ]
        return [matvec_obj]

    def get_arg_spec(self):
        import numpy as np

        batch_dim = (self.num_batches,) if self.num_batches > 1 else ()
        if self.weight_dtype == "bf16":
            matrix_spec = AIERuntimeArgSpec("in", batch_dim + (self.M, self.K))
        else:
            from iron.operators.gemv.quant import row_stride_bytes

            stride = row_stride_bytes(self.K, self.group_size, self.weight_dtype)
            # Flat byte buffer (int8-typed purely so the emitted shim BDs type as `i8`, matching
            # decode_ddr_bytes.py's parser): M rows of `stride` packed bytes each, row layout in
            # quant.py. num_batches is asserted ==1 for a non-bf16 weight_dtype in __post_init__.
            matrix_spec = AIERuntimeArgSpec("in", batch_dim + (self.M * stride,), dtype=np.int8)
        return [
            matrix_spec,  # matrix (A)
            AIERuntimeArgSpec("in", batch_dim + (self.K,)),  # vector (B, always bf16)
            AIERuntimeArgSpec("out", batch_dim + (self.M,)),  # output (C, always bf16)
        ]

    def reference(self, A, B):
        """CPU reference: (optionally batched) matrix-vector product."""
        from iron.operators.gemv.reference import reference

        return reference(A, B)
