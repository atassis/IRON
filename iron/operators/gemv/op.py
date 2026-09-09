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
    # How many CONSECUTIVE batches share one matrix. 1 = every batch has its own (the old
    # behaviour). >1 expresses GQA directly: gqa_group query heads attend to one kv head, so the
    # matrix operand holds num_batches//batch_group heads and the access pattern repeats each one
    # instead of a Repeat op materialising a duplicate in DDR.
    batch_group: int = 1
    # Rows ALLOCATED per matrix in A, when that differs from the rows COMPUTED (`M`). None means
    # they are equal -- the old behaviour, byte for byte. Set it to read a NARROW WINDOW out of a
    # buffer sized for a wider one: decode attention computes n_past scores against a KV cache
    # allocated at max_seq, so M=n_past while the per-matrix stride must stay max_seq*K. Only the
    # buffer size and the batch stride move; the run, the C tile and the core loop all follow M.
    # repr=False + the `name` override below, matching the epilogue/weight_dtype convention.
    alloc_M: int | None = field(default=None, repr=False)
    # Rows ALLOCATED per batch in C (the output), when that differs from the rows COMPUTED (`M`).
    # None means they are equal -- the old behaviour, byte for byte. The write-side mirror of
    # alloc_M: decode attention writes n_past scores into a scores buffer allocated at max_seq, so
    # M=n_past while the per-batch stride C is written at must stay max_seq. Only the buffer size
    # and the per-batch stride move; the run and the core loop still follow M.
    alloc_M_out: int | None = field(default=None, repr=False)
    # How many batches share one TaskGroup, i.e. one device-side drain wait, on the per-batch
    # fallback path. 1 is the historical behaviour. Measured on the scores shape at a wide
    # allocation: 16 barriers -> 4 is -20.7% at an IDENTICAL descriptor count, so the fallback's
    # cost is the waits and not the BDs. Bounded above by the shim's 16-BD budget -- chunk 8 and 16
    # do not build. A FIELD rather than an env read, because it changes the emitted design and
    # anything that changes the design must reach the artifact name (see `name` below).
    barrier_chunk: int = field(default=1, repr=False)
    kernel_vector_size: int = field(default=64, repr=False)
    # Optional fused activation applied to each output tile in the producing core.
    # "none" (default) leaves the output unchanged; "gelu" applies GELU(tanh approx); "silu" applies
    # SiLU(tanh approx), the same math as the standalone SiLU operator's kernel. Folding the
    # activation in deletes a whole design from the graph, and a design costs an aiex.configure per
    # layer whether or not it moves any bytes.
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
        "batch_group": "bgrp",
    }

    def __post_init__(self):
        if self.alloc_M is not None and self.alloc_M < self.M:
            raise ValueError(
                f"alloc_M ({self.alloc_M}) must be >= M ({self.M}): it is the ALLOCATED row "
                f"count per matrix, not a second window"
            )
        if self.alloc_M_out is not None and self.alloc_M_out < self.M:
            raise ValueError(
                f"alloc_M_out ({self.alloc_M_out}) must be >= M ({self.M}): it is the ALLOCATED "
                f"row count per output batch, not a second window"
            )
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
        if self.batch_group < 1 or self.num_batches % self.batch_group != 0:
            raise ValueError(
                f"num_batches ({self.num_batches}) must be a positive multiple of batch_group "
                f"({self.batch_group})"
            )
        if self.epilogue not in ("none", "gelu", "silu"):
            raise ValueError(
                f"unknown epilogue {self.epilogue!r} (expected 'none', 'gelu' or 'silu')"
            )
        # Both tile epilogues walk the C tile 32 lanes at a time from a 16-lane-aligned base.
        if self.epilogue != "none" and self.tile_size_output % 32 != 0:
            raise ValueError(
                f"{self.epilogue} epilogue needs tile_size_output % 32 == 0 "
                f"(got {self.tile_size_output})"
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
        # A windowed read is a DIFFERENT design from the plain GEMV of the same M: same compute
        # extent, different buffer size and per-matrix stride. Without this they collide in the
        # build dir and a cached plain build silently satisfies the windowed op. alloc_M == M is
        # the same design as alloc_M=None, so it keeps the stable name.
        if self.alloc_M is not None and self.alloc_M != self.M:
            base = f"{base}_am{self.alloc_M}"
        # Same reasoning, write side: a wide-strided C is a different design from the plain one at
        # the same M, and alloc_M_out == M is the same design as None.
        if self.alloc_M_out is not None and self.alloc_M_out != self.M:
            base = f"{base}_amo{self.alloc_M_out}"
        if self.barrier_chunk != 1:
            base = f"{base}_bc{self.barrier_chunk}"
        return base

    def design_key(self):
        """Every argument that reaches design.py, so two GEMVs sharing this key emit the same MLIR.

        The decode graph builds gate and up as separate GEMV instances of the SAME shape, adjacent
        in the runlist, and without sharing they compile to two designs and cost two
        `aiex.configure` per layer where one would do. Listed explicitly rather than derived from
        `name`, because `kernel_vector_size` is repr=False and absent from it.
        """
        return "|".join(str(x) for x in (
            "GEMV",
            self.num_aie_columns, self.M, self.K,
            self.tile_size_input, self.tile_size_output,
            self.num_batches, self.batch_group,
            self.epilogue, self.weight_dtype, self.group_size,
            self.alloc_M, self.alloc_M_out, self.kernel_vector_size, self.barrier_chunk,
            self._kernel_link_file,
        ))

    @property
    def _kernel_link_file(self):
        # With the gelu epilogue the core also links the gelu kernel, so the object becomes an
        # archive of (matvec, gelu); the plain matvec stays a single object.
        if self.epilogue != "none":
            return f"gemv_{self.K}k_{self.kernel_vector_size}vs_{self.epilogue}_kernels.a"
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
                    self.batch_group,
                ),
                {
                    "verbose": mlir_verbose,
                    "kernel_object": self._kernel_link_file,
                    "epilogue": self.epilogue,
                    "weight_dtype": self.weight_dtype,
                    "group_size": self.group_size,
                    "alloc_M": self.alloc_M,
                    "alloc_M_out": self.alloc_M_out,
                    "barrier_chunk": self.barrier_chunk,
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
        if self.epilogue != "none":
            # Both epilogue kernels live under aie2p/, so a fused epilogue is NPU2-only.
            if get_kernel_dir() != "aie2p":
                raise NotImplementedError(
                    f"gemv {self.epilogue} epilogue is only available on NPU2 (aie2p); "
                    f"current kernel dir is {get_kernel_dir()!r}"
                )
            epi_obj = KernelObjectArtifact(
                f"{self.epilogue}.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir
                        / "aie_kernels"
                        / "aie2p"
                        / f"{self.epilogue}.cc"
                    )
                ],
            )
            return [
                KernelArchiveArtifact(
                    self._kernel_link_file, dependencies=[matvec_obj, epi_obj]
                )
            ]
        return [matvec_obj]

    def get_arg_spec(self):
        import numpy as np

        batch_dim = (self.num_batches,) if self.num_batches > 1 else ()
        # A is indexed by MATRIX, not by batch: with batch_group>1 several batches read the same
        # one, which is the whole point -- the operand shrinks by exactly that factor.
        n_matrices = self.num_batches // self.batch_group
        a_batch_dim = (n_matrices,) if n_matrices > 1 else ()
        # Sized by the ALLOCATION: a windowed read (alloc_M > M) still addresses a buffer whose
        # per-matrix stride is alloc_M*K, so the host operand must be that big or every matrix
        # after the first reads past its end.
        a_rows = self.M if self.alloc_M is None else self.alloc_M
        if self.weight_dtype == "bf16":
            matrix_spec = AIERuntimeArgSpec("in", a_batch_dim + (a_rows, self.K))
        else:
            from iron.operators.gemv.quant import row_stride_bytes

            stride = row_stride_bytes(self.K, self.group_size, self.weight_dtype)
            # Flat byte buffer (int8-typed purely so the emitted shim BDs type as `i8`, matching
            # decode_ddr_bytes.py's parser): M rows of `stride` packed bytes each, row layout in
            # quant.py. num_batches is asserted ==1 for a non-bf16 weight_dtype in __post_init__.
            matrix_spec = AIERuntimeArgSpec("in", a_batch_dim + (self.M * stride,), dtype=np.int8)
        # Sized by the ALLOCATION, mirroring A: a wide-strided write still lands each batch at
        # `batch * alloc_M_out`, so the host operand must be that big or batches after the first
        # overlap the next one's region.
        c_rows = self.M if self.alloc_M_out is None else self.alloc_M_out
        return [
            matrix_spec,  # matrix (A)
            AIERuntimeArgSpec("in", batch_dim + (self.K,)),  # vector (B, always bf16)
            AIERuntimeArgSpec("out", batch_dim + (c_rows,)),  # output (C, always bf16)
        ]

    def reference(self, A, B):
        """CPU reference: (optionally batched) matrix-vector product."""
        from iron.operators.gemv.reference import reference

        return reference(A, B)
