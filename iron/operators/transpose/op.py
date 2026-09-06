# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import aie.utils as aie_utils
from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)


@dataclass
class Transpose(MLIROperator):
    """AIE-accelerated transpose operator.

    ``num_batches`` > 1 performs that many independent (M,N)->(N,M) transposes on
    contiguous matrices laid back-to-back in memory (results concatenated), mirroring
    GEMV's batching — the per-batch tile work rides the same ObjectFifos, so B batched
    transposes cost ONE dispatch instead of B unrolled ones.
    """

    M: int
    N: int
    num_aie_columns: int
    num_channels: int
    m: int
    n: int
    s: int
    num_batches: int = 1
    # How many CONSECUTIVE batches transpose the SAME input matrix. 1 = old behaviour. >1 lets a
    # GQA kv head feed gqa_group query heads with no Repeat materialising a duplicate in DDR.
    batch_group: int = 1
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_batches": "batch",
        "batch_group": "bgrp",
    }

    def __post_init__(self):
        if self.batch_group < 1 or self.num_batches % self.batch_group != 0:
            raise ValueError(
                f"num_batches ({self.num_batches}) must be a positive multiple of batch_group "
                f"({self.batch_group})"
            )
        if self.M % self.m != 0:
            raise ValueError(f"Matrix rows ({self.M}) must be a multiple of {self.m}")
        if self.N % self.n != 0:
            raise ValueError(
                f"Matrix columns ({self.N}) must be a multiple of {self.n}"
            )
        if self.m % self.s != 0:
            raise ValueError(f"AIE tile rows ({self.m}) must be a multiple of {self.s}")
        if self.n % self.s != 0:
            raise ValueError(
                f"AIE tile columns ({self.n}) must be a multiple of {self.s}"
            )
        if (
            self.M
            * self.N
            % (self.m * self.n * self.num_aie_columns * self.num_channels)
            != 0
        ):
            raise ValueError(
                "Transfer size must be divisible by m*n*num_columns*num_channels"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "shuffle_transpose",
                (
                    aie_utils.get_current_device(),
                    self.M,
                    self.N,
                    self.num_aie_columns,
                    self.num_channels,
                    self.m,
                    self.n,
                    self.s,
                    self.num_batches,
                    self.batch_group,
                ),
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                f"transpose_{self.m}x{self.n}.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir
                        / "aie_kernels"
                        / "generic"
                        / "transpose.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_m={self.m}",
                    f"-DDIM_n={self.n}",
                ],
            ),
        ]

    def get_arg_spec(self):
        batch_dim = (self.num_batches,) if self.num_batches > 1 else ()
        # The INPUT is indexed by matrix, the OUTPUT by batch: batch_group batches share one
        # source and each still produces its own transposed result.
        n_matrices = self.num_batches // self.batch_group
        in_batch_dim = (n_matrices,) if n_matrices > 1 else ()
        return [
            AIERuntimeArgSpec("in", in_batch_dim + (self.M * self.N,)),
            AIERuntimeArgSpec("out", batch_dim + (self.N * self.M,)),
        ]

    def reference(self, x):
        """CPU reference: 2D transpose of an (M, N) matrix stored row-major."""
        from iron.operators.transpose.reference import reference

        return reference(x.reshape(self.M, self.N))
