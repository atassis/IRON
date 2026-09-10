# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np

from iron.common import (
    KernelArchiveArtifact,
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
from iron.common.device_utils import get_kernel_dir
import aie.utils as aie_utils


@dataclass
class GEMM(MLIROperator):
    """AIE-accelerated General Matrix Multiplication (GEMM) layer"""

    M: int
    K: int
    N: int
    tile_m: int = 64
    tile_k: int = 64
    tile_n: int = 64
    b_col_maj: bool = False
    c_col_maj: bool = False
    # B stored BLOCKED: its rows in groups of `b_block_rows`, consecutive groups `b_block_stride`
    # elements apart, other matrices interleaved in the gap. `None` (both) is the flat layout every
    # caller had before, down to the descriptor -- see iron.common.kv_layout.blocked_access_pattern.
    # Both are excluded from the operator NAME when None, so a flat GEMM's artifact does not move.
    b_block_rows: int | None = None
    b_block_stride: int | None = None
    # A read from, and C written into, a per-head SLICE of a wider token-major buffer: the same
    # [M, K] / [M, N] tile shape, only the row stride differs. `None` is the dense operand every
    # caller had before, down to the descriptor. Lets a graph drop the rearrange either side of an
    # op instead of materialising a head-major copy -- see iron.common.kv_layout.restride_rows.
    a_row_stride: int | None = None
    c_row_stride: int | None = None
    # A kernel run over the whole C tile once it is complete, before it leaves L1. "none" (default)
    # is the operator every caller had. This is the mechanism a fused block needs: the stage after
    # a GEMM stops being a separate design reading C back out of DDR. gemv has carried the same
    # hook since 354cb38; this is its GEMM twin.
    epilogue: str = field(default="none", repr=False)
    num_aie_columns: int = field(default=8)
    emulate_bf16_mmul_with_bfp16: bool = field(default=True, repr=False)
    prio_accuracy: bool = field(default=False, repr=False)
    round_conv_even: bool = field(default=True, repr=False)
    dtype_in: str = field(default="bf16", repr=False)
    dtype_out: str = field(default="bf16", repr=False)
    use_scalar: bool = field(default=False, repr=False)
    separate_c_tiles: bool = field(default=False, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "tile_m": "tm",
        "tile_k": "tk",
        "tile_n": "tn",
        "b_col_maj": "bc",
        "c_col_maj": "cc",
        "b_block_rows": "bbr",
        "b_block_stride": "bbs",
        "a_row_stride": "ars",
        "c_row_stride": "crs",
    }

    def __post_init__(self):
        num_aie_rows = 4
        min_M = self.tile_m * num_aie_rows
        min_K = self.tile_k
        min_N = self.tile_n * self.num_aie_columns
        if self.M % min_M != 0:
            raise ValueError(f"M ({self.M}) must be a multiple of {min_M}")
        if self.K % min_K != 0:
            raise ValueError(f"K ({self.K}) must be a multiple of {min_K}")
        if self.N % min_N != 0:
            raise ValueError(f"N ({self.N}) must be a multiple of {min_N}")

        if self.emulate_bf16_mmul_with_bfp16:
            min_tile_m, min_tile_k, min_tile_n = 8, 8, 8
        else:
            min_tile_m, min_tile_k, min_tile_n = 4, 8, 8
        if self.tile_m < min_tile_m:
            raise ValueError(f"tile_m ({self.tile_m}) must be >= {min_tile_m}")
        if self.tile_k < min_tile_k:
            raise ValueError(f"tile_k ({self.tile_k}) must be >= {min_tile_k}")
        if self.tile_n < min_tile_n:
            raise ValueError(f"tile_n ({self.tile_n}) must be >= {min_tile_n}")

        if self.epilogue not in ("none", "gelu", "silu"):
            raise ValueError(f"unknown epilogue {self.epilogue!r} (want 'none', 'gelu' or 'silu')")
        # Both tile epilogues walk the C tile 32 lanes at a time from a 16-lane-aligned base, so a
        # tile that is only 16-aligned makes the last iteration read and write past its end.
        if self.epilogue != "none" and (self.tile_m * self.tile_n) % 32:
            raise ValueError(
                f"{self.epilogue} epilogue needs tile_m*tile_n % 32 == 0, got "
                f"{self.tile_m}*{self.tile_n}")
        if (self.b_block_rows is None) != (self.b_block_stride is None):
            raise ValueError(
                "b_block_rows and b_block_stride go together: a block size with no stride cannot "
                "be addressed, and a stride with no block size has nothing to step over")
        if self.b_block_rows is not None:
            if self._b_rows % self.b_block_rows:
                raise ValueError(
                    f"b_block_rows ({self.b_block_rows}) must divide B's "
                    f"{'N' if self.b_col_maj else 'K'} ({self._b_rows})")
            if self.b_block_stride < self.b_block_rows * self._b_row_width:
                raise ValueError(
                    f"b_block_stride ({self.b_block_stride}) is under one block "
                    f"({self.b_block_rows} x {self._b_row_width}): blocks would overlap")

        for nm, stride, dense in (("a_row_stride", self.a_row_stride, self.K),
                                  ("c_row_stride", self.c_row_stride, self.N)):
            if stride is not None and stride < dense:
                raise ValueError(
                    f"{nm} ({stride}) is narrower than the operand it strides ({dense}); it is the "
                    f"row pitch of the buffer the slice lives in, not the slice's own width")

        MLIROperator.__init__(self, context=self.context)

    @property
    def a_elems(self):
        """A's allocated extent. Strided, the operand runs from its first row to the END of its
        last -- `(M-1)*pitch + K`, not `M*pitch`, so the slice of the final head still fits."""
        return self.M * self.K if self.a_row_stride is None \
            else (self.M - 1) * self.a_row_stride + self.K

    @property
    def c_elems(self):
        return self.M * self.N if self.c_row_stride is None \
            else (self.M - 1) * self.c_row_stride + self.N

    @property
    def _b_rows(self):
        """B's PHYSICAL leading extent -- N when it is read column-major, K otherwise. The blocked
        axis in both cases, because both store the same [position, head_dim] cache."""
        return self.N if self.b_col_maj else self.K

    @property
    def _b_row_width(self):
        return self.K if self.b_col_maj else self.N

    @property
    def b_elems(self):
        """B's allocated extent in elements. Blocked, that is not `K*N`: this matrix's rows reach
        to the far end of its last block, past the interleaved peers in between."""
        if self.b_block_rows is None:
            return self.K * self.N
        blocks = self._b_rows // self.b_block_rows
        return (blocks - 1) * self.b_block_stride + self.b_block_rows * self._b_row_width

    @property
    def name(self):
        # epilogue is repr=False so an unfused GEMM keeps the name it has always had; a fused one
        # must NOT share that artifact, because the design and the linked object both differ.
        base = super().name
        return base if self.epilogue == "none" else f"{base}_epi{self.epilogue}"

    @property
    def _kernel_flags_suffix(self):
        """Suffix encoding compile-time flags that affect the kernel binary."""
        return f"_{int(self.prio_accuracy)}_{int(self.emulate_bf16_mmul_with_bfp16)}_{int(self.round_conv_even)}"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_matmul",
                (),
                {
                    "dev": aie_utils.get_current_device(),
                    "M": self.M,
                    "K": self.K,
                    "N": self.N,
                    "m": self.tile_m,
                    "k": self.tile_k,
                    "n": self.tile_n,
                    "n_aie_cols": self.num_aie_columns,
                    "dtype_in_str": self.dtype_in,
                    "dtype_out_str": self.dtype_out,
                    "b_col_maj": int(self.b_col_maj),
                    "b_block_rows": self.b_block_rows,
                    "b_block_stride": self.b_block_stride,
                    "a_row_stride": self.a_row_stride,
                    "c_row_stride": self.c_row_stride,
                    "c_col_maj": int(self.c_col_maj),
                    "use_scalar": self.use_scalar,
                    "emulate_bf16_mmul_with_bfp16": self.emulate_bf16_mmul_with_bfp16,
                    "prio_accuracy": self.prio_accuracy,
                    "separate_c_tiles": int(self.separate_c_tiles),
                    "epilogue": self.epilogue,
                    "trace_size": 0,
                    "generate_taps": False,
                    "kernel_object": self._kernel_link_file,
                },
            ),
        )

    @property
    def _mm_object(self):
        return (f"gemm_{self.tile_m}x{self.tile_k}x{self.tile_n}"
                f"_{int(self.b_col_maj)}_{int(self.c_col_maj)}{self._kernel_flags_suffix}.o")

    @property
    def _kernel_link_file(self):
        """With an epilogue the core links two kernels, so the object becomes an archive."""
        if self.epilogue == "none":
            return self._mm_object
        return self._mm_object[:-2] + f"_{self.epilogue}_kernels.a"

    def get_kernel_artifacts(self):
        base_dir = self.context.base_dir
        kernel_flags = [
            f"-DDIM_M={self.tile_m}",
            f"-DDIM_K={self.tile_k}",
            f"-DDIM_N={self.tile_n}",
        ]
        if self.prio_accuracy:
            kernel_flags.append("-Dbf16_f32_ONLY")
        else:
            kernel_flags.append("-Dbf16_bf16_ONLY")
        if self.round_conv_even:
            kernel_flags.append("-DROUND_CONV_EVEN")
        if self.emulate_bf16_mmul_with_bfp16:
            kernel_flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
        if self.b_col_maj:
            kernel_flags.append("-DB_COL_MAJ")
        if self.c_col_maj:
            kernel_flags.append("-DC_COL_MAJ")

        kernel_dir = get_kernel_dir()
        mm_obj = KernelObjectArtifact(
            self._mm_object,
            extra_flags=kernel_flags,
            dependencies=[SourceArtifact(base_dir / "aie_kernels" / kernel_dir / "mm.cc")],
        )
        if self.epilogue != "none":
            # Both epilogue kernels live under aie2p/, so a fused epilogue is NPU2-only.
            if kernel_dir != "aie2p":
                raise NotImplementedError(
                    f"GEMM {self.epilogue} epilogue is only available on NPU2 (aie2p); "
                    f"current kernel dir is {kernel_dir!r}")
            mm_obj = KernelArchiveArtifact(
                self._kernel_link_file,
                dependencies=[mm_obj, KernelObjectArtifact(
                    f"{self.epilogue}.o",
                    dependencies=[SourceArtifact(
                        base_dir / "aie_kernels" / "aie2p" / f"{self.epilogue}.cc")],
                )],
            )
        return [
            mm_obj,
            KernelObjectArtifact(
                "convert_copy.o",
                [
                    SourceArtifact(
                        base_dir / "aie_kernels" / "generic" / "convert_copy.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec(  # input A
                "in", (self.a_elems,) if self.a_row_stride is not None else (self.M, self.K)),
            # input B (weights). Blocked, B's extent is not K*N and it is not 2-D contiguous
            # either, so it is declared as the flat run the descriptor actually addresses; flat, the
            # shape is left exactly as it was so no existing sequence sees a changed spec.
            AIERuntimeArgSpec(
                "in",
                (self.b_elems,)
                if self.b_block_rows is not None
                else ((self.N, self.K) if self.b_col_maj else (self.K, self.N)),
            ),
            AIERuntimeArgSpec(  # output C
                "out",
                (self.c_elems,) if self.c_row_stride is not None
                else ((self.M, self.N) if not self.c_col_maj else (self.N, self.M))),
        ]

    def reference(self, A, B):
        """CPU reference: ``C = A @ B`` honoring ``b_col_maj`` / ``c_col_maj``."""
        from iron.operators.gemm.reference import reference

        return reference(A, B, self.b_col_maj, self.c_col_maj)

    def pad_A(self, A_np):
        """Pad A matrix to match operator dimensions (M, K)"""
        M, K = A_np.shape
        if M > self.M:
            raise ValueError(f"A rows ({M}) exceeds operator M ({self.M})")
        if M == self.M and K == self.K:
            return A_np

        M_padded = ((M + self.M - 1) // self.M) * self.M
        A_padded = np.zeros((M_padded, self.K), dtype=A_np.dtype)
        A_padded[:M, :K] = A_np
        return A_padded

    def pad_B(self, B_np):
        """Pad B matrix to match operator dimensions based on layout"""
        if self.b_col_maj:
            N, K = B_np.shape
            if N > self.N or K > self.K:
                raise ValueError(
                    f"B (col-major) shape ({N}, {K}) exceeds operator N ({self.N}), K ({self.K})"
                )
            if N == self.N and K == self.K:
                return B_np
            B_padded = np.zeros((self.N, self.K), dtype=B_np.dtype)
            B_padded[:N, :K] = B_np
        else:
            K, N = B_np.shape
            if N > self.N or K > self.K:
                raise ValueError(
                    f"B (row-major) shape ({K}, {N}) exceeds operator K ({self.K}), N ({self.N})"
                )
            if K == self.K and N == self.N:
                return B_np
            B_padded = np.zeros((self.K, self.N), dtype=B_np.dtype)
            B_padded[:K, :N] = B_np
        return B_padded

    def partition_B(self, B, partition_N):
        B_parts = [None] * partition_N
        if B is None:
            return B_parts
        for i in range(partition_N):
            col_start = i * self.N
            col_end = (i + 1) * self.N

            if self.b_col_maj:
                B_parts[i] = self.pad_B(B[col_start:col_end, :])
            else:
                B_parts[i] = self.pad_B(B[:, col_start:col_end])
        return B_parts
