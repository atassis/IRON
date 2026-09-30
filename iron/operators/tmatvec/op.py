# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np
from ml_dtypes import bfloat16

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
)
import aie.utils as aie_utils


@dataclass
class TMatVec(MLIROperator):
    """Transposed-A matvec: C[b][j] = sum_p W[b][p] * A[b // batch_group][p][j].

    gemv reduces ALONG a row and gives one output per row. This reduces DOWN the rows and gives one
    output per COLUMN, which is what attention's context step needs against a V cache stored
    [S][head_dim]. Doing it as a gemv is what forces a physical transpose of the whole cache first.
    """

    M: int  # output width == the matrix's row width (head_dim)
    K: int  # reduction extent == the matrix's row COUNT (sequence length)
    num_aie_columns: int = 1
    num_batches: int = 1
    batch_group: int = 1
    rows_per_chunk: int = 64
    # Rows ALLOCATED per matrix when that differs from the rows REDUCED -- gemv's alloc_M one axis
    # over. The window is a row PREFIX here, so only the per-matrix stride moves.
    alloc_K: int | None = field(default=None, repr=False)
    # Store A's `alloc_K` rows in BLOCKS of `block_size` rows (`cols` heads interleaved every
    # block) instead of one `alloc_K`-row-per-matrix slab. None (default) is one block --
    # byte-identical to the pre-blocking layout. See design.py's block_size docstring.
    block_size: int | None = field(default=None, repr=False)
    # Compute the output in `m_chunk`-wide column slices instead of one M-wide pass. None (default)
    # is one chunk -- byte-identical to the unchunked design. It is the only knob that moves the
    # S-independent `C+acc` floor, and it also lets W stream per K-chunk, which is what makes the
    # L1 footprint independent of K. See design.py's m_chunk comment.
    m_chunk: int | None = field(default=None, repr=False)
    # Per-dispatch int32 ScratchpadParameter naming the REDUCTION EXTENT in rows; None (the
    # default) reduces over the build-time K. See gemv/design.py's RUNTIME ROW EXTENT block.
    vector_size_parameter: str | None = field(default=None, repr=False)
    # A's stream format, as GEMV takes it (iron/operators/gemv/op.py) -- "bf16" (default,
    # unchanged) or group-quantized "int4"/"int8". The row this quantizes is one CACHED POSITION
    # (DIM_N = M wide), not an output feature, so group_size divides M, not K -- see design.py's
    # module docstring. No affine family yet: nothing here has asked for it.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
    kernel_vector_size: int = field(default=64, repr=False)
    layout: str = field(default="header_first", repr=False)
    row_group: int | None = field(default=None, repr=False)
    scale_dtype: str = field(default="f32", repr=False)
    kwargs: dict = field(default_factory=dict, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "num_batches": "batch",
        "batch_group": "bgrp",
        "rows_per_chunk": "rpc",
    }

    def __post_init__(self):
        if self.num_batches % self.batch_group != 0:
            raise ValueError(
                f"num_batches ({self.num_batches}) must be a multiple of batch_group "
                f"({self.batch_group})"
            )
        n_matrices = self.num_batches // self.batch_group
        if n_matrices != self.num_aie_columns:
            raise ValueError(
                f"this design places one matrix per column: num_batches//batch_group "
                f"({n_matrices}) must equal num_aie_columns ({self.num_aie_columns})"
            )
        if self.K % self.rows_per_chunk != 0:
            raise ValueError(
                f"rows_per_chunk ({self.rows_per_chunk}) must divide K ({self.K})"
            )
        if self.alloc_K is not None and self.alloc_K < self.K:
            raise ValueError(
                f"alloc_K ({self.alloc_K}) must be >= K ({self.K}): it is the ALLOCATED row "
                f"count per matrix, not a second window"
            )
        _ak = self.K if self.alloc_K is None else self.alloc_K
        if self.block_size is not None and (self.block_size <= 0 or _ak % self.block_size != 0):
            raise ValueError(
                f"block_size ({self.block_size}) must be a positive divisor of alloc_K ({_ak})"
            )
        if self.m_chunk is not None and (self.m_chunk <= 0 or self.M % self.m_chunk != 0):
            raise ValueError(
                f"m_chunk ({self.m_chunk}) must be a positive divisor of M ({self.M})"
            )
        if self.weight_dtype not in ("bf16", "int4", "int8"):
            raise ValueError(
                f"unknown weight_dtype {self.weight_dtype!r} (expected 'bf16', 'int4' or 'int8')"
            )
        if self.layout not in ("header_first", "row_group_planar"):
            raise ValueError(
                f"unknown layout {self.layout!r} (expected 'header_first' or 'row_group_planar')"
            )
        if self.layout == "row_group_planar" and self.weight_dtype == "bf16":
            raise ValueError("layout='row_group_planar' needs weight_dtype != 'bf16'")
        if self.layout == "header_first" and self.row_group is not None:
            raise ValueError("row_group is only meaningful under layout='row_group_planar'")
        if self.row_group is not None and self.row_group < 1:
            raise ValueError(f"row_group ({self.row_group}) must be >= 1")
        if self.scale_dtype not in ("f32", "bf16"):
            raise ValueError(
                f"unknown scale_dtype {self.scale_dtype!r} (expected 'f32' or 'bf16')"
            )
        if self.scale_dtype != "f32" and self.weight_dtype == "bf16":
            raise ValueError("scale_dtype is only meaningful for a quantized weight_dtype")

        _row_bytes = None
        if self.weight_dtype != "bf16":
            # Composing with either is a second retiling of A's own byte width this design has
            # not derived; refuse rather than silently mis-tile (K007/K008).
            if self.m_chunk is not None or self.block_size is not None:
                raise NotImplementedError(
                    "TMatVec weight_dtype != 'bf16' does not compose with m_chunk or "
                    "block_size yet"
                )
            if self.vector_size_parameter is not None:
                raise NotImplementedError(
                    "TMatVec weight_dtype != 'bf16' does not support vector_size_parameter yet"
                )
            if self.group_size <= 0:
                raise ValueError("weight_dtype != 'bf16' needs an explicit group_size > 0")
            if self.M % self.group_size != 0:
                raise ValueError(
                    f"M={self.M} (one cached position's own channel count) must be a whole "
                    f"number of groups (group_size={self.group_size})"
                )
            from iron.common.quant import max_legal_vec_size, row_stride_bytes

            if self.layout == "row_group_planar":
                from iron.common.quant import derive_row_group, widest_chunk

                if self.row_group is None:
                    provisional_vec = widest_chunk(self.group_size, self.weight_dtype,
                                                   cap=self.kernel_vector_size)
                    object.__setattr__(self, "row_group", derive_row_group(
                        [self.M], self.group_size, self.weight_dtype,
                        vec_size=provisional_vec, max_rows=self.rows_per_chunk,
                        scale_dtype=self.scale_dtype))
                if self.rows_per_chunk % self.row_group != 0:
                    raise ValueError(
                        f"row_group_planar needs rows_per_chunk ({self.rows_per_chunk}) to be "
                        f"a multiple of row_group ({self.row_group})"
                    )
                legal = max_legal_vec_size([self.M], self.group_size, self.weight_dtype,
                                           scale_dtype=self.scale_dtype, layout=self.layout,
                                           row_group=self.row_group)
            else:
                legal = max_legal_vec_size([self.M], self.group_size, self.weight_dtype,
                                           scale_dtype=self.scale_dtype)
            if self.kernel_vector_size > legal:
                object.__setattr__(self, "kernel_vector_size", legal)
            if self.group_size % self.kernel_vector_size != 0:
                raise ValueError(
                    f"kernel_vector_size={self.kernel_vector_size} must divide "
                    f"group_size={self.group_size} -- mv_taccum_quant.cc has no double-group-"
                    "span form (mv_quant_taccum.cc's own narrowing)"
                )
            _row_bytes = row_stride_bytes(self.M, self.group_size, self.weight_dtype,
                                          self.scale_dtype)

        # K008 -- the tiling must FIT, not merely divide. Checked HERE, at construction, because
        # the only other thing that notices is aiecc, which reports it as a placement failure
        # naming a tile and not a size. Arithmetic lives once, in design.py.
        from iron.operators.tmatvec.design import check_l1_fits

        msg = check_l1_fits(
            self.M, self.K, self.batch_group, self.rows_per_chunk, m_chunk=self.m_chunk,
            a_row_bytes=_row_bytes,
        )
        if msg is not None:
            raise ValueError(msg)
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # A windowed read is a different design from the plain one at the same K: same reduction
        # extent, different buffer size and per-matrix stride. Without this they collide in the
        # build dir and a cached plain build silently satisfies the windowed op.
        base = super().name
        # weight_dtype is repr=False (keeps the default bf16 path's name stable, matching GEMV's
        # convention), so a quantized build must earn its own name explicitly or a cached bf16
        # build would silently satisfy it.
        if self.weight_dtype != "bf16":
            base = f"{base}_wdt{self.weight_dtype}g{self.group_size}"
            if self.layout == "row_group_planar":
                base = f"{base}_planar{self.row_group}"
            if self.scale_dtype != "f32":
                base = f"{base}_s{self.scale_dtype}"
        if self.alloc_K is not None and self.alloc_K != self.K:
            base = f"{base}_ak{self.alloc_K}"
        if self.block_size is not None and self.block_size != (self.alloc_K or self.K):
            base = f"{base}_blk{self.block_size}"
        # m_chunk changes the core program, the fifo depths and all three access patterns, so it
        # must not share a cache key with the unchunked build at the same shape.
        if self.m_chunk is not None and self.m_chunk != self.M:
            base = f"{base}_mc{self.m_chunk}"
        # A different core program at the same shape.
        if self.vector_size_parameter is not None:
            base = f"{base}_rtk{self.vector_size_parameter}"
        return base

    @property
    def _dim_n(self) -> int:
        """The output width one core owns per call -- the chunk when chunked, else the whole row."""
        return self.M if self.m_chunk is None else self.m_chunk

    @property
    def _kernel_object(self) -> str:
        # Keyed on the compiled width, not on M: an m_chunk=128 slice of M=512 is the same object
        # as an unchunked M=128 op, and sharing it is correct. Quantized is an ARCHIVE (zero/finish
        # from mv_taccum.cc + the quantized accumulate from mv_taccum_quant.cc), same convention as
        # gemv's epilogue archive.
        if self.weight_dtype == "bf16":
            return f"tmv_{self._dim_n}n.o"
        suffix = f"_planar{self.row_group}" if self.layout == "row_group_planar" else ""
        suffix += "" if self.scale_dtype == "f32" else f"_s{self.scale_dtype}"
        return f"tmv_{self._dim_n}n_wdt{self.weight_dtype}g{self.group_size}{suffix}.a"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "transposed_matvec",
                (
                    aie_utils.get_current_device(),
                    self.num_aie_columns,
                    self.M,
                    self.K,
                    self.num_batches,
                    self.batch_group,
                    self.rows_per_chunk,
                ),
                {
                    **self.kwargs,
                    "kernel_object": self._kernel_object,
                    "alloc_K": self.alloc_K,
                    "block_size": self.block_size,
                    "m_chunk": self.m_chunk,
                    "vector_size_parameter": self.vector_size_parameter,
                    "weight_dtype": self.weight_dtype,
                    "group_size": self.group_size,
                    "kernel_vector_size": self.kernel_vector_size,
                    "layout": self.layout,
                    "row_group": self.row_group,
                    "scale_dtype": self.scale_dtype,
                },
            ),
        )

    def get_kernel_artifacts(self):
        # zero/finish never depend on A's dtype -- reused verbatim, same object whether standalone
        # (bf16) or archived (quantized), matching GEMV's epilogue-archive convention.
        zero_finish_obj = KernelObjectArtifact(
            f"tmv_{self._dim_n}n.o",
            dependencies=[
                SourceArtifact(
                    self.context.base_dir / "aie_kernels" / "generic" / "mv_taccum.cc"
                )
            ],
            # DIM_N is the output width each core owns, and the kernel has no loop over it, so
            # it IS the A row length. Unchunked that is the whole row, which keeps the read
            # contiguous -- a per-core column slice would be M/cols wide, 32 B at head_dim 128
            # over 8 columns, under the measured ~128 B contiguity knee. m_chunk trades exactly
            # that: contiguity falls to m_chunk*2 bytes, so keep the chunk at or above 64.
            extra_flags=[f"-DDIM_N={self._dim_n}"],
        )
        if self.weight_dtype == "bf16":
            return [zero_finish_obj]
        extra_flags = [
            f"-DDIM_N={self._dim_n}",
            f"-DVEC_SIZE={self.kernel_vector_size}",
            f"-DGROUP_SIZE={self.group_size}",
            # Emit only this dtype's wrapper -- same reason gemv/mv_quant.cc do it.
            f"-DQUANT_EMIT_{self.weight_dtype.upper()}=1",
        ]
        if self.layout == "row_group_planar":
            extra_flags += [f"-DPLANAR=1", f"-DROW_GROUP={self.row_group}"]
        if self.scale_dtype == "bf16":
            extra_flags.append("-DSCALE_BF16=1")
        rows_obj = KernelObjectArtifact(
            f"tmv_{self._dim_n}n_wdt{self.weight_dtype}g{self.group_size}_rows.o",
            dependencies=[
                SourceArtifact(
                    self.context.base_dir / "aie_kernels" / "generic" / "mv_taccum_quant.cc"
                )
            ],
            extra_flags=extra_flags,
        )
        return [
            KernelArchiveArtifact(
                self._kernel_object, dependencies=[zero_finish_obj, rows_obj]
            )
        ]

    def get_arg_spec(self):
        n_matrices = self.num_batches // self.batch_group
        a_rows = self.K if self.alloc_K is None else self.alloc_K
        if self.weight_dtype == "bf16":
            a_spec = AIERuntimeArgSpec("in", (n_matrices, a_rows, self.M))  # matrix (A)
        else:
            from iron.common.quant import row_stride_bytes

            stride = row_stride_bytes(self.M, self.group_size, self.weight_dtype,
                                      self.scale_dtype)
            # Flat packed-byte buffer, same convention as GEMV's quantized matrix_spec.
            a_spec = AIERuntimeArgSpec("in", (n_matrices, a_rows * stride), dtype=np.int8)
        return [
            a_spec,
            AIERuntimeArgSpec("in", (self.num_batches, self.K)),  # vector (W)
            AIERuntimeArgSpec("out", (self.num_batches, self.M)),  # output (C)
        ]
