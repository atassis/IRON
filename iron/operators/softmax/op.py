# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np

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
    """AIE-accelerated Softmax operation

    ``vector_size_source="rows"`` opts into per-row mask widths: the op takes a
    third runtime buffer of ``rows`` int32 values and row ``i`` is masked past
    ``widths[i]`` instead of every row sharing one scalar width. Batched prefill
    needs this -- row ``i`` of a chunk at absolute position ``base`` may only
    attend positions ``<= base + i``. Default (``None``) keeps the single-width
    behaviour and generates byte-identical MLIR.

    That mode costs a third shim DMA channel per core, so it places on up to 8
    cores where the default reaches 16 (measured on npu2: 8 OK, 16 fails with
    "no ShimNOCTile has sufficient DMA capacity"). Every configuration this repo
    builds is at most 4 cores. To lift it, broadcast one widths buffer to all
    cores and index it by row instead of giving each core its own fifo -- one
    shim tile instead of one per core, at ``rows * 4`` bytes of L1 per core."""

    rows: int
    cols: int
    num_aie_columns: int = 1
    num_channels: int = 1
    rtp_vector_size: int | None = None
    vector_size_parameter: str | None = None
    vector_size_source: str | None = None
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
        # Keep repr ON: MLIROperator.name filters fields that are None, so the
        # None default stays out of the artifact key and "rows" enters it. A
        # design-affecting flag hidden from that key is how two different
        # designs come to share one xclbin.
        if self.vector_size_source not in (None, "rows"):
            raise ValueError(
                f"vector_size_source must be None or 'rows', got "
                f"{self.vector_size_source!r}"
            )
        if self.vector_size_source == "rows" and self.vector_size_parameter is not None:
            raise ValueError(
                "vector_size_source='rows' takes each row's unmasked width from the "
                "streamed widths buffer, so vector_size_parameter must be None."
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def per_row_vector_size(self) -> bool:
        """Whether the unmasked width is streamed per row rather than shared."""
        return self.vector_size_source == "rows"

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
                    "vector_size_source": self.vector_size_source,
                    "kernel_obj_file": self._kernel_link_file,
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
        # The widths buffer sits between in and out: OperatorSequence splits a
        # runlist step as `*in_specs, out_spec`, so the output must stay last.
        # dtype is explicit -- AIERuntimeArgSpec defaults to bfloat16, and
        # calculate_buffer_layout sizes the host buffer off it, so leaving it
        # would under-allocate the widths buffer by 2x.
        specs = [AIERuntimeArgSpec("in", (self.size,))]
        if self.per_row_vector_size:
            specs.append(AIERuntimeArgSpec("in", (self.rows,), dtype=np.int32))
        specs.append(AIERuntimeArgSpec("out", (self.size,)))
        return specs

    def reference(self, x, widths=None):
        """CPU reference: row-wise softmax over ``cols``.

        With ``vector_size_source="rows"``, ``widths`` is the per-row unmasked
        width vector (``rows`` int32 values) and row ``i`` is softmaxed over
        ``x[i, :widths[i]]``, with the tail zeroed -- mirroring ``mask_bf16``
        writing ``-inf`` past ``widths[i]`` before ``softmax_bf16`` runs.

        Note: ignores the runtime ``vector_size_parameter`` (if any); in that
        mode the reference always softmaxes over the full ``cols``. For
        decode-style usage with a masked tail, the trailing positions will not
        match the NPU output."""
        from iron.operators.softmax.reference import reference, masked_reference

        rows = x.reshape(self.rows, self.cols)
        if not self.per_row_vector_size:
            return reference(rows)
        if widths is None:
            raise ValueError(
                "vector_size_source='rows' needs the per-row widths vector to "
                "produce a reference"
            )
        return masked_reference(rows, widths)
