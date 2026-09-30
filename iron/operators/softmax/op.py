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


# aie_kernels/aie2p/flash_contract.h's FLASH_SM_VEC_LEN. Third copy of that constant, for the same
# reason its own comment gives: the side that would corrupt should carry the check.
SEGMENT_GRANULARITY = 64


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
    shim tile instead of one per core, at ``rows * 4`` bytes of L1 per core.

    ``vector_size_source="rows_hole"`` is the sliding-window RING variant: past
    the wrap the valid columns are a hole in the middle
    ``[hole_lo, hole_hi)`` excluded, not a suffix, so each row streams a triple
    (``hole_lo``, ``hole_hi``, ``width``) instead of one width and the third
    buffer is ``rows * 3`` int32. See
    ``docs/superpowers/specs/2026-09-17-prefill-batch-past-the-sliding-window-design.md``
    sec 1.3/2.1. Same 8-core DMA-channel cost as ``"rows"``.

    ``segment=L`` opts into the CHUNKED form: the per-acquire L1 tile is ``L``
    columns rather than the whole row, so L1 stops scaling with ``cols``.
    Default (``None``) is the unchunked op and generates byte-identical MLIR.
    Bit-identical arithmetic, at three streams of the input instead of one --
    see ``design.py::_softmax_chunked``. aie2p only; it links its own kernel
    object so the unchunked arms' ELFs cannot move.

    ``segment=`` composes with ``vector_size_source="rows"``: each row's width is
    re-read from the widths fifo once per pass (three times total) instead of
    once per core. Still refused with ``"rows_hole"`` -- the chunked kernels have
    no hole-aware unmasked-width form."""

    rows: int
    cols: int
    num_aie_columns: int = 1
    num_channels: int = 1
    rtp_vector_size: int | None = None
    vector_size_parameter: str | None = None
    vector_size_source: str | None = None
    segment: int | None = None
    # Per-core buffer allocation strategy ('basic-sequential' or 'bank-aware'), forwarded to each
    # Worker -- see GEMM's twin field (iron/operators/gemm/op.py) for the measurement this mirrors.
    allocation_scheme: str | None = field(default=None, repr=False)
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
        if self.vector_size_source not in (None, "rows", "rows_hole"):
            raise ValueError(
                f"vector_size_source must be None, 'rows' or 'rows_hole', got "
                f"{self.vector_size_source!r}"
            )
        if self.per_row_vector_size and self.vector_size_parameter is not None:
            raise ValueError(
                f"vector_size_source={self.vector_size_source!r} takes each row's mask from "
                "the streamed widths buffer, so vector_size_parameter must be None."
            )
        if self.segment is not None:
            if self.cols % self.segment != 0:
                raise ValueError(
                    f"cols ({self.cols}) must be a multiple of segment ({self.segment})"
                )
            if self.segment % SEGMENT_GRANULARITY != 0:
                raise ValueError(
                    f"segment ({self.segment}) must be a multiple of "
                    f"{SEGMENT_GRANULARITY}: the kernel loops step that many elements with "
                    f"no scalar tail and would silently drop the remainder"
                )
            if self.hole_mode:
                # rows_hole's triple (hole_lo, hole_hi, width) has no chunked-kernel form: the
                # ring window's excluded middle needs a hole-aware unmasked-width computation
                # the segment kernels don't have, unlike plain "rows" below, which reuses
                # k_max/k_sum/k_apply's existing scalar `unmasked` argument unchanged.
                raise ValueError(
                    f"segment= and vector_size_source={self.vector_size_source!r} are not "
                    "implemented together"
                )
        MLIROperator.__init__(self, context=self.context)

    @property
    def per_row_vector_size(self) -> bool:
        """Whether the mask is streamed per row rather than shared."""
        return self.vector_size_source in ("rows", "rows_hole")

    @property
    def hole_mode(self) -> bool:
        """Whether the per-row mask is a (hole_lo, hole_hi, width) triple, not one width."""
        return self.vector_size_source == "rows_hole"

    @property
    def _kernel_link_file(self):
        kernel_dir = get_kernel_dir()
        if self.segment is not None:
            if kernel_dir != "aie2p":
                raise ValueError(
                    f"segment= needs aie_kernels/aie2p/softmax_chunked.cc; this device resolves "
                    f"to '{kernel_dir}', which has no chunked kernel"
                )
            return "softmax_chunked.o"
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
                    "allocation_scheme": self.allocation_scheme,
                    "segment": self.segment,
                },
            ),
        )

    def get_kernel_artifacts(self):
        kernel_dir = get_kernel_dir()
        if self.segment is not None:
            return [
                KernelObjectArtifact(
                    "softmax_chunked.o",
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir
                            / "aie_kernels"
                            / kernel_dir
                            / "softmax_chunked.cc"
                        )
                    ],
                )
            ]
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
            n = self.rows * 3 if self.hole_mode else self.rows
            specs.append(AIERuntimeArgSpec("in", (n,), dtype=np.int32))
        specs.append(AIERuntimeArgSpec("out", (self.size,)))
        return specs

    def reference(self, x, widths=None):
        """CPU reference: row-wise softmax over ``cols``.

        With ``vector_size_source="rows"``, ``widths`` is the per-row unmasked
        width vector (``rows`` int32 values) and row ``i`` is softmaxed over
        ``x[i, :widths[i]]``, with the tail zeroed -- mirroring ``mask_bf16``
        writing ``-inf`` past ``widths[i]`` before ``softmax_bf16`` runs.

        With ``vector_size_source="rows_hole"``, ``widths`` is ``[rows, 3]``
        (``hole_lo``, ``hole_hi``, ``width``) and row ``i`` is masked on
        ``[hole_lo, hole_hi)`` and ``[width, cols)`` -- mirroring
        ``mask_hole_bf16``.

        Note: ignores the runtime ``vector_size_parameter`` (if any); in that
        mode the reference always softmaxes over the full ``cols``. For
        decode-style usage with a masked tail, the trailing positions will not
        match the NPU output."""
        from iron.operators.softmax.reference import reference, masked_reference, masked_reference_hole

        rows = x.reshape(self.rows, self.cols)
        if not self.per_row_vector_size:
            return reference(rows)
        if widths is None:
            raise ValueError(
                f"vector_size_source={self.vector_size_source!r} needs the per-row widths "
                "vector to produce a reference"
            )
        if self.hole_mode:
            return masked_reference_hole(rows, widths)
        return masked_reference(rows, widths)
