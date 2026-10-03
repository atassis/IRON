# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
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
    # Store A's `alloc_M` rows in BLOCKS of `block_size` rows (n_matrices interleaved every block)
    # instead of one `alloc_M`-row-per-matrix slab. None (default) is one block -- byte-identical
    # to the pre-blocking layout. See design.py's block_size docstring for the exact addressing;
    # this operator stays KV-agnostic, the caller (e.g. gen_llm_decode.py, via
    # iron.common.kv_layout) is what knows why a particular block_size was chosen.
    block_size: int | None = field(default=None, repr=False)
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
    # "int4"/"int8" group-quantized with a per-row f32 scale per `group_size` columns, or the
    # AFFINE "int4a"/"int8a" (w = q*s + m, a bf16 scale and a bf16 min per group -- same row
    # stride as the symmetric f32-scale form at every K and group; see gemv/quant.py),
    # dequantized on-core right before the same bf16 MAC (see design.py / mv_quant.cc / quant.py).
    # This is a byte-stream lever on the WEIGHT only -- B and C stay bf16 regardless.
    # repr=False + the name/kernel-file overrides below keep the default path's artifact names
    # stable, matching the epilogue field's convention.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
    # Per-core buffer allocation strategy ('basic-sequential' or 'bank-aware'), forwarded to each
    # Worker. None leaves the compiler default (bank-aware first, falling back to basic-sequential
    # on failure) -- see GEMM's twin field for the measurement this mirrors. Not yet measured for
    # GEMV specifically (M=1 decode shapes have a much smaller buffer footprint than batched
    # prefill's GEMM, so bank-aware may actually succeed here) -- mechanism only, no policy change.
    allocation_scheme: str | None = field(default=None, repr=False)
    # Byte layout of a quantized row -- see iron/common/quant.py. "header_first" (default) is
    # unchanged; "row_group_planar" moves the per-row scale out of the row, which is what stops
    # g64 from being punished to a 128-bit load at K=3840 (Gemma-4-12B's shipped shape) -- see
    # quant.max_legal_vec_size's docstring for the mechanism. Meaningless at weight_dtype="bf16".
    layout: str = field(default="header_first", repr=False)
    # Planar block height; see iron/common/quant.py::derive_row_group.
    row_group: int | None = field(default=None, repr=False)
    # Header precision; see iron/common/quant.py.
    scale_dtype: str = field(default="f32", repr=False)
    # Per-dispatch int32 ScratchpadParameter naming how many output rows to COMPUTE; None (the
    # default) computes all M and is byte for byte the pre-existing design. Decode attention reads
    # the `sm_mask` slot the softmax already drives. See design.py's RUNTIME ROW EXTENT block.
    vector_size_parameter: str | None = field(default=None, repr=False)
    # Tile count per core from an RTP the sequence writes (design.py, TILE COUNT FROM THE SEQUENCE).
    tiles_rtp: bool = field(default=False, repr=False)
    # Fused RMSNorm / residual-add-and-norm prologue on B, run redundantly on every core before
    # its own row tiles (gemma4-12b-decode-map.md S6, variant B). "none" (default) leaves B
    # untouched, byte for byte. "off"/"on"/"residual" are RUNTIME modes of ONE compiled core body
    # sized by `prologue_capability` (see design.py's PROLOGUE block), which is what lets a device
    # merging several GEMVs (MERGE_WEIGHT_GEMVS) mix them on ONE device: "on" needs one extra
    # `gain` argument, "residual" needs the residual stream plus two gains and an extra x1 output
    # ("off" dummy-refills B instead of any of them). Needs tiles_rtp=True, whose barrier/RTP
    # channel this shares.
    prologue: str = field(default="none", repr=False)
    # "auto" (default) derives the compiled capability from `prologue` itself (off/on -> "prenorm",
    # residual -> "residual"). Set explicitly to force a WIDER capability than this instance's own
    # mode needs, so every member of a merged family compiles the same core body -- e.g. Wqkv
    # stays prologue="on" but takes prologue_capability="residual" once any sibling in its family
    # needs the residual prologue.
    prologue_capability: str = field(default="auto", repr=False)
    prologue_epsilon: float = field(default=1e-6, repr=False)
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
        _am = self.M if self.alloc_M is None else self.alloc_M
        if self.block_size is not None and (self.block_size <= 0 or _am % self.block_size != 0):
            raise ValueError(
                f"block_size ({self.block_size}) must be a positive divisor of alloc_M ({_am})"
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
        if self.weight_dtype not in ("bf16", "int4", "int8", "int4a", "int8a"):
            raise ValueError(
                f"unknown weight_dtype {self.weight_dtype!r} (expected 'bf16', 'int4', 'int8', "
                f"'int4a' or 'int8a')"
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
            # TWO constraints bound the dequant chunk width and only the first was checked.
            # (a) it must never straddle a quant-group boundary, or a chunk needs two scales;
            # (b) the payload it loads starts header_bytes into a row, and aie::load_v needs the
            # pointer on its access width -- 32 B for int4's 256-bit load, 64 B for int8's
            # 512-bit one. (b) is what made int8 compute garbage on device while int4 was
            # correct, and nothing expressed it. The operator derives the widest legal value
            # rather than making every caller know the rule; the width is in the object name, so
            # the result is visible rather than silent.
            from iron.common.quant import max_legal_vec_size
            if self.layout == "row_group_planar":
                from iron.common.quant import derive_row_group
                if self.row_group is None:
                    # The provisional target is the docstring's own claim: once the row is
                    # correctly derived, planar always reaches min(cap, group_size) -- confirmed
                    # by the max_legal_vec_size call right below, not assumed here.
                    from iron.common.quant import widest_chunk
                    provisional_vec = widest_chunk(self.group_size, self.weight_dtype,
                                                   cap=self.kernel_vector_size)
                    object.__setattr__(self, "row_group", derive_row_group(
                        [self.K], self.group_size, self.weight_dtype,
                        vec_size=provisional_vec, max_rows=self.tile_size_input,
                        scale_dtype=self.scale_dtype))
                # K008: a planar block cannot be cut -- row i's payload and its header sit
                # row_group*payload apart -- so the L1 tile must be a whole number of blocks.
                if self.tile_size_input % self.row_group != 0:
                    raise ValueError(
                        f"row_group_planar needs tile_size_input ({self.tile_size_input}) to be "
                        f"a multiple of row_group ({self.row_group}): raise tile_size_input to a "
                        f"multiple of {self.row_group}, or pick a group whose row stride needs a "
                        f"smaller block"
                    )
                legal = max_legal_vec_size([self.K], self.group_size, self.weight_dtype,
                                            scale_dtype=self.scale_dtype,
                                            layout=self.layout, row_group=self.row_group)
            else:
                legal = max_legal_vec_size([self.K], self.group_size, self.weight_dtype,
                                            scale_dtype=self.scale_dtype)
            if self.kernel_vector_size > legal:
                object.__setattr__(self, "kernel_vector_size", legal)
            if not (self.group_size % self.kernel_vector_size == 0
                    or self.kernel_vector_size == 2 * self.group_size):
                raise ValueError(
                    f"kernel_vector_size={self.kernel_vector_size} must divide group_size="
                    f"{self.group_size} or be exactly twice it"
                )
            if self.num_batches != 1:
                raise NotImplementedError(
                    "GEMV weight_dtype != 'bf16' does not support num_batches>1 yet"
                )
        if self.prologue not in ("none", "off", "on", "residual"):
            raise ValueError(
                f"unknown prologue {self.prologue!r} (expected 'none', 'off', 'on' or 'residual')"
            )
        if self.prologue_capability not in ("auto", "none", "prenorm", "residual"):
            raise ValueError(f"unknown prologue_capability {self.prologue_capability!r}")
        if self.prologue == "residual" and self.prologue_capability not in ("auto", "residual"):
            raise ValueError("prologue='residual' needs prologue_capability in ('auto', 'residual')")
        # NOT "prologue implies tiles_rtp=True" here: unify_weight_gemvs builds a family with
        # tiles_rtp still False and turns it on afterward via dataclasses.replace, same as it
        # already does for tile_size_input/output. design.py's my_matvec asserts it instead, once
        # tiles_rtp is final.

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
            if self.layout == "row_group_planar":
                base = f"{base}_planar{self.row_group}"
            if self.scale_dtype != "f32":
                base = f"{base}_s{self.scale_dtype}"
        # A windowed read is a DIFFERENT design from the plain GEMV of the same M: same compute
        # extent, different buffer size and per-matrix stride. Without this they collide in the
        # build dir and a cached plain build silently satisfies the windowed op. alloc_M == M is
        # the same design as alloc_M=None, so it keeps the stable name.
        if self.alloc_M is not None and self.alloc_M != self.M:
            base = f"{base}_am{self.alloc_M}"
        # Same reasoning: a blocked A is a different design from the flat one at the same alloc_M.
        if self.block_size is not None and self.block_size != (self.alloc_M or self.M):
            base = f"{base}_blk{self.block_size}"
        if self.barrier_chunk != 1:
            base = f"{base}_bc{self.barrier_chunk}"
        # A different core program at the same shape.
        if self.vector_size_parameter is not None:
            base = f"{base}_rtm{self.vector_size_parameter}"
        if self.tiles_rtp:
            base = f"{base}_trtp"
        if self.prologue != "none":
            base = f"{base}_pro{self.prologue}_cap{self._resolved_prologue_capability}"
        return base

    @property
    def _resolved_prologue_capability(self):
        """`prologue_capability` with "auto" resolved -- what design.py itself would derive."""
        if self.prologue_capability != "auto":
            return self.prologue_capability
        return {"none": "none", "off": "prenorm", "on": "prenorm",
               "residual": "residual"}[self.prologue]

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
            self.layout, self.row_group, self.scale_dtype,
            self.alloc_M, self.kernel_vector_size, self.barrier_chunk,
            self.block_size, self.vector_size_parameter, self.tiles_rtp,
            # The resolved CAPABILITY, not the mode: "off"/"on"/"residual" under the SAME
            # capability compile the identical core body (design.py), so instances differing only
            # in prologue mode are the same design; different capabilities are NOT (2 vs 4 B
            # objects), so this must distinguish them even though both have prologue != "none".
            self._resolved_prologue_capability,
            self._kernel_link_file,
        ))

    @property
    def _rowbatch(self):
        return int(os.environ.get("GEMV_ROWBATCH", "1"))

    @property
    def _rowbatch_tag(self):
        # The row-batch count must ride in the OBJECT NAME, because the artifact cache keys on the
        # name: two builds differing only by a -D would otherwise share one .o and the second would
        # silently measure the first's kernel. ROWBATCH=1 keeps the historical name and is
        # byte-identical to the pre-rowbatch ELF.
        return f"_rb{self._rowbatch}" if self._rowbatch > 1 else ""

    @property
    def _kernel_link_file(self):
        # With the gelu epilogue the core also links the gelu kernel, so the object becomes an
        # archive of (matvec, gelu); the plain matvec stays a single object.
        if self.epilogue != "none":
            return f"gemv_{self.K}k_{self.kernel_vector_size}vs_{self.epilogue}_kernels.a"
        if self.weight_dtype != "bf16":
            suffix = f"_planar{self.row_group}" if self.layout == "row_group_planar" else ""
            suffix += "" if self.scale_dtype == "f32" else f"_s{self.scale_dtype}"
            return (f"gemv_{self.K}k_{self.kernel_vector_size}vs_{self.weight_dtype}"
                    f"g{self.group_size}{suffix}.o")
        return f"gemv_{self.K}k_{self.kernel_vector_size}vs{self._rowbatch_tag}.o"

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
                    "scale_dtype": self.scale_dtype,
                    "alloc_M": self.alloc_M,
                    "barrier_chunk": self.barrier_chunk,
                    "block_size": self.block_size,
                    "allocation_scheme": self.allocation_scheme,
                    "vector_size_parameter": self.vector_size_parameter,
                    "tiles_rtp": self.tiles_rtp,
                    "prologue": self.prologue,
                    "prologue_capability": self.prologue_capability,
                    "prologue_epsilon": self.prologue_epsilon,
                },
            ),
        )

    def get_kernel_artifacts(self):
        if self.weight_dtype != "bf16":
            extra_flags = [
                f"-DDIM_K={self.K}",
                f"-DVEC_SIZE={self.kernel_vector_size}",
                f"-DGROUP_SIZE={self.group_size}",
                # Emit only this dtype's wrapper -- all four instantiate otherwise, and
                # their static_asserts fire on instantiation, so one VEC_SIZE would have
                # to be legal for every dtype at once.
                f"-DQUANT_EMIT_{self.weight_dtype.upper()}=1",
            ]
            if self.layout == "row_group_planar":
                extra_flags += [f"-DPLANAR=1", f"-DROW_GROUP={self.row_group}"]
            if self.scale_dtype == "bf16":
                extra_flags.append("-DSCALE_BF16=1")
            objs = [
                KernelObjectArtifact(
                    self._kernel_link_file,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / "generic" / "mv_quant.cc"
                        )
                    ],
                    extra_flags=extra_flags,
                )
            ]
            if self.prologue != "none":
                # A separate object, not an archive: RMSNorm's own `weighted` variant links
                # rms_norm.o alongside mul.o the same way (op.py's get_kernel_artifacts).
                objs.append(
                    KernelObjectArtifact(
                        "rms_norm.o",
                        dependencies=[
                            SourceArtifact(
                                self.context.base_dir / "aie_kernels" / get_kernel_dir()
                                / "rms_norm.cc"
                            )
                        ],
                    )
                )
            return objs
        matvec_obj = KernelObjectArtifact(
            f"gemv_{self.K}k_{self.kernel_vector_size}vs{self._rowbatch_tag}.o",
            dependencies=[
                SourceArtifact(
                    self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                )
            ],
            extra_flags=[
                f"-DDIM_K={self.K}",
                f"-DVEC_SIZE={self.kernel_vector_size}",
                f"-DGEMV_ROWBATCH={self._rowbatch}",
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
            from iron.common.quant import row_stride_bytes

            stride = row_stride_bytes(self.K, self.group_size, self.weight_dtype,
                                      self.scale_dtype)
            # Flat byte buffer (int8-typed purely so the emitted shim BDs type as `i8`, matching
            # decode_ddr_bytes.py's parser): M rows of `stride` packed bytes each, row layout in
            # quant.py. num_batches is asserted ==1 for a non-bf16 weight_dtype in __post_init__.
            matrix_spec = AIERuntimeArgSpec("in", a_batch_dim + (self.M * stride,), dtype=np.int8)
        specs = [
            matrix_spec,  # matrix (A)
            AIERuntimeArgSpec("in", batch_dim + (self.K,)),  # vector (B, always bf16)
        ]
        if self.prologue == "on":
            # "off" never binds this: its sequence dummy-refills B instead (design.py).
            specs.append(AIERuntimeArgSpec("in", (self.K,)))  # gain
        if self.prologue == "residual":
            # order matches design.py's Program arg list: B above is `o`, then x and both gains.
            specs += [
                AIERuntimeArgSpec("in", (self.K,)),  # x (residual stream)
                AIERuntimeArgSpec("in", (self.K,)),  # gain_a (post-attn-norm)
                AIERuntimeArgSpec("in", (self.K,)),  # gain_b (pre-ffn-norm)
            ]
        specs.append(AIERuntimeArgSpec("out", batch_dim + (self.M,)))  # output (C, always bf16)
        if self.prologue == "residual":
            specs.append(AIERuntimeArgSpec("out", (self.K,)))  # x1, column 0 only
        return specs

    def reference(self, A, B):
        """CPU reference: (optionally batched) matrix-vector product."""
        from iron.operators.gemv.reference import reference

        return reference(A, B)
