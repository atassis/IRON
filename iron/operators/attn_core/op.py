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
from iron.common.device_utils import get_kernel_dir


@dataclass
class AttnCore(MLIROperator):
    """Fused attention core (KV-append + scores + softmax + context + output projection).

    See design.py's module docstring for the full derivation. One `aie.device` replacing 7
    consecutive runlist designs (op_sck, op_scv, op_scores, op_scale, op_softmax, op_ctx, op_o) in
    xdna-engine's Qwen3-0.6B fused decode.
    """

    D: int = 1024
    Hq: int = 16
    Hkv: int = 8
    HD: int = 128
    S: int = 2048
    cols: int = 8
    rows_per_chunk: int = 64
    attn_scale: float | None = field(default=None, repr=False)
    scores_m_input: int = field(default=64, repr=False)
    o_m_input: int = field(default=4, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "cols": "col",
        "rows_per_chunk": "rpc",
    }

    def __post_init__(self):
        if self.Hq % self.Hkv != 0:
            raise ValueError(f"Hq ({self.Hq}) must be a multiple of Hkv ({self.Hkv})")
        if self.Hkv != self.cols:
            raise ValueError(
                f"TMatVec places one matrix per column: Hkv ({self.Hkv}) must equal cols "
                f"({self.cols})"
            )
        if self.S % self.rows_per_chunk != 0:
            raise ValueError(f"rows_per_chunk ({self.rows_per_chunk}) must divide S ({self.S})")
        from iron.operators.tmatvec.design import check_l1_fits as tmv_l1_fits

        group = self.Hq // self.Hkv
        msg = tmv_l1_fits(self.HD, self.S, group, self.rows_per_chunk)
        if msg is not None:
            raise ValueError(f"op_ctx (TMatVec): {msg}")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _attn_scale_value(self) -> float:
        return self.HD ** -0.5 if self.attn_scale is None else self.attn_scale

    @property
    def _kernel_object_taccum(self) -> str:
        return f"attn_tmv_{self.HD}n.o"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "attn_core",
                (aie_utils.get_current_device(),),
                {
                    "cols": self.cols,
                    "D": self.D,
                    "Hq": self.Hq,
                    "Hkv": self.Hkv,
                    "HD": self.HD,
                    "S": self.S,
                    "rows_per_chunk": self.rows_per_chunk,
                    "attn_scale": self.attn_scale,
                    "scores_m_input": self.scores_m_input,
                    "o_m_input": self.o_m_input,
                    "kernel_object_mv_scores": "attn_scores_mv.o",
                    "kernel_object_mv_o": "attn_o_mv.o",
                    "kernel_name_mv_o": "o_matvec_vectorized_bf16_bf16",
                    "kernel_object_scale": "attn_scale_tile.o",
                    "kernel_object_softmax": "softmax.o",
                    "kernel_object_taccum": self._kernel_object_taccum,
                    "verbose": getattr(self.context, "mlir_verbose", False),
                },
            ),
        )

    def get_kernel_artifacts(self):
        QD = self.Hq * self.HD
        kernel_dir = get_kernel_dir()
        scale_val = self._attn_scale_value
        return [
            KernelObjectArtifact(
                "attn_scores_mv.o",
                dependencies=[
                    SourceArtifact(self.context.base_dir / "aie_kernels" / "generic" / "mv.cc")
                ],
                extra_flags=[f"-DDIM_K={self.HD}", "-DVEC_SIZE=64"],
            ),
            # mv.cc's symbol names are fixed (not parameterized by DIM_K), so compiling it twice
            # under two different macros -- once for op_scores (DIM_K=HD), once here for op_o
            # (DIM_K=QD) -- gives two objects that both define `matvec_vectorized_bf16_bf16`.
            # Both land in the SAME aie.device (one device is the whole point of this operator),
            # and mlir-aie's symbol table is module-global, not per-core: "redefinition of symbol
            # named 'matvec_vectorized_bf16_bf16'" if both keep the stock name. rename_symbols
            # (an objcopy --redefine-sym post-process) gives op_o's object a distinct symbol;
            # design.py's `o_mv` Kernel() declaration must reference that same renamed name.
            KernelObjectArtifact(
                "attn_o_mv.o",
                dependencies=[
                    SourceArtifact(self.context.base_dir / "aie_kernels" / "generic" / "mv.cc")
                ],
                extra_flags=[f"-DDIM_K={QD}", "-DVEC_SIZE=64"],
                rename_symbols={
                    "matvec_vectorized_bf16_bf16": "o_matvec_vectorized_bf16_bf16"
                },
            ),
            KernelObjectArtifact(
                "attn_scale_tile.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "scale_tile_bf16.cc"
                    )
                ],
                extra_flags=[f"-DATTN_SCALE={scale_val!r}f"],
            ),
            KernelObjectArtifact(
                "softmax.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / kernel_dir / "softmax.cc"
                    )
                ],
            ),
            KernelObjectArtifact(
                self._kernel_object_taccum,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv_taccum.cc"
                    )
                ],
                extra_flags=[f"-DDIM_N={self.HD}"],
            ),
        ]

    def get_arg_spec(self):
        KVD = self.Hkv * self.HD
        QD = self.Hq * self.HD
        cache_len = self.Hkv * self.S * self.HD
        return [
            AIERuntimeArgSpec("in", (KVD,)),                # k
            AIERuntimeArgSpec("in", (KVD,)),                # v
            AIERuntimeArgSpec("inout", (cache_len,)),        # kc (persistent cache)
            AIERuntimeArgSpec("inout", (cache_len,)),        # vc (persistent cache)
            AIERuntimeArgSpec("in", (QD,)),                  # q
            AIERuntimeArgSpec("in", (self.D * QD,)),         # Wo
            AIERuntimeArgSpec("out", (self.D,)),             # a
            AIERuntimeArgSpec("inout", (self.Hq * self.S,)),  # sc scratch (not resident, see design.py)
            AIERuntimeArgSpec("inout", (self.Hq * self.S,)),  # sw scratch (not resident, see design.py)
            AIERuntimeArgSpec("inout", (QD,)),               # cx scratch (not resident, see design.py)
        ]
