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
class QKVHead(MLIROperator):
    """Fused decode QKV-head: RMSNorm(in) -> {Wq,Wk,Wv} GEMV -> per-head qk-RMSNorm -> RoPE(q,k).

    ONE aie.device replacing 6 consecutive designs (RMSNorm, 2x GEMV [Wq,Wk shared with Wv],
    24x per-head RMSNorm, 2x RoPE) in the Qwen3 decode runlist -- see
    xdna-engine/designs/decode_fused/gen_llm_decode.py. `hn` (the normalized input to the three
    projections) stays in L1 and never reaches DDR.
    """

    D: int
    HD: int
    Hq: int
    Hkv: int
    QD: int
    KVD: int
    num_aie_columns: int
    epsilon: float = 1e-6
    stack_size: int = 0xD00
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
        "stack_size": "ss",
    }

    def __post_init__(self):
        if self.QD != self.Hq * self.HD:
            raise ValueError(f"QD ({self.QD}) must equal Hq*HD ({self.Hq * self.HD})")
        if self.KVD != self.Hkv * self.HD:
            raise ValueError(f"KVD ({self.KVD}) must equal Hkv*HD ({self.Hkv * self.HD})")
        if self.Hq % self.num_aie_columns != 0:
            raise ValueError(
                f"Hq ({self.Hq}) must be divisible by num_aie_columns ({self.num_aie_columns})"
            )
        if self.Hkv % self.num_aie_columns != 0:
            raise ValueError(
                f"Hkv ({self.Hkv}) must be divisible by num_aie_columns ({self.num_aie_columns})"
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def _mv_kernel_object(self):
        return f"qkv_head_gemv_{self.D}k_64vs.o"

    @property
    def _rms_kernel_object(self):
        return "qkv_rms_norm.o"

    @property
    def _rope_kernel_object(self):
        return "rope_0.o"

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qkv_head_fused",
                (
                    aie_utils.get_current_device(),
                    self.D, self.HD, self.Hq, self.Hkv, self.QD, self.KVD,
                    self.num_aie_columns,
                ),
                {
                    "epsilon": self.epsilon,
                    "mv_kernel_object": self._mv_kernel_object,
                    "rms_kernel_object": self._rms_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                    "stack_size": self.stack_size,
                    "verbose": mlir_verbose,
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._mv_kernel_object,
                dependencies=[
                    SourceArtifact(self.context.base_dir / "aie_kernels" / "generic" / "mv.cc")
                ],
                extra_flags=[f"-DDIM_K={self.D}", "-DVEC_SIZE=64"],
            ),
            KernelObjectArtifact(
                self._rms_kernel_object,
                dependencies=[
                    SourceArtifact(self.operator_dir / "rms_norm_qkv.cc")
                ],
            ),
            KernelObjectArtifact(
                self._rope_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "rope.cc"
                    )
                ],
                extra_flags=["-DTWO_HALVES"],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.D,)),                 # cur
            AIERuntimeArgSpec("in", (self.D,)),                 # n_in
            AIERuntimeArgSpec("in", (self.QD * self.D,)),       # Wq
            AIERuntimeArgSpec("in", (self.KVD * self.D,)),      # Wk
            AIERuntimeArgSpec("in", (self.KVD * self.D,)),      # Wv
            AIERuntimeArgSpec("in", (self.HD,)),                # n_qn
            AIERuntimeArgSpec("in", (self.HD,)),                # n_kn
            AIERuntimeArgSpec("in", (self.HD,)),                # ang
            AIERuntimeArgSpec("out", (self.QD,)),               # q
            AIERuntimeArgSpec("out", (self.KVD,)),              # k
            AIERuntimeArgSpec("out", (self.KVD,)),              # v
        ]

    def reference(self, cur, n_in, wq, wk, wv, n_qn, n_kn, ang):
        """CPU reference composing the existing per-op references (welcome, not load-bearing)."""
        from iron.operators.qkv_head.reference import reference as _ref

        return _ref(
            cur, n_in, wq, wk, wv, n_qn, n_kn, ang,
            self.D, self.HD, self.Hq, self.Hkv, self.epsilon,
        )
