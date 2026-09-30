# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import numpy as np

import aie.utils as aie_utils

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext


@dataclass
class GatedDeltaRule(MLIROperator):
    """`tokens` steps of the gated delta rule (Gated DeltaNet linear attention), gates included.

    Arguments, in order, each per-token buffer [tokens, row] row-major:
      ab     bf16 [ab_len]      raw gate logits, a at ab_off, b at ab_off + v_heads
      params bf16 [dk]          the bytes of f32 [neg_a | dt_bias], neg_a = -exp(A_log)
      count  bf16 [dk]          `counted` only: an int32 in the first four bytes, the tokens to
                                step; the rest leave the state and their o rows untouched
      mixed  bf16 [mixed_len]   q (k_heads x dk) at q_off, k at k_off, v (v_heads x dv) at v_off;
                                q and k already L2-normalised and q scaled by dk^-0.5, unless
                                `l2_qk` (tokens > 1 only) has the op do it
      s_in   f32 [v_heads, dk, dv]  recurrent state
      s_out  f32 [v_heads, dk, dv]  updated state -- pass the same buffer as s_in
      o      bf16 [v_heads x dv]    S^T q per head, before the gated output norm
    """

    v_heads: int
    k_heads: int
    dk: int
    dv: int
    ab_len: int
    ab_off: int
    mixed_len: int
    q_off: int
    k_off: int
    v_off: int
    dvc: int = 16
    num_aie_columns: int = 8
    # None, not 1 / False: a None field stays out of the artifact name, so decode's op keeps its name.
    tokens: int | None = None
    l2_qk: bool | None = None
    counted: bool | None = None
    # The state as two bf16 limbs, every product a native bf16 MAC (see the kernel). With tokens > 1
    # only the counted entry has it.
    limbs: bool | None = None
    # Policy, checked: aiecc measures each core's frame and refuses a smaller stack. Measured 3392 B
    # (32 heads, 8 columns) and 3584 B (16 heads, 2 columns).
    stack_size: int | None = 4096
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.v_heads % self.num_aie_columns or self.v_heads % self.k_heads:
            raise ValueError(f"v_heads ({self.v_heads}) must divide by num_aie_columns "
                             f"({self.num_aie_columns}) and by k_heads ({self.k_heads})")
        self.tokens = self.tokens if self.tokens != 1 else None
        self.l2_qk = self.l2_qk or None
        self.counted = self.counted or None
        self.limbs = self.limbs or None
        if (self.l2_qk or self.counted) and not self.tokens:
            raise ValueError("l2_qk and counted are tokens > 1 options")
        if self.limbs and self.tokens and not self.counted:
            raise ValueError("with tokens > 1, limbs needs counted: gdn_seq has no limb path")
        if self.dv % self.dvc:
            raise ValueError(f"dv ({self.dv}) must be a multiple of dvc ({self.dvc})")
        super().__init__(context=self.context)

    @property
    def _obj(self) -> str:
        t = (f"_t{self.tokens}" if self.tokens else "") + ("_limbs" if self.limbs else "")
        return f"gated_delta_rule_dk{self.dk}_dvc{self.dvc}_h{self.v_heads}{t}.o"

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        s = (self.v_heads * self.dk * self.dv,)
        n = self.tokens or 1
        return [AIERuntimeArgSpec("in", (n * self.ab_len,)), AIERuntimeArgSpec("in", (self.dk,)),
                *([AIERuntimeArgSpec("in", (self.dk,))] if self.counted else []),
                AIERuntimeArgSpec("in", (n * self.mixed_len,)),
                AIERuntimeArgSpec("in", s, np.float32), AIERuntimeArgSpec("out", s, np.float32),
                AIERuntimeArgSpec("out", (n * self.v_heads * self.dv,))]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        seq = bool(self.tokens)
        args = (aie_utils.get_current_device(), self.v_heads, self.k_heads, self.dk, self.dv,
                self.dvc, self.num_aie_columns, *((self.tokens,) if seq else ()), self.ab_len,
                self.ab_off, self.mixed_len, self.q_off, self.k_off, self.v_off, self._obj)
        fn = "gated_delta_rule_seq_design" if seq else "gated_delta_rule_design"
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "design.py", fn, args,
                            {"stack_size": self.stack_size, **({"l2_qk": True} if self.l2_qk else {}),
                             **({"counted": True} if self.counted else {})}),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [KernelObjectArtifact(
            self._obj,
            dependencies=[SourceArtifact(self.context.base_dir / "aie_kernels" / "aie2p" /
                                         "gated_delta_rule.cc")],
            extra_flags=[f"-DGDR_DK={self.dk}", f"-DGDR_DVC={self.dvc}",
                         f"-DGDR_HEADS={self.v_heads}",
                         *([f"-DGDR_T={self.tokens}", f"-DGDR_REP={self.v_heads // self.k_heads}"]
                           if self.tokens else []),
                         *(["-DGDR_LIMBS=1"] if self.limbs else [])],
        )]
