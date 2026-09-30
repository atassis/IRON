# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional

import numpy as np

import aie.utils as aie_utils
from iron.common import (AIERuntimeArgSpec, DesignGenerator, KernelObjectArtifact, MLIROperator,
                         PythonGeneratedMLIRArtifact, SourceArtifact)

FLAGS = ["-Dbf16_f32_ONLY", "-DROUND_CONV_EVEN", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"]


def fa_objects(ctx, rows=16):
    """The QK/softmax object and the x V object for one row count: S^T = K . Q^T is
    [64 keys x rows], O = P . V is [rows x 64]."""
    k = ctx.base_dir / "aie_kernels" / "aie2p"
    deps = [SourceArtifact(k / n) for n in ("fused_attn.cc", "mm.cc", "softmax.cc")]
    sfx = "" if rows == 16 else str(rows)
    r = [f"-DFA_ROWS={rows}"]
    return [KernelObjectArtifact(f"fa_qk{sfx}.o", dependencies=deps,
                                 extra_flags=FLAGS + r + ["-DDIM_M=64", "-DDIM_K=64", f"-DDIM_N={rows}"]),
            KernelObjectArtifact(f"fa_pv{sfx}.o", dependencies=deps,
                                 extra_flags=FLAGS + r + [f"-DDIM_M={rows}", "-DDIM_K=64", "-DDIM_N=64"])]


@dataclass
class TiledFeed(MLIROperator):
    """Q and K through both memtile hops, copied out as delivered."""

    n_blocks: int
    head_dim: int = 512
    slice_w: int = 64
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases, "n_blocks": "nb", "head_dim": "hd", "slice_w": "sw"}

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "feed_design.py", "tiled_feed",
                            (aie_utils.get_current_device(), self.n_blocks, self.head_dim,
                             self.slice_w)))

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact(
            "fa_feed_passThrough.o",
            dependencies=[SourceArtifact(self.context.base_dir / "aie_kernels" / "generic"
                                         / "passThrough.cc")],
            extra_flags=["-DBIT_WIDTH=16"])]

    def get_arg_spec(self):
        q, k = 16 * self.head_dim, self.n_blocks * 64 * self.head_dim
        return [AIERuntimeArgSpec("in", (q,)), AIERuntimeArgSpec("in", (k,)),
                AIERuntimeArgSpec("out", (q,)), AIERuntimeArgSpec("out", (k,))]


@dataclass
class KernelCheck(MLIROperator):
    """One of kernel_checks_design's harnesses; `payload` is its resident inputs."""

    check: str
    rows: int = 16
    payload: object = field(default=None, repr=False, compare=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {**MLIROperator._name_aliases, "check": "ck",
                                               "rows": "r"}

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "kernel_checks_design.py", f"{self.check}_check",
                            (aie_utils.get_current_device(), self.rows) + tuple(self.payload)))

    def get_kernel_artifacts(self):
        return fa_objects(self.context, self.rows)

    def get_arg_spec(self):
        r = self.rows
        if self.check == "qt":
            return [AIERuntimeArgSpec("out", (512 * r,))]
        if self.check == "smt":
            return [AIERuntimeArgSpec("out", (r * 64,)),
                    AIERuntimeArgSpec("out", (2 * r,), dtype=np.float32)]
        return [AIERuntimeArgSpec("out", (len(self.payload[0]),))]


@dataclass
class FusedAttnOnePipe(MLIROperator):
    """16 query rows over n_blocks x 64 keys, one pipeline of three workers."""

    n_blocks: int
    head_dim: int = 512
    slice_w: int = 64
    mask: Optional[int] = None
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases, "n_blocks": "nb", "head_dim": "hd", "slice_w": "sw",
        "mask": "mk"}

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "design.py", "fused_attn_one_pipe",
                            (aie_utils.get_current_device(), self.n_blocks, self.head_dim,
                             self.slice_w, 1 if self.mask is None else self.mask)))

    def get_kernel_artifacts(self):
        return fa_objects(self.context)

    def get_arg_spec(self):
        kv = self.n_blocks * 64 * self.head_dim
        return [AIERuntimeArgSpec("in", (16 * self.head_dim,)),
                AIERuntimeArgSpec("in", (16,), dtype=np.int32),
                AIERuntimeArgSpec("in", (kv,)), AIERuntimeArgSpec("in", (kv,)),
                AIERuntimeArgSpec("out", (16 * self.head_dim,))]


@dataclass
class VPrepass(MLIROperator):
    """Gemma-4's attention_k_eq_v: recompute a whole K block's V up front (no stored V, no
    v_proj), so fused_attn's V-tap reads a real buffer unchanged. See v_prepass_design.py."""

    n_blocks: int
    head_dim: int = 512
    epsilon: float = 1e-6
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases, "n_blocks": "nb", "head_dim": "hd", "epsilon": "eps"}

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "v_prepass_design.py", "v_prepass",
                            (aie_utils.get_current_device(), self.n_blocks, self.head_dim,
                             self.epsilon)))

    def get_kernel_artifacts(self):
        k = self.context.base_dir / "aie_kernels" / "aie2p"
        return [
            KernelObjectArtifact(
                "fa_v_prepass.o", dependencies=[SourceArtifact(k / "fused_attn_v_prepass.cc")],
                extra_flags=[f"-DHEAD_DIM={self.head_dim}"]),
            KernelObjectArtifact(
                "fa_feed_passThrough.o",
                dependencies=[SourceArtifact(self.context.base_dir / "aie_kernels" / "generic"
                                             / "passThrough.cc")],
                extra_flags=["-DBIT_WIDTH=16"]),
        ]

    def get_arg_spec(self):
        kv = self.n_blocks * 64 * self.head_dim
        return [AIERuntimeArgSpec("in", (kv,)), AIERuntimeArgSpec("in", (self.head_dim,)),
                AIERuntimeArgSpec("in", (kv,)), AIERuntimeArgSpec("out", (kv,))]


@dataclass
class HalfFeed(MLIROperator):
    """One head-dim half of a [keys, head_dim] tensor through both memtile hops, copied out."""

    n_blocks: int
    half: int
    head_dim: int = 512
    slice_w: int = 64
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases, "n_blocks": "nb", "half": "h", "head_dim": "hd",
        "slice_w": "sw"}

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "feed_design.py", "half_feed",
                            (aie_utils.get_current_device(), self.n_blocks, self.head_dim,
                             self.slice_w, self.half)))

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact(
            "fa_feed_passThrough.o",
            dependencies=[SourceArtifact(self.context.base_dir / "aie_kernels" / "generic"
                                         / "passThrough.cc")],
            extra_flags=["-DBIT_WIDTH=16"])]

    def get_arg_spec(self):
        keys = self.n_blocks * 64
        return [AIERuntimeArgSpec("in", (keys * self.head_dim,)),
                AIERuntimeArgSpec("out", (keys * self.head_dim // 2,))]


@dataclass
class FusedAttnX2(MLIROperator):
    """32 query rows over n_blocks x 64 keys: QK, softmax and two x V halves in one column."""

    n_blocks: int
    head_dim: int = 512
    slice_w: int = 64
    mask: Optional[int] = None
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases, "n_blocks": "nb", "head_dim": "hd", "slice_w": "sw",
        "mask": "mk"}

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(self.operator_dir / "design_x2.py", "fused_attn_x2",
                            (aie_utils.get_current_device(), self.n_blocks, self.head_dim,
                             self.slice_w, 1 if self.mask is None else self.mask)))

    def get_kernel_artifacts(self):
        return fa_objects(self.context, 32)

    def get_arg_spec(self):
        kv = self.n_blocks * 64 * self.head_dim
        return [AIERuntimeArgSpec("in", (32 * self.head_dim,)),
                AIERuntimeArgSpec("in", (32,), dtype=np.int32),
                AIERuntimeArgSpec("in", (kv,)), AIERuntimeArgSpec("in", (kv,)),
                AIERuntimeArgSpec("out", (32 * self.head_dim,))]
