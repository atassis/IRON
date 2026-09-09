# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from iron.common.operator_bases import lut_based_ops_artifacts


@dataclass
class AttnBlockDataParallel(MLIROperator):
    """Decode attention block as ONE `aie.device`, data-parallel with one KV HEAD per core.

    Fuses RMSNorm(in) -> concatenated QKV GEMV -> per-head qk-RMSNorm -> RoPE -> KV append ->
    scores -> softmax -> context. See design.py for why the head->core mapping is the design and
    why the four operators this replaces cannot be fused as they stand.

    Runtime interface: cur, n_in, Wqkv, n_qn, n_kn, ang, kc, vc -> cx, with kc/vc `inout` (this
    token's row is appended at `kv_off`, then the whole window is read back).

    `wqkv_head_major=False` (the default) keeps Wqkv's stock [Wq | Wk | Wv] row-major layout: the
    head re-mapping is expressed in the FILLS, not in the blob, so it is a drop-in against the
    existing weight artifact at the cost of three fills per core instead of one. True expects the
    blob pre-permuted per core and spends one. Same bytes; see design.py for the await-count trade.
    """

    D: int
    HD: int
    Hq: int
    Hkv: int
    max_seq: int
    num_aie_columns: int = 8
    epsilon: float = 1e-6
    tile_size_input: int = 4
    stack_size: int = 0xD00
    kv_offset_parameter: str | None = "kv_off"
    mask_parameter: str = "sm_mask"
    wqkv_head_major: bool = False
    weight_depth: int = field(default=2, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
        "tile_size_input": "tsi",
        "stack_size": "ss",
        "max_seq": "S",
        "kv_offset_parameter": "kvpar",
        "mask_parameter": "mpar",
        "weight_depth": "wd",
        "wqkv_head_major": "hm",
    }

    def __post_init__(self):
        # Every rule here is design.py's, checked at construction so a graph fails where it is
        # built rather than three frames down in taplib.
        if self.Hkv != self.num_aie_columns:
            raise ValueError(
                f"one KV head per core: Hkv ({self.Hkv}) must equal num_aie_columns "
                f"({self.num_aie_columns})"
            )
        if self.Hq % self.Hkv:
            raise ValueError(f"Hq ({self.Hq}) must be a multiple of Hkv ({self.Hkv})")
        if self.D % self.HD:
            raise ValueError(
                f"d_model ({self.D}) must be a whole number of head_dim ({self.HD}) chunks -- "
                "`cur` and `n_in` ride the HD-wide misc channel"
            )
        if self.HD % self.tile_size_input:
            raise ValueError(
                f"head_dim ({self.HD}) must be divisible by tile_size_input "
                f"({self.tile_size_input})"
            )
        tile = self.tile_size_input * self.D
        if tile % self.HD:
            raise ValueError(
                f"the shared stream tile (tsi*D = {tile}) must be a whole number of cache rows "
                f"(head_dim {self.HD})"
            )
        rpc = tile // self.HD
        if self.max_seq % rpc:
            raise ValueError(
                f"max_seq ({self.max_seq}) must divide by the cache rows per stream tile ({rpc})"
            )
        # K008: the tiling must FIT, not merely divide -- and sc/sw are new L1 terms that scale
        # with max_seq, so a window this design cannot hold has to fail here and not as
        # "'aie.tile' op Basic sequential allocation also failed".
        from iron.operators.attn_block_dp.design import l1_footprint_bytes, L1_BYTES

        used = l1_footprint_bytes(self.D, self.HD, self.Hq // self.Hkv, self.max_seq, tile,
                                 self.weight_depth, self.stack_size)
        if used > L1_BYTES:
            gqa = self.Hq // self.Hkv
            raise ValueError(
                f"L1 use {used} B exceeds {L1_BYTES} B at max_seq={self.max_seq}: sc+sw alone are "
                f"{2 * gqa * self.max_seq * 2} B and are the terms max_seq drives"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "attn_block_dp",
                (aie_utils.get_current_device(), self.D, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                {
                    "epsilon": self.epsilon,
                    "kv_offset_parameter": self.kv_offset_parameter,
                    "mask_parameter": self.mask_parameter,
                    "weight_depth": self.weight_depth,
                    "tile_size_input": self.tile_size_input,
                    "stack_size": self.stack_size,
                    "n_aie_cols": self.num_aie_columns,
                    "wqkv_head_major": self.wqkv_head_major,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        copy_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        rms_obj = KernelObjectArtifact(
            f"rms_norm_{self.D}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.D}"],
        )
        rms_hd_obj = KernelObjectArtifact(
            f"hd_rms_norm_{self.HD}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.HD}"],
            prefix_symbols="hd_",
        )
        # Two DIM_Ks of one source: the projection reduces over d_model, the scores over head_dim.
        # mv.cc bakes DIM_K in, and a func.func symbol is keyed by name, so the second needs its
        # own prefixed object -- swiglu_mlp_dp's mv_gu/down_/o_ mechanism.
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_64vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.D}", "-DVEC_SIZE=64"],
        )
        sc_mv_obj = KernelObjectArtifact(
            f"sc_gemv_{self.HD}k_64vs.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.HD}", "-DVEC_SIZE=64"],
            prefix_symbols="sc_",
        )
        rope_obj = KernelObjectArtifact(
            "rope_0.o",
            dependencies=[SourceArtifact(kdir / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        softmax_obj = KernelObjectArtifact(
            "softmax.o", dependencies=[SourceArtifact(kdir / arch_dir / "softmax.cc")]
        )
        tmv_obj = KernelObjectArtifact(
            f"tmv_{self.HD}n.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv_taccum.cc")],
            extra_flags=[f"-DDIM_N={self.HD}"],
        )
        deps = [copy_obj, rms_obj, rms_hd_obj, mv_obj, sc_mv_obj, rope_obj, softmax_obj, tmv_obj]
        deps += lut_based_ops_artifacts(arch_dir)   # softmax's exp2 LUT, when the arch needs one
        return [KernelArchiveArtifact("attn_block_dp_core.a", dependencies=deps)]

    def get_arg_spec(self):
        QD, KVD = self.Hq * self.HD, self.Hkv * self.HD
        cache = self.Hkv * self.max_seq * self.HD
        return [
            AIERuntimeArgSpec("in", (self.D,)),                    # cur
            AIERuntimeArgSpec("in", (self.D,)),                    # n_in
            AIERuntimeArgSpec("in", ((QD + 2 * KVD) * self.D,)),   # Wqkv, flat [QD+2KVD, D]
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_qn
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_kn
            AIERuntimeArgSpec("in", (self.HD,)),                   # ang
            AIERuntimeArgSpec("inout", (cache,)),                  # kc: appended, then read back
            AIERuntimeArgSpec("inout", (cache,)),                  # vc: appended, then read back
            AIERuntimeArgSpec("out", (QD,)),                       # cx
        ]
