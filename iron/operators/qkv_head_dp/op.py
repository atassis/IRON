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


@dataclass
class QKVHeadDataParallel(MLIROperator):
    """Decode QKV head as ONE `aie.device`, data-parallel across `num_aie_columns` cores.

    Fuses RMSNorm(in) -> concatenated [Hq*HD + 2*Hkv*HD, D] QKV GEMV -> per-head qk-RMSNorm ->
    RoPE(q, k). Every core owns a contiguous row slice of the concatenated weight and runs every
    stage on it; see design.py for why that shape and not fuse/qkv-head's spatial one.

    Runtime interface: cur, n_in, Wqkv, n_qn, n_kn, ang -> qkv, where `qkv` is the concatenated
    [q | k | v] the caller then slices. The weight and the output are concatenated because the
    caller's graph already concatenates them (gen_llm_decode.py's FUSE_QKV_GEMV), not to save an
    argument.
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
    # Weight-stream format axis, as GEMV/SwiGLUMLPDataParallel take it (iron/operators/gemv/op.py).
    # bf16 (default) is the byte-for-byte pre-existing path; a quantized dtype dequantizes Wqkv
    # on-core via mv_quant.cc, same mechanism, same design-key/name/kernel-object rules.
    weight_dtype: str = field(default="bf16", repr=False)
    group_size: int = field(default=0, repr=False)
    layout: str = field(default="header_first", repr=False)
    row_group: int | None = field(default=None, repr=False)
    scale_dtype: str = field(default="f32", repr=False)
    # The dequant chunk width -- GEMV's kernel_vector_size, narrowed the same way in __post_init__.
    # Meaningless at weight_dtype="bf16", where mv.cc's own VEC_SIZE=64 stays hardcoded.
    quant_vec_size: int = field(default=64, repr=False)
    # Weight ObjectFifo depth; the L1 budget check in design.py follows it.
    weight_depth: int = field(default=2, repr=False)
    # KV-cache block size (iron.common.kv_layout.KVLayout's T). None (default) is one block ==
    # max_seq, byte-identical to the pre-blocking append. See design.py. repr=True (the dataclass
    # default) DELIBERATELY, unlike its siblings on this line -- the base `name` property includes
    # every repr=True non-None field automatically, and this is a graph-changing knob (a different
    # T addresses the cache differently), so it must never share an artifact name with the
    # unblocked design the way an un-named flag has before (see sequence_name()'s FUSE_QKV_DP
    # history in gen_llm_decode.py).
    kv_block_size: int | None = None
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
        "tile_size_input": "tsi",
        "stack_size": "ss",
        "max_seq": "S",
        "kv_offset_parameter": "kvpar",
        "weight_depth": "wd",
        "kv_block_size": "kvblk",
        "weight_dtype": "wdt",
        "group_size": "g",
        "layout": "lay",
        "row_group": "rg",
        "scale_dtype": "sdt",
    }

    @property
    def name(self) -> str:
        # weight_dtype/group_size/layout/row_group/scale_dtype are repr=False so the default path
        # keeps its stable name -- mirrors gemv/op.py's identical convention.
        base = super().name
        if self.weight_dtype != "bf16":
            base = f"{base}_wdt{self.weight_dtype}g{self.group_size}"
            if self.layout == "row_group_planar":
                base = f"{base}_planar{self.row_group}"
            if self.scale_dtype != "f32":
                base = f"{base}_s{self.scale_dtype}"
        return base

    def __post_init__(self):
        heads = self.Hq + 2 * self.Hkv
        if heads % self.num_aie_columns:
            raise ValueError(
                f"Hq + 2*Hkv ({heads}) must be divisible by num_aie_columns "
                f"({self.num_aie_columns}) -- every core owns a whole number of heads"
            )
        if self.HD % self.tile_size_input:
            raise ValueError(
                f"head_dim ({self.HD}) must be divisible by tile_size_input "
                f"({self.tile_size_input})"
            )
        if self.D % self.HD:
            raise ValueError(
                f"d_model ({self.D}) must be a whole number of head_dim ({self.HD}) chunks -- "
                "`cur` and `n_in` ride the HD-wide misc channel"
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
            if self.group_size <= 0:
                raise ValueError("weight_dtype != 'bf16' needs an explicit group_size > 0")
            if self.D % self.group_size != 0:
                raise ValueError(
                    f"D={self.D} must be a whole number of groups (group_size={self.group_size})"
                )
            from iron.common.quant import max_legal_vec_size
            # Complete-block requirement; see iron/common/quant.py::derive_row_group.
            if self.layout == "row_group_planar":
                from iron.common.quant import derive_row_group, widest_chunk
                if self.row_group is None:
                    provisional_vec = widest_chunk(self.group_size, self.weight_dtype, cap=64)
                    object.__setattr__(self, "row_group", derive_row_group(
                        [self.D], self.group_size, self.weight_dtype,
                        vec_size=provisional_vec, max_rows=self.tile_size_input,
                        scale_dtype=self.scale_dtype))
                if self.tile_size_input % self.row_group != 0:
                    raise ValueError(
                        f"row_group_planar needs tile_size_input ({self.tile_size_input}) to be "
                        f"a multiple of row_group ({self.row_group})"
                    )
                legal = max_legal_vec_size([self.D], self.group_size, self.weight_dtype,
                                            scale_dtype=self.scale_dtype, layout=self.layout,
                                            row_group=self.row_group)
            else:
                legal = max_legal_vec_size([self.D], self.group_size, self.weight_dtype,
                                            scale_dtype=self.scale_dtype)
            if self.quant_vec_size > legal:
                object.__setattr__(self, "quant_vec_size", legal)
            if not (self.group_size % self.quant_vec_size == 0
                    or self.quant_vec_size == 2 * self.group_size):
                raise ValueError(
                    f"quant_vec_size={self.quant_vec_size} must divide group_size="
                    f"{self.group_size} or be exactly twice it"
                )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qkv_head_dp",
                (aie_utils.get_current_device(), self.D, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                {
                    "epsilon": self.epsilon,
                    "kv_offset_parameter": self.kv_offset_parameter,
                    "weight_depth": self.weight_depth,
                    "tile_size_input": self.tile_size_input,
                    "stack_size": self.stack_size,
                    "n_aie_cols": self.num_aie_columns,
                    "kv_block_size": self.kv_block_size,
                    "weight_dtype": self.weight_dtype,
                    "group_size": self.group_size,
                    "layout": self.layout,
                    "row_group": self.row_group,
                    "scale_dtype": self.scale_dtype,
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
        # Same source, second symbol: this core calls weighted_rms_norm at D and at HD, and one
        # Kernel() binding fixes one signature per symbol. Prefixing a second object is what
        # swiglu_mlp_dp does for its two matvec DIM_Ks; the alternative -- a local copy of the
        # vendored kernel under two names -- is the duplication one-kernel-three-repos warns about.
        rms_hd_obj = KernelObjectArtifact(
            f"hd_rms_norm_{self.HD}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.HD}"],
            prefix_symbols="hd_",
        )
        # Same collision class as the two rms_norm widths above: -DPLANAR/-DSCALE_BF16 change the
        # emitted code while the exported symbol stays identical (mv_quant.cc's own contract), so
        # weight_dtype/layout/scale_dtype all ride in the object NAME -- see gemv/op.py's twin.
        _qsrc = kdir / "generic" / ("mv.cc" if self.weight_dtype == "bf16" else "mv_quant.cc")
        _qtag = "" if self.weight_dtype == "bf16" else f"_{self.weight_dtype}g{self.group_size}"
        if self.weight_dtype != "bf16":
            if self.layout == "row_group_planar":
                _qtag += f"_planar{self.row_group}"
            if self.scale_dtype != "f32":
                _qtag += f"_s{self.scale_dtype}"
        _vec = 64 if self.weight_dtype == "bf16" else self.quant_vec_size
        _qflags = ([] if self.weight_dtype == "bf16"
                   else [f"-DGROUP_SIZE={self.group_size}",
                         f"-DQUANT_EMIT_{self.weight_dtype.upper()}=1"])
        if self.weight_dtype != "bf16":
            if self.layout == "row_group_planar":
                _qflags += ["-DPLANAR=1", f"-DROW_GROUP={self.row_group}"]
            if self.scale_dtype == "bf16":
                _qflags.append("-DSCALE_BF16=1")
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_{_vec}vs{_qtag}.o",
            dependencies=[SourceArtifact(_qsrc)],
            extra_flags=[f"-DDIM_K={self.D}", f"-DVEC_SIZE={_vec}"] + _qflags,
        )
        rope_obj = KernelObjectArtifact(
            "rope_0.o",
            dependencies=[SourceArtifact(kdir / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        return [
            KernelArchiveArtifact(
                f"qkv_head_dp_core{_qtag}.a",
                dependencies=[copy_obj, rms_obj, rms_hd_obj, mv_obj, rope_obj],
            )
        ]

    def get_arg_spec(self):
        QD, KVD = self.Hq * self.HD, self.Hkv * self.HD
        TOT = QD + 2 * KVD
        cache = self.Hkv * self.max_seq * self.HD
        if self.weight_dtype == "bf16":
            wqkv_spec = AIERuntimeArgSpec("in", (TOT * self.D,))
        else:
            import numpy as np
            from iron.common.quant import row_stride_bytes

            stride = row_stride_bytes(self.D, self.group_size, self.weight_dtype,
                                      self.scale_dtype)
            wqkv_spec = AIERuntimeArgSpec("in", (TOT * stride,), dtype=np.int8)
        return [
            AIERuntimeArgSpec("in", (self.D,)),                    # cur
            AIERuntimeArgSpec("in", (self.D,)),                    # n_in
            wqkv_spec,                                             # Wqkv, flat [QD+2KVD, D|stride]
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_qn
            AIERuntimeArgSpec("in", (self.HD,)),                   # n_kn
            AIERuntimeArgSpec("in", (self.HD,)),                   # ang
            AIERuntimeArgSpec("out", (QD,)),                       # q
            AIERuntimeArgSpec("inout", (cache,)),                  # kc, appended at kv_off
            AIERuntimeArgSpec("inout", (cache,)),                  # vc, appended at kv_off
        ]

    def reference(self, cur, n_in, wqkv, n_qn, n_kn, ang):
        """Returns the concatenated [q | k | v]; the caller places k and v itself. The device
        appends them to the caches directly, so there is no single output to compare against."""
        from iron.operators.qkv_head_dp.reference import reference

        return reference(cur, n_in, wqkv, n_qn, n_kn, ang,
                         self.D, self.HD, self.Hq, self.Hkv, self.epsilon)
