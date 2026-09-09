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
class DecodeLayerDataParallel(MLIROperator):
    """A whole decoder layer as ONE `aie.device`: attention on 8 cores, the MLP on 4 others.

    See design.py for why the phases are on different CORES and not one (program memory, which
    unlike L1 cannot be borrowed from a neighbour), why no donor row is used (neither half
    overruns its own 64 KB), why the column counts differ (8 and 4 -- one device does not require
    one column count), and what actually bounds the design (aiecc's 16 host-buffer cap).

    Runtime interface is the two halves' concatenated, with `cur` and `cx` shared:
      cur, norms, Wqkv, ang, kc, vc, cx, n_pf, Wo, Wg, Wu, Wd, gh, a_scr -> nxt
    where `norms` packs n_in | n_qn | n_kn -- see get_arg_spec for why the count matters.
    """

    D: int
    FF: int
    HD: int
    Hq: int
    Hkv: int
    max_seq: int
    attn_cols: int = 8
    mlp_cols: int = 4
    eps_attn: float = 1e-6
    eps_mlp: float = 1e-5
    tile_size_input: int = 4
    attn_stack_size: int = 0xD00
    mlp_stack_size: int = 0x800
    weight_depth: int = field(default=2, repr=False)
    tile_rows_gu: int | None = field(default=None, repr=False)
    wqkv_head_major: bool = False
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "attn_cols": "acol", "mlp_cols": "mcol", "max_seq": "S",
        "eps_attn": "ea", "eps_mlp": "em", "tile_size_input": "tsi",
        "attn_stack_size": "ass", "mlp_stack_size": "mss",
        "weight_depth": "wd", "tile_rows_gu": "trg", "wqkv_head_major": "hm",
    }

    def __post_init__(self):
        # Delegate every per-half rule to the half that owns it rather than restating it here --
        # the two operators already raise at construction with their own messages. What is checked
        # HERE is only what is true of the MERGED device and of neither half alone.
        if self.Hkv != self.attn_cols:
            raise ValueError(
                f"attention places one KV head per core: Hkv ({self.Hkv}) must equal attn_cols "
                f"({self.attn_cols})"
            )
        if self.D % self.mlp_cols or self.FF % self.mlp_cols:
            raise ValueError(
                f"D ({self.D}) and FF ({self.FF}) must both divide mlp_cols ({self.mlp_cols})"
            )
        # SHIM CHANNELS. The halves sit on different cores, so they do NOT share their misc /
        # weight / output fifos and the budget is the SUM: attention misc(1)+stream(attn_cols),
        # out(attn_cols); MLP misc(1)+weight(mlp_cols), out(mlp_cols). Checked here because it is
        # exactly the constraint that is invisible to each half and that rejected the naive
        # 4-operator fusion (49 in / 32 out) -- and because aiecc reports it as "no ShimNOCTile on
        # the device has 0 input/1 output DMA channel(s) free", which names no operator.
        shim_in = (1 + self.attn_cols) + (1 + self.mlp_cols)
        shim_out = self.attn_cols + self.mlp_cols
        limit = 16   # get_shim_dma_limit(NPU2): 8 ShimNOCTiles x 2 channels each direction
        if shim_in > limit or shim_out > limit:
            raise ValueError(
                f"merged shim DMA budget exceeded: {shim_in} input / {shim_out} output channels "
                f"against {limit} each (attention {1+self.attn_cols}/{self.attn_cols}, MLP "
                f"{1+self.mlp_cols}/{self.mlp_cols})"
            )
        cores = self.attn_cols + self.mlp_cols
        if cores > 32:
            raise ValueError(f"{cores} workers exceeds NPU2's 32 core tiles (4 rows x 8 columns)")
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "decode_layer_dp",
                (aie_utils.get_current_device(), self.D, self.FF, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                {
                    "eps_attn": self.eps_attn, "eps_mlp": self.eps_mlp,
                    "attn_cols": self.attn_cols, "mlp_cols": self.mlp_cols,
                    "tile_size_input": self.tile_size_input,
                    "attn_stack_size": self.attn_stack_size,
                    "mlp_stack_size": self.mlp_stack_size,
                    "weight_depth": self.weight_depth,
                    "tile_rows_gu": self.tile_rows_gu,
                    "wqkv_head_major": self.wqkv_head_major,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch, kdir = get_kernel_dir(), self.context.base_dir / "aie_kernels"
        QD = self.Hq * self.HD
        gen, a2 = kdir / "generic", kdir / arch

        def obj(name, src, flags=(), prefix=None):
            return KernelObjectArtifact(name, dependencies=[SourceArtifact(src)],
                                        extra_flags=list(flags), prefix_symbols=prefix)

        # EVERY symbol in this device carries its half's prefix. Both halves declare
        # `copy_offset_bf16_vector` and `matvec_vectorized_bf16_bf16` at different memref shapes,
        # and a func.func symbol is keyed by NAME only -- MLIR rejects that as "redefinition of
        # symbol", not as an overload. Object FILENAMES are prefixed too: the archive is keyed by
        # name, and an unprefixed `add.o` built for one half would be reused for the other.
        attn = [
            obj("attn_add.o", gen / "add.cc", prefix="attn_"),
            obj(f"attn_rms_{self.D}.o", a2 / "rms_norm.cc",
                [f"-DRMS_COLS={self.D}"], "attn_"),
            obj(f"attn_hd_rms_{self.HD}.o", a2 / "rms_norm.cc",
                [f"-DRMS_COLS={self.HD}"], "attn_hd_"),
            obj(f"attn_gemv_{self.D}k.o", gen / "mv.cc",
                [f"-DDIM_K={self.D}", "-DVEC_SIZE=64"], "attn_"),
            obj(f"attn_sc_gemv_{self.HD}k.o", gen / "mv.cc",
                [f"-DDIM_K={self.HD}", "-DVEC_SIZE=64"], "attn_sc_"),
            obj("attn_rope.o", gen / "rope.cc", ["-DTWO_HALVES"], "attn_"),
            obj("attn_softmax.o", a2 / "softmax.cc", (), "attn_"),
            obj(f"attn_tmv_{self.HD}n.o", gen / "mv_taccum.cc",
                [f"-DDIM_N={self.HD}"], "attn_"),
        ] + lut_based_ops_artifacts(arch)
        mlp = [
            obj("mlp_add.o", gen / "add.cc", (), "mlp_"),
            obj("mlp_mul.o", gen / "mul.cc", (), "mlp_"),
            obj(f"mlp_rms_{self.D}.o", a2 / "rms_norm.cc",
                [f"-DRMS_COLS={self.D}"], "mlp_"),
            obj("mlp_silu.o", a2 / "silu.cc", (), "mlp_"),
            obj(f"mlp_gemv_{self.D}k.o", gen / "mv.cc",
                [f"-DDIM_K={self.D}", "-DVEC_SIZE=64"], "mlp_"),
            obj(f"mlp_down_gemv_{self.FF}k.o", gen / "mv.cc",
                [f"-DDIM_K={self.FF}", "-DVEC_SIZE=64"], "mlp_down_"),
            obj(f"mlp_o_gemv_{QD}k.o", gen / "mv.cc",
                [f"-DDIM_K={QD}", "-DVEC_SIZE=64"], "mlp_o_"),
            obj("mlp_add_cxcopy.o", gen / "add.cc", (), "mlp_cx_"),
            obj("mlp_add_oacopy.o", gen / "add.cc", (), "mlp_oa_"),
        ]
        # Archive names must match what each design's own CORE_ARCHIVE builds from its
        # func_prefix -- "attn_" and "mlp_", set by design.py's decode_layer_dp.
        return [
            KernelArchiveArtifact("attn_attn_block_dp_core.a", dependencies=attn),
            KernelArchiveArtifact("mlp_swiglu_mlp_dp_core.a", dependencies=mlp),
        ]

    def get_arg_spec(self):
        D, FF, HD, Hq, Hkv, S = self.D, self.FF, self.HD, self.Hq, self.Hkv, self.max_seq
        QD, KVD = Hq * HD, Hkv * HD
        cache = Hkv * S * HD
        # Wo carries the fused-o overlap padding: swiglu_mlp_dp reads ceil(D_PER_CORE/TSI_O) full
        # TSI_O-row tiles per core, so its last core's window runs past Wo's D real rows.
        tsi_gu = self.tile_rows_gu or 6
        TSI_O = (tsi_gu * D) // QD
        O_WINDOW = -(-(D // self.mlp_cols) // TSI_O) * TSI_O
        wo_rows = D + (O_WINDOW - D // self.mlp_cols)
        # FIFTEEN host buffers, and the count is load-bearing: aiecc caps a device at 16
        # (kMaxHostBOs, tools/aiecc/SidecarFiles.h -- "conservative, hardware-verified ceiling",
        # policy rather than silicon, and not overridable). The unpacked form is 17 and is
        # REJECTED; packing the three static norm gains into one blob is what buys the margin.
        return [
            AIERuntimeArgSpec("in", (D,)),                    # cur   (shared)
            AIERuntimeArgSpec("in", (D + 2 * HD,)),           # norms = n_in | n_qn | n_kn
            AIERuntimeArgSpec("in", ((QD + 2 * KVD) * D,)),   # Wqkv
            AIERuntimeArgSpec("in", (HD,)),                   # ang   (per token, so not packed)
            AIERuntimeArgSpec("inout", (cache,)),             # kc
            AIERuntimeArgSpec("inout", (cache,)),             # vc
            AIERuntimeArgSpec("inout", (QD,)),                # cx    (shared: attn out, mlp in)
            AIERuntimeArgSpec("in", (D,)),                    # n_pf
            AIERuntimeArgSpec("in", (wo_rows * QD,)),         # Wo
            AIERuntimeArgSpec("in", (FF * D,)),               # Wg
            AIERuntimeArgSpec("in", (FF * D,)),               # Wu
            AIERuntimeArgSpec("in", (D * FF,)),               # Wd
            AIERuntimeArgSpec("inout", (FF,)),                # gh all-gather scratch
            AIERuntimeArgSpec("inout", (D,)),                 # a all-gather scratch
            AIERuntimeArgSpec("out", (D,)),                   # nxt
        ]
