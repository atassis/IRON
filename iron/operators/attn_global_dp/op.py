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
from iron.operators.attn_global_dp.design import (
    FLASH_SM_VEC_LEN, L1_BYTES, column_splits, worker_l1_footprint_bytes,
)
from iron.common.operator_bases import lut_based_ops_artifacts


@dataclass
class AttnGlobalWorker(MLIROperator):
    """gemma4-weightless-attention-block variant A_g, worker half: one core per column, ALL Hq
    heads applied to that column's own position slice. See design.py's module docstring for the
    algebra and design.py's `attn_global_worker` for the runtime interface
    (q, kc, vc -> partials).
    """

    HD: int
    Hq: int
    S: int
    num_aie_columns: int = 8
    head_groups: int = 1
    weight_depth: int = 1
    stack_size: int = 0xD00
    mask_parameter: str = "sm_mask"
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "head_groups": "hg",
        "weight_depth": "wd",
        "stack_size": "ss",
        "mask_parameter": "mpar",
    }

    def __post_init__(self):
        if self.Hq % self.head_groups:
            raise ValueError(f"Hq ({self.Hq}) must be a multiple of head_groups ({self.head_groups})")
        # K007/K008 at construction, not three frames down in taplib -- same discipline
        # attn_block_dp's own __post_init__ follows.
        column_splits(self.S, self.num_aie_columns, FLASH_SM_VEC_LEN)  # raises if S is illegal
        gqa = self.Hq // self.head_groups
        used = worker_l1_footprint_bytes(self.HD, gqa, self.weight_depth, self.stack_size)
        if used > L1_BYTES:
            raise ValueError(
                f"L1 use {used} B exceeds {L1_BYTES} B at head_groups={self.head_groups} "
                f"(gqa={gqa} heads resident at once). Raise head_groups."
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "attn_global_worker",
                (aie_utils.get_current_device(), self.HD, self.Hq, self.S),
                {
                    "n_aie_columns": self.num_aie_columns,
                    "head_groups": self.head_groups,
                    "weight_depth": self.weight_depth,
                    "stack_size": self.stack_size,
                    "mask_parameter": self.mask_parameter,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        copy_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        # sc_matvec_rtk_bf16_bf16 is a __attribute__((alias)) onto the RUNTIME-K
        # matvec_rtk_bf16_bf16 (mv.cc), so DIM_K plays no role in what this op calls -- it is
        # passed only because mv.cc references it unconditionally elsewhere in the same
        # translation unit (a compile-time-K sibling this op never calls).
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.HD}k_{FLASH_SM_VEC_LEN}vs_sc.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.HD}", f"-DVEC_SIZE={FLASH_SM_VEC_LEN}",
                        "-DGEMV_ALIAS_SC"],
        )
        softmax_obj = KernelObjectArtifact(
            "softmax.o", dependencies=[SourceArtifact(kdir / arch_dir / "softmax.cc")]
        )
        tmv_obj = KernelObjectArtifact(
            f"tmv_{self.HD}n.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv_taccum.cc")],
            extra_flags=[f"-DDIM_N={self.HD}"],
        )
        deps = [copy_obj, mv_obj, softmax_obj, tmv_obj]
        deps += lut_based_ops_artifacts(arch_dir)
        return [KernelArchiveArtifact("attn_global_dp_core.a", dependencies=deps)]

    def get_arg_spec(self):
        QD = self.Hq * self.HD
        cache = self.S * self.HD
        partials = self.num_aie_columns * self.Hq * (self.HD + 2)
        return [
            AIERuntimeArgSpec("in", (QD,)),          # q, already normed and RoPE'd
            AIERuntimeArgSpec("in", (cache,)),        # kc, read-only
            AIERuntimeArgSpec("in", (cache,)),        # vc, read-only
            AIERuntimeArgSpec("out", (partials,)),    # partials, [column][head]-major
        ]


@dataclass
class AttnGlobalFlash(MLIROperator):
    """Worker cores merging over the cascade in one design, K/V length set per token by a
    scratchpad parameter. See design.py's `attn_global_flash` (q, kc, vc -> cx).
    """

    HD: int
    Hq: int
    capacity: int
    num_aie_columns: int = 8
    heads_per_core: int = 16
    block: int = 64
    rows_per_element: int = 1
    mask_parameter: str = "sm_mask"
    len_parameter: str = "gf_len"
    loop_parameter: str = "gf_loop"
    stack_size: int = 0xD00
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "heads_per_core": "hpc",
        "capacity": "cap",
        "rows_per_element": "rpe",
        "stack_size": "ss",
        "mask_parameter": "mpar",
        "len_parameter": "lpar",
        "loop_parameter": "npar",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "attn_global_flash",
                (aie_utils.get_current_device(), self.HD, self.Hq, self.capacity),
                {
                    "n_aie_columns": self.num_aie_columns,
                    "heads_per_core": self.heads_per_core,
                    "block": self.block,
                    "rows_per_element": self.rows_per_element,
                    "mask_parameter": self.mask_parameter,
                    "len_parameter": self.len_parameter,
                    "loop_parameter": self.loop_parameter,
                    "stack_size": self.stack_size,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        copy_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        # See AttnGlobalWorker.get_kernel_artifacts for why DIM_K is passed.
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.HD}k_{FLASH_SM_VEC_LEN}vs_sc.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv.cc")],
            extra_flags=[f"-DDIM_K={self.HD}", f"-DVEC_SIZE={FLASH_SM_VEC_LEN}",
                        "-DGEMV_ALIAS_SC"],
        )
        softmax_obj = KernelObjectArtifact(
            "softmax.o", dependencies=[SourceArtifact(kdir / arch_dir / "softmax.cc")]
        )
        tmv_obj = KernelObjectArtifact(
            f"tmv_{self.HD}n.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv_taccum.cc")],
            extra_flags=[f"-DDIM_N={self.HD}"],
        )
        cascade_obj = KernelObjectArtifact(
            "flash_cascade_f32.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "flash_cascade_f32.cc")],
        )
        deps = [copy_obj, mv_obj, softmax_obj, tmv_obj, cascade_obj]
        deps += lut_based_ops_artifacts(arch_dir)
        return [KernelArchiveArtifact("attn_global_dp_core.a", dependencies=deps)]

    def get_arg_spec(self):
        QD = self.Hq * self.HD
        cache = self.capacity * self.HD
        return [
            AIERuntimeArgSpec("in", (QD,)),          # q, already normed and RoPE'd
            AIERuntimeArgSpec("in", (cache,)),        # kc, read-only
            AIERuntimeArgSpec("in", (cache,)),        # vc, read-only
            AIERuntimeArgSpec("out", (QD,)),          # cx
        ]


@dataclass
class AttnGlobalMerge(MLIROperator):
    """gemma4-weightless-attention-block variant A_g, merge half: folds AttnGlobalWorker's
    per-column partials into the true softmax result. See design.py's `attn_global_merge`.
    """

    HD: int
    Hq: int
    num_aie_columns: int = 8
    stack_size: int = 0xD00
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "stack_size": "ss",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "attn_global_merge",
                (aie_utils.get_current_device(), self.HD, self.Hq),
                {
                    "n_aie_columns": self.num_aie_columns,
                    "stack_size": self.stack_size,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        tmv_obj = KernelObjectArtifact(
            f"tmv_{self.HD}n.o",
            dependencies=[SourceArtifact(kdir / "generic" / "mv_taccum.cc")],
            extra_flags=[f"-DDIM_N={self.HD}"],
        )
        softmax_obj = KernelObjectArtifact(
            "softmax.o", dependencies=[SourceArtifact(kdir / arch_dir / "softmax.cc")]
        )
        deps = [tmv_obj, softmax_obj]
        return [KernelArchiveArtifact("attn_global_merge_core.a", dependencies=deps)]

    def get_arg_spec(self):
        QD = self.Hq * self.HD
        partials = self.num_aie_columns * self.Hq * (self.HD + 2)
        return [
            AIERuntimeArgSpec("in", (partials,)),   # partials, [column][head]-major
            AIERuntimeArgSpec("out", (QD,)),         # cx
        ]
