# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-free build gate for AttnCore.

No NPU on this box for this task: this test only checks that aiecc PLACES and BUILDS the fused
design (kernel compile -> MLIR -> xclbin), never that it runs correctly on hardware. Correctness
against a reference is out of scope here (see design.py's module docstring).
"""

import re

import aie.utils as aie_utils
from aie.iron.device import NPU2

from iron.common import AIEContext
from iron.operators.attn_core.op import AttnCore


def test_attn_core_builds(tmp_path):
    aie_utils.set_current_device(NPU2())
    ctx = AIEContext(build_dir=tmp_path)
    op = AttnCore(context=ctx)
    op.compile()

    mlir_text = op.get_mlir_artifact().filename
    mlir_path = ctx.build_dir / mlir_text
    assert mlir_path.exists(), f"no MLIR emitted at {mlir_path}"
    text = mlir_path.read_text()
    n_devices = len(re.findall(r"aie\.device", text))
    assert n_devices == 1, f"expected exactly 1 aie.device, found {n_devices}"

    xclbin_path = ctx.build_dir / op.xclbin_artifact.filename
    assert xclbin_path.exists(), f"no xclbin built at {xclbin_path}"
