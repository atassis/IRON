# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.common.test_utils import run_test
from iron.operators.act_poly.op import SigmoidFast, SigmoidPoly, SiLUFast, SiLUPoly


@pytest.mark.parametrize("cls,fn", [(SiLUPoly, torch.nn.functional.silu), (SigmoidPoly, torch.sigmoid),
                                    (SiLUFast, torch.nn.functional.silu), (SigmoidFast, torch.sigmoid)])
@pytest.mark.parametrize("size,cols", [(4096, 8), (8192, 8)])
def test_act_poly(cls, fn, size, cols, aie_context):
    x = (torch.randn(size, generator=torch.Generator().manual_seed(0)) * 3).to(torch.bfloat16)
    want = fn(x.double()).to(torch.bfloat16)          # correctly rounded
    op = cls(size=size, num_aie_columns=cols, tile_size=size // cols, context=aie_context)
    # one bf16 ulp (2^-8 relative) is all the rounding of a correct result can explain
    errors, latency_us, _ = run_test(op, {"x": x}, {"y": want}, rel_tol=2 ** -7, abs_tol=1e-6)
    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"Test failed with errors: {errors}"
