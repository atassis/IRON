# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.common.test_utils import run_test
from iron.operators.conv1d_step.op import Conv1dStep
from iron.operators.conv1d_step.reference import generate_golden_reference


@pytest.mark.parametrize(
    "channels,num_aie_columns,tile_channels,tokens",
    [(1024, 4, 256, 1), (8192, 8, 256, 1),  # 8192 is Qwen3.5-4B's mixed q|k|v width
     (1024, 4, 256, 16), (8192, 8, 256, 16)],
)
def test_conv1d_step(channels, num_aie_columns, tile_channels, tokens, aie_context):
    golden = generate_golden_reference(channels, tokens=tokens)
    op = Conv1dStep(channels=channels, num_aie_columns=num_aie_columns,
                    tile_channels=tile_channels, tokens=tokens, context=aie_context)
    errors, latency_us, _ = run_test(
        op, {"window": golden["window"], "w": golden["w"]}, {"out": golden["out"]},
        rel_tol=0.01, abs_tol=1e-2)
    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"Test failed with errors: {errors}"
