# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.common.test_utils import run_test
from iron.operators.dequant_rows.op import DequantRows
from iron.operators.dequant_rows.reference import generate_golden_reference


@pytest.mark.parametrize("rows,K,out_stride,out_col", [
    (256, 2560, None, 0),      # Qwen3.5-4B in_proj / o_proj rows
    (2560, 4608, 9216, 4608),  # the second K-chunk of its down projection, into the full row
])
def test_dequant_rows(rows, K, out_stride, out_col, aie_context):
    g = generate_golden_reference(rows, K, out_stride=out_stride, out_col=out_col)
    op = DequantRows(rows=rows, K=K, out_stride=out_stride, out_col=out_col, context=aie_context)
    errors, latency_us, _ = run_test(op, {"a": g["a"]}, {"c": g["c"]}, rel_tol=0, abs_tol=0)
    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"Test failed with errors: {errors}"
