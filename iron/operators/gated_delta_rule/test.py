# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from iron.common.test_utils import run_test
from iron.operators.gated_delta_rule.op import GatedDeltaRule
from iron.operators.gated_delta_rule.reference import generate_golden_reference


@pytest.mark.parametrize("limbs", [False, True])
@pytest.mark.parametrize("v_heads,k_heads,num_aie_columns,tokens,l2_qk,count", [
    (16, 8, 2, 1, False, None),
    (32, 16, 8, 1, False, None),  # Qwen3.5-4B: 32 value heads over 16 k-heads, dk = dv = 128
    (16, 8, 2, 4, False, None),
    (32, 16, 8, 16, False, None),
    (32, 16, 8, 16, True, None),
    (32, 16, 8, 16, True, 5),     # a padded chunk: 5 real tokens of 16
    (32, 16, 8, 16, True, 0),     # an all-pad call leaves the state as it was
])
def test_gated_delta_rule(v_heads, k_heads, num_aie_columns, tokens, l2_qk, count, limbs,
                          aie_context):
    if limbs and tokens > 1 and count is None:
        pytest.skip("with tokens > 1 the limb path is the counted entry's")
    g = generate_golden_reference(v_heads=v_heads, k_heads=k_heads, tokens=tokens, l2_qk=l2_qk,
                                  count=count)
    op = GatedDeltaRule(v_heads=v_heads, k_heads=k_heads, dk=128, dv=128, ab_len=g["ab_len"],
                        ab_off=0, mixed_len=g["mixed_len"], q_off=g["q_off"], k_off=g["k_off"],
                        v_off=g["v_off"], num_aie_columns=num_aie_columns, tokens=tokens,
                        l2_qk=l2_qk, counted=count is not None, limbs=limbs or None,
                        context=aie_context)
    ins = {"ab": g["ab"], "params": g["params"], **({"count": g["count"]} if count is not None else {}),
           "mixed": g["mixed"], "s_in": g["s_in"]}
    errors, latency_us, _ = run_test(
        op, ins,
        {"s_out": g["s_out"], "o": g["o"]}, rel_tol=2e-3, abs_tol=1e-3)
    print(f"\nLatency (us): {latency_us:.1f}")
    assert not errors, f"Test failed with errors: {errors}"
