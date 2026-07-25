# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conformance test: the upstream host-runtime Tensor provides ``subview``.

This package dropped its former ``XRTSubBuffer`` fork and now relies on the
upstream ``Tensor.subview`` primitive (via the ``mlir_aie`` wheel). This test
guards against a stale wheel pin silently regressing that primitive: if
``subview`` is missing or misbehaves, it fails loudly here instead of the
arena runner silently re-forking behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor


def test_xrttensor_has_subview():
    # Attribute-level guard: catches an old mlir_aie wheel without the primitive.
    assert hasattr(XRTTensor, "subview"), (
        "mlir_aie wheel is too old: XRTTensor.subview is missing. Bump the "
        "mlir_aie pin to a build that includes the host-runtime subview primitive."
    )


def test_subview_shares_storage_and_syncs_own_slice():
    import ml_dtypes

    parent = XRTTensor((4, 8), dtype=ml_dtypes.bfloat16)  # 32 elements
    view = parent.subview(8, (8,), dtype=ml_dtypes.bfloat16)  # elements [8:16] == row 1
    assert view.shape == (8,)
    # Write through the view, push only its slice to the device, then read the
    # whole parent back: the write lands in row 1 and row 0 is untouched.
    view.torch_view()[:] = 3.0
    view.to("npu")
    parent.to("cpu")
    got = parent.numpy().astype(np.float32)
    assert np.allclose(got[1], 3.0)
    assert np.allclose(got[0], 0.0)


def test_subview_bounds_checked():
    parent = XRTTensor((4,), dtype=np.float32)
    with pytest.raises(ValueError):
        parent.subview(2, (4,), dtype=np.float32)  # 2 + 4 == 6 elements > 4
