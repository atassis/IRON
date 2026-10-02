# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aie.iron import Worker
from iron.common.worker_compat import create_worker


def test_create_worker_omits_none_allocation_scheme_for_the_bound_iron_api():
    assert isinstance(create_worker(None, allocation_scheme=None), Worker)


def test_create_worker_rejects_non_none_allocation_scheme_when_bound_iron_lacks_it():
    with pytest.raises(NotImplementedError, match="allocation_scheme"):
        create_worker(None, allocation_scheme="basic-sequential")
