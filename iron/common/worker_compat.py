# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from typing import Any

from aie.iron import Worker


def create_worker(
    core_fn: Callable | None,
    *args: Any,
    allocation_scheme: Any | None = None,
    **kwargs: Any,
) -> Worker:
    if allocation_scheme is not None:
        if "allocation_scheme" not in inspect.signature(Worker).parameters:
            raise NotImplementedError(
                "The bound aie.iron.Worker API does not support allocation_scheme"
            )
        kwargs["allocation_scheme"] = allocation_scheme
    return Worker(core_fn, *args, **kwargs)
