# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw kernel objects are scoped to ONE compile() call, not locked/reused.

An earlier version of this fix (387612d) let _compile_raw_kernel_object skip
recompiling when kobj_<key>.o already existed, guarded by a cross-process
file lock. That reintroduced the failure the dedup key can't rule out on its
own: the key covers only what the compile command varies on WITHIN one
compile() call (source bytes, flags, target arch, include dirs, Chess vs
Peano) -- not the #include closure or compiler identity -- so a build dir
that persists across builds and toolchain re-pins (e.g. $XDNA_CACHE's
llm-build/<spec>) could silently link an object compiled by an older
toolchain or before a header edit.

The fix instead gives every compile() call its own kobj/<run_id>/ directory
(run_id = f"{pid}-{uuid4().hex[:12]}"), always compiles fresh into it, and
removes the directory once the copy stage is done (_ScopedRawCompileStage).
Two builds -- concurrent, or the same script run twice -- then never share a
raw path at all, so there is nothing to lock and nothing stale to read.
"""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aie.utils as aie_utils
from aie.iron.device import NPU2

import iron.common.compilation.base as base_mod
from iron.common.compilation.base import (
    CompilationArtifactGraph,
    KernelCompilationRule,
    KernelObjectArtifact,
    SourceArtifact,
)
from iron.common.device_utils import get_kernel_dir


def _kernel_compilation_rule(monkeypatch):
    aie_utils.set_current_device(NPU2())
    monkeypatch.setattr(base_mod, "compile_cxx_core_function", lambda **kwargs: None)
    return KernelCompilationRule(peano_dir="/peano", mlir_aie_dir="/mlir")


def test_a_stale_raw_object_left_in_kernel_dir_is_never_used(monkeypatch, tmp_path):
    """A crashed build (or a build from before this fix) can leave a
    kobj_<key>.o sitting directly under kernel_dir, outside any run_id
    directory. The new design must never resolve a fresh compile() call to
    that path."""
    rule = _kernel_compilation_rule(monkeypatch)
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(src_file), available=True)
    a1 = KernelObjectArtifact(str(tmp_path / "op0_add.o"), dependencies=[src])

    kernel_dir = get_kernel_dir()
    include_dirs = [str(Path("/mlir") / "aie_runtime_lib" / kernel_dir.upper())]
    key = base_mod._kernel_object_key(
        source_path=str(src_file),
        compile_args=["-Wno-missing-template-arg-list-after-template-kw"],
        target_arch=kernel_dir,
        include_dirs=include_dirs,
        use_chess=False,
    )
    stale = tmp_path / f"kobj_{key}.o"  # the old, pre-run_id flat path
    stale.write_bytes(b"STALE")

    monkeypatch.setattr(
        base_mod,
        "compile_cxx_core_function",
        lambda *, output_path, **kwargs: Path(output_path).write_bytes(b"FRESH"),
    )

    graph = CompilationArtifactGraph([a1])
    commands = rule.compile(graph)
    assert len(commands) == 1
    assert commands[0].run()

    assert Path(a1.filename).read_bytes() == b"FRESH"
    assert stale.read_bytes() == b"STALE"  # never touched


def test_concurrent_compile_calls_never_share_a_raw_path(monkeypatch, tmp_path):
    """Two builds racing on the same key AND the same kernel_dir (the
    build_prefill.sh pattern: a fixed WORK dir, no per-build lock) must
    never read or write each other's raw object."""
    aie_utils.set_current_device(NPU2())
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")

    lock = threading.Lock()
    raw_paths_used = []

    def fake_compile(*, output_path, **kwargs):
        with lock:
            raw_paths_used.append(output_path)
        # Content deterministically tied to THIS call's own raw path, so a
        # final artifact's bytes can be traced back to the raw path that
        # produced them without any other cross-thread bookkeeping.
        Path(output_path).write_bytes(hashlib.sha256(output_path.encode()).digest())

    monkeypatch.setattr(base_mod, "compile_cxx_core_function", fake_compile)

    def one_build(idx):
        rule = KernelCompilationRule(peano_dir="/peano", mlir_aie_dir="/mlir")
        src = SourceArtifact(str(src_file), available=True)
        artifact = KernelObjectArtifact(
            str(tmp_path / f"op0_add_{idx}.o"), dependencies=[src]
        )
        graph = CompilationArtifactGraph([artifact])
        commands = rule.compile(graph)
        assert len(commands) == 1
        assert commands[0].run()
        return artifact

    with ThreadPoolExecutor(max_workers=2) as pool:
        artifacts = list(pool.map(one_build, [0, 1]))

    assert len(raw_paths_used) == 2
    assert raw_paths_used[0] != raw_paths_used[1], "raw paths must never collide"

    contents = [Path(a.filename).read_bytes() for a in artifacts]
    assert contents[0] != contents[1], "each artifact must trace back to its OWN raw compile"
    expected = {hashlib.sha256(p.encode()).digest() for p in raw_paths_used}
    assert set(contents) == expected


def test_scratch_dir_is_removed_after_success_and_after_failure(monkeypatch, tmp_path):
    rule = _kernel_compilation_rule(monkeypatch)
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(src_file), available=True)

    monkeypatch.setattr(
        base_mod,
        "compile_cxx_core_function",
        lambda *, output_path, **kwargs: Path(output_path).write_bytes(b"OK"),
    )
    a1 = KernelObjectArtifact(str(tmp_path / "op0_add.o"), dependencies=[src])
    graph = CompilationArtifactGraph([a1])
    commands = rule.compile(graph)
    assert commands[0].run()
    assert not (tmp_path / "kobj").exists() or not any((tmp_path / "kobj").iterdir()), (
        "the per-call kobj/<run_id>/ directory must be gone after success"
    )

    def failing_compile(**kwargs):
        raise RuntimeError("compiler crashed")

    monkeypatch.setattr(base_mod, "compile_cxx_core_function", failing_compile)
    a2 = KernelObjectArtifact(str(tmp_path / "op1_add.o"), dependencies=[src])
    graph2 = CompilationArtifactGraph([a2])
    commands2 = rule.compile(graph2)

    import pytest

    with pytest.raises(RuntimeError):
        commands2[0].run()

    assert not (tmp_path / "kobj").exists() or not any((tmp_path / "kobj").iterdir()), (
        "the per-call kobj/<run_id>/ directory must be gone after a failure too"
    )
