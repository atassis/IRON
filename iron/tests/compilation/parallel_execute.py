# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ParallelCompilationCommand: independent kernel-compile chains run concurrently.

execute() ran every CompilationCommand from a rule's plan step one at a time
(base.py execute(), a serial `for command in commands: command.run()`); a
cold gemma3-270m build spent 13.6 s of a 20 s build in 16
compile_cxx_core_function calls on the main thread. Kernel object compiles of
one design have no edges between them, so they can run in a thread pool sized
by IRON_COMPILE_JOBS. A rename/prefix-symbols step touches the very file its
own compile just produced, so it must stay ordered after it -- these tests
check both: independent chains overlap, and a dependent command never runs
before its input.
"""

import os
import time

import pytest

import aie.utils as aie_utils
from aie.iron.device import NPU2

import iron.common.compilation.base as base_mod
from iron.common.compilation.base import (
    CompilationArtifactGraph,
    CompilationCommand,
    KernelCompilationRule,
    KernelObjectArtifact,
    ParallelCompilationCommand,
    SourceArtifact,
)


class _Record(CompilationCommand):
    """Sleeps, then appends (label, start, end) to a shared list."""

    def __init__(self, label, seconds, log):
        self.label = label
        self.seconds = seconds
        self.log = log

    def run(self) -> bool:
        start = time.monotonic()
        time.sleep(self.seconds)
        end = time.monotonic()
        self.log.append((self.label, start, end))
        return True

    def __repr__(self) -> str:
        return f"Record({self.label})"


class _Fail(CompilationCommand):
    def run(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Fail()"


def test_independent_chains_overlap_under_iron_compile_jobs(monkeypatch):
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    log = []
    chains = [[_Record(i, 0.5, log)] for i in range(4)]
    start = time.monotonic()
    ParallelCompilationCommand(chains).run()
    wall = time.monotonic() - start
    assert wall < 0.5 * 4 / 2
    assert len(log) == 4


def test_iron_compile_jobs_1_runs_serially_in_submission_order(monkeypatch):
    monkeypatch.setenv("IRON_COMPILE_JOBS", "1")
    log = []
    chains = [[_Record(i, 0.05, log)] for i in range(3)]
    ParallelCompilationCommand(chains).run()
    assert [label for label, _, _ in log] == [0, 1, 2]
    # Serial: each command's start is at or after the previous one's end.
    for (_, _, prev_end), (_, next_start, _) in zip(log, log[1:]):
        assert next_start >= prev_end


def test_a_dependent_command_runs_after_its_own_chains_input(monkeypatch):
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    log = []
    # Chain 0: compile then a dependent (e.g. prefix-symbols objcopy) that
    # touches the file the compile just produced -- must run after it.
    dependent_chain = [_Record("compile", 0.3, log), _Record("prefix", 0.0, log)]
    other_chain = [_Record("other", 0.0, log)]
    ParallelCompilationCommand([dependent_chain, other_chain]).run()
    compile_end = next(end for label, _, end in log if label == "compile")
    prefix_start = next(start for label, start, _ in log if label == "prefix")
    assert prefix_start >= compile_end


def test_first_failure_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("IRON_COMPILE_JOBS", "2")
    with pytest.raises(RuntimeError):
        ParallelCompilationCommand([[_Fail()]]).run()


def test_force_serial_ignores_iron_compile_jobs(monkeypatch):
    """Chess's xchesscc_wrapper is not verified safe under a shared cwd, so
    KernelCompilationRule forces force_serial=True when use_chess is set."""
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    log = []
    chains = [[_Record(i, 0.05, log)] for i in range(3)]
    ParallelCompilationCommand(chains, force_serial=True).run()
    assert [label for label, _, _ in log] == [0, 1, 2]


def test_default_jobs_is_half_cpu_count(monkeypatch):
    monkeypatch.delenv("IRON_COMPILE_JOBS", raising=False)
    from iron.common.compilation.base import _compile_jobs

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert _compile_jobs() == 4


def _kernel_compilation_rule(monkeypatch):
    """A KernelCompilationRule whose Peano compile is a no-op, so .compile()
    can be exercised without a real toolchain."""
    aie_utils.set_current_device(NPU2())
    monkeypatch.setattr(
        base_mod, "compile_cxx_core_function", lambda **kwargs: None
    )
    return KernelCompilationRule(peano_dir="/peano", mlir_aie_dir="/mlir")


def test_two_artifacts_with_the_same_filename_share_one_chain(monkeypatch, tmp_path):
    """SeparateDispatch (iron/common/sequence.py) gives two operators of the
    same class no per-operator filename prefix, so operator_bases.py's
    get_kernel_artifacts() can produce two distinct KernelObjectArtifact
    objects that both resolve to e.g. "add.o". Grouping chains by object
    identity (the old behaviour) would compile and objcopy the same file from
    two threads at once; grouping by filename must instead serialize them
    into one chain.

    KernelCompilationRule.compile() now returns ONE _ScopedRawCompileStage
    wrapping [raw_stage, copy_stage] -- one raw compile per unique (source,
    flags, arch) key, one copy-then-rename chain per output filename
    (kernel_object_dedup.py covers the dedup itself); this test only
    re-checks the same-filename invariant on the copy stage.
    """
    rule = _kernel_compilation_rule(monkeypatch)
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    (tmp_path / "add.cc").write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(tmp_path / "add.cc"), available=True)
    a1 = KernelObjectArtifact(str(tmp_path / "add.o"), dependencies=[src])
    a2 = KernelObjectArtifact(str(tmp_path / "add.o"), dependencies=[src])
    assert a1 is not a2 and a1.filename == a2.filename

    graph = CompilationArtifactGraph([a1, a2])
    commands = rule.compile(graph)

    assert len(commands) == 1
    raw, copy = commands[0].raw_stage, commands[0].copy_stage
    assert isinstance(raw, ParallelCompilationCommand)
    assert isinstance(copy, ParallelCompilationCommand)
    assert len(copy.chains) == 1, (
        "two artifacts with the same output path must land in ONE copy chain, "
        f"got {len(copy.chains)}"
    )
    assert len(copy.chains[0]) == 2


def test_two_artifacts_with_distinct_filenames_get_separate_chains(
    monkeypatch, tmp_path
):
    rule = _kernel_compilation_rule(monkeypatch)
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    (tmp_path / "add.cc").write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(tmp_path / "add.cc"), available=True)
    a1 = KernelObjectArtifact(str(tmp_path / "add.o"), dependencies=[src])
    a2 = KernelObjectArtifact(str(tmp_path / "mul.o"), dependencies=[src])

    graph = CompilationArtifactGraph([a1, a2])
    commands = rule.compile(graph)

    copy = commands[0].copy_stage
    assert len(copy.chains) == 2
