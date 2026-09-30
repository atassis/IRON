# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KernelCompilationRule compiles each unique (source, flags, arch) kernel once.

Measured on the served qwen3-0.6b decode: 83 kernel compiles for 14 unique
(source, flags) pairs -- FusedDispatch gives every operator instance its own
output filename (``op{idx}_<name>.o``), so the old filename-keyed grouping in
KernelCompilationRule.compile() never saw the duplication: two operators of
the same kernel with the same flags each got their own
``compile_cxx_core_function`` call, differing only in the ``op{idx}_`` symbol
prefix objcopy applies afterwards. These tests check that identical
(source path, source bytes, compile_args, target_arch, include_dirs,
use_chess) artifacts share ONE raw compile, copied into each artifact's own
filename before that artifact's own rename/prefix steps run -- and that a
flag or source difference still gets its own compile.
"""

from pathlib import Path

import aie.utils as aie_utils
from aie.iron.device import NPU2

import iron.common.compilation.base as base_mod
from iron.common.compilation.base import (
    CompilationArtifactGraph,
    KernelCompilationRule,
    KernelObjectArtifact,
    ParallelCompilationCommand,
    SourceArtifact,
    _ScopedRawCompileStage,
)


def _kernel_compilation_rule(monkeypatch):
    """A KernelCompilationRule whose Peano compile is a no-op, so .compile()
    can be exercised without a real toolchain."""
    aie_utils.set_current_device(NPU2())
    monkeypatch.setattr(base_mod, "compile_cxx_core_function", lambda **kwargs: None)
    return KernelCompilationRule(peano_dir="/peano", mlir_aie_dir="/mlir")


def _raw_stage(commands):
    assert len(commands) == 1, (
        f"compile() must return one _ScopedRawCompileStage, got {len(commands)}"
    )
    scoped = commands[0]
    assert isinstance(scoped, _ScopedRawCompileStage)
    assert isinstance(scoped.raw_stage, ParallelCompilationCommand)
    assert isinstance(scoped.copy_stage, ParallelCompilationCommand)
    return scoped.raw_stage, scoped.copy_stage


def test_same_source_same_flags_different_prefix_share_one_compile(
    monkeypatch, tmp_path
):
    rule = _kernel_compilation_rule(monkeypatch)
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(src_file), available=True)

    a1 = KernelObjectArtifact(
        str(tmp_path / "op0_add.o"), dependencies=[src], prefix_symbols="op0_"
    )
    a2 = KernelObjectArtifact(
        str(tmp_path / "op1_add.o"), dependencies=[src], prefix_symbols="op1_"
    )

    graph = CompilationArtifactGraph([a1, a2])
    commands = rule.compile(graph)

    raw, copy = _raw_stage(commands)
    assert len(raw.chains) == 1, (
        "two artifacts with the same source and flags must compile ONCE, "
        f"got {len(raw.chains)} raw compiles"
    )
    assert len(raw.chains[0]) == 1  # one compile command

    assert len(copy.chains) == 2, "distinct output filenames get independent copy chains"
    for chain in copy.chains:
        # copy-from-raw, then the objcopy prefix-rename step (nm + redefine-syms).
        assert len(chain) == 3


def test_different_flags_get_separate_compiles(monkeypatch, tmp_path):
    rule = _kernel_compilation_rule(monkeypatch)
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(src_file), available=True)

    a1 = KernelObjectArtifact(
        str(tmp_path / "op0_add.o"), dependencies=[src], extra_flags=["-DDIM_K=64"]
    )
    a2 = KernelObjectArtifact(
        str(tmp_path / "op1_add.o"), dependencies=[src], extra_flags=["-DDIM_K=128"]
    )

    graph = CompilationArtifactGraph([a1, a2])
    commands = rule.compile(graph)

    raw, copy = _raw_stage(commands)
    assert len(raw.chains) == 2, (
        "different extra_flags must not share a raw compile, "
        f"got {len(raw.chains)}"
    )


def test_different_source_files_get_separate_compiles(monkeypatch, tmp_path):
    """Guards the key against trusting flags/arch alone: two different source
    files with identical extra_flags must not share a raw compile."""
    rule = _kernel_compilation_rule(monkeypatch)
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")

    src1_file = tmp_path / "add.cc"
    src1_file.write_text("int f() { return 1; }\n")
    src1 = SourceArtifact(str(src1_file), available=True)
    a1 = KernelObjectArtifact(str(tmp_path / "op0_add.o"), dependencies=[src1])

    src2_file = tmp_path / "mul.cc"
    src2_file.write_text("int f() { return 2; }\n")
    src2 = SourceArtifact(str(src2_file), available=True)
    a2 = KernelObjectArtifact(str(tmp_path / "op1_mul.o"), dependencies=[src2])

    graph = CompilationArtifactGraph([a1, a2])
    commands = rule.compile(graph)

    raw, copy = _raw_stage(commands)
    assert len(raw.chains) == 2, "different source files must not share a raw compile"


def test_raw_compile_stage_runs_before_any_copy_chain(monkeypatch, tmp_path):
    """_ScopedRawCompileStage.run() must finish the raw compile before the
    copy chain reads it -- copy() reads bytes the compile just wrote. Runs
    the real returned command end to end with a fake compiler that writes a
    marker, so a reordering (or the two stages folded into one
    ParallelCompilationCommand, which could interleave them) shows up as
    wrong bytes rather than only as a structural assertion."""
    rule = _kernel_compilation_rule(monkeypatch)
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(src_file), available=True)
    a1 = KernelObjectArtifact(str(tmp_path / "op0_add.o"), dependencies=[src])

    monkeypatch.setattr(
        base_mod,
        "compile_cxx_core_function",
        lambda *, output_path, **kwargs: Path(output_path).write_bytes(b"COMPILED"),
    )

    graph = CompilationArtifactGraph([a1])
    commands = rule.compile(graph)
    assert len(commands) == 1
    assert commands[0].run()
    assert Path(a1.filename).read_bytes() == b"COMPILED"


def test_two_artifacts_with_the_same_filename_still_share_one_copy_chain(
    monkeypatch, tmp_path
):
    """Preserves the pre-existing invariant (see parallel_execute.py): two
    artifacts resolving to the same output path must never copy/objcopy it
    from two threads at once."""
    rule = _kernel_compilation_rule(monkeypatch)
    monkeypatch.setenv("IRON_COMPILE_JOBS", "4")
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(src_file), available=True)
    a1 = KernelObjectArtifact(str(tmp_path / "add.o"), dependencies=[src])
    a2 = KernelObjectArtifact(str(tmp_path / "add.o"), dependencies=[src])
    assert a1 is not a2 and a1.filename == a2.filename

    graph = CompilationArtifactGraph([a1, a2])
    commands = rule.compile(graph)
    raw, copy = _raw_stage(commands)

    assert len(raw.chains) == 1  # same source+flags -> one raw compile too
    assert len(copy.chains) == 1, (
        "two artifacts with the same output path must land in ONE copy chain, "
        f"got {len(copy.chains)}"
    )
    assert len(copy.chains[0]) == 2  # two copy-from-raw steps, in order


def test_depfile_is_copied_next_to_each_artifact(monkeypatch, tmp_path):
    """compile_cxx_core_function writes the Peano ``-MF`` depfile beside its
    output object; once the compile targets the shared raw path instead of
    each artifact's own filename, the depfile has to be copied out too, or
    every artifact but one silently loses its .d file."""
    rule = _kernel_compilation_rule(monkeypatch)
    src_file = tmp_path / "add.cc"
    src_file.write_text("int f() { return 1; }\n")
    src = SourceArtifact(str(src_file), available=True)
    a1 = KernelObjectArtifact(str(tmp_path / "op0_add.o"), dependencies=[src])
    a2 = KernelObjectArtifact(str(tmp_path / "op1_add.o"), dependencies=[src])

    def fake_compile(*, output_path, **kwargs):
        Path(output_path).write_bytes(b"OBJ")
        Path(f"{output_path}.d").write_text("op0_add.o: add.cc\n")

    monkeypatch.setattr(base_mod, "compile_cxx_core_function", fake_compile)

    graph = CompilationArtifactGraph([a1, a2])
    commands = rule.compile(graph)
    for command in commands:
        assert command.run()

    for artifact in (a1, a2):
        assert Path(artifact.filename).read_bytes() == b"OBJ"
        assert Path(f"{artifact.filename}.d").read_text() == "op0_add.o: add.cc\n"
