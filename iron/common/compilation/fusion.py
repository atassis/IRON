# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Temporal fusion of multiple MLIR modules into one module with multiple devices and a main runtime sequence that calls into them.
"""

from __future__ import annotations

import numpy as np
import importlib.util
from functools import partial
from pathlib import Path
from aie import ir
from aie.dialects import aie, aiex, memref
from aie.extras.context import mlir_mod_ctx
import ml_dtypes

from typing import Any

from . import (
    CompilationArtifact,
    CompilationArtifactGraph,
    CompilationRule,
    CompilationCommand,
    PythonCallbackCompilationCommand,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)

# Compilation Artifacts
# ##########################################################################


class FusedMLIRSource(CompilationArtifact):
    def __init__(
        self,
        filename: str,
        operator_mlir_map: dict[str, PythonGeneratedMLIRArtifact],
        runlist: list[tuple[str, ...]],
        subbuffer_layout: dict[str, tuple[str, int, int]],
        buffer_sizes: tuple[int, int, int],
        slice_info: dict[str, tuple[str, int, int]] | None = None,
    ) -> None:
        dependencies = list(operator_mlir_map.values())
        super().__init__(filename, dependencies)
        self.operator_mlir_map = operator_mlir_map
        self.runlist = runlist
        self.subbuffer_layout = subbuffer_layout
        self.buffer_sizes = buffer_sizes
        self.slice_info = slice_info or {}


# Helper Functions
# ##########################################################################


def extract_runtime_sequence_arg_types(dev_op: Any) -> list[Any]:
    """MLIR helper: Extract argument types from a device operation's runtime sequence."""
    for nested_op in dev_op.body_region.blocks[0].operations:
        op_name = nested_op.operation.name
        if op_name == "aie.runtime_sequence":
            if hasattr(nested_op, "body") and hasattr(nested_op.body, "blocks"):
                if len(nested_op.body.blocks) > 0:
                    entry_block = nested_op.body.blocks[0]
                    arg_types = [
                        entry_block.arguments[i].type
                        for i in range(len(entry_block.arguments))
                    ]
                    return arg_types
    raise RuntimeError("Could not find runtime sequence in device operation")


def get_child_mlir_module(mlir_artifact: PythonGeneratedMLIRArtifact) -> Any:
    """Extract MLIR module from a PythonGeneratedMLIRArtifact.

    Uses the artifact's DesignGenerator to dynamically import the design
    module and call the callback, returning the raw (non-stringified) MLIR
    module object for further inspection by the fusion pass.
    """
    if not isinstance(mlir_artifact, PythonGeneratedMLIRArtifact):
        raise TypeError(
            f"Expected PythonGeneratedMLIRArtifact, got {type(mlir_artifact).__name__}"
        )
    gen = mlir_artifact.generator
    spec = importlib.util.spec_from_file_location(gen.source_path.name, gen.source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    callback_function = getattr(module, gen.fn_name)
    return callback_function(*gen.args, **gen.kwargs)


def fuse_mlir(artifact: FusedMLIRSource) -> None:
    """Fuse multiple MLIR modules by inlining their device operations and adding a new main device and runtime sequence that call into sequence of operations based on a runlist."""

    input_buffer_size, output_buffer_size, scratch_buffer_size = artifact.buffer_sizes

    # Extract device operations from each operator's MLIR artifact
    device_mlir_strings = {}
    device_ty = None
    sequence_arg_types = {}
    # Deep-C: module-scope `aiex.scratchpad_parameter` decls are device-global and live OUTSIDE the
    # aie.device op, so the device-only stitch below would drop them (the read_scratchpad_parameter /
    # sync preamble / dma_bd offset_parameter that USE them live inside the device and survive the
    # round-trip + RunOp inlining — only the decl is lost). Collect the unique decls here and re-emit
    # them at the fused module scope. Dedup by symbol so operators sharing a parameter name (e.g. one
    # KV-write offset shared by every layer's strided-copy) collapse to a single decl.
    scratchpad_decls = {}  # sym_name -> decl asm string
    for op_name, mlir_artifact in artifact.operator_mlir_map.items():
        mlir_module = get_child_mlir_module(mlir_artifact)
        device_ops = [
            op for op in mlir_module.body.operations if isinstance(op, aie.DeviceOp)
        ]
        if len(device_ops) != 1:
            raise ValueError(
                f"Expected exactly one device operation in MLIR artifact for operator '{op_name}', "
                f"got {len(device_ops)}"
            )
        device_op = device_ops[0]
        if device_ty is None:
            device_ty = device_op.device
        device_mlir_strings[op_name] = str(device_op)
        sequence_arg_types[op_name] = extract_runtime_sequence_arg_types(device_op)
        for op in mlir_module.body.operations:
            if op.operation.name == "aiex.scratchpad_parameter":
                sym = ir.StringAttr(op.operation.attributes["sym_name"]).value
                decl = str(op.operation)
                if scratchpad_decls.setdefault(sym, decl) != decl:
                    raise ValueError(
                        f"Conflicting scratchpad_parameter decls for symbol '{sym}': "
                        f"{scratchpad_decls[sym]!r} vs {decl!r}"
                    )

    # Build fused MLIR module
    with mlir_mod_ctx() as ctx:

        # Deep-C: re-emit the hoisted module-scope scratchpad_parameter decls FIRST (before the
        # devices), matching the module layout the single-op scratchpad designs use.
        for sym, decl_str in scratchpad_decls.items():
            ctx.module.body.append(ir.Operation.parse(decl_str))

        # Concatenate aie.device ops. A device that reads/uses scratchpad parameters fails standalone
        # verification (aie.DeviceOp.parse) because read_scratchpad_parameter's verifier requires the
        # module-scope decl in scope; parse it inside a decls-wrapper module so the decl is visible,
        # then move the (renamed) device into the fused module which carries the same decls.
        for op_name, device_str in device_mlir_strings.items():
            if scratchpad_decls:
                wrapper = (
                    "module {\n"
                    + "\n".join(scratchpad_decls.values())
                    + "\n"
                    + device_str
                    + "\n}"
                )
                wmod = ir.Module.parse(wrapper)
                dev_op = next(
                    o
                    for o in wmod.body.operations
                    if isinstance(o, aie.DeviceOp)
                )
                dev_op.operation.detach_from_parent()
            else:
                dev_op = aie.DeviceOp.parse(device_str)
            dev_op.sym_name = ir.StringAttr.get(op_name)
            ctx.module.body.append(dev_op)

        # Create the main device -- this contains the runtime sequence calling into the other devices
        @aie.device(device_ty)
        def main():
            buf_dtype = np.dtype[
                ml_dtypes.bfloat16
            ]  # TODO: support for other data types
            itemsize = np.dtype(ml_dtypes.bfloat16).itemsize

            # RuntimeSequenceOp
            @aiex.runtime_sequence(
                np.ndarray[(input_buffer_size // itemsize,), buf_dtype],
                np.ndarray[(output_buffer_size // itemsize,), buf_dtype],
                np.ndarray[(scratch_buffer_size // itemsize,), buf_dtype],
            )
            def sequence(input_buf, output_buf, scratch_buf):
                consolidated_buffers = {
                    "input": input_buf,
                    "output": output_buf,
                    "scratch": scratch_buf,
                }

                # Execute operations in runlist order
                configure_op = None
                last_op_name = None
                for op_name, *buffer_names in artifact.runlist:
                    expected_arg_types = sequence_arg_types[op_name]

                    # Avoid reconfiguring altogether if the same op is called multiple times consecutively
                    if configure_op is None or op_name != last_op_name:
                        # Configure Op
                        configure_sym_ref_attr = ir.FlatSymbolRefAttr.get(op_name)
                        configure_op = aiex.ConfigureOp(
                            configure_sym_ref_attr
                        )  # TODO: optimization -- if previous op was in the same device, skip reconfiguration
                        configure_body = configure_op.body.blocks.append()
                        last_op_name = op_name

                    with ir.InsertionPoint(configure_body):

                        # For each buffer, add subview and reinterpret_cast ops
                        buffer_ssa_values = []
                        for idx, buf_name in enumerate(buffer_names):
                            # Check if this is a sliced buffer
                            if buf_name in artifact.slice_info:
                                base_name, start, end = artifact.slice_info[buf_name]
                                # Get parent buffer info
                                buf_type, parent_offset, parent_length = (
                                    artifact.subbuffer_layout[base_name]
                                )
                                # Calculate actual offset and length for slice
                                offset = parent_offset + start
                                length = end - start
                            else:
                                # Regular buffer
                                buf_type, offset, length = artifact.subbuffer_layout[
                                    buf_name
                                ]

                            # Subview Op
                            consolidated_buf = consolidated_buffers[buf_type]
                            offset_elements = offset // itemsize
                            size_elements = length // itemsize
                            subview = memref.subview(
                                consolidated_buf,
                                [offset_elements],
                                [size_elements],
                                [1],
                            )

                            # Reinterpret_cast Op
                            target_type = expected_arg_types[idx]
                            expected_memref = ir.MemRefType(target_type)
                            target_shape = [
                                expected_memref.shape[i]
                                for i in range(expected_memref.rank)
                            ]
                            expected_size = np.prod(target_shape)
                            assert (
                                expected_size == size_elements
                            ), f"Size mismatch for buffer '{buf_name}': MLIR runtime sequence expected {expected_size}, Python fused operator provided {size_elements}"
                            strides = []
                            stride = 1
                            for dim in reversed(target_shape):
                                strides.insert(0, stride)
                                stride *= dim
                            result_type = ir.MemRefType.get(
                                target_shape, ir.BF16Type.get()
                            )
                            reinterpreted = memref.reinterpret_cast(
                                result=result_type,
                                source=subview,
                                offsets=[],
                                sizes=[],
                                strides=[],
                                static_offsets=[0],
                                static_sizes=target_shape,
                                static_strides=strides,
                            )
                            buffer_ssa_values.append(reinterpreted)

                        # Run Op
                        sequence_sym_ref_attr = ir.FlatSymbolRefAttr.get("sequence")
                        run_op = aiex.RunOp(sequence_sym_ref_attr, buffer_ssa_values)

        # Write the fused MLIR to file
        with open(artifact.filename, "w") as f:
            f.write(str(ctx.module))


# Compilation Rules
# ##########################################################################


class FusePythonGeneratedMLIRCompilationRule(CompilationRule):
    """Compilation rule that fuses multiple MLIR modules into one."""

    def matches(self, graph: CompilationArtifactGraph) -> bool:
        return any(graph.get_worklist(FusedMLIRSource))

    def compile(self, graph: CompilationArtifactGraph) -> list[CompilationCommand]:
        commands: list[CompilationCommand] = []
        worklist = graph.get_worklist(FusedMLIRSource)
        for artifact in worklist:
            callback = partial(fuse_mlir, artifact)
            commands.append(PythonCallbackCompilationCommand(callback))
            new_artifact = SourceArtifact(artifact.filename)
            new_artifact.available = True
            graph.replace(artifact, new_artifact)
        return commands
