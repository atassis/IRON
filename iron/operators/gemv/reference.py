# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from ml_dtypes import bfloat16


def reference(A, B):
    """CPU reference: matrix-vector product ``C = A @ B`` (ground truth)."""
    return A @ B


def generate_golden_reference(
    M=128, K=128, seed=42
):  # Defaults are tile-aligned minimums; tests always pass explicit values
    """
    Generate golden reference data for GEMV (General Matrix-Vector Multiplication).

    Parameters:
        M: Number of rows of matrix A
        K: Number of columns of matrix A (equals vector B length)
        seed: Random seed

    Returns:
        dict: Contains 'A' (matrix), 'B' (vector), 'C' (output vector)
    """
    torch.manual_seed(seed)

    # Generate golden inputs
    val_range = 4
    A = torch.randn(M, K, dtype=torch.bfloat16) * val_range
    B = torch.randn(K, dtype=torch.bfloat16) * val_range

    # Generate golden outputs
    C = reference(A, B)

    return {
        "A": A,
        "B": B,
        "C": C,
    }


def generate_golden_reference_batched(M=128, K=128, num_batches=2, seed=42):
    """
    Generate golden reference data for a batched GEMV (num_batches independent
    matrix-vector products stacked contiguously, matching the GEMV op layout).

    Parameters:
        M: Number of rows of each matrix A
        K: Number of columns of each matrix A (equals vector B length)
        num_batches: Number of independent GEMVs
        seed: Random seed

    Returns:
        dict: Contains 'A' (matrices), 'B' (vectors), 'C' (output vectors)
    """
    torch.manual_seed(seed)
    val_range = 4
    A = torch.randn(num_batches, M, K, dtype=torch.bfloat16) * val_range
    B = torch.randn(num_batches, K, dtype=torch.bfloat16) * val_range
    C = torch.empty(num_batches, M, dtype=torch.bfloat16)
    for b in range(num_batches):
        C[b] = A[b] @ B[b]
    return {"A": A, "B": B, "C": C}


def gelu_tanh_approx(x):
    """Tanh-approximation GELU, matching aie_kernels/aie2p/gelu.cc.

    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). Computed in float32.
    """
    xf = np.asarray(x, dtype=np.float32)
    inner = 0.79788456 * (xf + 0.044715 * xf**3)
    return 0.5 * xf * (1.0 + np.tanh(inner))


def generate_golden_reference_grouped(M=128, K=128, num_batches=4, batch_group=2, seed=42):
    """Golden for a GQA-shaped batched GEMV: `batch_group` consecutive batches share one matrix.

    A holds num_batches//batch_group matrices, B and C stay per-batch, and batch b multiplies
    A[b // batch_group] by B[b]. That is exactly what a Repeat op produces today by materialising
    A[b // batch_group] into every b -- the point of the grouped form is to skip the copy.
    """
    import torch

    assert num_batches % batch_group == 0
    torch.manual_seed(seed)
    n_matrices = num_batches // batch_group
    val_range = 1.0
    A = torch.randn(n_matrices, M, K, dtype=torch.bfloat16) * val_range
    B = torch.randn(num_batches, K, dtype=torch.bfloat16) * val_range
    C = torch.empty(num_batches, M, dtype=torch.bfloat16)
    for b in range(num_batches):
        C[b] = (A[b // batch_group].float() @ B[b].float()).to(torch.bfloat16)
    return A, B, C

def generate_golden_reference_windowed(
    M=128, K=128, alloc_M=256, num_batches=4, batch_group=1, seed=42
):
    """Golden for a NARROW WINDOW read: A is allocated with `alloc_M` rows per matrix but only the
    first `M` are computed against.

    This is decode attention's shape: the KV cache is allocated at max_seq and only n_past rows are
    live, so the per-matrix stride must stay alloc_M*K while the GEMV computes M outputs.

    Rows beyond M are POISONED, and what that catches is the PER-MATRIX STRIDE, not the window.
    A GEMV output row is a dot product of one A row, so rows past the window cannot contribute to
    the first M outputs however they are addressed -- poison there is invisible to a correct-stride
    read (verified: rel-L2 1.7e-3, bf16 noise). It becomes visible the moment the stride is wrong:
    at stride M*K instead of alloc_M*K, matrix 1 starts inside matrix 0's poisoned tail and the
    comparison goes to rel-L2 1.0e+3. So this golden needs n_matrices > 1 to have any teeth --
    with a single matrix there is no stride to get wrong.
    """
    import torch

    assert num_batches % batch_group == 0
    assert alloc_M >= M
    torch.manual_seed(seed)
    n_matrices = num_batches // batch_group
    A = torch.randn(n_matrices, alloc_M, K, dtype=torch.bfloat16)
    A[:, M:, :] = 1000.0  # poison: any read past the window blows up the comparison
    B = torch.randn(num_batches, K, dtype=torch.bfloat16)
    C = torch.empty(num_batches, M, dtype=torch.bfloat16)
    for b in range(num_batches):
        C[b] = (A[b // batch_group, :M].float() @ B[b].float()).to(torch.bfloat16)
    return {"A": A, "B": B, "C": C}

