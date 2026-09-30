# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming (online) decode attention against the materialised three-step form.

Both compute the SAME function; they differ only in what they keep. The question is
whether the running-max rescale costs accuracy, and how the error tracks n_past.
"""
import numpy as np

def bf16(x):
    """Round-to-nearest-even into bf16, kept in f32. The device's storage format."""
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).view(np.float32)

def materialised(q, K, V):
    """What ships: scores[N] -> softmax[N] -> weighted sum. Three N-wide passes."""
    s = bf16(K @ q)                      # scores GEMV, bf16 out (mv.cc rounds per partial)
    m = s.max()
    p = bf16(np.exp(s - m))
    l = p.sum()
    return bf16((p @ V) / l)

def streaming(q, K, V, B):
    """Running max / running sum / rescaled accumulator. L1 is O(head_dim), not O(N)."""
    N, hd = K.shape
    m, l = np.float32(-np.inf), np.float32(0.0)
    acc = np.zeros(hd, np.float32)       # f32 accumulator, per the catalog's precision rule
    for i in range(0, N, B):
        Kb, Vb = K[i:i+B], V[i:i+B]
        s  = bf16(Kb @ q)
        mn = max(m, s.max())
        c  = np.exp(m - mn) if np.isfinite(m) else np.float32(0.0)
        e  = bf16(np.exp(s - mn))
        l   = l * c + e.sum()
        acc = acc * c + (e @ Vb)
        m   = mn
    return bf16(acc / l)

rng = np.random.default_rng(0)
print("%8s %6s %14s %14s   %s" % ("n_past","block","rel-L2 vs mat","max |diff|","note"))
for N in (1024, 6912, 32768, 262144):
    hd = 512
    q = bf16(rng.standard_normal(hd, dtype=np.float32) / np.sqrt(hd))
    K = bf16(rng.standard_normal((N, hd), dtype=np.float32))
    V = bf16(rng.standard_normal((N, hd), dtype=np.float32))
    ref = materialised(q, K, V)
    for B in (256,):
        out = streaming(q, K, V, B)
        rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        print("%8d %6d %14.3e %14.3e   %s" % (N, B, rel, np.abs(out-ref).max(),
              "materialised needs %d KB of N-wide buffers" % (3*N*2//1024)))

print()
print("=== against an EXACT f32 golden: which form is actually more accurate? ===")
def golden(q, K, V):
    s = (K.astype(np.float64) @ q.astype(np.float64))
    p = np.exp(s - s.max()); return (p @ V.astype(np.float64)) / p.sum()
print("%8s %6s %14s %14s  %s" % ("n_past","seeds","materialised","streaming","verdict"))
for N in (1024, 6912, 32768):
    hd, rm, rs = 512, [], []
    for sd in range(5):
        r = np.random.default_rng(sd)
        q = bf16(r.standard_normal(hd, dtype=np.float32)/np.sqrt(hd))
        K = bf16(r.standard_normal((N,hd), dtype=np.float32))
        V = bf16(r.standard_normal((N,hd), dtype=np.float32))
        g = golden(q,K,V); ng = np.linalg.norm(g)
        rm.append(np.linalg.norm(materialised(q,K,V)-g)/ng)
        rs.append(np.linalg.norm(streaming(q,K,V,256)-g)/ng)
    mm, ms = np.mean(rm), np.mean(rs)
    print("%8d %6d %14.3e %14.3e  %s" % (N, 5, mm, ms,
          "streaming %.2fx BETTER" % (mm/ms) if ms < mm else "materialised %.2fx better" % (ms/mm)))

print()
print("=== is N=1024 really bit-identical, or a lucky seed? ===")
for B in (128, 256, 512):
    ident = 0
    for sd in range(20):
        r = np.random.default_rng(100+sd); hd = 256
        q = bf16(r.standard_normal(hd, dtype=np.float32)/np.sqrt(hd))
        K = bf16(r.standard_normal((1024,hd), dtype=np.float32))
        V = bf16(r.standard_normal((1024,hd), dtype=np.float32))
        if np.array_equal(streaming(q,K,V,B), materialised(q,K,V)): ident += 1
    print("   block=%-4d n_past=1024 (the sliding window): %2d/20 bit-identical" % (B, ident))
