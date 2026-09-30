# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free check: does attn_block_dp's existing split-K flash attention reach O(head_dim) L1,
flat in the context window, by pinning `attn_split` to its floor instead of letting it grow to fill
L1? No new kernel or design -- attn_block_dp/design.py already carries the running-max/rescale
recurrence (aie_kernels/aie2p/softmax.cc::partial_softmax_f32state_bf16); this only chooses a
different value for an existing parameter.

Three checks, each runnable alone:
    l1_sweep()      pure Python, l1_footprint_bytes() at attn_split=64 across S -- and the SAME
                    function showing the default derivation (attn_split=None) has no answer once S
                    exceeds the direct-fit cap when kv_block_size is left at its own default.
    build_check()   device-free aiecc placement at HD in {256, 512}, two S four decades apart,
                    same attn_split -- needs an activated toolchain env (see any operator's test.py).
    oracle_check()  numpy model of the ACTUAL kernel composition (scores gemv -> log2e/exp2 running
                    softmax -> f32 accumulator rescale) against stream_attn.py's f64 golden.

    python attn_flat_l1_check.py
"""
import numpy as np

from iron.operators.attn_block_dp.design import (
    l1_footprint_bytes, derive_attn_split, L1_BYTES, FLASH_SM_VEC_LEN,
)


def l1_sweep():
    D, tsi, weight_depth, stack_size = 1024, 4, 2, 0xD00
    print("=== L1 footprint: attn_split pinned at the floor (64) vs the default derivation ===")
    print(f"{'HD':>4} {'gqa':>3} {'S':>7}  {'pinned L=64':>12} {'fits':>5}  "
          f"{'default-derived L':>18} {'fits':>5}")
    for HD, gqa in ((256, 2), (512, 2)):
        tile_elems = tsi * D
        rpc = tile_elems // HD
        assert FLASH_SM_VEC_LEN % rpc == 0 or rpc % FLASH_SM_VEC_LEN == 0
        for S in (2048, 8192, 32768, 262144):
            pinned = l1_footprint_bytes(D, HD, gqa, 64, tile_elems, weight_depth, stack_size)
            derived = derive_attn_split(D, HD, gqa, S, tile_elems, weight_depth, stack_size)
            derived_str = str(derived) if derived else "NONE (raises)"
            derived_fits = "yes" if derived else "n/a"
            print(f"{HD:>4} {gqa:>3} {S:>7}  {pinned:>12} {'yes' if pinned <= L1_BYTES else 'NO':>5}  "
                  f"{derived_str:>18} {derived_fits:>5}")
    print(f"(L1 budget: {L1_BYTES} B. Pinned column is IDENTICAL across every S at fixed HD/gqa --")
    print(" the whole point. Default column is None past the direct-fit cap because")
    print(" derive_attn_split's own gran = lcm(rpc, S, 64) is a multiple of S whenever")
    print(" kv_block_size is left at its default None, so the search range it builds is empty.)")


def build_check():
    try:
        from iron.common import AIEContext
        from iron.operators.attn_block_dp.op import AttnBlockDataParallel
    except ImportError as e:
        print(f"=== build check SKIPPED: toolchain env not active ({e}) ===")
        return
    from pathlib import Path
    import tempfile
    print("\n=== device-free aiecc placement, attn_split=64 ===")
    for D, HD, Hq, Hkv, stack_size in ((1024, 256, 8, 4, 0xD00), (1024, 512, 4, 2, 0xD80)):
        for S in (2048, 32768):
            with tempfile.TemporaryDirectory() as td:
                op = AttnBlockDataParallel(
                    D=D, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S, num_aie_columns=Hkv,
                    attn_split=64, stack_size=stack_size,
                    context=AIEContext(build_dir=Path(td)),
                )
                op.compile()
                print(f"  HD={HD} S={S:>6}: PLACED ({op.name})")


def oracle_check():
    """Same math as tests/test_split_k_golden.py's split_k_context, plus the scores gemv that
    file takes as a given input (matvec_rtk_bf16_bf16: f32-accumulate reduce, one bf16 round)."""
    LOG2E = np.float32(1.4453125)

    def bf16(x):
        x = np.asarray(x, np.float32)
        u = x.view(np.uint32)
        r = ((u >> 16) & 1) + 0x7FFF
        return ((u + r) & 0xFFFF0000).view(np.float32)

    def kernel_model(q, K, V, L):
        N, hd = K.shape
        m, l, acc = np.float32(-np.inf), np.float32(0.0), np.zeros(hd, np.float32)
        for start in range(0, N, L):
            Kb, Vb = K[start:start + L], V[start:start + L]
            s = bf16(Kb.astype(np.float32) @ q.astype(np.float32))
            scaled = bf16(s * LOG2E)
            seg_max = np.float32(scaled.max())
            m_new = max(m, seg_max) if np.isfinite(m) else seg_max
            corr = np.float32(0.0) if not np.isfinite(m) else bf16(np.exp2(m - m_new))
            p = bf16(np.exp2(scaled - m_new))
            acc = acc * corr + p.astype(np.float32) @ Vb.astype(np.float32)
            l = l * corr + np.float32(p.astype(np.float32).sum())
            m = m_new
        return bf16(acc / l)

    def golden(q, K, V):
        s = K.astype(np.float64) @ q.astype(np.float64)
        p = np.exp(s - s.max())
        return (p @ V.astype(np.float64)) / p.sum()

    print("\n=== kernel_model vs f64 golden, at the pinned split (L=64) ===")
    print(f"{'hd':>4} {'N':>7}  {'rel-L2 vs golden':>16}")
    for hd in (256, 512):
        for N in (1024, 6912, 32768, 262144):
            rng = np.random.default_rng(0)
            q = bf16(rng.standard_normal(hd, dtype=np.float32) / np.sqrt(hd))
            K = bf16(rng.standard_normal((N, hd), dtype=np.float32))
            V = bf16(rng.standard_normal((N, hd), dtype=np.float32))
            g = golden(q, K, V)
            rel = np.linalg.norm(kernel_model(q, K, V, 64) - g) / np.linalg.norm(g)
            print(f"{hd:>4} {N:>7}  {rel:>16.3e}")

    print("\n=== splitting cost: L=64 vs one segment (L=N), same log2e/exp2/bf16 convention ===")
    print(f"{'hd':>4} {'N':>7}  {'rel-L2, split vs full-row':>26}")
    for hd in (256, 512):
        for N in (1024, 6912, 32768):
            rng = np.random.default_rng(0)
            q = bf16(rng.standard_normal(hd, dtype=np.float32) / np.sqrt(hd))
            K = bf16(rng.standard_normal((N, hd), dtype=np.float32))
            V = bf16(rng.standard_normal((N, hd), dtype=np.float32))
            full = kernel_model(q, K, V, N)
            rel = np.linalg.norm(kernel_model(q, K, V, 64) - full) / np.linalg.norm(full)
            print(f"{hd:>4} {N:>7}  {rel:>26.3e}")


if __name__ == "__main__":
    l1_sweep()
    build_check()
    oracle_check()
