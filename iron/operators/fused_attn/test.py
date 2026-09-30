# SPDX-License-Identifier: Apache-2.0
"""Fused prefill attention, one pipeline: host layout checks, build checks, gated device tests.

Build-only tests need AIE_DEVICE=npu2. Device tests also need FA_DEVICE=1 and npu_lock.sh.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from aie.iron.device import from_name
from iron.common import AIEContext

from iron.operators.fused_attn.layouts import relay, slice_tiles

if os.environ.get("AIE_DEVICE"):
    aie_utils.set_current_device(from_name(os.environ["AIE_DEVICE"], n_cols=None))

FA = Path(os.environ["FA_BUILD"])
ON_DEVICE = pytest.mark.skipif(os.environ.get("FA_DEVICE") != "1",
                               reason="device run: set FA_DEVICE=1 and run under npu_lock.sh")


def ctx(name):
    return AIEContext(build_dir=FA / name)


def pattern(n, mod=251):
    """Distinct-enough bf16-exact values: integers below 251, a prime, so no layout shift aliases."""
    return (np.arange(n) % mod).astype(np.float32)


@pytest.mark.parametrize("rows", [64, 16])
def test_host_relay_emits_slice_tiles(rows):
    x = pattern(rows * 512).reshape(rows, 512)
    assert np.array_equal(relay(x, 64), slice_tiles(x, 64))


def test_host_slice_tiles_is_not_the_identity():
    x = pattern(64 * 512).reshape(64, 512)
    assert (slice_tiles(x, 64) != x.reshape(-1)).mean() > 0.9


from iron.operators.fused_attn.op import TiledFeed


def run_once(op, *arrays):
    """One invocation; returns every output as numpy, in arg-spec order."""
    op.compile()
    f = op.get_callable()
    args, outs = [], []
    it = iter(arrays)
    for spec in op.get_arg_spec():
        if spec.direction == "in":
            a = next(it)
            t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a))
            args.append(aie_utils.DEFAULT_TENSOR_CLASS.from_torch(t))
        else:
            b = aie_utils.DEFAULT_TENSOR_CLASS(spec.shape, dtype=spec.dtype)
            args.append(b)
            outs.append(b)
    f(*args)
    # An f32 output's numpy view points into the XRT buffer, which is freed when `outs` goes.
    return [o.to_torch().float().numpy() if o.to_torch().dtype == torch.bfloat16
            else o.to_torch().numpy().copy() for o in outs]


def bf(x):
    return torch.from_numpy(np.asarray(x, np.float32)).to(torch.bfloat16)


def test_feed_builds():
    TiledFeed(n_blocks=4, context=ctx("feed_nb4")).compile()


@ON_DEVICE
def test_feed_is_bit_exact_on_device():
    q = pattern(16 * 512).reshape(16, 512)
    k = pattern(4 * 64 * 512, mod=241).reshape(4 * 64, 512)
    oq, ok = run_once(TiledFeed(n_blocks=4, context=ctx("feed_nb4")), bf(q), bf(k))
    want_k = np.concatenate([slice_tiles(k[b * 64:(b + 1) * 64], 64) for b in range(4)])
    assert np.array_equal(oq, slice_tiles(q, 64)), f"q: {np.sum(oq != slice_tiles(q, 64))} differ"
    assert np.array_equal(ok, want_k), f"k: {np.sum(ok != want_k)} differ"
    aie_utils.DefaultNPURuntime.cleanup()


from iron.operators.fused_attn.layouts import o_tiles, p_tiles, st_tiles
from iron.operators.fused_attn.op import KernelCheck

EXP2_TABLE = Path(os.environ["FA_EXP2_TABLE"])


def aie_exp2(x):
    """aie::exp2<bfloat16> from the engine's device-sampled table; -inf is exactly 0."""
    z = np.load(EXP2_TABLE)
    lo, per, tab = float(z["lo"]), int(z["per"]), np.asarray(z["hw"], np.float32)
    x = np.asarray(x, np.float32)
    out = np.exp2(np.clip(x, -400.0, 128.0)).astype(np.float32)
    out[np.isneginf(x)] = 0.0
    inside = (x >= lo) & (x < lo + len(tab) / per) & np.isfinite(x)
    out[inside] = tab[np.clip(((x[inside] - lo) * per).astype(np.int64), 0, len(tab) - 1)]
    return out


def b16(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def smt_golden(s_blocks, widths):
    """The kernel's arithmetic in numpy: bf16 block max, exp2 from the device table, f32 state.
    s_blocks are [64 keys, rows] f32. Returns the last block's p [keys, rows], corr, l."""
    rows = widths.size
    m = np.full(rows, -np.inf, np.float32)
    l = np.zeros(rows, np.float32)
    for b, s in enumerate(s_blocks):
        vis = (b * 64 + np.arange(64))[:, None] < widths[None, :]
        sm = np.where(vis, s, -np.inf).astype(np.float32)
        bmax = b16(sm).max(axis=0)
        m_new = bmax if b == 0 else np.maximum(b16(m), bmax)
        corr = np.ones(rows, np.float32) if b == 0 else b16(aie_exp2(m - m_new))
        p = np.where(vis, b16(aie_exp2(s - m_new[None, :])), 0.0).astype(np.float32)
        bsum = p.sum(axis=0, dtype=np.float32)
        l = bsum if b == 0 else (l * corr + bsum).astype(np.float32)
        m = m_new
    return p, corr, l


def smt_case(rows=16):
    """Block 0 fully visible, max 0 per row. Block 1 partial: row 0 fully hidden, the first half
    of the rows cut part way, the rest fully visible; every third row rises by 0.5."""
    k, r = np.meshgrid(np.arange(64), np.arange(rows), indexing="ij")
    s0 = -((k + 3 * r) % 11) / 4.0
    s1 = s0 + np.where(r % 3 == 0, 0.5, -0.25)
    widths = (64 + (128 // rows) * np.arange(rows)).astype(np.int32)
    return [s0.astype(np.float32), s1.astype(np.float32)], widths


def finish_case(rows=16):
    """One x V worker's accumulator: the whole head at 16 rows, its half at 32."""
    hd = 512 if rows == 16 else 256
    o_nat = (pattern(rows * hd, mod=199).reshape(rows, hd) - 99.0) / 4.0
    corr = np.ones(rows, np.float32)
    corr[[9, 12] + ([25] if rows > 16 else [])] = 0.5
    l = np.array([2.0 ** (r % 4) for r in range(rows)], np.float32)
    return o_nat.astype(np.float32), corr, l


def kernel_check(check, rows=16):
    if check == "qt":
        n = 8 if rows == 16 else 2
        payload = (pattern(n * rows * 64, mod=97).reshape(n, rows * 64) / 8.0,)
    elif check == "smt":
        s_blocks, widths = smt_case(rows)
        payload = ([st_tiles(s) for s in s_blocks], widths, 1)
    else:
        o_nat, corr, l = finish_case(rows)
        payload = (o_tiles(o_nat), np.concatenate([corr, l]).astype(np.float32))
    return KernelCheck(check=check, rows=rows, payload=payload,
                       context=ctx(f"check_{check}" + ("" if rows == 16 else f"_r{rows}")))


@pytest.mark.parametrize("rows", [16, 32])
@pytest.mark.parametrize("check", ["qt", "smt", "finish"])
def test_kernel_check_builds(check, rows):
    kernel_check(check, rows).compile()


@ON_DEVICE
@pytest.mark.parametrize("rows", [16, 32])
def test_qt_slice_on_device(rows):
    op = kernel_check("qt", rows)
    (got,) = run_once(op)
    nj = rows // 8
    q = op.payload[0].reshape(-1, nj, 8, 8, 8)
    want = np.stack([b16(q[sl % len(q), rt, kt] * 1.4453125).T
                     for sl in range(8) for kt in range(8) for rt in range(nj)]).reshape(-1)
    assert np.array_equal(got, want), f"{np.sum(got != want)} of {got.size} differ"
    aie_utils.DefaultNPURuntime.cleanup()


@ON_DEVICE
@pytest.mark.parametrize("rows", [16, 32])
def test_smt_block_on_device(rows):
    op = kernel_check("smt", rows)
    p, cl = run_once(op)
    s_blocks, widths = smt_case(rows)
    wp, wc, wl = smt_golden(s_blocks, widths)
    bad = ~np.isclose(p, p_tiles(wp), rtol=4e-3, atol=0)
    assert not bad.any(), f"P: {bad.sum()} differ"
    assert np.array_equal(cl[:rows], wc), f"corr {cl[:rows]} vs {wc}"
    assert np.allclose(cl[rows:], wl, rtol=1e-5), f"l {cl[rows:]} vs {wl}"
    aie_utils.DefaultNPURuntime.cleanup()


@ON_DEVICE
@pytest.mark.parametrize("rows", [16, 32])
def test_finish_on_device(rows):
    op = kernel_check("finish", rows)
    (got,) = run_once(op)
    o_nat, corr, l = finish_case(rows)
    want = b16(o_nat * corr[:, None] / l[:, None]).reshape(-1)
    assert np.array_equal(got, want), f"{np.sum(got != want)} of {got.size} differ"
    aie_utils.DefaultNPURuntime.cleanup()


from iron.operators.fused_attn.op import FusedAttnOnePipe


def one_pipe(W, mask=None):
    return FusedAttnOnePipe(n_blocks=W // 64, mask=mask,
                            context=ctx(f"one_pipe_w{W}" + ("" if mask is None else f"_mk{mask}")))


@pytest.mark.parametrize("W", [1024, 4096, 8192, 32768, 262144])
def test_one_pipe_builds(W):
    one_pipe(W).compile()


def test_one_pipe_mask_off_builds():
    one_pipe(1024, mask=0).compile()


# ##########################################################################
# V pre-pass operator: attention_k_eq_v's V, recomputed from kc, off fused_attn's tiled path
# ##########################################################################
#
# Replaces Task 1.4's in-kernel kv_skip_v recompute (reverted: it derived V's RMSNorm from
# ROTATED K, not raw K -- see the revert commit). This operator is separate from fused_attn
# entirely: it produces a real, stored V buffer, byte-identical in shape/dtype to what
# FusedAttnOnePipe's V argument always expected, so fused_attn itself needs zero changes.


from iron.operators.fused_attn.op import VPrepass
from iron.operators.fused_attn.v_prepass_design import v_prepass


def test_v_prepass_mlir_calls_the_recompute_kernel():
    """MLIR generation only (no aiecc): the worker binds fa_v_prepass_rows."""
    dev = aie_utils.get_current_device()
    mlir = str(v_prepass(dev, 2, 512))
    assert "func.call @fa_v_prepass_rows(" in mlir


def test_v_prepass_arg_spec_matches_fused_attn_v_slot():
    """Drop-in check: VPrepass's V-out arg must be the exact same shape/dtype as
    FusedAttnOnePipe's V-in arg for the same (n_blocks, head_dim), so the pre-pass's output
    buffer can be fed to fused_attn as V unmodified."""
    vp = VPrepass(n_blocks=2, context=ctx("vprepass_argspec"))
    fa = FusedAttnOnePipe(n_blocks=2, context=ctx("vprepass_argspec_fa"))
    v_out = vp.get_arg_spec()[-1]
    v_in = fa.get_arg_spec()[3]     # Q, widths, K, V, O -- V is index 3
    assert (v_out.shape, v_out.dtype) == (v_in.shape, v_in.dtype)


def test_v_prepass_builds():
    """Full aiecc build: fa_v_prepass_rows (fused_attn_v_prepass.cc) and the reused
    passThroughLine gain-copy (fa_feed_passThrough.o) link into one core."""
    VPrepass(n_blocks=2, context=ctx("vprepass_build")).compile()



def _gate_llm_reference():
    """CPU oracle lives in the sibling wt-kv-skip-v worktree's scripts/, not this repo -- same
    cross-worktree pattern the decode task's analogous test uses (wt-kv-skip-v-iron's
    iron/operators/attn_global_dp/test.py)."""
    scripts_dir = Path(__file__).resolve().parents[4] / "wt-kv-skip-v" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import gate_llm_reference as glr
    import test_gate_llm as tgl
    return glr, tgl


def test_recompute_from_kc_oracle_holds_under_prefill_batching():
    """The pre-pass implements exactly `v_mode="recompute_from_kc"` on the corrected CPU oracle
    (scripts/gate_llm_reference.py, sibling wt-kv-skip-v worktree): RoPE-inversion on the already-
    cached rotated/qk-normed kc, divide by qk-norm's K gain, then the existing gainless RMSNorm --
    no raw-K array. The sibling decode test exercises this oracle with a 2-token prompt (mostly
    the free-running tail, layer_step -- one row at a time, same as decode). This pre-pass's own
    use case is PREFILL: a whole K block computed before any generation, which run_numpy takes
    through layer_step_batch instead -- a materially different code path (batched GEMMs, positions
    filled in one array-write, not walked one at a time). Reusing the oracle, not re-deriving the
    math (per this task's own instruction): a longer known-prefix prompt makes run_numpy spend the
    bulk of its work in that batched path before its single free-running step."""
    glr, tgl = _gate_llm_reference()
    sp = tgl._tiny_attention_k_eq_v_spec(qk_norm=True)
    with tempfile.TemporaryDirectory() as tmp:
        tgl._write_tiny_weights(tmp, sp)
        prompt_ids, n_tokens, k = [0, 1, 0, 1, 0, 1, 0, 1], 1, 2
        stored = glr.run_numpy(sp, tmp, prompt_ids, n_tokens, k, v_mode="store_at_write")
        recomputed = glr.run_numpy(sp, tmp, prompt_ids, n_tokens, k, v_mode="recompute_from_kc")

    gen_stored, _tops_s, margins_s = stored
    gen_recomputed, _tops_r, margins_r = recomputed
    assert gen_stored == gen_recomputed
    assert sp.eps == 1e-6, "atol below assumes this fixture's eps; re-check if eps changes"
    np.testing.assert_allclose(margins_r, margins_s, rtol=0, atol=1e-4)


FLASH_REF = Path(os.environ["FA_FLASH_REF"])
sys.path.insert(0, str(FLASH_REF))
from flash_attn_ref import full_attention, visible_from_widths  # noqa: E402

# rel-L2 against f64 is a note: aie::exp2's table alone puts a faithful model of these kernels at
# 3.8e-2..6.6e-2 on this data (5e-3 with exact exp2). The ceiling only catches gross breakage; the
# exact structural tests below are the correctness gates.
GROSS = 1e-1


def case(W, rows, n=16):
    """n rows of one head: 'last' = the last n tokens of the window's last chunk, 'first' = its
    first n, 'mid' = the first n of a chunk halfway in (most blocks fully hidden)."""
    base = W // 2 if rows == "mid" else W - 64
    i0 = 64 - n if rows == "last" else 0
    return np.minimum(base + i0 + np.arange(n) + 1, W).astype(np.int32)


def data(W, seed=0, n=16):
    rng = np.random.default_rng(seed)
    q = b16(rng.standard_normal((n, 512)) * 0.08)
    k = b16(rng.standard_normal((W, 512)))
    v = b16(rng.standard_normal((W, 512)))
    return q, k, v


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_rel_l2_bound_can_fail():
    """The gross bound must reject a layout error: rows permuted score far above it."""
    q, k, v = data(1024)
    widths = case(1024, "last")
    ref = full_attention(q, k, v, visible_from_widths(widths, 1024), 1.0)
    assert rel_l2(ref[::-1], ref) > 10 * GROSS


@ON_DEVICE
@pytest.mark.parametrize("W", [1024, 4096])
@pytest.mark.parametrize("rows", ["last", "first", "mid"])
def test_one_pipe_matches_reference(W, rows):
    q, k, v = data(W)
    widths = case(W, rows)
    op = one_pipe(W)
    (o1,) = run_once(op, bf(q), torch.from_numpy(widths), bf(k), bf(v))
    (o2,) = run_once(op, bf(q), torch.from_numpy(widths), bf(k), bf(v))
    ref = full_attention(q, k, v, visible_from_widths(widths, W), 1.0)
    err = rel_l2(o1.reshape(16, 512), ref)
    print(f"\none_pipe W={W} rows={rows}: rel-L2 vs f64 reference {err:.3e}")
    assert np.array_equal(o1, o2), "two runs differ"
    assert np.isfinite(o1).all()
    assert err < GROSS
    aie_utils.DefaultNPURuntime.cleanup()


def v_prepass_ref(kc, ang, gain, epsilon=1e-6):
    """Numpy model of fa_v_prepass_rows (fused_attn_v_prepass.cc): rotate-half inverse RoPE
    (cos even / sin odd, interleaved over HEAD_DIM/2, same layout as rope.cc's `lut`), divide
    out qk-norm's K gain, then gainless RMSNorm -- same op sequence as the kernel, float32."""
    kc, ang, gain = (np.asarray(x, np.float32) for x in (kc, ang, gain))
    half = kc.shape[-1] // 2
    cos_val, sin_val = ang[..., 0::2], ang[..., 1::2]
    y1, y2 = kc[..., :half], kc[..., half:]
    raw1 = y1 * cos_val + y2 * sin_val
    raw2 = y2 * cos_val - y1 * sin_val
    k_row = np.concatenate([raw1 / gain[:half], raw2 / gain[half:]], axis=-1)
    inv_rms = 1.0 / np.sqrt(np.mean(k_row ** 2, axis=-1, keepdims=True) + epsilon)
    return k_row * inv_rms


def v_prepass_case(n_blocks, seed, head_dim=512):
    """kc/ang/gain inputs with no external model behind them: fa_v_prepass_rows's math is
    checked against v_prepass_ref directly, so any well-formed rotation table exercises it."""
    rng = np.random.default_rng(seed)
    keys, half = n_blocks * 64, head_dim // 2
    kc = b16(rng.standard_normal((keys, head_dim)) * 0.3)
    theta = rng.uniform(0, 2 * np.pi, (keys, half)).astype(np.float32)
    ang = np.empty((keys, head_dim), np.float32)
    ang[:, 0::2], ang[:, 1::2] = np.cos(theta), np.sin(theta)
    gain = b16(0.5 + rng.random(head_dim))
    return kc, b16(ang), gain


@ON_DEVICE
def test_v_prepass_composes_with_fused_attn():
    """Two separate dispatches, not a fused one (per the plan's preference when nothing forces a
    single dispatch): VPrepass's own device output V, fed unmodified into FusedAttnOnePipe's
    V-input, must reproduce reference attention over the same K -- chaining actual device output,
    not just the arg-spec shape/dtype match test_v_prepass_arg_spec_matches_fused_attn_v_slot
    already covers."""
    W = 1024
    n_blocks = W // 64
    kc, ang, gain = v_prepass_case(n_blocks, seed=7)
    # Device order is (KC, GAIN, ANG) per v_prepass_design.sequence's arg order; v_prepass_ref
    # and v_prepass_case above use (KC, ANG, GAIN) -- gain/ang are swapped between the two, easy
    # to get backwards, verify against v_prepass_design.py before touching either call site.
    (v_dev,) = run_once(VPrepass(n_blocks=n_blocks, context=ctx(f"vprepass_compose_w{W}")),
                        bf(kc), bf(gain), bf(ang))
    v_dev = v_dev.reshape(n_blocks * 64, 512)
    v_ref = v_prepass_ref(kc, ang, gain)
    # bf16 chains ~6 rounding steps here (mul, add, mul, mean, invsqrt, mul -- see the RMSNorm
    # in v_prepass_ref); rtol/atol match this file's own bf16-pipeline precision budget elsewhere
    # (test_one_pipe_one_hot_picks_the_target_not_the_decoy's 2^-3/2^-7 accounting). The 2% slack
    # fraction allows the occasional element near a bf16 rounding boundary without hiding a real
    # formula error, which would fail on most elements, not a couple of percent of them.
    bad = ~np.isclose(v_dev, v_ref, rtol=2e-2, atol=2e-3)
    assert bad.mean() < 0.02, f"V pre-pass vs reference: {bad.sum()} of {bad.size} differ"

    q = b16(np.random.default_rng(9).standard_normal((16, 512)) * 0.08)
    widths = case(W, "last")
    (o,) = run_once(one_pipe(W), bf(q), torch.from_numpy(widths), bf(kc), bf(v_dev))
    ref = full_attention(q, kc, v_ref, visible_from_widths(widths, W), 1.0)
    err = rel_l2(o.reshape(16, 512), ref)
    print(f"\nvprepass->one_pipe compose W={W}: rel-L2 vs reference {err:.3e}")
    assert np.isfinite(o).all()
    assert err < GROSS
    aie_utils.DefaultNPURuntime.cleanup()


def v_exact(W, seed):
    """V that bfp16 holds exactly: nonzero multiples of 2^-6 (a zero keeps other keys' 2^-30
    leakage), every 8-key block led by |v| in [0.5, 1)."""
    rng = np.random.default_rng(seed)
    v = rng.integers(1, 64, (W, 512)) * np.where(rng.integers(0, 2, (W, 512)) == 1, 1, -1) / 64.0
    lead = (32 + rng.integers(0, 32, (W // 8, 512))) / 64.0
    v[::8] = np.where(rng.integers(0, 2, (W // 8, 512)) == 1, lead, -lead)
    return v.astype(np.float32)


@ON_DEVICE
@pytest.mark.parametrize("W", [1024, 4096])
@pytest.mark.parametrize("rows", ["last", "first", "mid"])
def test_one_pipe_uniform_is_the_visible_mean(W, rows):
    """q = 0: every score is 0 and every P exactly 1, so row r is the mean of V[:width_r]."""
    widths = case(W, rows)
    k = b16(np.random.default_rng(3).standard_normal((W, 512)))
    v = v_exact(W, 4)
    (o,) = run_once(one_pipe(W), bf(np.zeros((16, 512))), torch.from_numpy(widths), bf(k), bf(v))
    want = np.stack([v[:w].astype(np.float64).mean(axis=0) for w in widths])
    bad = ~np.isclose(o.reshape(16, 512), want, rtol=1e-2, atol=1e-5)
    assert not bad.any(), f"{bad.sum()} differ; rows {sorted(set(np.nonzero(bad)[0]))}"
    aie_utils.DefaultNPURuntime.cleanup()


@ON_DEVICE
@pytest.mark.parametrize("W", [1024, 4096])
@pytest.mark.parametrize("rows", ["first", "mid"])
def test_one_pipe_one_hot_picks_the_target_not_the_decoy(W, rows):
    """Row r's query is its target key scaled, so the target's score beats every other visible
    key by ~30 in log2; each row also has a decoy at twice the target, placed where no row may
    look. Out must be v[target] up to one row-wide factor: the target's P sits within 2^-3 of 1,
    where x V's bfp16 keeps a bit or two fewer than the softmax's bf16 sum, so O/l can be 2^-7 off.
    A wrong key tile (another row's value) or a failed mask (the decoy's) is off by ~1."""
    widths = case(W, rows)
    lo, hi = int(widths.min()), int(widths.max())
    targets = np.array([w - 1 if r % 2 == 0 else (r * 131) % (lo - 64)
                        for r, w in enumerate(widths)])
    k = b16(np.random.default_rng(5).standard_normal((W, 512)))
    decoys = hi + np.arange(16)
    assert decoys.max() < W and len(set(targets)) == 16
    k[decoys] = b16(2 * k[targets])
    q = b16(0.05 * k[targets])
    v = v_exact(W, 6)
    (o,) = run_once(one_pipe(W), bf(q), torch.from_numpy(widths), bf(k), bf(v))
    o = o.reshape(16, 512)
    wrong = [r for r in range(16) if not np.allclose(o[r], v[targets[r]], rtol=2e-2, atol=0)]
    assert not wrong, f"rows {wrong}: max|o-v[target]| {[float(np.abs(o[r] - v[targets[r]]).max()) for r in wrong]}"
    aie_utils.DefaultNPURuntime.cleanup()


def poisoned(W, widths, k, v):
    """NaN in every key and 1e30 in every value no row may see (at or past the largest width)."""
    k, v = k.copy(), v.copy()
    hidden = np.arange(W) >= widths.max()
    k[hidden] = np.nan
    v[hidden] = 1e30
    return k, v


@ON_DEVICE
@pytest.mark.parametrize("mask,identical", [(None, True), (0, False)])
def test_one_pipe_poison(mask, identical):
    W = 1024
    q, k, v = data(W, seed=1)
    widths = case(W, "mid")
    op = one_pipe(W, mask=mask)
    kp, vp = poisoned(W, widths, k, v)
    (clean,) = run_once(op, bf(q), torch.from_numpy(widths), bf(k), bf(v))
    (dirty,) = run_once(op, bf(q), torch.from_numpy(widths), bf(kp), bf(vp))
    same = np.array_equal(clean, dirty, equal_nan=False)
    print(f"\npoison mask={mask}: identical={same}, dirty finite={np.isfinite(dirty).all()}")
    assert same is identical
    aie_utils.DefaultNPURuntime.cleanup()


from iron.common.test_utils import run_test


@ON_DEVICE
@pytest.mark.parametrize("W", [8192, 32768, 8192, 32768])
def test_one_pipe_timing(W):
    q, k, v = data(W, seed=2)
    widths = np.full(16, W, np.int32)
    _, latency_us, _ = run_test(one_pipe(W), {"Q": bf(q), "W": torch.from_numpy(widths),
                                              "K": bf(k), "V": bf(v)}, {"O": None},
                                warmup_iters=2, timed_iters=10)
    print(f"\none_pipe timing W={W}: {latency_us:.1f} us, {1e3 * latency_us / W:.2f} ns/key")
    aie_utils.DefaultNPURuntime.cleanup()


from iron.operators.fused_attn.op import HalfFeed


@pytest.mark.parametrize("half", [0, 1])
def test_half_feed_builds(half):
    HalfFeed(n_blocks=4, half=half, context=ctx(f"half_feed_h{half}")).compile()


@ON_DEVICE
@pytest.mark.parametrize("half", [0, 1])
def test_half_feed_is_bit_exact_on_device(half):
    k = pattern(4 * 64 * 512, mod=241).reshape(4 * 64, 512)
    (got,) = run_once(HalfFeed(n_blocks=4, half=half, context=ctx(f"half_feed_h{half}")), bf(k))
    kh = k[:, half * 256:(half + 1) * 256]
    want = np.concatenate([slice_tiles(kh[b * 64:(b + 1) * 64], 64) for b in range(4)])
    assert np.array_equal(got, want), f"{np.sum(got != want)} of {got.size} differ"
    aie_utils.DefaultNPURuntime.cleanup()


from iron.operators.fused_attn.op import FusedAttnX2


def x2(W, mask=None):
    return FusedAttnX2(n_blocks=W // 64, mask=mask,
                       context=ctx(f"x2_w{W}" + ("" if mask is None else f"_mk{mask}")))


@pytest.mark.parametrize("W", [1024, 4096, 8192, 32768, 262144])
def test_x2_builds(W):
    x2(W).compile()


def test_x2_mask_off_builds():
    x2(1024, mask=0).compile()


@ON_DEVICE
@pytest.mark.parametrize("W", [1024, 4096])
@pytest.mark.parametrize("rows", ["last", "first", "mid"])
def test_x2_uniform_is_the_visible_mean(W, rows):
    widths = case(W, rows, n=32)
    k = b16(np.random.default_rng(3).standard_normal((W, 512)))
    v = v_exact(W, 4)
    (o,) = run_once(x2(W), bf(np.zeros((32, 512))), torch.from_numpy(widths), bf(k), bf(v))
    want = np.stack([v[:w].astype(np.float64).mean(axis=0) for w in widths])
    bad = ~np.isclose(o.reshape(32, 512), want, rtol=1e-2, atol=1e-5)
    assert not bad.any(), f"{bad.sum()} differ; rows {sorted(set(np.nonzero(bad)[0]))}"
    aie_utils.DefaultNPURuntime.cleanup()


@ON_DEVICE
@pytest.mark.parametrize("W", [1024, 4096])
@pytest.mark.parametrize("rows", ["first", "mid"])
def test_x2_one_hot_picks_the_target_not_the_decoy(W, rows):
    """As the 16-row test; the two halves of each row come from different x V workers, so a
    half-swap or a wrong V half shows as a wrong row too."""
    widths = case(W, rows, n=32)
    lo, hi = int(widths.min()), int(widths.max())
    targets = np.array([w - 1 if r % 2 == 0 else (r * 131) % (lo - 64)
                        for r, w in enumerate(widths)])
    k = b16(np.random.default_rng(5).standard_normal((W, 512)))
    decoys = hi + np.arange(32)
    assert decoys.max() < W and len(set(targets)) == 32
    k[decoys] = b16(2 * k[targets])
    q = b16(0.05 * k[targets])
    v = v_exact(W, 6)
    (o,) = run_once(x2(W), bf(q), torch.from_numpy(widths), bf(k), bf(v))
    o = o.reshape(32, 512)
    wrong = [r for r in range(32) if not np.allclose(o[r], v[targets[r]], rtol=2e-2, atol=0)]
    assert not wrong, f"rows {wrong}"
    aie_utils.DefaultNPURuntime.cleanup()


@ON_DEVICE
@pytest.mark.parametrize("W", [1024, 4096])
@pytest.mark.parametrize("rows", ["last", "first", "mid"])
def test_x2_reference_note(W, rows):
    q, k, v = data(W, n=32)
    widths = case(W, rows, n=32)
    op = x2(W)
    (o1,) = run_once(op, bf(q), torch.from_numpy(widths), bf(k), bf(v))
    (o2,) = run_once(op, bf(q), torch.from_numpy(widths), bf(k), bf(v))
    ref = full_attention(q, k, v, visible_from_widths(widths, W), 1.0)
    err = rel_l2(o1.reshape(32, 512), ref)
    print(f"\nx2 W={W} rows={rows}: rel-L2 vs f64 reference {err:.3e}")
    assert np.array_equal(o1, o2), "two runs differ"
    assert np.isfinite(o1).all()
    assert err < GROSS
    aie_utils.DefaultNPURuntime.cleanup()


@ON_DEVICE
@pytest.mark.parametrize("mask,identical", [(None, True), (0, False)])
def test_x2_poison(mask, identical):
    W = 1024
    q, k, v = data(W, seed=1, n=32)
    widths = case(W, "mid", n=32)
    op = x2(W, mask=mask)
    kp, vp = poisoned(W, widths, k, v)
    (clean,) = run_once(op, bf(q), torch.from_numpy(widths), bf(k), bf(v))
    (dirty,) = run_once(op, bf(q), torch.from_numpy(widths), bf(kp), bf(vp))
    same = np.array_equal(clean, dirty, equal_nan=False)
    print(f"\nx2 poison mask={mask}: identical={same}, dirty finite={np.isfinite(dirty).all()}")
    assert same is identical
    aie_utils.DefaultNPURuntime.cleanup()


@ON_DEVICE
@pytest.mark.parametrize("arm", ["x2_8192", "one_8192", "x2_32768", "one_32768",
                                 "x2_8192", "one_8192", "x2_32768", "one_32768"])
def test_x2_vs_one_pipe_timing(arm):
    kind, W = arm.split("_")
    W = int(W)
    n = 32 if kind == "x2" else 16
    q, k, v = data(W, seed=2, n=n)
    widths = np.full(n, W, np.int32)
    op = x2(W) if kind == "x2" else one_pipe(W)
    _, latency_us, _ = run_test(op, {"Q": bf(q), "W": torch.from_numpy(widths), "K": bf(k),
                                     "V": bf(v)}, {"O": None}, warmup_iters=2, timed_iters=10)
    print(f"\ntiming {kind} rows={n} W={W}: {latency_us:.1f} us, {1e3 * latency_us / W:.2f} ns/key, "
          f"{1e3 * latency_us / W / n:.3f} ns/key/row")
    aie_utils.DefaultNPURuntime.cleanup()
