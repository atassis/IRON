#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-only gate for the whole decoder layer as ONE `aie.device`.

THE QUESTION IS PLACEMENT, and it is the only thing this gate can answer. Both halves' per-core
budgets are already measured and neither is at risk: `.text` 11,760 B and 8,048 B against a 16 KB
region, L1 45,568 B and 50,180 B against 64 KB. What is NOT known in advance is whether aiecc can
place and route 12 workers over two compute rows at two different column counts while the two
halves' fifos together spend 14 of the device's 16 shim input channels and 12 of its 16 output.

Placement is NOT correctness, and it is not freedom from deadlock. Both need the device.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from iron.common import AIEContext
from iron.operators.decode_layer_dp.op import DecodeLayerDataParallel

PROGRAM_MEM_BYTES = 0x4000

# Qwen3-0.6B decode shapes, shared by every device-free test below.
_D, _FF, _HD, _HQ, _HKV = 1024, 3072, 128, 16, 8


def _op(max_seq=2048, **kw):
    return DecodeLayerDataParallel(D=_D, FF=_FF, HD=_HD, Hq=_HQ, Hkv=_HKV, max_seq=max_seq,
                                   attn_cols=8, mlp_cols=4, **kw)


# kv_alloc SIZES THE CACHE, not the window: max_seq stays what the attention math iterates (sc/sw,
# the KV-chunk loop, the mask), kv_alloc only sizes kc/vc and the per-head stride -- same split as
# gemv's alloc_M/tmatvec's alloc_K, one level up.
def test_kv_alloc_sizes_the_cache_not_the_window():
    spec = _op(max_seq=256, kv_alloc=4096).get_arg_spec()
    assert spec[4].shape == (_HKV * 4096 * _HD,), f"kc must follow kv_alloc, got {spec[4].shape}"
    assert spec[5].shape == (_HKV * 4096 * _HD,), f"vc must follow kv_alloc, got {spec[5].shape}"


def test_kv_alloc_none_is_the_old_cache_shape():
    """The default must not move: kv_alloc=None and kv_alloc==max_seq both leave kc/vc sized by
    max_seq, exactly as before this field existed."""
    a = _op(kv_alloc=None).get_arg_spec()[4].shape
    b = _op().get_arg_spec()[4].shape
    c = _op(kv_alloc=2048).get_arg_spec()[4].shape
    assert a == b == c == (_HKV * 2048 * _HD,), a


def test_kv_alloc_below_max_seq_is_refused():
    with pytest.raises(ValueError, match="kv_alloc"):
        _op(kv_alloc=1024)  # < max_seq=2048


def test_kv_block_size_must_divide_kv_alloc():
    with pytest.raises(ValueError, match="kv_block_size"):
        _op(kv_alloc=4096, kv_block_size=300)


def test_kv_alloc_windowed_does_not_share_a_name_with_the_plain_design():
    """A cached plain build must not be able to satisfy a wide-cache/blocked op (see the `name`
    override on DecodeLayerDataParallel)."""
    plain = _op().name
    wide = _op(kv_alloc=4096).name
    assert wide != plain, f"wide-cache design shares an artifact name with the plain one: {plain}"
    assert "kva4096" in wide, wide
    # kv_alloc == max_seq is the same design as None, so it must NOT perturb the stable name.
    assert _op(kv_alloc=2048).name == plain

    blocked = _op(max_seq=256, kv_alloc=16384, kv_block_size=128).name
    assert "kvblk128" in blocked, blocked


def test_default_mlir_is_byte_identical():
    """kv_alloc/kv_block_size unset (or set to their own default value) must not perturb one byte
    of the emitted MLIR -- the whole point of repr=False plus the None-default convention."""
    plain = _op().get_mlir_artifact().generator()
    explicit = _op(kv_alloc=2048, kv_block_size=2048).get_mlir_artifact().generator()
    assert plain == explicit


# GRANULE (op.py __post_init__): max_seq must be a multiple of lcm(rpc, kv_block), rpc = tsi*D//HD
# being the stream tile in cache rows and kv_block the KV block (or the alloc when unblocked).
def test_granule_qwen3_shape_is_legal():
    """The doctrine's own worked example: D=1024, HD=128, tsi=4 -> rpc=32; unblocked, kv_block
    collapses to max_seq itself, so lcm(32, max_seq)==max_seq iff rpc divides max_seq -- true at
    max_seq=128 (128/32=4)."""
    _op(max_seq=128)  # must not raise


def test_granule_illegal_max_seq_raises():
    """max_seq=100 is not a multiple of rpc=32 (100/32 is not whole), so the unblocked granule
    (lcm(32,100)=800) is not a multiple of max_seq either -- refused at construction, not three
    frames down in aiecc's placer."""
    with pytest.raises(ValueError, match="granule"):
        _op(max_seq=100)


def test_granule_kv_block_widens_it_past_rpc_alone():
    """kv_block_size=48 does not share rpc=32's power-of-two factor (lcm(32,48)=96), so max_seq=160
    -- itself a clean multiple of rpc alone (160/32=5) -- must still be refused: a check that only
    tested rpc would have missed this."""
    with pytest.raises(ValueError, match="granule"):
        _op(max_seq=160, kv_alloc=384, kv_block_size=48)
    _op(max_seq=192, kv_alloc=384, kv_block_size=48)  # 192 is a multiple of lcm(32,48)=96


def test_window_parameter_defaults_off_and_is_not_in_the_name():
    """The switch must be invisible when unused: same operator name, so a shared build dir cannot
    let a windowed build silently satisfy a plain one. Mirrors attn_block_dp/test.py."""
    plain = _op()
    off = _op(window_parameter=None)
    assert off.name == plain.name
    on = _op(window_parameter="attn_window")
    assert on.name != plain.name, "a windowed build must not share a name with a plain one"
    assert "winattn_window" in on.name, on.name


def test_window_parameter_off_mlir_is_byte_identical():
    plain = _op().get_mlir_artifact().generator()
    off = _op(window_parameter=None).get_mlir_artifact().generator()
    assert plain == off


def test_window_parameter_on_reaches_attn_block_dp():
    """Confirms the pass-through, not just the name suffix: attn_block_dp is the half that owns
    the ScratchpadParameter, so its name must reach the generated MLIR text."""
    off = _op(window_parameter=None).get_mlir_artifact().generator()
    on = _op(window_parameter="attn_window").get_mlir_artifact().generator()
    assert off != on
    assert "attn_window" in on


def main():
    D, FF, HD, Hq, Hkv = 1024, 3072, 128, 16, 8          # Qwen3-0.6B
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    build_dir = Path(__file__).resolve().parents[4] / "build" / f"decode_layer_dp_S{S}"
    op = DecodeLayerDataParallel(D=D, FF=FF, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S,
                                 attn_cols=8, mlp_cols=4,
                                 context=AIEContext(build_dir=build_dir))
    print(f"operator: {op.name}")
    op.compile()

    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    text = mlir_path.read_text()
    n_dev = len(re.findall(r"\baie\.device\b", text))
    n_cores = len(re.findall(r"\baie\.core\(", text))
    n_fifo = len(re.findall(r"aie\.objectfifo @", text))
    assert n_dev == 1, f"expected ONE aie.device (that is the whole point), found {n_dev}"
    print(f"aie.device={n_dev}  aie.core={n_cores}  objectfifos={n_fifo}")

    sub = mlir_path.parent / f"{mlir_path.stem}.mlir.d"
    tiles = sorted(tuple(int(x) for x in d.name.removeprefix("elfs_main_core_").split("_"))
                   for d in sub.glob("elfs_main_core_*") if d.is_dir())
    rows = sorted({r for _, r in tiles})
    print(f"placed compute tiles (col,row): {tiles}")
    print(f"compute ROWS used: {rows}")

    if shutil.which("llvm-size"):
        by_row = {}
        for col, row in tiles:
            elf = sub / f"elfs_main_core_{col}_{row}" / f"elfs_main_core_{col}_{row}.elf"
            out = subprocess.run(["llvm-size", "-A", str(elf)], capture_output=True, text=True)
            tb = next((int(ln.split()[1]) for ln in out.stdout.splitlines()
                       if ln.startswith(".text")), None)
            by_row.setdefault(row, []).append((col, tb))
        for row in sorted(by_row):
            ts = [t for _, t in by_row[row] if t]
            print(f"  row {row}: {len(by_row[row])} cores, .text "
                  f"{min(ts)}-{max(ts)} B ({100.0*max(ts)/PROGRAM_MEM_BYTES:.1f}% of region)")

    print(f"PLACES: ONE aie.device, {n_cores} cores over {len(rows)} compute rows, max_seq={S}")
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
