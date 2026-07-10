# Ad-hoc repro: IRON aie2p softmax.cc `max_val=0` init bug on all-negative rows.
# softmax.cc:33 `float max_val=0;` + :48 `if(running_max>max_val)` clamps the
# subtracted max to >=0. After log2e scaling, a strongly-negative row keeps
# max_val=0, so exp2(scaled) underflows to 0 -> sum 0 -> 1/0 -> NaN/garbage,
# whereas true-max subtraction keeps the max element at exp2(0)=1.
import numpy as np, torch, ml_dtypes
import aie.utils as aie_utils
from iron.common import AIEContext
from iron.operators.softmax.op import Softmax
from iron.common.test_utils import run_test

def run_case(label, lo, hi, rows=16, cols=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(rows, cols, generator=g) * (hi - lo) + lo)  # uniform [lo,hi]
    x_bf16 = x.to(torch.bfloat16)
    ref = torch.softmax(x_bf16.float(), dim=-1)  # valid distribution (shift-invariant)
    ctx = AIEContext(mlir_verbose=False)
    op = Softmax(rows=rows, cols=cols, num_aie_columns=1, num_channels=1, context=ctx)
    inb = {"in": x_bf16.flatten()}
    outb = {"out": ref.to(torch.bfloat16).flatten()}
    errors, lat, bw = run_test(op, inb, outb, rel_tol=0.05, abs_tol=0.02,
                               max_error_rate=1.0, warmup_iters=1, timed_iters=3)
    # read back actual device output by re-running raw
    op2 = op
    nerr = sum(len(v) for v in errors.values()) if errors else 0
    # summarize device vs ref on row 0
    print(f"\n[{label}]  input range [{lo},{hi}], scaled(*1.4427) ~[{lo*1.4427:.0f},{hi*1.4427:.0f}]")
    print(f"  mismatching elems (tol 5%/0.02): {nerr} / {rows*cols}")
    print(f"  ref row0 sum={ref[0].sum():.4f} (valid softmax sums to ~1)")
    aie_utils.DefaultNPURuntime.cleanup()
    return nerr

if __name__ == "__main__":
    # control: moderate negative -> shift-invariance holds, no underflow -> should PASS
    run_case("control: moderate-negative [-8,-2]", -8, -2)
    # bug: strongly negative -> exp2 underflows with max=0 -> device garbage vs valid ref
    run_case("BUG: strong-negative [-130,-110]", -130, -110)
