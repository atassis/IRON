# Direct device drive to inspect raw softmax output (NaN check) on all-negative rows.
import numpy as np, torch
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.common import AIEContext
from iron.operators.softmax.op import Softmax

def run_case(label, lo, hi, rows=16, cols=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(rows, cols, generator=g) * (hi - lo) + lo).to(torch.bfloat16)
    ref = torch.softmax(x.float(), dim=-1)
    ctx = AIEContext(mlir_verbose=False)
    op = Softmax(rows=rows, cols=cols, num_aie_columns=1, num_channels=1, context=ctx)
    op.compile(); fn = op.get_callable()
    a = XRTTensor.from_torch(x.flatten())
    c = XRTTensor((rows * cols,), dtype=a.dtype)
    fn(a, c)
    out = c.to_torch().reshape(rows, cols).float().numpy()
    nan = np.isnan(out).sum(); inf = np.isinf(out).sum()
    row0_sum = np.nan_to_num(out[0]).sum()
    print(f"\n[{label}]  input [{lo},{hi}]  scaled*1.4427 ~[{lo*1.4427:.0f},{hi*1.4427:.0f}]")
    print(f"  device out: NaN={nan} Inf={inf} / {rows*cols}   row0 sum(device)={row0_sum:.4f}  ref row0 sum={ref[0].sum():.4f}")
    print(f"  device row0[:6]={out[0,:6]}")
    print(f"  ref    row0[:6]={ref[0,:6].numpy()}")
    aie_utils.DefaultNPURuntime.cleanup()

if __name__ == "__main__":
    run_case("control moderate [-8,-2]", -8, -2)
    run_case("BUG strong-negative [-130,-110]", -130, -110)
