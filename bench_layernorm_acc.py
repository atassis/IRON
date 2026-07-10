# Ad-hoc bench: IRON aie2p LayerNorm accuracy (rel-L2 vs f32 golden) + latency.
# Run twice: once with the default extern"C" entry (bf16-accum, layer_norm<bf16,16>),
# once after swapping it to layer_norm_bf16_f32_calculation (f32-accum, 8-wide).
import sys, numpy as np, torch
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.common import AIEContext
from iron.operators.layer_norm.op import LayerNorm

def bench(rows, cols, ncols, nchan, seed=42, iters=20, warm=3):
    torch.manual_seed(seed)
    x = (torch.rand(rows, cols) * 4).to(torch.bfloat16)
    golden = torch.nn.functional.layer_norm(x.float(), (cols,))  # f32 ref from same bf16 input
    ctx = AIEContext(mlir_verbose=False)
    op = LayerNorm(size=rows * cols, num_aie_columns=ncols, num_channels=nchan,
                   tile_size=cols, context=ctx)
    op.compile(); fn = op.get_callable()
    a = XRTTensor.from_torch(x.flatten())
    o = XRTTensor((rows * cols,), dtype=a.dtype)
    for _ in range(warm):
        fn(a, o)
    tot = 0
    for _ in range(iters):
        r = fn(a, o); tot += r.npu_time
    lat_us = (tot / iters) / 1e3
    dev = o.to_torch().reshape(rows, cols).float()
    err = (dev - golden)
    rel_l2 = (err.norm(dim=1) / golden.norm(dim=1)).mean().item()  # mean per-row rel-L2
    max_abs = err.abs().max().item()
    nan = torch.isnan(dev).sum().item()
    bw = (rows * cols * 2 * 2) / (lat_us * 1e-6) / 1e9  # in+out bf16
    print(f"  rows={rows} cols={cols} ncol={ncols} nchan={nchan}: "
          f"lat={lat_us:7.1f}us  BW={bw:6.3f} GB/s  rel-L2={rel_l2*100:6.3f}%  "
          f"max_abs={max_abs:.4f}  NaN={nan}")
    aie_utils.DefaultNPURuntime.cleanup()
    return rel_l2, lat_us

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "?"
    print(f"== LayerNorm variant: {tag} ==")
    bench(8, 2048, 8, 1)
    bench(8, 8192, 8, 1)
    bench(16, 1024, 8, 2)
