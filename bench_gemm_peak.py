# Ad-hoc bench: IRON aie2p reference GEMM %-of-peak, bf16 vs bfp16-emulation.
# Throwaway (not committed to the engine repo). Reuses the operator's own harness.
import aie.utils as aie_utils
from iron.common import AIEContext
from iron.operators.gemm.op import GEMM
from iron.operators.gemm.reference import generate_golden_reference
from iron.common.test_utils import run_test

PEAK_TFLOPS = 25.0  # KB denominator (exp-task0-results: ~25-29 TFLOPS bf16 peak, aie2p/Strix)

def bench(M, K, N, cols, m, k, n, b_col_maj, c_col_maj, emulate_bfp16, prio_accuracy):
    gr = generate_golden_reference(M=M, K=K, N=N, b_col_maj=b_col_maj, c_col_maj=c_col_maj)
    ctx = AIEContext(mlir_verbose=False)
    op = GEMM(M=M, K=K, N=N, tile_m=m, tile_k=k, tile_n=n,
              num_aie_columns=cols, prio_accuracy=prio_accuracy,
              emulate_bf16_mmul_with_bfp16=emulate_bfp16,
              b_col_maj=b_col_maj, c_col_maj=c_col_maj, context=ctx)
    inb = {"A": gr["input"].flatten(), "B": gr["input_b"][0].flatten()}
    outb = {"C": gr["output"][0].flatten()}
    # loose tol so a bfp16/precision miss doesn't abort timing; we report errors separately
    errors, lat_us, bw = run_test(op, inb, outb, rel_tol=0.06, abs_tol=0.06,
                                  max_error_rate=0.02, warmup_iters=3, timed_iters=20)
    gflops = (2.0 * M * K * N) / (lat_us * 1e-6) / 1e9
    pct = gflops / (PEAK_TFLOPS * 1000) * 100
    nerr = sum(len(v) for v in errors.values()) if errors else 0
    print(f"  cols={cols} bcol={int(b_col_maj)} ccol={int(c_col_maj)} "
          f"emul_bfp16={int(emulate_bfp16)} prio_acc={int(prio_accuracy)} "
          f"tile={m}x{k}x{n}: {lat_us:8.1f} us  {gflops:8.1f} GFLOP/s  "
          f"{pct:5.2f}% of {PEAK_TFLOPS:.0f}T  err_elems={nerr}")
    aie_utils.DefaultNPURuntime.cleanup()
    return gflops, pct, nerr

if __name__ == "__main__":
    M = K = N = 2048
    print("== 2048^3 bf16 GEMM, 8 col, b/c col-major (best bf16 config) ==")
    bench(M, K, N, 8, 64, 64, 64, True, True, emulate_bfp16=False, prio_accuracy=True)
    print("== same shape, bfp16-emulation ON (the mlir-air flash-attn lever) ==")
    bench(M, K, N, 8, 64, 64, 64, True, True, emulate_bfp16=True, prio_accuracy=False)
    print("== bf16, prio_accuracy OFF (bf16_bf16 path, f32->bf16 store earlier) ==")
    bench(M, K, N, 8, 64, 64, 64, True, True, emulate_bfp16=False, prio_accuracy=False)
