#!/usr/bin/env bash
# Compile-gate mv_quant.cc over (layout, K, group, width, dtype), with ROW_GROUP derived the way a
# real build derives it, so the gate exercises the same constant the operator would pass.
PEANO=${PEANO:?set PEANO to the llvm-aie install (provides bin/clang++)}
AIEAPI=${AIEAPI:?set AIEAPI to the aie_api include directory}
PY=${PY:?set PY to a python that can import iron/common/quant.py}
derive() { $PY -c "
import importlib.util,sys
sp=importlib.util.spec_from_file_location('q','iron/common/quant.py'); q=importlib.util.module_from_spec(sp); sp.loader.exec_module(q)
try: print(q.derive_row_group([$1], $2, '$3', $4))
except ValueError: print(0)"; }
run() { # planar K g r dtype EMIT expect
  rg=1; [ "$1" = "1" ] && rg=$(derive $2 $3 "$5" $4)
  [ "$rg" = "0" ] && { printf '   PLANAR=%s K=%-5s g=%-3s r=%-2s %-6s -> no legal ROW_GROUP\n' "$1" "$2" "$3" "$4" "$6"; return; }
  out=$("$PEANO/bin/clang++" --target=aie2p-none-unknown-elf -std=c++2b -Wno-parentheses \
        -Wno-attributes -O2 -I"$AIEAPI" -DPLANAR=$1 -DROW_GROUP=$rg -DDIM_K=$2 -DGROUP_SIZE=$3 \
        -DVEC_SIZE=$4 -DQUANT_EMIT_$6=1 -c aie_kernels/generic/mv_quant.cc -o /dev/null 2>&1)
  got=ok; [ -n "$out" ] && got=fail
  mark="  "; [ "$got" != "$7" ] && mark="!!"
  printf '%s PLANAR=%s K=%-5s g=%-3s r=%-2s G=%-2s %-6s -> %-4s (want %s)\n' \
         "$mark" "$1" "$2" "$3" "$4" "$rg" "$6" "$got" "$7"
  [ "$got" != "$7" ] && echo "$out" | grep -E "error" | head -2
}
echo "--- header_first: today's shipped widths compile, the illegal ones do not"
run 0 3840 32  32 int8 INT8  ok
run 0 3840 64  16 int8 INT8  ok
run 0 3840 64  64 int8 INT8  fail
run 0 3840 128 8  int8 INT8  fail
echo "--- row_group_planar at the DERIVED block: min(64,g) compiles, wider than the group does not"
run 1 3840 32  32 int8 INT8  ok
run 1 3840 64  64 int8 INT8  ok
run 1 3840 128 64 int8 INT8  ok
run 1 3840 32  64 int8 INT8  fail
run 1 640  32  32 int4 INT4  ok
run 1 640  64  64 int8 INT8  ok
run 1 15360 64 64 int8 INT8  ok
run 1 4096 64  64 int8a INT8A ok
run 1 3840 64  64 int4a INT4A ok
