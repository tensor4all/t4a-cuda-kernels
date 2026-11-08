#!/bin/bash
# usage: knsys.sh <bench-binary> <outdir> <case...>  -> per case: pass-1 / pass-2 device ns per launch
set -euo pipefail
BIN=$1; OUT=$2; shift 2; mkdir -p "$OUT"
source ~/work/gpu-env.sh
for c in "$@"; do
  taskset -c 0-7 nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "$OUT/$c" --force-overwrite true "$BIN" "$c" > "$OUT/$c.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/$c.nsys-rep" 2>/dev/null \
    | grep -E 't4a_cuda_reduce' | sed "s/^/$c,/" >> "$OUT/kern.csv"
done
