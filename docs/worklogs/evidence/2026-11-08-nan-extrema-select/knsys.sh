#!/bin/bash
# usage: knsys.sh <bench-binary> <outdir> <case...>  -> per case: device ns per launch of each t4a_cuda_ kernel
#
# <outdir> must not exist or must be empty: one invocation owns its kern.csv, reports and logs, so a
# second run into the same directory is refused instead of appending a second execution to kern.csv
# (the rule bench/run-paired.sh applies). A case listed twice in one invocation is refused too.
set -euo pipefail
BIN=$1; OUT=$2; shift 2
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "knsys.sh: refusing to reuse $OUT: it is not empty; start each run in a fresh directory" >&2
  exit 2
fi
if [ "$(printf '%s\n' "$@" | sort | uniq -d)" != "" ]; then
  echo "knsys.sh: a case is listed twice" >&2
  exit 2
fi
mkdir -p "$OUT"
source ~/work/gpu-env.sh
# gpu-env.sh does not put the 12.6 toolkit on PATH; the bench compiles its cubins with `nvcc`.
export PATH=/usr/local/cuda-12.6/bin:$PATH
for c in "$@"; do
  taskset -c 0-7 nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "$OUT/$c" "$BIN" "$c" > "$OUT/$c.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/$c.nsys-rep" 2>/dev/null \
    | grep -E 't4a_cuda_' | sed "s/^/$c,/" >> "$OUT/kern.csv"
done
