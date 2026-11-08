#!/bin/bash
# usage: run-pass1.sh <tenferro-benchmark binary> <outdir>  (direct + eager + pytorch, probe-pass1)
set -euo pipefail
S=/tmp/claude-2000/-home-shinaoka-work-t4a-reduce-2406/7036dc7e-cb48-4619-846e-725e49469709/scratchpad
BIN=$1; OUT=$2; mkdir -p "$OUT"
source ~/work/gpu-env.sh
cd ~/work/t4a-pass1-bench
export BENCH_RUNS=15 BENCH_WARMUPS=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
for arm in tenferro-cuda-direct tenferro-cuda-eager; do
  taskset -c 0-7 "$BIN" --fixtures $S/fixtures-pass1/manifest.json --backend $arm --output "$OUT/$arm.jsonl"
done
taskset -c 0-7 uv run python scripts/benchmark_gpu_public_api.py --suite $S/probe-pass1.json --fixtures $S/fixtures-pass1 --backend pytorch-cuda --output "$OUT/pytorch-cuda.jsonl"
