#!/bin/bash
# Kernel (device) time per launch for each arm: nsys cuda_gpu_kern_sum, rounds with rotated arm order.
set -euo pipefail
A=/home/shinaoka/work/arms-30; OUT=${OUT:?}; ROUNDS=${ROUNDS:-3}
K=/home/shinaoka/work/t4a-cuda-kernels-combine/docs/worklogs/evidence/2026-11-08-nan-extrema-select/knsys.sh
arms=(${ARMS:-base-ab7395c p29-4ea91e6 cand-w2 cand-w4 cand-w8})
cases="${CASES:-reduce-sum-f32-full-1048576 reduce-prod-f32-full-1048576 reduce-max-f32-full-1048576 reduce-min-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-prod-f64-full-1048576 reduce-max-f64-full-1048576 reduce-min-f64-full-1048576 reduce-sumsquares-f32-full-1048576 reduce-sum-f32-full-1000000 reduce-sum-f32-axis-513 reduce-sum-f32-4096sq-axis0 reduce-max-f32-4x131072-axis1}"
mkdir -p "$OUT"; { date -Is; nvidia-smi; } > "$OUT/env.txt"
n=${#arms[@]}
for r in $(seq 1 $ROUNDS); do
  for i in $(seq 0 $((n-1))); do
    arm=${arms[$(( (i + r) % n ))]}
    for t in $(seq 1 40); do
      apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
      [ -z "$apps" ] && break
      echo "wait r$r $arm gpu=[$apps]" | tee -a "$OUT/env.txt"; sleep 15
      [ $t = 40 ] && { echo "GPU never idle" | tee -a "$OUT/env.txt"; exit 3; }
    done
    echo "r$r $arm $(date -Is)" >> "$OUT/env.txt"
    bash $K $A/$arm "$OUT/r$r-$arm" $cases
  done
done
echo sweep-done
