#!/bin/bash
set -euo pipefail
E=$(dirname "$0"); B=/home/shinaoka/work/t4a-cuda-kernels-base/target/release/examples/bench_kernels
C=/home/shinaoka/work/t4a-cuda-kernels-combine/target/release/examples/bench_kernels
cases="reduce-sum-f32-full-1048576 reduce-prod-f32-full-1048576 reduce-max-f32-full-1048576 reduce-min-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-prod-f64-full-1048576 reduce-max-f64-full-1048576 reduce-min-f64-full-1048576 reduce-sumsquares-f32-full-1048576"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then order="base cand"; else order="cand base"; fi
  for side in $order; do
    [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "GPU busy"; exit 3; }
    bin=$B; [ $side = cand ] && bin=$C
    bash /home/shinaoka/work/t4a-cuda-kernels-combine/docs/worklogs/evidence/2026-11-08-nan-extrema-select/knsys.sh $bin ${NSYS_OUT:?}/r$r-$side $cases
  done
done
echo nsys-done
