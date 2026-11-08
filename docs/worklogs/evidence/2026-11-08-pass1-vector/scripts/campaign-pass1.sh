#!/bin/bash
set -euo pipefail
S=/tmp/claude-2000/-home-shinaoka-work-t4a-reduce-2406/7036dc7e-cb48-4619-846e-725e49469709/scratchpad
for side in base cand base cand; do
  n=$((${n:-0}+1)); out=$S/tf$n-$side
  python3 $S/idle.py "before $out"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  $S/run-pass1.sh $S/tbench-$side $out > $out.log 2>&1
  python3 $S/idle.py "after  $out"
done
