#!/usr/bin/env bash
# usage: flock /tmp/gpu-slot.lock taskset -c 0-7 measure.sh <outdir> <rounds>
#
# Run it under primerose's GPU slot lock and pinned to CPUs 0-7, as the other
# evidence in docs/worklogs does; meta.txt records the affinity and whether the
# lock is held by this process tree.
#
# Kernel-time measurement for tenferro-rs#2529 on one A100, one process at a time.
# Arms, all at len = 1<<20:
#   rev13 - examples/bench_flat_vector with T4A_CUDA_SOURCE_ROOT=$BASE (revision-13
#           kernels, ab7395c), width 1: the kernel tenferro runs today;
#   w1    - this branch's kernels, width 1 (scalar plan);
#   auto  - this branch's kernels, width from the real address alignment;
#   torch - PyTorch, same protocol (torch_flat.py).
# Each (round, arm, case) is one process under nsys; per-launch kernel durations come
# from the cuda_gpu_trace report. Arms rotate per round so drift does not favour one.
# gpu_state.sh logs nvidia-smi compute apps, utilization and python processes
# before and after the whole run and between rounds.
set -euo pipefail
OUT=$1; ROUNDS=${2:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$HOME/tensor4all/t4a-cuda-kernels-2529vw}
BASE=${BASE:-$HOME/tensor4all/t4a-cuda-kernels-2529base}
STATE=${STATE:-$HOME/scratch/claude-2529/gpu_state.sh}
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "measure.sh: $OUT is not empty; use a fresh directory" >&2; exit 2
fi
mkdir -p "$OUT"
source ~/work/gpu-env.sh
export PATH=/usr/local/cuda-12.6/bin:$PATH
BIN=$REPO/target/release/examples/bench_flat_vector
CASES=(add_f64 exp_f64 tanh_f64 clamp_f64 compare_gt_f64)
{
  echo "repo=$(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
  echo "base=$(git -C "$BASE" rev-parse HEAD)"
  nvcc --version | tail -1; nsys --version; python -c 'import torch; print("torch", torch.__version__, torch.version.cuda)'
  uptime
  echo "command: $0 $*"
  echo "affinity: $(taskset -pc $$)"
  if flock -n /tmp/gpu-slot.lock true 2>/dev/null; then echo "gpu-slot.lock: NOT held (free to take)"; else echo "gpu-slot.lock: held"; fi
} > "$OUT/meta.txt" 2>&1
"$STATE" before > "$OUT/gpu_state.log" 2>&1
ARMS=(rev13 w1 auto torch)
for round in $(seq 1 "$ROUNDS"); do
  shift_by=$(( (round - 1) % ${#ARMS[@]} ))
  order=("${ARMS[@]:$shift_by}" "${ARMS[@]:0:$shift_by}")
  for arm in "${order[@]}"; do
    arm_cases=("${CASES[@]}")
    if [ "$arm" = torch ]; then arm_cases+=(clamp_full_f64); fi
    for c in "${arm_cases[@]}"; do
      tag="r${round}-${arm}-${c}"
      case $arm in
        rev13) cmd=(env T4A_CUDA_SOURCE_ROOT="$BASE" "$BIN" "$c" 1) ;;
        w1)    cmd=("$BIN" "$c" 1) ;;
        auto)  cmd=("$BIN" "$c" auto) ;;
        torch) cmd=(python "$HERE/torch_flat.py" "$c") ;;
      esac
      nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=false -o "$OUT/$tag" "${cmd[@]}" > "$OUT/$tag.log" 2>&1
      nsys stats --report cuda_gpu_trace --format csv --output "$OUT/$tag" "$OUT/$tag.nsys-rep" > /dev/null 2>&1
    done
  done
  "$STATE" "after-round-$round" >> "$OUT/gpu_state.log" 2>&1
done
python "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
