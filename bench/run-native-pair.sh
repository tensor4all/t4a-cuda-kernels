#!/bin/bash
# Native/native paired campaign (docs/design/slice-reduction-throughput.md §6): the same harness
# source built against two library revisions, one process per case per side per repetition,
# AB/BA order, fresh campaign directory, and the GPU checked idle before it starts.
#
#   BASE_BIN=path/to/baseline/bench_kernels CAND_BIN=path/to/candidate/bench_kernels \
#     OUT=/tmp/campaign REPS=3 bench/run-native-pair.sh
#
# Each record is the harness's own BENCH_META line, so the host time, launch count, input
# digest and check result travel with the measurement. A failed process is recorded as a
# failure, never dropped: a campaign with a failure is incomplete and gets repeated whole.
set -euo pipefail
out="${OUT:?set OUT to a fresh directory}"
base="${BASE_BIN:?baseline binary}"
cand="${CAND_BIN:?candidate binary}"
reps="${REPS:-3}"
cpus="${CPUS:-60-62}"
[[ -e "$out" ]] && { echo "$out already exists; a campaign starts fresh" >&2; exit 1; }
if [ "${ALLOW_BUSY:-0}" != 1 ] && [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "another process is using the GPU; wait or set ALLOW_BUSY=1" >&2
    exit 1
fi
mkdir -p "$out"
cases="${CASES:-reduce-sum-f32-axis-511 reduce-sum-f32-axis-512 reduce-sum-f32-axis-513 \
reduce-sum-f32-axis-600 reduce-sum-f32-4096sq-axis0 reduce-sum-f32-4096sq-axis1 \
reduce-sum-f64-4096sq-axis0 reduce-sum-c64-4096sq-axis0 reduce-max-i32-4096sq-axis0 \
reduce-sum-f32-full-1000000 reduce-sum-f32-full-10000000 \
reduce-sum-f32-full-1048576 reduce-prod-f32-full-1048576 reduce-max-f32-full-1048576 \
reduce-min-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-prod-f64-full-1048576 \
reduce-max-f64-full-1048576 reduce-min-f64-full-1048576 reduce-sumsquares-f32-full-1048576 \
reduce-max-f32-4x131072-axis1}"
echo "campaign $out reps=$reps cpus=$cpus" | tee "$out/campaign.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a "$out/campaign.txt"
echo "baseline $base" | tee -a "$out/campaign.txt"
echo "candidate $cand" | tee -a "$out/campaign.txt"
for case in $cases; do
    for rep in $(seq 1 "$reps"); do
        # Alternating order: odd repetitions run the baseline first, even ones the
        # candidate first, so a drifting clock cannot favour one side.
        if [ $((rep % 2)) -eq 1 ]; then order="base cand"; else order="cand base"; fi
        for side in $order; do
            bin="$base"; [ "$side" = cand ] && bin="$cand"
            line="$(taskset -c "$cpus" "$bin" "$case" | grep '^BENCH_META' || true)"
            if [ -z "$line" ]; then
                echo "{\"case\":\"$case\",\"side\":\"$side\",\"rep\":$rep,\"status\":\"failed\"}" \
                    >> "$out/records.jsonl"
                echo "FAILED $case $side rep$rep" >&2
                continue
            fi
            ns="$(sed -n 's/.*host_ns_per_op=\([0-9.]*\).*/\1/p' <<<"$line")"
            launches="$(sed -n 's/.*launches_per_iter=\([0-9]*\).*/\1/p' <<<"$line")"
            printf '{"case":"%s","side":"%s","rep":%d,"host_ns_per_op":%s,"launches":%s}\n' \
                "$case" "$side" "$rep" "$ns" "$launches" >> "$out/records.jsonl"
        done
    done
done
echo done
