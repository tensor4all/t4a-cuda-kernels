#!/bin/bash
# Paired campaign for docs/design/slice-combine-load-ahead.md. The same bench_kernels source
# (with the prod 16M cells) built on ab7395c (BASE_BIN) and on the candidate (CAND_BIN).
# Per repetition: one process per side that runs every case (each case: 2000 settling + 10
# warm-up launches, then 100 timed iterations closed by one synchronize), the side order
# alternating AB/BA. WARMUP_REPS repetitions are run and recorded but excluded; REPS follow.
# Isolation: the measured processes run on CPUS (default 0-7), which must be exactly one L3
# domain; this driver and its probes (nvidia-smi, the busy check) re-pin themselves to
# OBSERVER_CPUS (default 8-15, another L3 domain), and both sets plus the topology are recorded.
# Before every process: no compute process on the GPU and CPUS below 5% busy over 1 s; the
# driver waits for that (recorded) and stops after 10 minutes without it (the campaign is then
# incomplete and is rerun whole). Every BENCH_META line
# is kept verbatim.
set -euo pipefail
out="${OUT:?fresh directory}"; base="${BASE_BIN:?}"; cand="${CAND_BIN:?}"
reps="${REPS:-15}"; warm="${WARMUP_REPS:-3}"; cpus="${CPUS:-0-7}"; observer="${OBSERVER_CPUS:-8-15}"
if [ "${CAMPAIGN_PINNED:-0}" != 1 ]; then
  CAMPAIGN_PINNED=1 exec taskset -c "$observer" "$0" "$@"
fi
first="${cpus%-*}"; last="${cpus#*-}"
l3="$(cat /sys/devices/system/cpu/cpu$first/cache/index3/shared_cpu_list)"
[ "$l3" = "$cpus" ] || { echo "CPUS=$cpus is not one L3 domain (cpu$first shares L3 with $l3)" >&2; exit 1; }
[[ -e "$out" ]] && { echo "$out exists" >&2; exit 1; }
mkdir -p "$out"
cases="reduce-sum-f32-full-1048576 reduce-prod-f32-full-1048576 reduce-max-f32-full-1048576 \
reduce-min-f32-full-1048576 reduce-sum-f64-full-1048576 reduce-prod-f64-full-1048576 \
reduce-max-f64-full-1048576 reduce-min-f64-full-1048576 \
reduce-sum-f32-full-16777216 reduce-prod-f32-full-16777216 reduce-max-f32-full-16777216 \
reduce-min-f32-full-16777216 reduce-sum-f64-full-16777216 reduce-prod-f64-full-16777216 \
reduce-max-f64-full-16777216 reduce-min-f64-full-16777216 reduce-sumsquares-f32-full-1048576 \
reduce-sum-f32-axis-511 reduce-sum-f32-axis-512 reduce-sum-f32-axis-513 reduce-sum-f32-axis-600 \
reduce-sum-f32-4096sq-axis0 reduce-sum-f32-4096sq-axis1 reduce-sum-f64-4096sq-axis0 \
reduce-sum-c64-4096sq-axis0 reduce-max-i32-4096sq-axis0 reduce-sum-f32-full-1000000 \
reduce-sum-f32-full-10000000 reduce-max-f32-4x131072-axis1"
cpu_busy() { # percent busy of $cpus over 1 s
  python3 - "$cpus" <<'PY'
import sys,time
lo,hi=map(int,sys.argv[1].split('-')); ids=[f"cpu{i}" for i in range(lo,hi+1)]
def snap():
    d={}
    for l in open('/proc/stat'):
        f=l.split()
        if f[0] in ids: v=list(map(int,f[1:])); d[f[0]]=(sum(v),v[3]+v[4])
    return d
a=snap(); time.sleep(1); b=snap()
t=sum(b[k][0]-a[k][0] for k in ids); i=sum(b[k][1]-a[k][1] for k in ids)
print(f"{100*(t-i)/max(t,1):.1f}")
PY
}
{ echo "campaign $(date -Is) reps=$reps warmup_reps=$warm cpus=$cpus observer=$observer"
  echo "measured L3 domain: cpu$first shares L3 with $l3; driver affinity: $(taskset -pc $$ | sed 's/.*: //')"
  lscpu | grep -E "Model name|^CPU\(s\)|L3"; echo "base $base"; echo "cand $cand"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used --format=csv
  nvidia-smi; nvcc --version | tail -2; } > "$out/campaign.txt" 2>&1
for rep in $(seq 1 $((warm + reps))); do
  phase=sample; [ "$rep" -le "$warm" ] && phase=warmup
  if [ $((rep % 2)) -eq 1 ]; then order="base cand"; else order="cand base"; fi
  for side in $order; do
    bin="$base"; [ "$side" = cand ] && bin="$cand"
    # Wait (15 s steps, at most 40) until the GPU has no compute process and $cpus are below 5 %
    # busy; every wait is recorded, and the campaign stops if the machine never goes quiet.
    for attempt in $(seq 1 41); do
      apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
      busy="$(cpu_busy)"
      if [ -z "$apps" ] && awk -v b="$busy" 'BEGIN{exit !(b<5.0)}'; then break; fi
      [ "$attempt" = 41 ] && { echo "machine not quiet before rep $rep $side: gpu=[$apps] cpus=$busy%" | tee -a "$out/campaign.txt"; exit 3; }
      echo "wait before rep $rep $side: gpu=[$apps] cpus $cpus busy $busy%" >> "$out/campaign.txt"
      sleep 15
    done
    echo "rep $rep $phase $side cpus_busy=$busy% $(date -Is)" >> "$out/campaign.txt"
    taskset -c "$cpus" "$bin" $cases > "$out/raw-$rep-$side.txt" 2>&1 || { echo "FAILED rep $rep $side" | tee -a "$out/campaign.txt"; exit 4; }
    grep '^BENCH_META' "$out/raw-$rep-$side.txt" | sed "s/^/rep=$rep phase=$phase side=$side /" >> "$out/records.txt"
  done
done
{ echo "end $(date -Is)"; nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv; nvidia-smi --query-compute-apps=pid,name --format=csv; } >> "$out/campaign.txt"
echo done
