#!/bin/bash
# Paired before/after campaign for elementwise MAX/MIN (`t4a_cuda_binary`), PR #26 review item 2.
#
# usage: elementwise-pair.sh <main-bin> <fix-bin> <outdir> <rounds>
#
# For every round and every case, both binaries run back to back, each in its own process under
# knsys.sh (nsys over the bench's 100 timed launches); the order alternates per round (odd: main
# first, even: fix first), so neither side always runs warm or cold. <outdir> must not exist or
# must be empty. Before every process the GPU must have no compute process and cores 0-7 must be
# idle (< 5% busy over 1 s); otherwise the campaign stops. Run it under the GPU slot lock.
set -euo pipefail
MAIN=$1; FIX=$2; OUT=$3; ROUNDS=$4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES=()
for len in 1000 1048576 16777216; do
  for op in max min; do
    for dt in f32 f64; do CASES+=("elementwise-$op-$dt-$len"); done
  done
done
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "elementwise-pair.sh: refusing to reuse $OUT: it is not empty" >&2
  exit 2
fi
mkdir -p "$OUT"

busy_0_7() { # percent busy of cores 0-7 over one second, from /proc/stat
  python3 - <<'PY'
import time
def snap():
    rows = {}
    for line in open('/proc/stat'):
        f = line.split()
        if f[0].startswith('cpu') and f[0][3:].isdigit() and int(f[0][3:]) < 8:
            v = list(map(int, f[1:]))
            rows[f[0]] = (sum(v), v[3] + v[4])
    return rows
a = snap(); time.sleep(1); b = snap()
total = sum(b[k][0] - a[k][0] for k in a); idle = sum(b[k][1] - a[k][1] for k in a)
print(f"{100.0 * (total - idle) / max(total, 1):.1f}")
PY
}

guard() {
  local apps busy
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  if [ -n "$apps" ]; then echo "GPU busy: $apps" >&2; exit 3; fi
  busy="$(busy_0_7)"
  if python3 -c "import sys; sys.exit(0 if float('$busy') < 5.0 else 1)"; then :; else
    echo "cores 0-7 busy: $busy%" >&2; exit 3
  fi
  echo "$1 cores0-7=$busy% gpu-apps=none" >> "$OUT/guard.log"
}

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="main fix"; else order="fix main"; fi
  for c in "${CASES[@]}"; do
    for side in $order; do
      if [ "$side" = main ]; then bin=$MAIN; else bin=$FIX; fi
      guard "r$r $side $c"
      "$HERE/knsys.sh" "$bin" "$OUT/r$r/$side-$c" "$c"
    done
  done
done
echo "complete: $ROUNDS rounds x ${#CASES[@]} cases x 2 sides" >> "$OUT/guard.log"
