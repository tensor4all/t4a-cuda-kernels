"""Median and IQR of per-launch kernel durations from measure.sh's cuda_gpu_trace CSVs.

usage: python summarize.py <outdir>

Every kernel row of a report counts (memcpy/memset rows are excluded; the
measured range has none). One (round, arm, case) report holds 200 launches;
rounds are pooled per (arm, case) and also printed per round.
"""
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

out = sys.argv[1]
pooled = defaultdict(list)
per_round = defaultdict(list)
names = defaultdict(set)
for path in sorted(glob.glob(os.path.join(out, "r*-*_cuda_gpu_trace.csv"))):
    tag = os.path.basename(path).removesuffix("_cuda_gpu_trace.csv")
    m = re.match(r"r(\d+)-(rev13|w1|auto|torch)-(.+)", tag)
    rnd, arm, case = int(m[1]), m[2], m[3]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "")
            if name.startswith("[CUDA mem"):
                continue
            ns = float(row["Duration (ns)"])
            pooled[(arm, case)].append(ns)
            per_round[(rnd, arm, case)].append(ns)
            names[(arm, case)].add(name[:60])


def stats(v):
    q = statistics.quantiles(v, n=4)
    return statistics.median(v), q[2] - q[0], len(v)


print("arm,case,launches,median_us,iqr_us,kernel")
for (arm, case), v in sorted(pooled.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    med, iqr, k = stats(v)
    print(f"{arm},{case},{k},{med/1e3:.3f},{iqr/1e3:.3f},{'|'.join(sorted(names[(arm, case)]))}")
print()
print("round,arm,case,launches,median_us,iqr_us")
for (rnd, arm, case), v in sorted(per_round.items()):
    med, iqr, k = stats(v)
    print(f"{rnd},{arm},{case},{k},{med/1e3:.3f},{iqr/1e3:.3f}")
print()
print("case,rev13/torch,w1/torch,auto/torch,rev13/auto")
med = {key: statistics.median(v) for key, v in pooled.items()}
for case in sorted({c for _, c in pooled}):
    t = med.get(("torch", case))
    r = med.get(("rev13", case))
    w = med.get(("w1", case))
    a = med.get(("auto", case))
    f = lambda p, q: f"{p/q:.2f}" if p and q else "-"
    print(f"{case},{f(r, t)},{f(w, t)},{f(a, t)},{f(r, a)}")
