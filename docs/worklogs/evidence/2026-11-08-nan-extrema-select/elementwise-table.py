#!/usr/bin/env python3
"""Median and IQR table of an elementwise-pair.sh campaign.

usage: elementwise-table.py <outdir> [rounds]

With [rounds], only r1..r<rounds> are used and every cell must have exactly that many, so a
campaign the guard stopped mid-round is summarized over its complete rounds only.

Per round and side, device time is nsys's Avg (ns) for t4a_cuda_binary over the 100 timed
launches, and host time is the bench's host_ns_per_op. Every run must report check=ok and
exactly one kernel row with 100 instances, or the script fails. IQR is the inclusive-method
first to third quartile over the rounds.
"""
import csv
import glob
import os
import re
import statistics
import sys

out = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
data = {}  # (case, side) -> {"dev": [...], "host": [...]}
for d in sorted(glob.glob(os.path.join(out, "r*", "*-elementwise-*"))):
    if limit is not None and int(os.path.basename(os.path.dirname(d))[1:]) > limit:
        continue
    side, case = os.path.basename(d).split("-", 1)
    rows = [r for r in csv.reader(open(os.path.join(d, "kern.csv")))]
    if len(rows) != 1 or rows[0][0] != case or rows[0][-1] != "t4a_cuda_binary":
        sys.exit(f"{d}: expected one t4a_cuda_binary row, got {rows}")
    # case, Time(%), Total Time (ns), Instances, Avg (ns), Med (ns), Min, Max, StdDev, Name
    if int(rows[0][3]) != 100:
        sys.exit(f"{d}: {rows[0][3]} instances, expected 100")
    log = open(os.path.join(d, f"{case}.log")).read()
    meta = re.search(r"BENCH_META case=(\S+) .*host_ns_per_op=([0-9.]+) .*check=(\S+)", log)
    if not meta or meta.group(1) != case or meta.group(3) != "ok":
        sys.exit(f"{d}: no BENCH_META with check=ok")
    cell = data.setdefault((case, side), {"dev": [], "host": []})
    cell["dev"].append(float(rows[0][4]) / 1000.0)
    cell["host"].append(float(meta.group(2)) / 1000.0)


def summary(values):
    q1, med, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return med, q1, q3


def fmt(values):
    med, q1, q3 = summary(values)
    return f"{med:.2f} [{q1:.2f}-{q3:.2f}]"


def verdict(main, fix):
    m, mq1, mq3 = summary(main)
    f, fq1, fq3 = summary(fix)
    ratio = f / m
    if ratio > 1.0 and fq1 > mq3:
        return ratio, "**slower** (IQRs disjoint)"
    if ratio < 1.0 and fq3 < mq1:
        return ratio, "faster (IQRs disjoint)"
    return ratio, "IQRs overlap"


def order(case):
    _, op, dt, n = case.split("-")
    return (int(n), op, dt)


cases = sorted({c for c, _ in data}, key=order)
rounds = {len(data[(c, s)]["dev"]) for c in cases for s in ("main", "fix")}
if limit is not None and rounds != {limit}:
    sys.exit(f"expected {limit} rounds in every cell, got {sorted(rounds)}")
print(f"rounds per cell: {sorted(rounds)}\n")
print("| case | device µs main | device µs fix | device ratio | device | host µs main | host µs fix | host ratio | host |")
print("|---|---|---|---|---|---|---|---|---|")
for c in cases:
    m, f = data[(c, "main")], data[(c, "fix")]
    dr, dv = verdict(m["dev"], f["dev"])
    hr, hv = verdict(m["host"], f["host"])
    print(f"| `{c}` | {fmt(m['dev'])} | {fmt(f['dev'])} | {dr:.3f} | {dv} | {fmt(m['host'])} | {fmt(f['host'])} | {hr:.3f} | {hv} |")
print("\nmedian [Q1-Q3] over the rounds; ratio = fix median / main median.")
print("\nper-round device µs (main / fix):")
for c in cases:
    m, f = data[(c, "main")], data[(c, "fix")]
    print(f"- `{c}`: main {' '.join(f'{v:.2f}' for v in m['dev'])}; fix {' '.join(f'{v:.2f}' for v in f['dev'])}")
