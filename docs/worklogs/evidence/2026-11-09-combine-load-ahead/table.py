"""Median (IQR) host us/op per case and side over the sample repetitions of records.txt."""
import re, statistics, sys
from collections import defaultdict
rows = defaultdict(lambda: defaultdict(list)); checks = set(); order = []
for line in open(sys.argv[1]):
    f = dict(re.findall(r"(\w+)=(\S+)", line))
    if f["phase"] != "sample": continue
    if f["case"] not in order: order.append(f["case"])
    rows[f["case"]][f["side"]].append(float(f["host_ns_per_op"]) / 1e3)
    checks.add(f["check"])
def q(v):
    v = sorted(v); n = len(v); med = statistics.median(v)
    qs = statistics.quantiles(v, n=4, method="inclusive"); return med, qs[2] - qs[0], n
print(f"checks: {sorted(checks)}")
print("| case | base µs (IQR) | candidate µs (IQR) | ratio | n |")
print("|---|---:|---:|---:|---:|")
for c in order:
    b, bi, bn = q(rows[c]["base"]); k, ki, kn = q(rows[c]["cand"])
    print(f"| `{c}` | {b:.3f} ({bi:.3f}) | {k:.3f} ({ki:.3f}) | {k/b:.3f} | {bn}/{kn} |")
