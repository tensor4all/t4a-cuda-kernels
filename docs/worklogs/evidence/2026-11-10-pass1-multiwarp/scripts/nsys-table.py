"""Device µs per launch: median over rounds of nsys's per-round median, per arm/case/kernel."""
import csv, glob, os, re, statistics, sys
from collections import defaultdict
root = sys.argv[1]; arms = sys.argv[2].split(",")
d = defaultdict(list)
for path in glob.glob(f"{root}/r*-*/kern.csv"):
    arm = re.match(r"r\d+-(.*)", os.path.basename(os.path.dirname(path))).group(1)
    for row in csv.reader(open(path)):
        case, kernel, med = row[0], row[-1], float(row[6])
        k = "pass1" if "partials" in kernel else "combine"
        d[(arm, case, k)].append(med / 1e3)
cases = sorted({c for (_, c, _) in d}, key=lambda c: (("full" not in c), c))
m = lambda a, c, k: statistics.median(d[(a, c, k)]) if d.get((a, c, k)) else float("nan")
print("| case | kernel | " + " | ".join(arms) + " | rounds |")
print("|---|---|" + "---:|" * len(arms) + "---:|")
for c in cases:
    for k in ("pass1", "combine", "sum"):
        if k == "sum":
            vals = [m(a, c, "pass1") + m(a, c, "combine") for a in arms]
        else:
            vals = [m(a, c, k) for a in arms]
        n = min(len(d.get((a, c, "combine"), [])) for a in arms)
        print(f"| `{c}` | {k} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {n} |")
