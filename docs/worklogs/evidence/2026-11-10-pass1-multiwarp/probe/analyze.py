"""Per-SM warp timelines from the pass-1 probe (SM-local clock64, so only within-SM comparisons)."""
import sys, statistics
from collections import defaultdict
for path in sys.argv[1:]:
    head = open(path).readline().strip()
    planes = int(head.split("planes=")[1].split()[0])
    rows = [tuple(map(int, l.split())) for l in open(path) if not l.startswith("#")]
    by_sm = defaultdict(list)
    for c, t0, t1, sm in rows:
        by_sm[sm].append((c, t0, t1))
    span, conc, held, tailf, peak_l, durs = [], [], [], [], [], []
    for sm, ws in by_sm.items():
        lo = min(w[1] for w in ws); hi = max(w[2] for w in ws); mk = hi - lo
        busy = sum(w[2] - w[1] for w in ws); durs += [w[2] - w[1] for w in ws]
        # A block's slots are held until its last warp ends (blocks = cube // planes).
        bend = defaultdict(int)
        for c, t0, t1 in ws: bend[c // planes] = max(bend[c // planes], t1)
        hold = sum(bend[c // planes] - t1 for c, t0, t1 in ws)
        ev = sorted([(w[1], 1) for w in ws] + [(w[2], -1) for w in ws])
        a = 0; pk = 0; tl = []
        for t, d in ev: a += d; pk = max(pk, a); tl.append((t, a))
        low = 0  # time with fewer than half the peak warps active
        for (t, a), (t2, _) in zip(tl, tl[1:]):
            if a < pk / 2: low += t2 - t
        span.append(mk); conc.append(busy / mk); held.append(hold / (busy + hold)); tailf.append(low / mk); peak_l.append(pk)
    med = statistics.median
    print(f"{head}\n  SMs {len(by_sm)}  warps/SM med {med([len(v) for v in by_sm.values()])}  "
          f"SM span med {med(span):.0f} clk max {max(span)}  warp dur med {med(durs):.0f} clk  "
          f"mean active warps/SM med {med(conc):.1f} (peak med {med(peak_l)})  "
          f"slot time held by finished warps {100*med(held):.1f}%  span with < half-peak active {100*med(tailf):.1f}%")
