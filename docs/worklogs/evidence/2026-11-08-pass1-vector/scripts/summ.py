import json,sys,statistics as st
r=[json.loads(l) for l in open(sys.argv[1])]
print(f"| case | base µs (IQR%) | cand µs (IQR%) | ratio |\n|---|---:|---:|---:|")
worst=0
for c in dict.fromkeys(x['case'] for x in r):
    def m(side):
        v=sorted(x['host_ns_per_op'] for x in r if x['case']==c and x['side']==side)
        med=st.median(v); return med/1e3,(v[-1]-v[0])/med*100
    b,bi=m('base'); a,ai=m('cand'); worst=max(worst,a/b)
    print(f"| `{c}` | {b:.2f} ({bi:.1f}) | {a:.2f} ({ai:.1f}) | {a/b:.3f} |")
print("worst", round(worst,3))
