import json,statistics as st,sys
S=sys.argv[1]
def load(run,arm):
    d={}
    for l in open(f"{S}/{run}/{arm}.jsonl"):
        r=json.loads(l)
        if r.get('status')!='ok': d[r['case_id']]=(None,None,r.get('status')); continue
        t=r['timing']; v=sorted(b/t['operations_per_sample']*1e3 for b in t['batch_ms'])
        q=st.quantiles(v,n=4); med=st.median(v)
        ver=r.get('verification') or {}
        d[r['case_id']]=(med,(q[2]-q[0])/med*100,ver.get('max_error_over_tolerance'))
    return d
arms=['tenferro-cuda-direct','tenferro-cuda-eager','pytorch-cuda']
runs={'b':['tf1-base','tf3-base'],'c':['tf2-cand','tf4-cand']}
D={(s,i,a):load(r,a) for s in runs for i,r in enumerate(runs[s]) for a in arms}
cases=list(D[('b',0,arms[0])])
worst_ver=0; worst_iqr=0
def ratio(arm,c):
    # pairwise: (b0,c0),(b1,c1); report mean of the two ratios
    rs=[D[('c',i,arm)][c][0]/D[('b',i,arm)][c][0] for i in (0,1)]
    return sum(rs)/2, rs
print("| cell | direct base µs | direct cand µs | direct ratio | PyTorch drift | normalised | eager ratio | PyTorch µs |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
prim=[]
for c in cases:
    for k,v in D.items():
        m=v[c]; 
        if m[2] is not None: worst_ver=max(worst_ver,m[2])
        if m[1] is not None: worst_iqr=max(worst_iqr,m[1])
    rd,rds=ratio(arms[0],c); re,_=ratio(arms[1],c); dp,_=ratio(arms[2],c)
    b=st.mean(D[('b',i,arms[0])][c][0] for i in (0,1)); a=st.mean(D[('c',i,arms[0])][c][0] for i in (0,1))
    pt=st.mean(D[(s,i,arms[2])][c][0] for s in 'bc' for i in (0,1))
    print(f"| `{c}` | {b:.2f} | {a:.2f} | {rd:.3f} ({rds[0]:.3f}/{rds[1]:.3f}) | {dp:.3f} | {rd/dp:.3f} | {re:.3f} | {pt:.2f} |")
    if c.endswith('n16777216') and 'prod' not in c: prim.append((rd, rd/dp))
print("primary (6 x n16777216, direct): median raw", round(st.median(x[0] for x in prim),3), "median normalised", round(st.median(x[1] for x in prim),3))
print("worst max_error_over_tolerance", worst_ver, "worst IQR% ", round(worst_iqr,1))
