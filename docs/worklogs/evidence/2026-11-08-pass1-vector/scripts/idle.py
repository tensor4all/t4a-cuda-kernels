import time,sys
# busy fraction of CPUs 0-7 over a 2 s window, from /proc/stat
def snap():
    r={}
    for l in open('/proc/stat'):
        p=l.split()
        if p[0].startswith('cpu') and p[0]!='cpu':
            c=int(p[0][3:]); v=list(map(int,p[1:])); r[c]=(sum(v), v[3]+v[4])
    return r
a=snap(); time.sleep(2); b=snap()
busy={c:1-(b[c][1]-a[c][1])/max(1,(b[c][0]-a[c][0])) for c in range(8)}
print(sys.argv[1], 'load', open('/proc/loadavg').read().split()[:3], 'cpu0-7 busy max %.3f'%max(busy.values()))
