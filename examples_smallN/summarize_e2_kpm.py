import json, numpy as np
rows = []
for nm in ("chain", "ssh", "kwon", "anderson"):
    for blk in json.load(open(f"results/e2kpm_{nm}.json")): rows += blk["rows"]
def get(sysn, N, M, R, est):
    for r in rows:
        if r["system"] == sysn and r["N"] == N and r["M"] == M and r["R"] == R and r["est"] == est: return r
print("KPM E_BS, M=100, R=4: per estimator, bias/sd, bias/se, rmse/rmse(rad); W-relative rmse")
ests = ["rad", "splice K=4", "splice K=16", "imp", "imp+K4"]
for nm in ("chain", "ssh", "kwon", "anderson"):
    Ns = sorted({r["N"] for r in rows if r["system"] == nm})
    for N in Ns:
        b = get(nm, N, 100, 4, "rad"); s = f"{nm:8s} N={N:5d} "
        for e in ests:
            r = get(nm, N, 100, 4, e); s += f"| {e[:11]:11s} b/sd {r['bias']/r['sd']:+.2f} b/se {r['bias']/r['se']:+5.1f} rmse/rad {r['rmse']/b['rmse']:.3f} "
        print(s)
