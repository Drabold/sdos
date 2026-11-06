"""E1: is the (spliced / sampled) moment list the moment list of a positive measure?  Toeplitz test lambda_min(T_m) >= 0.
usage: python e1_cone.py SYSTEM [nseeds]  -> results/e1_SYSTEM.json"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
name = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40; RAV = 16; NI1, NIR = 20, 6
KS = (2, 4, 8, 16); out = []
for size in SYSTEMS[name][1]:
    t0 = time.time(); ref = Ref(name, size); N = ref.N; M = int(min(4 * N, 400)); M = max(M, 64)
    T = ref.cheb(M); mu_ex = T.mean(1); rng = np.random.default_rng(100 + size)
    mgrid = np.unique(np.round(np.linspace(1, M - 1, 40)).astype(int))
    lists = {}     # label -> array (M, nseeds)
    lists["exact"] = mu_ex[:, None]
    Xr = ref.rademacher(rng, NS * RAV); mr = ref.vec_moments(Xr, M, T).reshape(M, NS, RAV)
    lists["rad R=1"] = mr[:, :, 0]; lists[f"rad R={RAV}"] = mr.mean(2)
    for K in KS: lists[f"splice K={K} R=1"] = splice(mr[:, :, 0], mu_ex, K)
    for K in (4, 16): lists[f"splice K={K} R={RAV}"] = splice(mr.mean(2), mu_ex, K)
    Xi = ref.impartial(rng, NI1); mi = ref.vec_moments(Xi, M, T); lists["impartial(3) R=1"] = mi
    lists["impartial(3)+splice K=4 R=1"] = splice(mi, mu_ex, 4)
    Xi2 = ref.impartial(rng, NIR * RAV); mi2 = ref.vec_moments(Xi2, M, T).reshape(M, NIR, RAV).mean(2); lists[f"impartial(3) R={RAV}"] = mi2
    lists[f"impartial(3)+splice K=4 R={RAV}"] = splice(mi2, mu_ex, 4)
    res = dict(system=name, size=size, N=N, M=M, mgrid=mgrid.tolist(), lists={})
    prof_labels = ("exact", "rad R=1", "splice K=4 R=1", "splice K=16 R=1", "impartial(3) R=1", f"rad R={RAV}", f"splice K=4 R={RAV}")
    for lab, arr in lists.items():
        fv = [first_violation(arr[:, s]) for s in range(arr.shape[1])]
        d = dict(nseeds=len(fv), frac_violating=float(np.mean([f >= 0 for f in fv])),
                 m_star=[int(f) for f in fv], lam_min_end=[float(lam_min_T(arr[:, s], M - 1)) for s in range(arr.shape[1])])
        if lab in prof_labels:
            ns = min(8, arr.shape[1]); d["profile_median"] = [float(np.median([lam_min_T(arr[:, s], m) for s in range(ns)])) for m in mgrid]
            d["profile_min"] = [float(np.min([lam_min_T(arr[:, s], m) for s in range(ns)])) for m in mgrid]
        res["lists"][lab] = d
    # exact: sanity threshold -1e-13
    res["exact_min_lam_over_m"] = float(min(lam_min_T(mu_ex, m) for m in mgrid))
    out.append(res)
    print(f"{name} N={N} M={M} done {time.time()-t0:.0f}s  exact min lam {res['exact_min_lam_over_m']:.2e}", flush=True)
    for lab, d in res["lists"].items():
        ms = [m for m in d["m_star"] if m >= 0]
        print(f"   {lab:32s} viol {d['frac_violating']:.2f}  m*/N median {np.median(ms)/N if ms else float('nan'):.3f}", flush=True)
    json.dump(out, open(f"results/e1_{name}.json", "w"))
