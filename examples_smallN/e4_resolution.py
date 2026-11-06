"""E4: the resolution-limit regime.  Scan M from N/4 to 4N (Maxent capped at M=1024 for memory).  R = 4 Rademacher vectors, mean over `ns` seeds.
Quantities vs the EXACT eigenvalues: cumulative error sup|N_est - N_exact| (true step function) and E_BS error per orbital; Newton steps/converged for Maxent;
Toeplitz first-violation order of the final moment list.  usage: python e4_resolution.py SYSTEM SIZE ns -> results/e4_SYSTEM_SIZE.json"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
name, size, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); R = 4
ref = Ref(name, size); N = ref.N; rng = np.random.default_rng(5)
Ms = [N // 4, N // 2, N, 2 * N, 4 * N]; Mmax = max(Ms); T = ref.cheb(Mmax); mex_full = T.mean(1)
X = ref.rademacher(rng, ns * R); mr_full = ref.vec_moments(X, Mmax, T).reshape(Mmax, ns, R)
step_cache = {}
rows = []
def kpm_row(A, M):
    E, rho, idos = reconstruct(A[None], ref.a, ref.b, 8192)
    eb = band_quantities_grid(E, None, idos[0], 1, ref.nocc / ref.N, 1.0)[1]
    return eb, np.abs(idos[0] - cum_exact(ref, E)).max()
for M in Ms:
    mex = mex_full[:M]; t0 = time.time()
    lists = {"plain": [mr_full[:M, s].mean(1) for s in range(ns)], "splice K=4": [splice(mr_full[:M, s].mean(1), mex, 4) for s in range(ns)], "exact": [mex]}
    for lab, L in lists.items():
        res = [kpm_row(A, M) for A in L]; fv = [first_violation(A) for A in L]
        rows.append(dict(M=M, M_over_N=M / N, method="kpm", est=lab, n=len(L), Ebs_err=float(np.mean([r[0] - ref.Ebs_exact / N for r in res])),
                         Ebs_err_sd=float(np.std([r[0] for r in res])), cum_err=float(np.mean([r[1] for r in res])), frac_noncone=float(np.mean([f >= 0 for f in fv])),
                         m_star_median=float(np.median([f for f in fv if f >= 0])) if any(f >= 0 for f in fv) else None))
    if M <= N:
        for lab, L, mode, fl in (("plain", lists["plain"], "sampled", 1e-4), ("splice K=4", lists["splice K=4"], "sampled", 1e-4),
                                 ("exact floor 1e-4", lists["exact"], "floor", 1e-4)):
            ee, cc, cv, it = [], [], [], []
            if lab.startswith("exact"): pass
            for s, A in enumerate(L if lab.startswith("exact") else L[:min(ns, 3)]):
                if lab.startswith("plain") or lab.startswith("splice"):
                    Rs = mr_full[:M, s, :].T.copy()
                    if lab.startswith("splice"): Rs = splice(Rs.T, mex, 4).T
                else: Rs = np.vstack([A, A])
                try: eb, E, idos, info = maxent_eval(Rs, ref, fl, mode)
                except Exception as ex_: cv.append(False); it.append(-1); continue
                ee.append(eb / N - ref.Ebs_exact / N); cc.append(np.abs(idos - cum_exact(ref, E)).max()); cv.append(bool(info["converged"])); it.append(info["iterations"])
            rows.append(dict(M=M, M_over_N=M / N, method="maxent", est=lab, n=len(ee), Ebs_err=float(np.mean(ee)) if ee else None, cum_err=float(np.mean(cc)) if cc else None,
                             converged_frac=float(np.mean(cv)), iters_median=float(np.median(it))))
    print(f"{name} N={N} M={M} done {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(system=name, size=size, N=N, R=R, ns=ns, rows=rows), open(f"results/e4_{name}_{size}.json", "w"))
