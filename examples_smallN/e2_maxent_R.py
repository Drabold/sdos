"""B4 of DIRECTIVE_sdos_final_fixes: Maxent with sampled sigma and exact-moment substitution K=4 (and plain Rademacher) for R = 2, 3, 4, 8: convergence fraction, Newton steps and E_BS error, M = 100.
usage: python e2_maxent_R.py SYSTEM SIZE [nseeds] -> results/e2maxentR_SYSTEM.json"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
name, size = sys.argv[1], int(sys.argv[2]); NS = int(sys.argv[3]) if len(sys.argv) > 3 else 20; M = 100; RS = (2, 3, 4, 8)
ref = Ref(name, size); N = ref.N; rng = np.random.default_rng(5 + size); T = ref.cheb(M); mex = T.mean(1)
Xr = ref.rademacher(rng, NS * max(RS)); mr = ref.vec_moments(Xr, M, T).reshape(M, NS, max(RS))
Eref, Er, Ir, info0 = maxent_eval(mex[None].repeat(2, 0), ref, 1e-4, "floor"); rows = []
for R in RS:
    for lab, K in (("rad", 0), ("splice K=4", 4)):
        errs, conv, its = [], [], []
        for s in range(NS):
            A = mr[:, s, :R].T.copy()
            if K: A = splice(A.T, mex, K).T
            try: eb, E, idos, info = maxent_eval(A, ref, 1e-4, "sampled")
            except Exception: conv.append(False); its.append(-1); continue
            errs.append(eb - Eref); conv.append(bool(info["converged"])); its.append(info["iterations"])
        e = np.array(errs); row = dict(system=name, size=size, N=N, M=M, R=R, est=lab, nseeds=NS, converged_frac=float(np.mean(conv)), iters_median=float(np.median(its)),
                                       rmse=float(np.sqrt((e ** 2).mean())) if len(e) else None, sd=float(e.std(ddof=1)) if len(e) > 1 else None)
        rows.append(row); print(row, flush=True)
json.dump(rows, open(f"results/e2maxentR_{name}{size}.json", "w"), indent=1)
