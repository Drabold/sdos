"""E2 (Maxent part, reduced): E_BS error and cumulative error of Maxent from plain / spliced / impartial vectors, M = 100.
R=1: sigma = floor 1e-4 (no sampled error exists for one vector); R=16: sigma = sampled standard error with the default floor 1e-4, and with no floor.
Reference: Maxent from the exact moment list (same M, floor).  usage: python e2_maxent.py SYSTEM [nseeds] -> results/e2maxent_SYSTEM.json"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
name = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 60; M = 100; RMAX = 16
out = []
SIZES = {'chain':[64,512],'ssh':[64,512],'kwon':[2,3],'anderson':[8]}
for size in SIZES[name]:
    t0 = time.time(); ref = Ref(name, size); N = ref.N; rng = np.random.default_rng(11 + size)
    T = ref.cheb(M); mex = T.mean(1)
    Xr = ref.rademacher(rng, NS * RMAX); mr = ref.vec_moments(Xr, M, T).reshape(M, NS, RMAX)
    Xi = ref.impartial(rng, NS * RMAX); mi = ref.vec_moments(Xi, M, T).reshape(M, NS, RMAX)
    Eref, Er, Ir, info0 = maxent_eval(mex[None].repeat(2, 0), ref, 1e-4, "floor")
    Nstep = cum_exact(ref, Er); rows = []
    cases = []   # (label, R, builder(sample)->(R,M) moments, sigma_mode, floor, nseeds)
    for R in (1, 16):
        for lab, src, K in (("rad", mr, 0), ("splice K=4", mr, 4), ("splice K=16", mr, 16), ("imp", mi, 0), ("imp+K4", mi, 4)):
            for mode, fl in ((("floor", 1e-4),) if R == 1 else (("sampled", 1e-4), ("sampled", 0.0))):
                if fl == 0.0: continue          # floor 0 dropped: too slow (see status note)
                elif R == 1 and K > 0: ns = 5   # known non-convergent (500 Newton steps each): 5 seeds only
                else: ns = NS
                cases.append((lab, R, src, K, mode, fl, ns))
    for lab, R, src, K, mode, fl, ns in cases:
        errs, cum, cumt, conv, its = [], [], [], [], []
        for s in range(ns):
            A = src[:, s, :R].T.copy()
            if K: A = splice(A.T, mex, K).T
            if R == 1: A = np.vstack([A, A]) if mode == "floor" else A      # maxent_fit needs R rows; sigma forced to 0 -> floor
            try:
                eb, E, idos, info = maxent_eval(A, ref, fl, mode)
            except Exception as ex:
                conv.append(False); its.append(-1); continue
            errs.append(eb - Eref); cum.append(np.abs(idos - Ir).max()); cumt.append(np.abs(idos - Nstep).max())
            conv.append(bool(info["converged"])); its.append(info["iterations"])
        e = np.array(errs); n = len(e)
        row = dict(system=name, size=size, N=N, M=M, R=R, est=lab, sigma=mode, floor=fl, nseeds=ns, converged_frac=float(np.mean(conv)), iters_median=float(np.median(its)))
        if n > 1:
            row.update(bias=float(e.mean()), se=float(e.std(ddof=1) / np.sqrt(n)), sd=float(e.std(ddof=1)), rmse=float(np.sqrt((e ** 2).mean())),
                       cum_pipe_mean=float(np.mean(cum)), cum_true_mean=float(np.mean(cumt)))
        rows.append(row)
    out.append(dict(size=size, N=N, rows=rows, ref_converged=bool(info0["converged"])))
    print(f"{name} N={N} done {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(f"results/e2maxent_{name}.json", "w"))
