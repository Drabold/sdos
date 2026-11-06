"""E2 (KPM part): bias / variance of E_BS and of the cumulative distribution for plain, spliced and impartial vectors.
Error = estimate - estimate from the EXACT moment list with the same M and kernel (isolates the stochastic estimator; kernel truncation is common),
and, for the cumulative curve, also against the exact step function N(E)/N.   E_BS per orbital, half filling, spin 1.
usage: python e2_kpm.py SYSTEM  -> results/e2kpm_SYSTEM.json"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
name = sys.argv[1]; RS = (1, 2, 4, 16); RMAX = 16; KS = (2, 4, 8, 16); NPTS = 4096
def ebs_batch(A, ref):
    """KPM E_BS per orbital and idos for a batch of moment lists A (S, M)."""
    E, rho, idos = reconstruct(A, ref.a, ref.b, NPTS)
    eb = np.array([band_quantities_grid(E, None, idos[s], 1, ref.nocc / ref.N * 1.0, 1.0)[1] for s in range(A.shape[0])])
    return eb, E, idos
out = []
for size in SYSTEMS[name][1]:
    t0 = time.time(); ref = Ref(name, size); N = ref.N; W = float(ref.ev[-1] - ref.ev[0])
    NS = 500 if N <= 1024 else 200
    Ms = [50, 100, 200] + [m for m in (2 * N, 4 * N) if m > 200 and m <= 512]
    rng = np.random.default_rng(7 + size); Mmax = max(Ms); T = ref.cheb(Mmax); mu_ex_full = T.mean(1)
    Xr = ref.rademacher(rng, NS * RMAX); mr = ref.vec_moments(Xr, Mmax, T).reshape(Mmax, NS, RMAX); del Xr
    Xi = ref.impartial(rng, NS * RMAX); mi = ref.vec_moments(Xi, Mmax, T).reshape(Mmax, NS, RMAX); del Xi
    # sanity: impartial first three moments exact per vector
    imp_res = float(np.abs(mi[:3] - mu_ex_full[:3, None, None]).max())
    rows = []
    for M in Ms:
        mex = mu_ex_full[:M]
        Eref, Egrid, Iref = ebs_batch(mex[None], ref); Eref = Eref[0]; Iref = Iref[0]
        Nstep = cum_exact(ref, Egrid)
        cum_floor = float(np.abs(Iref - Nstep).max())
        for R in RS:
            base = {"rad": mr[:M, :, :R].mean(2), "imp": mi[:M, :, :R].mean(2)}
            ests = {"rad": base["rad"], "imp": base["imp"], "imp+K4": splice(base["imp"], mex, 4)}
            for K in KS: ests[f"splice K={K}"] = splice(base["rad"], mex, K)
            for lab, A in ests.items():
                eb, _, idos = ebs_batch(A.T.copy(), ref)
                err = (eb - Eref) / 1.0                         # per orbital (band_quantities_grid with n_orb=1 -> per orbital)
                cp = np.abs(idos - Iref[None]).max(1); ct = np.abs(idos - Nstep[None]).max(1)
                rows.append(dict(system=name, size=size, N=N, W=W, M=M, R=R, est=lab, nseeds=NS,
                                 bias=float(err.mean()), se=float(err.std(ddof=1) / np.sqrt(NS)), sd=float(err.std(ddof=1)),
                                 rmse=float(np.sqrt((err ** 2).mean())), cum_pipe_mean=float(cp.mean()), cum_true_mean=float(ct.mean()),
                                 cum_pipe_rms=float(np.sqrt((cp ** 2).mean())), kpm_trunc_cum=cum_floor))
    out.append(dict(size=size, N=N, W=W, imp_moment_resid=imp_res, rows=rows))
    print(f"{name} N={N} done {time.time()-t0:.0f}s (impartial mu_0..2 resid {imp_res:.1e})", flush=True)
    json.dump(out, open(f"results/e2kpm_{name}.json", "w"))
