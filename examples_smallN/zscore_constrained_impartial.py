"""Reproducible test behind the statement about impartial vectors constrained on K moments: rms z-score of the moment residuals of the ensemble.
Vector x (|x|^2 = N) is built by conjugate gradients from a Gaussian start, minimising P(x) = sum_{n=1..K} (x^T T_n(Ht) x / N - mu_n)^2 + (|x|^2/N - 1)^2
with the EXACT moments mu_n (K = 2: the three constrained moments mu_0, mu_1, mu_2 of Drabold-Sankey; K = 4, 8, 16 constrain more).
Implemented in the eigenbasis (x = V c, moments linear in w = c^2) -- identical mathematics to the matrix-vector version, O(N) per CG step.
z_n = (mean over V vectors of mu_n(x) - mu_n) / (sd_n / sqrt(V)) for the UNCONSTRAINED moments n = K+1 .. 63; rms z is ~1 for an unbiased ensemble.
usage: python zscore_constrained_impartial.py -> results/zscore_constrained.json"""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize
from common import *
def vectors(ref, K, V, rng, T):
    N = ref.N; mex = T[:K + 1].mean(1); out = np.empty((N, V))
    for v in range(V):
        c0 = rng.normal(size=N); c0 *= np.sqrt(N) / np.linalg.norm(c0)
        def fun(c):
            w = c * c; r = T[:K + 1] @ w / N - mex; r[0] = w.sum() / N - 1.0
            g = (2.0 / N) * c * 2.0 * (T[:K + 1].T @ r)
            return float(r @ r), g
        res = minimize(fun, c0, jac=True, method="CG", options=dict(maxiter=2000, gtol=1e-16))
        out[:, v] = res.x
    return out          # coefficients c in the eigenbasis
out = []; M = 64
for name, size, V in (("kwon", 3, 1500), ("kwon", 4, 1000), ("anderson", 10, 1500), ("chain", 512, 1500)):
    ref = Ref(name, size); N = ref.N; rng = np.random.default_rng(9); T = ref.cheb(M); mex = T.mean(1)
    rec = dict(system=name, N=N, V=V, rms_z={})
    for K in (2, 4, 8, 16):
        C = vectors(ref, K, V, rng, T); m = T @ (C * C) / N
        z = (m.mean(1) - mex) / (m.std(1, ddof=1) / np.sqrt(V))
        rec["rms_z"][K] = float(np.sqrt((z[K + 1:] ** 2).mean())); rec.setdefault("max_resid", {})[K] = float(np.abs(m[:K + 1] - mex[:K + 1, None]).max())
    cR = rng.choice([-1.0, 1.0], size=(N, V)); mR = ref.vec_moments(cR, M, T)
    zR = (mR.mean(1) - mex) / (mR.std(1, ddof=1) / np.sqrt(V)); rec["rad_control"] = float(np.sqrt((zR[3:] ** 2).mean()))
    out.append(rec); print(name, N, rec, flush=True)
    json.dump(out, open("results/zscore_constrained.json", "w"), indent=1)
