"""E3: (1) empirical covariance of mu_hat_n for random vectors vs the Gaussian formula N^-1 (mu_{m+n} + mu_|m-n|) and the exact Rademacher formula;
(2) control-variate tail: mu_cv_n(r) = mu_n(r) - beta_n^T (mu_head(r) - mu_head^exact), beta from INDEPENDENT pilot vectors; head set exact.
usage: python e3_cov.py -> results/e3.json"""
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
CASES = [("chain", 64), ("chain", 256), ("chain", 1024), ("ssh", 256), ("ssh", 1024), ("kwon", 2), ("kwon", 3), ("kwon", 4), ("anderson", 8), ("anderson", 16)]
M = 100; MC = 64; NP = 400; NS = 500; out = []
def Ebs_cum(A, ref, Iref, Eref):
    E, rho, idos = reconstruct(A, ref.a, ref.b, 4096)
    eb = np.array([band_quantities_grid(E, None, idos[s], 1, ref.nocc / ref.N, 1.0)[1] for s in range(A.shape[0])])
    return eb - Eref, np.abs(idos - Iref[None]).max(1)
for name, size in CASES:
    t0 = time.time(); ref = Ref(name, size); N = ref.N; rng = np.random.default_rng(31 + size)
    T = ref.cheb(M); mex = T.mean(1); rec = dict(system=name, size=size, N=N)
    # ---- (1) covariance, Rademacher vectors, n = 0..MC-1
    NV = 20000 if N <= 1024 else 8000
    mv = np.concatenate([ref.vec_moments(ref.rademacher(rng, 2000), MC, T[:MC]) for _ in range(NV // 2000)], axis=1)   # (MC, NV)
    C = np.cov(mv)
    mm, nn = np.meshgrid(np.arange(MC), np.arange(MC), indexing="ij")
    ext = np.concatenate([mex[:MC], ref.cheb(2 * MC).mean(1)[MC:]])          # mu_0 .. mu_{2MC-1}
    Cg = (ext[mm + nn] + ext[np.abs(mm - nn)]) / N
    # exact Rademacher: 2[ tr(T_m T_n) - sum_i (T_m)_ii (T_n)_ii ] / N^2 ; diag_n(i) = sum_k V_ik^2 T_n(x_k)
    D = (ref.V ** 2) @ T[:MC].T                                              # (N, MC)
    Cr = Cg - 2.0 * (D.T @ D) / N ** 2
    s = N ** -1.0 * 0 + np.sqrt((np.diag(C)[:, None] * np.diag(C)[None]) / NV)    # rough statistical scale of each entry
    sel = (mm >= 1) & (nn >= 1)
    rec["cov"] = dict(NV=NV, rel_dev_gauss=float(np.abs(C - Cg)[sel].max() / np.abs(C[sel]).max()), rel_dev_rad=float(np.abs(C - Cr)[sel].max() / np.abs(C[sel]).max()),
                      zmax_gauss=float((np.abs(C - Cg) / (s + 1e-300))[sel].max()), zmax_rad=float((np.abs(C - Cr) / (s + 1e-300))[sel].max()),
                      diag_term_frac=float(np.abs(Cg - Cr)[sel].max() / np.abs(Cg[sel]).max()), var_n1_emp=float(C[1, 1]), var_n1_gauss=float(Cg[1, 1]), var_n1_rad=float(Cr[1, 1]))
    # ---- (2) control variate
    Eref, Ir_, _ = None, None, None
    Er, Egrid, Iref = None, None, None
    E0, rho0, idos0 = reconstruct(mex[None], ref.a, ref.b, 4096); Eref = band_quantities_grid(E0, None, idos0[0], 1, ref.nocc / ref.N, 1.0)[1]; Iref = idos0[0]
    pil = ref.vec_moments(ref.rademacher(rng, NP), M, T)                      # (M, NP) independent pilot
    Xs = ref.rademacher(rng, NS * 4); ms = ref.vec_moments(Xs, M, T).reshape(M, NS, 4)
    rec["cv"] = []
    for K in (4, 8, 16):
        Cp = np.cov(pil)                                                      # (M, M)
        h = np.arange(1, K + 1); tl = np.arange(K + 1, M)
        beta = Cp[np.ix_(tl, h)] @ np.linalg.inv(Cp[np.ix_(h, h)])            # (tail, head)
        for R in (1, 4):
            v = ms[:, :, :R]                                                  # (M, NS, R)
            plain = v.mean(2)
            spl = splice(plain, mex, K)
            cvv = v.copy()
            dh = v[h] - mex[h][:, None, None]                                 # (K, NS, R) head deviation per vector
            cvv[tl] = v[tl] - np.einsum("tk,ksr->tsr", beta, dh)
            cv = splice(cvv.mean(2), mex, K)
            res = {}
            for lab, A in (("rad", plain), ("splice", spl), ("cv", cv)):
                e, c = Ebs_cum(A.T.copy(), ref, Iref, Eref)
                res[lab] = dict(bias=float(e.mean()), se=float(e.std(ddof=1) / np.sqrt(NS)), sd=float(e.std(ddof=1)), cum=float(c.mean()))
            res.update(K=K, R=R, var_ratio_cv_over_splice=float(res["cv"]["sd"] ** 2 / res["splice"]["sd"] ** 2), cum_ratio=float(res["cv"]["cum"] / res["splice"]["cum"]))
            rec["cv"].append(res)
    out.append(rec)
    print(f"{name} N={N} {time.time()-t0:.0f}s  cov dev gauss {rec['cov']['rel_dev_gauss']:.3f} rad {rec['cov']['rel_dev_rad']:.3f} | cv var ratio K=4,8,16 (R=1): "
          + " ".join(f"{r['var_ratio_cv_over_splice']:.2f}" for r in rec['cv'] if r['R'] == 1), flush=True)
    json.dump(out, open("results/e3.json", "w"))
