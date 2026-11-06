"""Directive B3: is the stochastic estimator of the GENERALIZED problem (DTW4096 DFTB, Gamma, N = 16 384, Chebyshev B^-1 mode) unbiased?  Seeds x R = 96, M = 100, K = 0.
(a) z-score per moment against the exact moments from the dense generalized spectrum: z_n = (mean_R mu_hat_n - mu_n) / (s_n / sqrt(R)); rms z over n = 1..99 should be 1 if unbiased and the errors are as estimated.
(b) E_BS: KPM(mu_hat) - KPM(mu_exact) at the same M (the truncation bias cancels), meV/atom, and its standard error over seeds.
usage: python dtw_dftb_unbiased.py [nseeds] -> dtw_dftb_unbiased.json"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from numpy.polynomial import chebyshev as C
from sdos.interfaces import skf
from sdos.results import _rebuild
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 8; M = 100; R = 96
d = np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz"); X, L = d["X"], d["L"]
sk = skf.SKSet("/home/drabold/opt/slako/pbc-0-3", ["Si"]); H, S = skf.skf_hamiltonian(X, ["Si"] * len(X), L, sk); n = H.shape[0]
ev = np.load("dtw_dftb_gamma_ev.npy"); out = []
for sd in range(41, 41 + NS):
    t = time.time(); r = sdos.dos(H, B=S, M=M, R=R, method="kpm", seed=sd, binv="chebyshev", verbose=False, exact_moments=0)
    x = np.clip((ev - r["b"]) / r["a"], -1, 1); mu_ex = np.array([C.chebval(x, np.eye(M)[k]).mean() for k in range(M)])
    ms = r["mu_samples"]; mu = ms.mean(0); se = ms.std(0, ddof=1) / np.sqrt(R); z = (mu - mu_ex)[1:] / se[1:]
    r2 = dict(r); r2["mu"] = mu_ex; r2["mu_samples"] = np.tile(mu_ex, (R, 1))
    k1 = _rebuild(r, M, "kpm", "flat", None, 8000); k2 = _rebuild(r2, M, "kpm", "flat", None, 8000)
    e1 = sdos.fermi_and_band_energy(k1, n)[1]; e2 = sdos.fermi_and_band_energy(k2, n)[1]
    out.append(dict(seed=sd, rms_z=float(np.sqrt((z ** 2).mean())), mean_z=float(z.mean()), max_abs_z=float(np.abs(z).max()), dEBS_meV_atom=float((e1 - e2) / 4096 * 1e3), sec=time.time() - t, z=z.tolist()))
    print({k: v for k, v in out[-1].items() if k != "z"}, flush=True); json.dump(out, open("dtw_dftb_unbiased.json", "w"))
