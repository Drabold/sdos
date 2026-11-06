"""Matrix-free ACE vibrational DOS of the 17^3 replica of R10 vs the exact spectrum: integrated mode count at low frequency (no broadening), all thermodynamic functions
(ZPE, U, F, S, Cv versus T) from sdos.harmonic_thermo for KPM and Maxent against the exact levels, and the low-frequency (propagon) region against the Debye law.
usage: python r10_vib_analysis.py result.npz exact_lambda.npy"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from sdos.results import _rebuild
from sdos.thermo import HBAR_OMEGA_METAL
f, fl = sys.argv[1], sys.argv[2]; r = sdos.load(f); lam = np.sort(np.load(fl)); N = len(lam); nat = N // 3; kb = 8.617333262e-5
T = np.array([5., 10., 20., 50., 100., 200., 300., 600.]); out = {}
def thermo_exact(T):
    om = np.sqrt(lam[lam > 1e-6]) * HBAR_OMEGA_METAL; x = om[None] / (kb * T[:, None])
    zpe = 0.5 * om.sum() / nat; U = (om[None] * (0.5 + 1 / np.expm1(x))).sum(1) / nat; F = (0.5 * om[None] + kb * T[:, None] * np.log1p(-np.exp(-x))).sum(1) / nat
    S = (x / np.expm1(x) - np.log1p(-np.exp(-x))).sum(1) / nat; Cv = (x ** 2 * np.exp(x) / np.expm1(x) ** 2).sum(1) / nat
    return dict(zpe=zpe, U=U, F=F, S=S, Cv=Cv)
ex = thermo_exact(T); res = {"kpm": r}
for M in (170, 300, r["M"]):
    try: res[f"maxent{M}"] = _rebuild(r, M, "maxent", "flat", None, 8000)
    except Exception as e: print("maxent", M, "failed", e)
res["kpm170"] = _rebuild(r, 170, "kpm", "flat", None, 8000)
for name, d in res.items():
    E = np.asarray(d["E"]); rho = np.asarray(d["rho"]); pos = E > 0; om = np.sqrt(E[pos]); g = 2 * om * rho[pos]
    dd = dict(omega=om, g=g, E=E, rho=rho, idos=d["idos"], N=d["N"], M=d.get("M", r["M"])); th = sdos.harmonic_thermo(dd, T)
    o = dict(zpe_err_pct=100 * (float(th.zpe) / ex["zpe"] - 1), Cv_err_pct=(100 * (np.asarray(th.Cv) / ex["Cv"] - 1)).round(2).tolist(), S_err_pct=(100 * (np.asarray(th.S) / ex["S"] - 1)).round(2).tolist(),
             F_err_meV=((np.asarray(th.F) - ex["F"]) * 1e3).round(4).tolist(), U_err_meV=((np.asarray(th.U) - ex["U"]) * 1e3).round(4).tolist())
    ex_cnt = np.searchsorted(lam, E, side="right") / N; sel = (E > 0) & (E < 0.02)
    o["max_count_err_lowfreq"] = float(np.abs(np.asarray(d["idos"])[sel] - ex_cnt[sel]).max() * N); o["max_count_err_all"] = float(np.abs(np.asarray(d["idos"]) - ex_cnt).max() * N)
    out[name] = o; print(name, {k: (v if not isinstance(v, list) else v) for k, v in o.items()}, flush=True)
out["exact"] = {k: (np.asarray(v).round(6).tolist() if not np.isscalar(v) else float(v)) for k, v in ex.items()}; out["T"] = T.tolist(); out["N"] = N
json.dump(out, open("r10_vib_analysis.json", "w"), indent=1)
