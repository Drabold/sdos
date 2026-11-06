"""C3840 additions for the report (marked-up draft, p.7-9):
(1) Maxent s0 (sigma_floor) with EXACT moments from the exact spectrum, M=150, relaxed geometry; also sigma_floor=0.
(2) The same Hamiltonian on the UNRELAXED spherical (Caspar-Klug, projected to sphere) geometry: exact spectrum, KPM/Maxent M=150 R=12, gap, DOS near E_F,
    and geometry measures (radial spread; bond-length spread) of the two geometries."""
import json, sys, time
import numpy as np, scipy.linalg as sla
import sdos
from sdos.maxent import maxent_fit, maxent_density
from sdos.interfaces import xu_carbon as X
import goldberg as G
K, R = 150, 12
S = json.load(open("ssc96_C3840.json")); ev_rel = np.array(S["ev"]); a, b = S["maxent"]["a"], S["maxent"]["b"]
def cheb_moments(ev, a, b, K):
    x = np.clip((ev - b) / a, -1, 1); th = np.arccos(x)
    return np.array([np.cos(n * th).mean() for n in range(K)])
def smooth(E, ev, eta=0.1):
    out = np.zeros_like(E)
    for ch in np.array_split(ev, 60): out += np.exp(-0.5 * ((E[:, None] - ch[None]) / eta) ** 2).sum(1)
    return out / (len(ev) * eta * np.sqrt(2 * np.pi))
def l1(E, r1, r2): return float(np.trapezoid(abs(r1 - r2), E))
out = {}
# (1) exact moments, relaxed
mu = cheb_moments(ev_rel, a, b, K)
Eg = np.linspace(a * -0.999 + b, a * 0.999 + b, 4000); ex = smooth(Eg, ev_rel)
for name, sf in (("s0_1e-4", 1e-4), ("s0_0", 0.0)):
    t = time.time(); lam, _, info = maxent_fit(mu[None, :], a, b, default="chebyshev", noise=0, sigma_floor=sf, jackknife=False, verbose=False)
    E, rho, _ = maxent_density(lam, a, b, default="chebyshev", mu=mu)
    rg = np.interp(Eg, E, rho)
    out["relaxed_exactmoments_" + name] = dict(converged=bool(info["converged"]), iters=int(info["iterations"]), resid=float(info["max_moment_residual"]), L1_vs_exact_0p1eV=l1(Eg, rg, ex), sec=time.time() - t)
    np.savez(f"c3840_maxent_exact_{name}.npz", E=E, rho=rho)
    print(name, out["relaxed_exactmoments_" + name], flush=True)
for key in ("kpm", "maxent"):
    E, r = np.array(S[key]["E"]), np.array(S[key]["rho"]); out["relaxed_stochastic_" + key + "_L1"] = l1(Eg, np.interp(Eg, E, r), ex)
print({k: v for k, v in out.items() if "stoch" in k}, flush=True)
# (2) sphere
pos_s, nb, T = G.fullerene(8, 8, 1.42); pos_r = np.load("C3840_lcbop.npy")
def geom(p):
    c = p - p.mean(0); r = np.linalg.norm(c, axis=1)
    d = np.linalg.norm(p[:, None] - p[None], axis=-1) if len(p) < 5000 else None
    d.sort(1); b1 = d[:, 1:4]
    return dict(N=len(p), R_mean=float(r.mean()), R_std=float(r.std()), R_min=float(r.min()), R_max=float(r.max()), bond_mean=float(b1.mean()), bond_std=float(b1.std()), bond_min=float(b1.min()), bond_max=float(b1.max()))
out["geom_sphere"] = geom(pos_s); out["geom_relaxed"] = geom(pos_r); print(out["geom_sphere"], out["geom_relaxed"], flush=True)
H = X.xu_hamiltonian(pos_s); n = H.shape[0]
t = time.time(); ev_s = sla.eigvalsh(H.toarray(), driver="evd", overwrite_a=True, check_finite=False); print("exact sphere", f"{time.time()-t:.0f}s", flush=True)
nocc = n // 2; ef_s = 0.5 * (ev_s[nocc] + ev_s[nocc - 1]); out["sphere"] = dict(gap=float(ev_s[nocc] - ev_s[nocc - 1]), ef=float(ef_s), emin=float(ev_s[0]), emax=float(ev_s[-1]))
out["relaxed"] = dict(gap=float(S["gap"]), ef=float(S["ef"]), emin=float(ev_rel[0]), emax=float(ev_rel[-1]))
res = {}
for m in ("kpm", "maxent"):
    r = sdos.dos(H, M=K, R=R, method=m, seed=3, exact_moments=0, verbose=False)
    res[m] = dict(E=np.asarray(r.E).tolist(), rho=np.asarray(r.rho).tolist(), a=float(r.a), b=float(r.b))
Eg2 = np.linspace(ev_s[0] - 0.5, ev_s[-1] + 0.5, 4000); exs = smooth(Eg2, ev_s)
for m in res: out["sphere_" + m + "_L1"] = l1(Eg2, np.interp(Eg2, res[m]["E"], res[m]["rho"], left=0, right=0), exs)
# DOS at E_F (0.1 eV broadened exact), weight within 0.6 eV of E_F, in states per atom
for nm, ev_, ef_ in (("sphere", ev_s, ef_s), ("relaxed", ev_rel, S["ef"])):
    out[nm]["dos_EF_0p1"] = float(smooth(np.array([ef_]), ev_)[0]); out[nm]["frac_within_0.6eV_EF"] = float((abs(ev_ - ef_) < 0.6).mean())
    out[nm]["n_within_0.6eV_EF"] = int((abs(ev_ - ef_) < 0.6).sum())
out["spectrum_rms_diff_sphere_vs_relaxed_0p1"] = float(np.sqrt(np.trapezoid((smooth(Eg2, ev_s) - smooth(Eg2, ev_rel)) ** 2, Eg2)))
json.dump(dict(out=out, sphere_curves=res, ev_sphere=ev_s.tolist()), open("c3840_extra.json", "w"))
print(json.dumps(out, indent=1))
