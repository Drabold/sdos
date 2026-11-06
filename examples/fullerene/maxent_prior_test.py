"""Maxent default model vs spectral-window width, with exact moments (A) and stochastic moments of a real Hamiltonian (B)."""
import sys, time, warnings
import numpy as np, scipy.linalg as sla
import sdos
from sdos.interfaces import xu_carbon as X
warnings.filterwarnings("ignore")
PRIORS = ("chebyshev", "flat", "kpm", "kpm:K/5")
def pr(d, K): return d if d != "kpm:K/5" else f"kpm:{max(8, K // 5)}"

print("(A) EXACT moments of a hard-edged density; window = support widened by f x width each side")
rng = np.random.default_rng(0)
E = np.concatenate([rng.uniform(-1, 0.15, 120000) * 18, 0.9 * np.sqrt(rng.uniform(0, 1, 80000)) * 8 + 4]); lo, hi = E.min(), E.max(); W = hi - lo
Eg = np.linspace(lo + 0.4, hi - 0.4, 1500); ref = np.zeros_like(Eg)
for ch in np.array_split(E, 200): ref += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.06) ** 2).sum(1)
ref /= len(E) * 0.06 * np.sqrt(2 * np.pi)
hdr = "  f     K  | " + " | ".join(f"{d:>22s}" for d in PRIORS) + " | KPM L1"
print(hdr); print("      (each: converged?/max|lambda|/L1 error)")
for f in (0.0, 0.02, 0.3, 1.5):
    l, h = lo - f * W, hi + f * W; A = 0.5 * (h - l); B = 0.5 * (h + l); th = np.arccos(np.clip((E - B) / A, -1, 1))
    for K in (60, 150):
        mu = np.array([np.cos(n * th).mean() for n in range(K)]); cells = []
        for d in PRIORS:
            dd = pr(d, K); lam, info = sdos.maxent(mu, default=dd); Em, rm, _ = sdos.maxent_density(lam, A, B, dd, 4000, mu=mu)
            cells.append(f"{'Y' if info['converged'] else 'n'} {info['max_abs_lambda']:8.1f} {np.trapezoid(np.abs(np.interp(Eg, Em, rm) - ref), Eg):.4f}")
        Ek, rk, _ = sdos.reconstruct(mu, A, B, 3000); l1k = np.trapezoid(np.abs(np.interp(Eg, Ek, rk) - ref), Eg)
        print(f"{f:5.2f} {K:4d} | " + " | ".join(f"{c:>22s}" for c in cells) + f" | {l1k:.4f}")

print("\n(B) STOCHASTIC moments (R=6 vectors) of the Xu C960 Hamiltonian; reference = exact eigenvalues, Gaussian 0.10 eV")
pos = np.load("C960_lcbop.npy"); H = X.xu_hamiltonian(pos); ev = sla.eigvalsh(H.toarray())
Er = np.linspace(ev[0] + 1, ev[-1] - 1, 1500); rr = np.zeros_like(Er)
for ch in np.array_split(ev, 40): rr += np.exp(-0.5 * ((Er[:, None] - ch[None]) / 0.10) ** 2).sum(1)
rr /= len(ev) * 0.10 * np.sqrt(2 * np.pi)
print(f"  margin   K | " + " | ".join(f"{d:>22s}" for d in PRIORS[:3]) + " | KPM L1 (Jackson, same moments)")
for margin in (0.01, 0.1, 0.5):
    for K in (60, 150):
        cells = []
        for d in PRIORS[:3]:
            r = sdos.estimate_dos(H, M=K, R=6, seed=3, method="maxent", maxent_default=d, margin=margin, exact_moments=0, verbose=False, npts=4000)
            cells.append(f"{'Y' if r.maxent_info['converged'] else 'n'} {r.maxent_info['max_abs_lambda']:8.1f} {np.trapezoid(np.abs(np.interp(Er, r.E, r.rho) - rr), Er):.4f}")
        rk = sdos.estimate_dos(H, M=K, R=6, seed=3, margin=margin, exact_moments=0, verbose=False, npts=3000)
        print(f"  {margin:6.2f} {K:4d} | " + " | ".join(f"{c:>22s}" for c in cells) + f" | {np.trapezoid(np.abs(np.interp(Er, rk.E, rk.rho) - rr), Er):.4f}")
