"""Graphene (Xu Hamiltonian, exact k-sum on a 240x240 mesh): Maxent from EXACT moments, M = 400, bootstrapped (maxent_bootstrap) against the plain solver (maxent),
with and without the uncertainty floor. Times, convergence, Newton steps, L1 distance to the exact spectrum broadened by 0.05 eV."""
import json, time, numpy as np
from sdos.interfaces import xu_carbon as X
from sdos.maxent import maxent, maxent_bootstrap, _maxent_grid
a0 = 1.42 * 3 ** 0.5
a1, a2 = np.array([a0, 0, 0]), np.array([a0 / 2, a0 * 3 ** 0.5 / 2, 0]); tau = [np.zeros(3), np.array([a0 / 2, a0 / (2 * 3 ** 0.5), 0])]
terms = []
for al in range(2):
    for be in range(2):
        for n1 in range(-4, 5):
            for n2 in range(-4, 5):
                Rv = n1 * a1 + n2 * a2; d = tau[be] + Rv - tau[al]
                if 1e-9 < np.linalg.norm(d) < X.RM: terms.append((al, be, Rv, X.hopping_blocks(d[None])[0]))
b1 = 2 * np.pi * np.array([1 / a0, -1 / (a0 * 3 ** 0.5), 0]); b2 = 2 * np.pi * np.array([0, 2 / (a0 * 3 ** 0.5), 0])
def bands(k):
    H = np.zeros((len(k), 8, 8), complex)
    for al in range(2): H[:, 4*al:4*al+4, 4*al:4*al+4] += np.diag([X.EPS_S, X.EPS_P, X.EPS_P, X.EPS_P])
    for al, be, Rv, B in terms: H[:, 4*al:4*al+4, 4*be:4*be+4] += B[None] * np.exp(1j * k @ Rv)[:, None, None]
    return np.linalg.eigvalsh(H)
nk = 240; g = (np.arange(nk) + 0.5) / nk
f1, f2 = np.meshgrid(g, g, indexing="ij"); K = f1.reshape(-1)[:, None] * b1 + f2.reshape(-1)[:, None] * b2
ev = np.concatenate([bands(K[i:i + 4000]).ravel() for i in range(0, len(K), 4000)])
emin, emax = ev.min(), ev.max(); w = emax - emin; a = 0.5 * w * 1.02; b = 0.5 * (emax + emin)
x = np.clip((ev - b) / a, -1, 1); th = np.arccos(x); M = 400
mu = np.array([np.cos(n * th).mean() for n in range(M)])
Eg = np.linspace(emin - 0.5, emax + 0.5, 6000); ex = np.zeros_like(Eg)
for ch in np.array_split(ev, 200): ex += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.05) ** 2).sum(1)
ex /= len(ev) * 0.05 * np.sqrt(2 * np.pi)
def l1(E, r): return float(np.trapezoid(abs(np.interp(Eg, E, r, left=0, right=0) - ex), Eg))
out = dict(n_eig=len(ev), a=float(a), b=float(b), M=M); sig = np.zeros(M)
for name, fl in (("bootstrap_floor_1e-4", 1e-4), ("bootstrap_floor_0", 0.0)):
    t = time.time(); xs, rho, idos, info = maxent_bootstrap(mu, default="flat", sigma_floor=fl); sec = time.time() - t
    out[name] = dict(sec=sec, converged=bool(info["converged"]), moments_used=int(info["moments_used"]), L1=l1(a * xs + b, rho / a), stages=[(int(s[0]), bool(s[3]), int(s[4])) for s in info["stages"]])
    print(name, {k: v for k, v in out[name].items() if k != "stages"}, flush=True)
for name, fl in (("plain_floor_1e-4", 1e-4), ("plain_floor_0", 0.0)):
    sg = np.maximum(sig, fl); sg[0] = 0.0
    t = time.time(); lam, info = maxent(mu, sg, "flat", tol=1e-9, maxiter=300); sec = time.time() - t
    from sdos.maxent import maxent_density
    E, rho, _ = maxent_density(lam, a, b, default="flat", mu=mu)
    out[name] = dict(sec=sec, converged=bool(info["converged"]), iters=int(info["iterations"]), resid=float(info["max_moment_residual"]), L1=l1(E, rho))
    print(name, out[name], flush=True)
json.dump(out, open("bootstrap_graphene.json", "w"), indent=1)
