"""Is the rise of the Maxent L1 error with M (DTW4096, Kwon, exact moments) due to the uncertainty floor s0?  Exact moments from the dense spectrum, Maxent with
floors 1e-3 ... 1e-6 (each M), L1 distance to the exact spectrum broadened by 0.05 eV.  usage: python maxent_floor_scan.py -> maxent_floor_scan.json"""
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from sdos.maxent import maxent_fit, maxent_density
ev = np.load("dtw4096_kwon_ev.npy"); N = len(ev); w = ev[-1] - ev[0]; lo, hi = ev[0] - 0.01 * w, ev[-1] + 0.01 * w; a, b = 0.5 * (hi - lo), 0.5 * (hi + lo)
x = (ev - b) / a; th = np.arccos(x); Eg = np.linspace(-14.5, 7.5, 3000); ex = np.zeros_like(Eg)
for ch in np.array_split(ev, 40): ex += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.05) ** 2).sum(1)
ex /= N * 0.05 * np.sqrt(2 * np.pi); out = []
for M in (170, 300, 400):
    mu = np.cos(np.outer(np.arange(M), th)).mean(1)
    for sf in (1e-3, 1e-4, 1e-5, 1e-6):
        t = time.time()
        lam, _, info = maxent_fit(np.vstack([mu, mu]), a, b, default="chebyshev", noise=0.0, jackknife=False, verbose=False, sigma_floor=sf)
        E, rho, _ = maxent_density(lam, a, b, default="chebyshev", mu=mu); rg = np.interp(Eg, E, rho)
        r = dict(M=M, floor=sf, converged=bool(info["converged"]), iters=int(info["iterations"]), L1=float(np.trapezoid(abs(rg - ex), Eg)), sec=time.time() - t)
        out.append(r); print(r, flush=True); json.dump(out, open("maxent_floor_scan.json", "w"), indent=1)
