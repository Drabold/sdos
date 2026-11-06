"""Maxent with a moment-uncertainty floor on exact moments of DTW4096 (Kwon TB): the exact-moment problem stops converging beyond M ~ 100; assigning every moment
(but mu_0) the uncertainty s0 restores convergence.  Same exact moments and metrics as bench_accuracy.py; writes bench_accuracy_floor.json."""
import json, time, warnings
import numpy as np
from sdos.reconstruct import reconstruct, band_quantities
from sdos.maxent import maxent, maxent_density, band_quantities_grid
warnings.filterwarnings("ignore")
ev = np.load("../dtw4096/dtw4096_kwon_ev.npy"); n = len(ev); nocc = n // 2
a_, b_ = 0.5 * (ev[-1] - ev[0]) * 1.01, 0.5 * (ev[-1] + ev[0])
th = np.arccos(np.clip((ev - b_) / a_, -1, 1))
mu = np.array([np.cos(m * th).mean() for m in range(400)])
ebs_ex = 2 * ev[:nocc].sum() / 4096
Eg = np.linspace(ev[0] - 0.5, ev[-1] + 0.5, 3000); ref = np.zeros_like(Eg)
for ch in np.array_split(ev, 40): ref += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.05) ** 2).sum(1)
ref /= n * 0.05 * np.sqrt(2 * np.pi)
out = []
for M in (20, 40, 70, 100, 170, 300, 400):
    for s0 in (0.0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3):
        sig = None if s0 == 0 else np.r_[0.0, np.full(M - 1, s0)]
        t = time.time(); lam, info = maxent(mu[:M], sig, default="flat", tol=1e-9, maxiter=300)
        Em, rm, im = maxent_density(lam, a_, b_, "flat", 4000, mu=mu[:M])
        try: d = (band_quantities_grid(Em, rm, im, n, n, 2.0)[1] / 4096 - ebs_ex) * 1e3
        except Exception: d = float("nan")
        L1 = float(np.trapezoid(np.abs(np.interp(Eg, Em, rm) - ref), Eg))
        out.append(dict(M=M, s0=s0, conv=bool(info["converged"]), it=int(info["iterations"]), dEBS=float(d), L1=L1, sec=time.time() - t))
        print(out[-1], flush=True)
json.dump(out, open("bench_accuracy_floor.json", "w"), indent=1)
