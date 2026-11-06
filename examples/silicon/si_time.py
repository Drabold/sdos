"""Timing of the matrix-free diamond-Si stencil (Kwon), KPM M=170, R=4, exact_moments=0, seed 7, at the current bound tolerance: bounds and moments timed separately."""
import sys, time, json, resource, numpy as np, sdos
from sdos.interfaces import kwon_si as K
out = []
for n in map(int, sys.argv[1:]):
    S = K.DiamondStencil(n); t = time.time(); lo, hi = sdos.spectral_bounds(S, verbose=False); tb = time.time() - t
    t = time.time(); r = sdos.dos(S, M=170, R=4, method="kpm", seed=7, exact_moments=0, bounds=(lo, hi)); tm = time.time() - t
    t = time.time(); rm = sdos.dos(S, M=170, R=4, method="maxent", seed=7, exact_moments=0, bounds=(lo, hi)); tx = time.time() - t
    row = dict(n=n, t_maxent_moments_plus_fit=tx, atoms=8 * n ** 3, orbitals=int(S.shape[0]), t_bounds=tb, t_moments=tm, t_total=tb + tm, bounds=[float(lo), float(hi)], peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    out.append(row); print(row, flush=True); json.dump(out, open("si_time.json", "w"), indent=1)
