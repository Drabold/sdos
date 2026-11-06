"""Cost scaling, orthogonal case: Kwon TB on tiled DTW4096, KPM M=100, R=2 (exact_moments=0), times of build / bounds / moments and peak RSS."""
import sys, json, time, warnings
import numpy as np
import sdos
from sdos.interfaces import kwon_si as K
from common import dtw_tiled, rss_gb, Timer
warnings.filterwarnings("ignore")
out = []
for n in map(int, sys.argv[1:]):
    X, L = dtw_tiled(n)
    with Timer() as tb: H = K.kwon_hamiltonian(X, L, chunk=16000)
    with Timer() as tbd: lo, hi = sdos.spectral_bounds(H, verbose=False)
    with Timer() as tm: r = sdos.dos(H, M=100, R=2, method="kpm", seed=1, exact_moments=0, bounds=(lo, hi))
    row = dict(n=n, atoms=len(X), orbitals=H.shape[0], nnz_per_row=H.nnz / H.shape[0], t_build=tb.s, t_bounds=tbd.s, t_moments=tm.s, peak_rss_gb=rss_gb(), window=[float(lo), float(hi)])
    out.append(row); print(row, flush=True)
    json.dump(out, open("bench_kwon.json", "w"), indent=1)
    del H, X
