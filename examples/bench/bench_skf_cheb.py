"""Cost scaling, non-orthogonal case: DFTB (.skf, pbc-0-3) H,S on tiled DTW4096, KPM M=100, R=2; RERUN of the Chebyshev row only, 16 GB cap."""
import sys, json, warnings
import numpy as np
import sdos
from sdos.interfaces import skf
from common import dtw_tiled, rss_gb, Timer
warnings.filterwarnings("ignore")
sk = skf.SKSet("/home/drabold/opt/slako/pbc-0-3", ["Si"])
out = []
for n in map(int, sys.argv[1:]):
    X, L = dtw_tiled(n)
    with Timer() as tb: H, S = skf.skf_hamiltonian(X, ["Si"] * len(X), L, sk, chunk=8000)
    row = dict(n=n, atoms=len(X), orbitals=H.shape[0], nnz_per_row=H.nnz / H.shape[0], t_build=tb.s, peak_rss_after_build_gb=rss_gb())
    for label, kw in (("chebyshev", dict(binv="chebyshev")),):
        if label == "pcg_ic" and n > 2:
            continue
        try:
            with Timer() as t: r = sdos.dos(H, B=S, M=100, R=2, method="kpm", seed=1, exact_moments=0, **kw)
            bi = r.get("binv_info", {})
            row[label] = dict(sec=t.s, peak_rss_gb=rss_gb(), binv={k: (float(v) if np.isscalar(v) else str(v)) for k, v in bi.items() if k in ("iterations", "nsolve", "degree", "kappa")})
        except Exception as exc:
            row[label] = dict(error=f"{type(exc).__name__}: {str(exc)[:150]}")
        print(n, label, row[label], flush=True)
    out.append(row)
    json.dump(out, open("bench_skf_cheb_rerun.json", "w"), indent=1)
    del H, S, X
