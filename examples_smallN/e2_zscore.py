"""rms z-score of the moment residuals of the impartial(3) ensemble vs N (Rademacher as the control): z_n = (mean_v mu_n(v) - mu_n) / (sd_n / sqrt(V)), n = 3 .. 63
(n = 0, 1, 2 are exact by construction).  ~1 for an unbiased ensemble.  usage: python e2_zscore.py -> results/e2_zscore.json"""
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
out = []; M = 64
for name in ("chain", "ssh", "kwon", "anderson"):
    for size in SYSTEMS[name][1]:
        ref = Ref(name, size); N = ref.N; V = 2000 if N <= 1024 else 600; rng = np.random.default_rng(3 + size); T = ref.cheb(M); mex = T.mean(1)
        rec = dict(system=name, N=N, V=V)
        for lab, X in (("rad", ref.rademacher(rng, V)), ("imp", ref.impartial(rng, V))):
            m = ref.vec_moments(X, M, T)
            z = (m.mean(1) - mex) / (m.std(1, ddof=1) / np.sqrt(V))
            rec[lab] = dict(rms_z=float(np.sqrt((z[3:] ** 2).mean())), mean_z=float(z[3:].mean()), max_abs_z=float(np.abs(z[3:]).max()))
        out.append(rec); print(name, N, {k: round(v["rms_z"], 2) for k, v in rec.items() if isinstance(v, dict)}, flush=True)
        json.dump(out, open("results/e2_zscore.json", "w"))
