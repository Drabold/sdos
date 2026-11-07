"""Plain Lanczos (full reorthogonalization) on the matrix-free ACE Hessian of the relaxed Morrow 1M model: extreme Ritz values vs steps.  Diagnoses the bounds problem (ARPACK gave [-29, 69])."""
import sys, time, json, warnings; warnings.filterwarnings("ignore")
import numpy as np
exec(open("morrow_vib.py").read().split("F = KKForces")[0].split("M, R = ")[0])   # imports
src = open("morrow_vib.py").read(); cls = src[src.index("class KKForces"):src.index("F = KKForces")]; exec(cls)
F = KKForces("morrow_relaxed.data"); DELTA = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3; op = HessianOperator(F, F.x0, F.mass, DELTA); n = op.shape[0]; K = int(sys.argv[1]); print("delta", DELTA, flush=True)
rng = np.random.default_rng(7); v = rng.standard_normal(n); v /= np.linalg.norm(v); V = [v]; al = []; be = []; w = None; out = []
for j in range(K):
    t = time.time(); w = op @ V[-1]; a = float(V[-1] @ w); al.append(a)
    w = w - a * V[-1] - (be[-1] * V[-2] if j else 0)
    for q in V: w -= (q @ w) * q                       # full reorthogonalization
    b = float(np.linalg.norm(w)); be.append(b); V.append(w / b)
    T = np.diag(al) + np.diag(be[:-1], 1) + np.diag(be[:-1], -1); ev = np.linalg.eigvalsh(T)
    out.append(dict(step=j + 1, lo=float(ev[0]), hi=float(ev[-1]), sec=time.time() - t)); print(out[-1], flush=True); json.dump(out, open("morrow_lanczos.json", "w"))
import os; os._exit(0)
