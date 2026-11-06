"""Number of matrix-vector products used by the ARPACK spectral bounds (tol 1e-3, two eigsh calls) on the diamond-Si stencil, against size."""
import sys, time, json, numpy as np, scipy.sparse.linalg as spla, sdos
from sdos.interfaces import kwon_si as K
out = []
for n in map(int, sys.argv[1:]):
    S = K.DiamondStencil(n); cnt = [0]
    def mv(x):
        cnt[0] += 1; return S @ x
    op = spla.LinearOperator(S.shape, matvec=mv, dtype=S.dtype)
    t = time.time(); lo, hi = sdos.spectral_bounds(op, verbose=False); dt = time.time() - t
    out.append(dict(n=n, orbitals=int(S.shape[0]), matvecs=cnt[0], seconds=dt)); print(out[-1], flush=True)
json.dump(out, open("bounds_count.json", "w"), indent=1)
