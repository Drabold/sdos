"""Eigenstates in an energy window with their IPR: extended (weak disorder) versus localized (strong disorder) states."""
import sys
sys.path.insert(0, "../tests")
import numpy as np
import scipy.linalg as sla
import sdos
from conftest import cubic

for W in (3.0, 30.0):                                                # 3D Anderson model: critical disorder ~ 16.5
    H = cubic(8, W=W, seed=2)
    ev = sla.eigvalsh(H.toarray())
    r = sdos.window_eigs(H, None, (ev[200] - 1e-6, ev[300] + 1e-6), method="shift-invert")   # exact, not O(N)
    f = sdos.window_eigs(H, None, (ev[200] - 1e-6, ev[300] + 1e-6), method="filter", degree=400, tol=1e-6)   # scalable route
    print(f"W = {W:4.1f}: {len(r.E)} states (filter found {len(f.E)} converged, edge width {f.info['edge_width']:.3f}); "
          f"mean N*IPR = {r.ipr.mean() * H.shape[0]:.1f}  (extended ~ 3, localized >> 3)")
