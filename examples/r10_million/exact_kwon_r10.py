"""Exact spectrum of the 17^3 replica of R10 with the Kwon tight-binding Hamiltonian = union of the 864x864 H(k) of the 216-atom cell over the 17^3 k-mesh
(k = 2 pi n / (17 L0), n = 0..16; H(-k) = conj H(k) has the same eigenvalues: half the points are computed).  usage: python exact_kwon_r10.py [n=17]"""
import sys, time, numpy as np, scipy.sparse as sp
from sdos.interfaces import kwon_si as K
from common_r10 import *
n = int(sys.argv[1]) if len(sys.argv) > 1 else 17
x0 = r10_positions(); box = np.full(3, L0)
seen = {}; ev = []; t = time.time()
pts = [(a, b, c) for a in range(n) for b in range(n) for c in range(n)]
for (a, b, c) in pts:
    m = ((-a) % n, (-b) % n, (-c) % n)
    if m in seen: seen[(a, b, c)] = seen[m]; continue
    k = 2 * np.pi * np.array([a, b, c]) / (n * L0)
    Hk = K.kwon_hamiltonian(x0, box, kpoint=k if (a, b, c) != (0, 0, 0) else None).toarray()
    w = np.linalg.eigvalsh(Hk); seen[(a, b, c)] = w; ev.append(w)
    if len(ev) % 200 == 0: print(len(ev), f"{time.time() - t:.0f} s", flush=True)
allev = np.sort(np.concatenate([seen[p] for p in pts]))
np.save(f"kwon_r10_exact_ev_{n}.npy", allev); print("done", len(allev), "levels", f"{time.time() - t:.0f} s")
