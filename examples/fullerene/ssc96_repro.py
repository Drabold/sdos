"""Reproduce the spirit of Fig. 2 of Drabold-Ordejon-Dong-Martin, Solid State Commun. 96, 833 (1995): C3840 DOS from ~150 moments (Maxent), here with the
Xu-Wang-Chan-Ho tight-binding Hamiltonian and LCBOP-relaxed geometry, against (i) exact eigenvalues and (ii) graphene from the same Hamiltonian and moment count."""
import json, sys, time
import numpy as np, scipy.linalg as sla
import sdos
from sdos.interfaces import xu_carbon as X

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3840
K, R = 150, 12
pos = np.load(f"C{N}_lcbop.npy")
H = X.xu_hamiltonian(pos)
n = H.shape[0]
t = time.time()
ev = sla.eigvalsh(H.toarray(), driver="evd", overwrite_a=True, check_finite=False)
print("exact eigenvalues", n, f"{time.time()-t:.0f}s", flush=True)
nocc = n // 2
gap = ev[nocc] - ev[nocc - 1]
ef = 0.5 * (ev[nocc] + ev[nocc - 1])
out = dict(N=N, n=n, K=K, R=R, gap=float(gap), ef=float(ef), ev=ev.tolist() if n <= 20000 else None)
res = {}
for m in ("kpm", "maxent"):
    r = sdos.dos(H, M=K, R=R, method=m, seed=3, exact_moments=0, verbose=False)
    res[m] = r
    out[m] = dict(E=np.asarray(r.E).tolist(), rho=np.asarray(r.rho).tolist(), a=float(r.a), b=float(r.b))
    print(m, "done", flush=True)
json.dump(out, open(f"ssc96_C{N}.json", "w"))
print("gap", gap, "ef", ef)
