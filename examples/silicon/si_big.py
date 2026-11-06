"""Diamond Si, Kwon TB, Gamma point of an n^3-cell supercell (8 n^3 atoms, 32 n^3 orbitals): matrix-free KPM and Maxent DOS against the exact k-sum
(Gamma-only supercell spectrum = the 8-atom cell bands on the n^3 mesh).  Cf. Extended Data Fig. 9 of Deringer et al., Nature 589, 59 (2021)."""
import sys, time, resource, json
import numpy as np
import sdos
from sdos.interfaces import kwon_si as K

n = int(sys.argv[1]); M = int(sys.argv[2]); R = int(sys.argv[3])
S = K.DiamondStencil(n)
print(f"n={n}: {8 * n ** 3} atoms, {S.shape[0]} orbitals", flush=True)
out = dict(n=n, M=M, R=R, atoms=8 * n ** 3)
for method in ("kpm", "maxent"):
    t = time.time()
    r = sdos.dos(S, M=M, R=R, method=method, seed=7, exact_moments=0)
    out[method] = dict(E=np.asarray(r.E).tolist(), rho=np.asarray(r.rho).tolist(), rho_err=np.asarray(r.rho_err).tolist() if r.R > 1 else None, seconds=time.time() - t,
                       a=float(r.a), b=float(r.b))
    print(f"  {method}: {time.time() - t:.0f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
    json.dump(out, open(f"si_big_n{n}.json", "w"))
# exact reference: Gamma-supercell spectrum = bands of the 8-atom cell on the n^3 mesh, binned in a fine histogram (no eigenvalue storage)
tab = K._neighbour_table()
G = 2 * np.pi / (n * K.A_LATTICE)
g = np.arange(n)
edges = np.linspace(-14.5, 7.5, 4401)
hist = np.zeros(len(edges) - 1)
grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * G
nk = len(grid)
for c0 in range(0, nk, 4096):
    ks = grid[c0:c0 + 4096]
    Hk = np.zeros((len(ks), 32, 32), complex)
    for al in range(8):
        Hk[:, 4 * al:4 * al + 4, 4 * al:4 * al + 4] += np.diag([K.ES, K.EP, K.EP, K.EP])
        for be, sh, B, d in tab[al]:
            Hk[:, 4 * al:4 * al + 4, 4 * be:4 * be + 4] += B[None] * np.exp(1j * ks @ d)[:, None, None]
    hist += np.histogram(np.linalg.eigvalsh(Hk).ravel(), bins=edges)[0]
out["exact_hist"] = dict(edges=edges.tolist(), counts=hist.tolist(), n_orb=int(hist.sum()))
json.dump(out, open(f"si_big_n{n}.json", "w"))
print("done", flush=True)
