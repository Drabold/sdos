"""Complex Hermitian check: the DOS of a diamond-Si supercell with a Bloch twist (complex H) from sdos against the exact primitive-cell k-sum."""
import sys, time
import numpy as np
import sdos
from sdos.interfaces import kwon_si as K

n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
M, R = 170, 6
k = np.array([0.17, -0.09, 0.23])                       # twist (1/A), generic: H is genuinely complex
S = K.DiamondStencil(n, k)
# exact: spectrum of the supercell at twist k = union of the 8-atom-cell bands at k + (2 pi / L)(i, j, l)
G = 2 * np.pi / (n * K.A_LATTICE)
g = np.arange(n)
ks = k + G * np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
tab = K._neighbour_table()
Hk = np.zeros((len(ks), 32, 32), complex)
for al in range(8):
    Hk[:, 4 * al:4 * al + 4, 4 * al:4 * al + 4] += np.diag([K.ES, K.EP, K.EP, K.EP])
    for be, sh, B, d in tab[al]:
        Hk[:, 4 * al:4 * al + 4, 4 * be:4 * be + 4] += B[None] * np.exp(1j * ks @ d)[:, None, None]
ev = np.linalg.eigvalsh(Hk).ravel()
print(f"n={n}: {8 * n ** 3} atoms, {S.shape[0]} orbitals; exact eigenvalue range {ev.min():.3f} .. {ev.max():.3f}; hermiticity defect {sdos.operators.check_hermitian(S, raise_error=False) if hasattr(sdos.operators,'check_hermitian') else 'n/a'}", flush=True)
E0 = np.linspace(-14.5, 7.5, 1800)
def broaden(E, eta):
    d = np.zeros_like(E)
    for ch in np.array_split(ev, max(1, len(ev) // 2000)):
        d += np.exp(-0.5 * ((E[:, None] - ch[None]) / eta) ** 2).sum(1)
    return d / (len(ev) * eta * np.sqrt(2 * np.pi))
res = {}
for method in ("kpm", "maxent"):
    t = time.time()
    r = sdos.dos(S, M=M, R=R, method=method, seed=5, exact_moments=0)
    dt = time.time() - t
    ref = broaden(r.E, 0.10)
    sel = (r.E > -14.5) & (r.E < 7.5)
    l1 = np.trapezoid(np.abs(r.rho[sel] - ref[sel]), r.E[sel])
    res[method] = (r.E, r.rho)
    print(f"  {method}: M={M} R={R}  {dt:.0f} s;  L1 to exact(0.1 eV Gaussian) = {l1:.4f};  window [{r.b - r.a:.2f}, {r.b + r.a:.2f}]", flush=True)
np.savez(f"si_complex_n{n}.npz", ev=ev, k=k, **{f"{m}_E": v[0] for m, v in res.items()}, **{f"{m}_rho": v[1] for m, v in res.items()})
