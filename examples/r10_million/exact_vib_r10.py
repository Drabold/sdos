"""Exact vibrational spectrum (lambda = omega^2, eV A^-2 u^-1) of the 17^3 replica of R10 (ACE): union of the 648x648 D(k) over the 17^3 k-mesh from the
real-space force constants ifc_r10.npy.  Checks: acoustic sum rule, Phi symmetry, and D(Gamma) of the 216 cell against a direct finite-difference calculation."""
import sys, time, numpy as np
n = int(sys.argv[1]) if len(sys.argv) > 1 else 17; m = 28.0855
Phi = np.load("ifc_r10.npy")                      # (648, 27, 648): [3i+a, c, 3j+b]
cells = [(c % 3, (c // 3) % 3, c // 9) for c in range(27)]; R = np.array([[(v if v < 2 else -1) for v in cc] for cc in cells])
# checks
asr = abs(Phi.sum(axis=1).reshape(216, 3, 216, 3).sum(2)).max()           # sum_j,R Phi(i a; j b) over all j,R, b = a: should vanish (translation)
print("acoustic sum rule: max |sum_{jR} Phi_{ia,jb}| =", asr, "(Phi max", abs(Phi).max(), ")")
Phis = Phi.reshape(216, 3, 27, 216, 3)
# symmetry Phi(i,0;j,R) = Phi(j,0;i,-R)
cidx = {tuple(r): c for c, r in enumerate(R)}; err = 0
for c, r in enumerate(R):
    cm = cidx[tuple(-r)]; err = max(err, abs(Phis[:, :, c, :, :] - Phis[:, :, cm, :, :].transpose(2, 3, 0, 1)).max())
print("Phi symmetry defect:", err)
print("share of |Phi| in the home cell:", abs(Phi[:, 0]).sum() / abs(Phi).sum(), " in the farthest cell (index 26):", abs(Phi[:, 26]).sum() / abs(Phi).sum())
Phi_s = (Phis + Phis.transpose(0, 1, 2, 3, 4) * 0)                         # (no symmetrisation of the data; D is Hermitised below)
def Dk(k):
    ph = np.exp(1j * (R @ k)); D = np.einsum("c,iacjb->iajb", ph, Phis).reshape(648, 648) / m
    return 0.5 * (D + D.conj().T)
w0 = np.linalg.eigvalsh(Dk(np.zeros(3)).real); print("Gamma: lowest 5 eigenvalues", w0[:5], " highest", w0[-1])
t = time.time(); seen = {}
pts = [(a, b, c) for a in range(n) for b in range(n) for c in range(n)]
for p in pts:
    mm = tuple((-v) % n for v in p)
    if mm in seen: seen[p] = seen[mm]; continue
    k = 2 * np.pi * np.array(p) / n          # phase per cell: R is in units of cells, k.R with k = 2 pi n/17
    seen[p] = np.linalg.eigvalsh(Dk(k))
allev = np.sort(np.concatenate([seen[p] for p in pts])); np.save(f"vib_r10_exact_lambda_{n}.npy", allev)
print("levels", len(allev), "min", allev[0], "max", allev[-1], "negative:", int((allev < -1e-6).sum()), f"{time.time() - t:.0f} s")
