"""From fullerenes to graphene: what large spectral computations tell you (runs in about a minute).

1. Exact HOMO-LUMO gaps of icosahedral C_N (Xu-Wang-Chan-Ho tight binding, LCBOP-relaxed geometries) by inertia bisection: no eigensolver.
2. hbar v_F of graphene from the bands of the same Hamiltonian, and the scaling  gap ~ c hbar v_F / R  of a confined Dirac cone (c -> ~1.9, a smooth sphere gives 2).
3. States nearest E_F by shift-invert, with their Ih multiplet structure and IPR (extended states: IPR ~ 1/N).
4. The non-orthogonal check: SIESTA Harris/single-zeta (H, S) of C240 and C540 (shipped in fullerene/data): sdos generalized problem vs dense eigh(H, S).
See fullerene/ for the larger studies (C150000 with 600k orbitals, the facet/pentagon analysis, KPM vs Maxent at 150 moments for C3840 after the 1995 SSC paper)."""
import pathlib
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from sdos.interfaces import xu_carbon as X

warnings.filterwarnings("ignore", message="incomplete Cholesky needed a diagonal shift")   # expected: the overlap of a SZ basis is not an M-matrix
HERE = pathlib.Path(__file__).parent / "fullerene"


def graphene_hbar_vf():
    a = 1.42 * 3 ** 0.5
    a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * 3 ** 0.5 / 2, 0])
    tau = [np.zeros(3), np.array([a / 2, a / (2 * 3 ** 0.5), 0])]
    terms = []
    for al in range(2):
        for be in range(2):
            for n1 in range(-4, 5):
                for n2 in range(-4, 5):
                    Rv = n1 * a1 + n2 * a2
                    d = tau[be] + Rv - tau[al]
                    if 1e-9 < np.linalg.norm(d) < X.RM:
                        terms.append((al, be, Rv, X.hopping_blocks(d[None])[0]))
    b1 = 2 * np.pi * np.array([1 / a, -1 / (a * 3 ** 0.5), 0])
    b2 = 2 * np.pi * np.array([0, 2 / (a * 3 ** 0.5), 0])

    def bands(k):
        H = np.zeros((8, 8), complex)
        for al in range(2):
            H[4 * al:4 * al + 4, 4 * al:4 * al + 4] += np.diag([X.EPS_S, X.EPS_P, X.EPS_P, X.EPS_P])
        for al, be, Rv, B in terms:
            H[4 * al:4 * al + 4, 4 * be:4 * be + 4] += B * np.exp(1j * k @ Rv)
        return np.linalg.eigvalsh(H)

    K = min(((np.diff(bands(f1 * b1 + f2 * b2)[3:5])[0], f1, f2) for f1 in (1 / 3, 2 / 3) for f2 in (1 / 3, 2 / 3)))
    K = K[1] * b1 + K[2] * b2
    hv = np.mean([np.diff(bands(K + 1e-3 * np.array([np.cos(t), np.sin(t), 0]))[3:5])[0] / 2e-3 for t in np.linspace(0, np.pi, 7)])
    return float(hv)


# 1-2. gaps and the Dirac scaling
hv = graphene_hbar_vf()
A_atom = 3 * 3 ** 0.5 / 2 * 1.42 ** 2 / 2                     # graphene area per atom
print(f"graphene (Xu Hamiltonian): hbar v_F = {hv:.2f} eV A")
print("  N      gap (eV)   R_eq (A)   gap R / hbar v_F")
for N in (240, 540, 960):
    pos = np.load(HERE / f"C{N}_lcbop.npy")
    H = X.xu_hamiltonian(pos)
    gap = sdos.homo_lumo(H, H.shape[0] // 2)["gap"]
    R = np.sqrt(N * A_atom / (4 * np.pi))
    print(f"{N:5d}   {gap:8.4f}   {R:7.2f}   {gap * R / hv:8.3f}")

# 3. states nearest E_F
H = X.xu_hamiltonian(np.load(HERE / "C960_lcbop.npy"))
hl = sdos.homo_lumo(H, H.shape[0] // 2)
w = sdos.window_eigs(H, window=(hl["fermi"] - 0.4, hl["fermi"] + 0.4), method="shift-invert")
E = w.E - hl["fermi"]
print(f"C960: {len(E)} states within 0.4 eV of E_F (multiplicity, E - E_F, mean IPR):")
s = 0
while s < len(E):
    t = s
    while t + 1 < len(E) and abs(E[t + 1] - E[s]) < 2e-4:
        t += 1
    print(f"   x{t - s + 1}  {E[s]:+.3f} eV   IPR {np.mean(w.ipr[s:t + 1]):.2e}  (1/N = {1 / H.shape[0]:.2e})")
    s = t + 1

# 4. non-orthogonal (SIESTA H, S)
for N in (240, 540):
    Hs = sp.load_npz(HERE / "data" / f"C{N}_siesta_H.npz")
    Ss = sp.load_npz(HERE / "data" / f"C{N}_siesta_S.npz")
    n = Hs.shape[0]
    evd = sla.eigh(Hs.toarray(), Ss.toarray(), eigvals_only=True)
    ref_gap = evd[n // 2] - evd[n // 2 - 1]
    g = sdos.homo_lumo(Hs, n // 2, B=Ss)
    r = sdos.dos(Hs, B=Ss, M=200, R=8, seed=2, precond="ic")
    ref = np.zeros_like(r.E)
    for ch in np.array_split(evd, 20):
        ref += np.exp(-0.5 * ((r.E[:, None] - ch[None]) / 0.2) ** 2).sum(1)
    ref /= n * 0.2 * np.sqrt(2 * np.pi)
    print(f"SIESTA C{N} ({n} orbitals, non-orthogonal): gap sdos {g['gap']:.6f} vs dense eigh(H,S) {ref_gap:.6f}   "
          f"|diff| = {abs(g['gap'] - ref_gap):.1e};  KPM(B=S) L1 to exact = {np.trapezoid(np.abs(r.rho - ref), r.E):.3f}")
    assert abs(g["gap"] - ref_gap) < 1e-6
