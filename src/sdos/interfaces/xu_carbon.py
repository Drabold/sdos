"""Xu, Wang, Chan, Ho orthogonal sp3 tight-binding Hamiltonian for carbon (band-structure part only; the repulsive energy is not implemented).

Source: C. H. Xu, C. Z. Wang, C. T. Chan and K. M. Ho, J. Phys.: Condens. Matter 4, 6047 (1992) (the PDF was read from the text layer; the OCR
garbles some exponents, so the tail coefficients below are recomputed from the matching conditions and checked against Table 2 to 1e-7).
Orthogonal basis s, px, py, pz per atom; on-site eps_s = -2.99 eV, eps_p = 3.71 eV; two-centre Slater-Koster hoppings V_ll'm(r) = V0_ll'm s(r) with
V0(ssσ, spσ, ppσ, ppπ) = -5.0, 4.7, 5.5, -1.55 eV and the Goodwin-Skinner-Pettifor scaling function
    s(r) = (r0/r)^n exp{n[-(r/rc)^nc + (r0/rc)^nc]}     (r < r1),     n = 2.0, nc = 6.5, rc = 2.18 A, r0 = 1.536329 A,
replaced for r1 <= r <= rm by a cubic t(r - r1) = c0 + c1 x + c2 x^2 + c3 x^3 that joins s smoothly (value and slope) at r1 = 2.45 A and goes to zero
with zero slope at the cutoff rm = 2.6 A; s = 0 beyond.  Orbital order per atom: s, px, py, pz (use block_size = 4).
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

EPS_S, EPS_P = -2.99, 3.71
R0, N_EXP, NC, RC = 1.536329, 2.00, 6.50, 2.18
R1, RM = 2.45, 2.6                                           # match point and cutoff of the original model
V0 = dict(sss=-5.00, sps=4.70, pps=5.50, ppp=-1.55)


def _s_body(r):
    return (R0 / r) ** N_EXP * np.exp(N_EXP * (-(r / RC) ** NC + (R0 / RC) ** NC))


def _tail_coefficients():
    """c0..c3 of t(x), x = r - r1: t(0) = s(r1), t'(0) = s'(r1), t(L) = 0, t'(L) = 0 with L = rm - r1."""
    s1 = _s_body(R1)
    ds1 = s1 * (-N_EXP / R1 - N_EXP * NC / RC * (R1 / RC) ** (NC - 1))
    L = RM - R1
    A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, L, L ** 2, L ** 3], [0, 1, 2 * L, 3 * L ** 2]], float)
    return np.linalg.solve(A, np.array([s1, ds1, 0.0, 0.0]))


TAIL = _tail_coefficients()


def scaling(r):
    """s(r): Goodwin-Skinner-Pettifor form below r1, cubic tail between r1 and rm, zero beyond; s(r0) = 1."""
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    body = r < R1
    out[body] = _s_body(r[body])
    tail = (r >= R1) & (r <= RM)
    x = r[tail] - R1
    out[tail] = TAIL[0] + TAIL[1] * x + TAIL[2] * x ** 2 + TAIL[3] * x ** 3
    return out


def hopping_blocks(d):
    """4x4 Slater-Koster blocks (s, px, py, pz) for displacement vectors d (m, 3) from atom i to atom j; returns (m, 4, 4)."""
    r = np.linalg.norm(d, axis=1)
    l = d / r[:, None]
    s = scaling(r)
    sss, sps, pps, ppp = (V0[k] * s for k in ("sss", "sps", "pps", "ppp"))
    B = np.zeros((len(r), 4, 4))
    B[:, 0, 0] = sss
    for a in range(3):
        B[:, 0, 1 + a] = l[:, a] * sps                       # s on i, p_a on j
        B[:, 1 + a, 0] = -l[:, a] * sps                      # p_a on i, s on j
        for b in range(3):
            B[:, 1 + a, 1 + b] = l[:, a] * l[:, b] * (pps - ppp)
    for a in range(3):
        B[:, 1 + a, 1 + a] += ppp
    return B


def xu_hamiltonian(pos, cutoff=RM, pairs=None):
    """Sparse real symmetric H (4N x 4N, eV) of an isolated cluster with atom positions ``pos`` (N, 3), hoppings for pairs closer than ``cutoff`` (default: the model's own cutoff, 2.6 A; 1.9 A keeps only the first neighbours of a fullerene)."""
    pos = np.asarray(pos, float)
    n = len(pos)
    if pairs is None:
        pairs = cKDTree(pos).query_pairs(cutoff, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    B = hopping_blocks(pos[j] - pos[i])                      # block (i, j)
    rows = (4 * i[:, None, None] + np.arange(4)[None, :, None]) + 0 * np.arange(4)[None, None, :]
    cols = (4 * j[:, None, None] + np.arange(4)[None, None, :]) + 0 * np.arange(4)[None, :, None]
    onsite = np.tile([EPS_S, EPS_P, EPS_P, EPS_P], n)
    r = np.concatenate([rows.ravel(), cols.ravel(), np.arange(4 * n)])
    c = np.concatenate([cols.ravel(), rows.ravel(), np.arange(4 * n)])
    v = np.concatenate([B.ravel(), B.ravel(), onsite])             # partner of (i a, j b) is (j b, i a) with the same value
    return sp.csr_matrix((v, (r, c)), shape=(4 * n, 4 * n))
