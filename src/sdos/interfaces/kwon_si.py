"""Kwon, Biswas, Wang, Ho, Soukoulis orthogonal sp3 tight-binding Hamiltonian for silicon (band-structure part only).

Source: I. Kwon, R. Biswas, C. Z. Wang, K. M. Ho and C. M. Soukoulis, Phys. Rev. B 49, 7242 (1994).  The constants were checked against Table I and Eqs. (4) and (8) of the paper (PDF read from its text layer): r0, n, E_s, E_p, the four h(r0), n_c, r_c,
the cutoff 4.16 A and the cubic join at r1 = 4.0 A.  The repulsive pair potential and embedding energy of the paper are not implemented.
Orthogonal basis s, px, py, pz per atom; E_s = -5.25 eV, E_p = 1.20 eV; Slater-Koster two-centre integrals V_ll'm(r) = V0 (r0/r)^n exp{n[-(r/rc)^nc + (r0/rc)^nc]}
with V0(sss, sps, pps, ppp) = -2.038, 1.745, 2.75, -1.075 eV, n = 2, r0 = 2.360352 A, nc = 9.5, 8.5, 7.5, 7.5, rc = 3.4, 3.55, 3.7, 3.7 A,
joined to a cubic polynomial (matching value and slope at r1 = 4.0 A) that vanishes with zero slope at rmax = 4.16 A.

``kwon_hamiltonian(pos, box, kpoint=None)``: sparse Hamiltonian for atoms in an orthorhombic periodic box (minimum image: every box edge must exceed 2 x 4.16 A).
With a Bloch wave vector ``kpoint`` (1/A) the hoppings carry exp(i k.d) (d = minimum-image bond vector): a complex Hermitian matrix whose spectrum is that of the
crystal at k (twisted boundary conditions).  ``DiamondStencil(ncell, kpoint)`` applies the same Hamiltonian of a cubic diamond supercell (8 ncell^3 atoms) without
storing it (matrix-free, 4 orbitals x 8 sites x ncell^3), for the 10^6-10^7-atom cases.  ``bulk_hamiltonian_k`` is the 8x8 primitive-cell H(k) used as the exact reference.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

ES, EP = -5.25, 1.20
R0, NEXP, R1, RMAX = 2.360352, 2.0, 4.0, 4.16
V0 = dict(sss=-2.038, sps=1.745, pps=2.75, ppp=-1.075)
NC = dict(sss=9.5, sps=8.5, pps=7.5, ppp=7.5)
RC = dict(sss=3.4, sps=3.55, pps=3.7, ppp=3.7)
A_LATTICE = 5.431            # cubic diamond lattice constant (A) used with the model (not relaxed: set by the user)


def _f(x, a):
    return (R0 / x) ** NEXP * np.exp(NEXP * (-(x / RC[a]) ** NC[a] + (R0 / RC[a]) ** NC[a]))


def scaling(r, a):
    """s_a(r): the radial function without V0 (cubic tail between r1 and rmax)."""
    r = np.asarray(r, float)
    h = 1e-6
    b0 = float(_f(R1, a)); b1 = float((_f(R1 + h, a) - _f(R1 - h, a)) / (2 * h)); D = RMAX - R1
    b2, b3 = np.linalg.solve(np.array([[D ** 2, D ** 3], [2 * D, 3 * D ** 2]]), [-b0 - b1 * D, -b1])
    u = r - R1
    return np.where(r >= RMAX, 0.0, np.where(r > R1, b0 + b1 * u + b2 * u ** 2 + b3 * u ** 3, _f(np.maximum(r, 1e-9), a)))


def hopping_blocks(d):
    """4x4 blocks <i,o|H|j,o'> for bond vectors d = x_j - x_i (N, 3) in Angstrom (orbital order s, px, py, pz)."""
    d = np.asarray(d, float)
    r = np.linalg.norm(d, axis=1)
    u = d / r[:, None]
    sc = {a: V0[a] * scaling(r, a) for a in V0}
    B = np.zeros((len(d), 4, 4))
    B[:, 0, 0] = sc["sss"]
    for k in range(3):
        B[:, 0, 1 + k] = u[:, k] * sc["sps"]
        B[:, 1 + k, 0] = -u[:, k] * sc["sps"]
        for m in range(3):
            B[:, 1 + k, 1 + m] = u[:, k] * u[:, m] * (sc["pps"] - sc["ppp"]) + (sc["ppp"] if k == m else 0.0)
    return B


def kwon_hamiltonian(pos, box, kpoint=None, chunk=None):
    """Sparse CSR Hamiltonian (4 N x 4 N) of atoms ``pos`` (N, 3) in the periodic orthorhombic ``box`` (3,); complex if ``kpoint`` is not None.

    ``chunk``: build in blocks of that many atoms (peak memory = the final matrix plus one block; use it above ~10^5 atoms)."""
    box = np.asarray(box, float)
    if chunk:
        from ._assemble import assemble_csr
        return assemble_csr(pos, box, RMAX, lambda d: [hopping_blocks(d)], 4, [np.array([ES, EP, EP, EP])], kpoint, chunk)[0]
    if np.any(box < 2 * RMAX):
        raise ValueError("every box edge must exceed 2 x 4.16 A (minimum-image convention)")
    X = np.mod(pos, box)
    N = len(X)
    pairs = cKDTree(X, boxsize=box).query_pairs(RMAX, output_type="ndarray")
    I, J = pairs[:, 0], pairs[:, 1]
    d = X[J] - X[I]
    d -= box * np.rint(d / box)
    B = hopping_blocks(d).astype(complex if kpoint is not None else float)
    if kpoint is not None:
        B = B * np.exp(1j * (d @ np.asarray(kpoint, float)))[:, None, None]       # <i|H|j> = h(d_ij) exp(i k.d_ij)
    o = np.arange(4)
    rows = (4 * I[:, None, None] + o[None, :, None]).repeat(4, 2).ravel()
    cols = (4 * J[:, None, None] + o[None, None, :]).repeat(4, 1).ravel()
    vals = B.ravel()
    M = 4 * N
    ons = np.tile([ES, EP, EP, EP], N)
    # (j, i) block is the Hermitian conjugate of the (i, j) block
    Bt = np.conj(np.transpose(B, (0, 2, 1))).ravel()
    rows_t = (4 * J[:, None, None] + o[None, :, None]).repeat(4, 2).ravel()
    cols_t = (4 * I[:, None, None] + o[None, None, :]).repeat(4, 1).ravel()
    H = sp.coo_matrix((np.concatenate([vals, Bt, ons.astype(vals.dtype)]),
                       (np.concatenate([rows, rows_t, np.arange(M)]), np.concatenate([cols, cols_t, np.arange(M)]))), shape=(M, M)).tocsr()
    return H


_BASIS = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0], [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])


def diamond_positions(ncell, a=A_LATTICE):
    """Atoms of a cubic diamond supercell ncell^3 (8 per cell), ordered (ix, iy, iz, basis)."""
    g = np.arange(ncell)
    cells = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pos = (cells[:, None, :] + _BASIS[None]) * a
    return pos.reshape(-1, 3), np.full(3, ncell * a)


def bulk_hamiltonian_k(k, a=A_LATTICE):
    """8x8 (spin-free, 32x32 including orbitals) H(k) of the 8-atom cubic diamond cell with Bloch phases exp(i k.d) in the atomic gauge, k in 1/A."""
    pos = _BASIS * a
    H = np.zeros((32, 32), complex)
    for al in range(8):
        H[4 * al:4 * al + 4, 4 * al:4 * al + 4] += np.diag([ES, EP, EP, EP])
    T = [np.array([i, j, l]) * a for i in range(-2, 3) for j in range(-2, 3) for l in range(-2, 3)]
    for al in range(8):
        for be in range(8):
            for t in T:
                d = pos[be] + t - pos[al]
                if 1e-9 < np.linalg.norm(d) < RMAX:
                    H[4 * al:4 * al + 4, 4 * be:4 * be + 4] += hopping_blocks(d[None])[0] * np.exp(1j * (d @ np.asarray(k, float)))
    return H


def _neighbour_table(a=A_LATTICE):
    """For each basis atom alpha: list of (beta, cell shift (3,), 4x4 real block B, bond vector d)."""
    pos = _BASIS * a
    table = []
    for al in range(8):
        rows = []
        for be in range(8):
            for sh in [(i, j, l) for i in range(-2, 3) for j in range(-2, 3) for l in range(-2, 3)]:
                d = pos[be] + np.array(sh) * a - pos[al]
                if 1e-9 < np.linalg.norm(d) < RMAX:
                    rows.append((be, np.array(sh), hopping_blocks(d[None])[0], d))
        table.append(rows)
    return table


class DiamondStencil(spla.LinearOperator):
    """Matrix-free H(k) of an ncell^3 cubic diamond supercell (periodic): dimension 32 ncell^3, vector layout (ix, iy, iz, basis atom, orbital)."""

    def __init__(self, ncell, kpoint=None, a=A_LATTICE):
        self.n = int(ncell)
        if self.n * a < 2 * RMAX:
            raise ValueError("supercell too small for the minimum-image convention")
        self.k = None if kpoint is None else np.asarray(kpoint, float)
        self.table = _neighbour_table(a)
        self.cplx = kpoint is not None
        dim = 32 * self.n ** 3
        super().__init__(dtype=complex if self.cplx else float, shape=(dim, dim))
        self._ons = np.array([ES, EP, EP, EP])
        self.nnz_per_row = 4 * (1 + len(self.table[0]))

    def _matvec(self, v):
        n = self.n
        x = np.asarray(v).reshape(n, n, n, 8, 4)
        y = x * self._ons
        for al in range(8):
            acc = np.zeros((n, n, n, 4), dtype=np.result_type(x.dtype, np.complex128 if self.cplx else np.float64))
            for be, sh, B, d in self.table[al]:
                Bm = B if self.k is None else B * np.exp(1j * (d @ self.k))
                src = np.roll(x[:, :, :, be, :], shift=(-sh[0], -sh[1], -sh[2]), axis=(0, 1, 2))      # value at cell + shift
                acc += src @ Bm.T
            y[:, :, :, al, :] += acc
        return y.reshape(-1)

    def _matmat(self, V):
        return np.stack([self._matvec(V[:, j]) for j in range(V.shape[1])], axis=1)

    def _adjoint(self):
        return self                                   # Hermitian by construction
