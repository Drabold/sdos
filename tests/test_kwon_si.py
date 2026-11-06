import numpy as np
import scipy.linalg as sla
import sdos
from sdos.interfaces import kwon_si as K


def test_hermitian_real_and_complex():
    pos, box = K.diamond_positions(2)
    H0 = K.kwon_hamiltonian(pos, box)
    assert abs(H0 - H0.T).max() < 1e-12
    k = np.array([0.13, -0.27, 0.41])
    H = K.kwon_hamiltonian(pos, box, k)
    assert np.iscomplexobj(H.data) and abs(H - H.getH()).max() < 1e-12


def test_supercell_equals_primitive_k_sum():
    n = 2
    pos, box = K.diamond_positions(n)
    k = np.array([0.21, 0.07, -0.33])
    ev = np.sort(np.linalg.eigvalsh(K.kwon_hamiltonian(pos, box, k).toarray()))
    G = 2 * np.pi / (n * K.A_LATTICE)
    ref = []
    for i in range(n):
        for j in range(n):
            for l in range(n):
                # H_super(k) spectrum = union over the folded points k + (2 pi / L) (i, j, l) of the 8-atom cell H(k')
                ref.append(np.linalg.eigvalsh(K.bulk_hamiltonian_k(k + G * np.array([i, j, l]))))
    ref = np.sort(np.concatenate(ref))
    assert np.max(np.abs(ev - ref)) < 1e-9


def test_stencil_matches_sparse():
    for kp in (None, np.array([0.1, 0.2, -0.3])):
        pos, box = K.diamond_positions(2)
        H = K.kwon_hamiltonian(pos, box, kp)
        S = K.DiamondStencil(2, kp)
        x = np.random.default_rng(0).normal(size=H.shape[0]) + (1j * np.random.default_rng(1).normal(size=H.shape[0]) if kp is not None else 0)
        assert np.max(np.abs(H @ x - S @ x)) < 1e-10


def test_band_edges_regression():
    """Regression values on an 8x8x8 k mesh of the 8-atom cell (they also match the energy axis of Extended Data Fig. 9 of Deringer et al., Nature 589, 59 (2021), read off the plot)."""
    nk, g = 8, 2 * np.pi / K.A_LATTICE
    vb, cb, lo, hi = -1e9, 1e9, 1e9, -1e9
    for i in range(nk):
        for j in range(nk):
            for l in range(nk):
                e = np.linalg.eigvalsh(K.bulk_hamiltonian_k(g * np.array([i, j, l]) / nk))
                vb, cb, lo, hi = max(vb, e[15]), min(cb, e[16]), min(lo, e[0]), max(hi, e[31])
    assert abs(vb - 0.462) < 2e-3 and abs(cb - 1.240) < 2e-3 and abs(lo + 13.497) < 2e-3 and abs(hi - 6.650) < 2e-3


def test_chunked_assembly_matches():
    pos, box = K.diamond_positions(3)
    pos = pos + np.random.default_rng(5).uniform(-0.1, 0.1, pos.shape)
    for kp in (None, np.array([0.1, -0.2, 0.3])):
        A = K.kwon_hamiltonian(pos, box, kp)
        B = K.kwon_hamiltonian(pos, box, kp, chunk=50)
        assert abs(A - B).max() < 1e-12
