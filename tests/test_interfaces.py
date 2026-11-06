"""Optional interfaces: the force-based Hessian operator (any force callable; the LAMMPS route) and the sisl adapter."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from sdos.interfaces.lmp import HessianOperator, LammpsForces
from conftest import exact_moments
from keating import keating_hessian, keating_model


@pytest.fixture(scope="module")
def keating():
    X0, forces = keating_model(nc=2)
    Phi = keating_hessian(nc=2)
    m = np.random.default_rng(4).choice([28.0855, 72.63], size=len(X0))
    return X0, forces, Phi, m


def test_hessian_operator_matches_the_explicit_matrix(keating):
    X0, forces, Phi, m = keating
    op = HessianOperator(forces, X0, m, delta=1e-4)
    w = 1 / np.sqrt(np.repeat(m, 3))
    D = (sp.diags(w) @ Phi @ sp.diags(w)).toarray()
    P = np.eye(3 * len(m)) - op.Tr @ op.Tr.T                       # translations projected out
    Dp = P @ D @ P
    v = np.random.default_rng(1).normal(size=3 * len(m))
    np.testing.assert_allclose(op @ v, Dp @ v, rtol=1e-5, atol=1e-6 * np.abs(Dp @ v).max())
    assert op.nprod == 1
    x = np.random.default_rng(2).normal(size=(2, 3 * len(m)))      # (near-)Hermitian: <x, D y> = <D x, y>
    assert abs(x[0] @ (op @ x[1]) - (op @ x[0]) @ x[1]) < 1e-5 * abs(x[0] @ (op @ x[1]))


def test_matrix_free_vdos_matches_exact_spectrum(keating):
    X0, forces, Phi, m = keating
    op = HessianOperator(forces, X0, m, delta=1e-4)
    w = 1 / np.sqrt(np.repeat(m, 3))
    D = (sp.diags(w) @ Phi @ sp.diags(w)).toarray()
    ev = np.sort(sla.eigvalsh(D))[3:]                              # the three translations are projected to ~0
    res = sdos.vdos(op, M=80, R=40, seed=2, bounds=(0.0, ev[-1] * 1.0005), hermitian_tol=op.hermitian_tol)
    a, b = res.a, res.b
    mex = np.cos(np.outer(np.arange(80), np.arccos(np.clip((np.concatenate([ev, np.zeros(3)]) - b) / a, -1, 1)))).mean(axis=1)
    se = sdos.moment_errors(res.mu_samples)
    assert np.all(np.abs(res.mu - mex)[2:] < 5 * se[2:] + 1e-6)
    assert res["exact_moments"] == 0                               # matrix-free: no exact local moments


def test_lammps_forces_needs_the_module_clearly():
    try:
        import lammps  # noqa: F401
        pytest.skip("lammps module present: this test is for its absence")
    except ImportError:
        with pytest.raises(ImportError, match="LAMMPS Python module"):
            LammpsForces("in.nothing")


sisl = pytest.importorskip("sisl")
from sdos.interfaces.sisl_if import from_sisl, sisl_dos  # noqa: E402


def _graphene(nt=8, orthogonal=True):
    g = sisl.geom.graphene().tile(nt, 0).tile(nt, 1)
    H = sisl.Hamiltonian(g, orthogonal=orthogonal)
    if orthogonal:
        H.construct([[0.1, 1.44], [0.0, -2.7]])
    else:
        H.construct([[0.1, 1.44], [(0.0, 1.0), (-2.7, 0.12)]])
    return H


@pytest.mark.parametrize("k", [(0.0, 0.0, 0.0), (0.2, 0.1, 0.0)])
@pytest.mark.parametrize("orthogonal", [True, False])
def test_sisl_adapter_spectrum_matches_sisl(k, orthogonal):
    H = _graphene(5, orthogonal)
    Hk, Sk = from_sisl(H, k)
    assert (Sk is None) == orthogonal and (k == (0.0, 0.0, 0.0)) == (not np.iscomplexobj(Hk.data) or np.abs(Hk.data.imag).max() < 1e-12)
    ev_sisl = np.sort(H.eigh(k=k))
    ev = sla.eigh(Hk.toarray(), None if Sk is None else Sk.toarray(), eigvals_only=True)
    np.testing.assert_allclose(ev, ev_sisl, atol=1e-8)
    res = sdos.dos(Hk, B=Sk, M=64, R=40, seed=1, bounds=(ev[0] - 1e-6, ev[-1] + 1e-6), exact_moments=0, diagonal_shortcut=False)
    mex = exact_moments(ev, res.a, res.b, 64)
    se = sdos.moment_errors(res.mu_samples)
    assert np.all(np.abs(res.mu - mex)[2:] < 5 * se[2:] + 1e-9)
    assert sisl_dos(H, k, M=64, R=2, seed=1).N == Hk.shape[0]
