"""Eigenpairs in a window (shift-invert and Chebyshev-filtered subspace iteration), IPR and Mulliken weights."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from conftest import cubic, random_hermitian
from test_generalized import spd_B


def exact_window(A, B, win):
    ev, V = sla.eigh(A.toarray(), None if B is None else B.toarray())
    m = (ev >= win[0]) & (ev <= win[1])
    return ev[m], V[:, m]


def exact_ipr(V, B):
    return sdos.ipr(V, B)


def test_ipr_and_mulliken_conventions():
    n = 50
    e = np.zeros((n, 2))
    e[3, 0] = 1.0
    e[:, 1] = 1.0
    ip = sdos.ipr(e)
    assert ip[0] == pytest.approx(1.0) and ip[1] == pytest.approx(1.0 / n)             # localized = 1, extended = 1/N
    B = sp.identity(n, format="csr")
    np.testing.assert_allclose(sdos.mulliken_weights(e, B), sdos.mulliken_weights(e))  # B = 1: |psi|^2
    rng = np.random.default_rng(0)
    Bs = spd_B(n, 20, False)
    w = sdos.mulliken_weights(rng.normal(size=(n, 3)), Bs)
    np.testing.assert_allclose(w.sum(axis=0), 1.0)
    assert sdos.ipr(np.ones((n, 1)), None, block_size=5)[0] == pytest.approx(1.0 / (n // 5))   # site-resolved


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("general", [False, True])
def test_shift_invert_window_matches_exact(cplx, general):
    n = 300
    A = random_hermitian(n, cplx, seed=5)
    B = spd_B(n, 50, cplx, seed=3) if general else None
    ev = sla.eigh(A.toarray(), None if B is None else B.toarray(), eigvals_only=True)
    win = (np.percentile(ev, 40), np.percentile(ev, 52))
    Ex, Vx = exact_window(A, B, win)
    r = sdos.window_eigs(A, B, win, method="shift-invert", bounds=(ev[0] - 1e-6, ev[-1] + 1e-6))
    assert len(r.E) == len(Ex) and r.info["complete"]
    np.testing.assert_allclose(r.E, Ex, atol=1e-8)
    assert r.residual.max() < 1e-8
    np.testing.assert_allclose(r.ipr, exact_ipr(Vx, B), rtol=1e-6)           # IPR is gauge-free (nondegenerate states)


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("general", [False, pytest.param(True, marks=pytest.mark.slow)])
def test_filtered_subspace_finds_the_window(cplx, general):
    n = 200
    A = random_hermitian(n, cplx, seed=5)
    B = spd_B(n, 30, cplx, seed=3) if general else None
    ev = sla.eigh(A.toarray(), None if B is None else B.toarray(), eigvals_only=True)
    win = (np.percentile(ev, 30), np.percentile(ev, 70))
    degree = 250
    a = 0.5 * (ev[-1] - ev[0])
    delta = 2.2 * np.pi * a / degree                                       # edge transition band of the filter (code: 2 pi a / degree)
    Ex, Vx = exact_window(A, B, (win[0] + delta, win[1] - delta))
    assert len(Ex) > 10
    r = sdos.window_eigs(A, B, win, method="filter", degree=degree, bounds=(ev[0] - 1e-6, ev[-1] + 1e-6), tol=1e-7, maxiter=40)
    assert r.E.min() >= win[0] and r.E.max() <= win[1] and r.info["edge_width"] > 0
    found = np.array([np.min(np.abs(r.E - e)) for e in Ex])
    assert found.max() < 1e-6, found.max()                                # every state of the shrunken window is present
    assert r.residual.max() < 1e-6 and r.info["products"] > 0


def test_ipr_separates_extended_from_localized_states():
    """3D Anderson model: critical disorder W_c ~ 16.5.  W = 3 (extended) vs W = 30 (localized): the IPR of states in the same window differs."""
    out = {}
    for W in (3.0, 30.0):
        H = cubic(8, W=W, seed=2)
        ev = sla.eigvalsh(H.toarray())
        r = sdos.window_eigs(H, None, (ev[200] - 1e-6, ev[300] + 1e-6), method="shift-invert")
        out[W] = r.ipr.mean() * H.shape[0]
    assert out[3.0] < 6.0 and out[30.0] > 4 * out[3.0], out


def test_auto_method_and_errors():
    A = random_hermitian(200, False, seed=4)
    ev = sla.eigvalsh(A.toarray())
    r = sdos.window_eigs(A, None, (ev[80] - 1e-6, ev[100] + 1e-6))
    assert r.info["method"] == "shift-invert" and len(r.E) == 21
    with pytest.raises(ValueError):
        sdos.window_eigs(A, None, (0, 1), method="nope")


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("general", [False, True])
def test_count_states_below_is_exact(cplx, general):
    n = 150
    A = random_hermitian(n, cplx, seed=6)
    B = spd_B(n, 30, cplx, seed=2) if general else None
    ev = sla.eigh(A.toarray(), None if B is None else B.toarray(), eigvals_only=True)
    for s in np.linspace(ev[0] - 1, ev[-1] + 1, 23) + 0.00731:
        assert sdos.count_states_below(A, s, B) == int((ev < s).sum())


@pytest.mark.parametrize("general", [False, True])
def test_homo_lumo_exact(general):
    n = 120
    A = random_hermitian(n, False, seed=8)
    B = spd_B(n, 20, False, seed=4) if general else None
    ev = sla.eigh(A.toarray(), None if B is None else B.toarray(), eigvals_only=True)
    r = sdos.homo_lumo(A, 47, B)
    assert r["homo"] == pytest.approx(ev[46], abs=1e-7) and r["lumo"] == pytest.approx(ev[47], abs=1e-7)
    assert r["gap"] == pytest.approx(ev[47] - ev[46], abs=1e-7) and ev[46] < r["fermi"] < ev[47]
