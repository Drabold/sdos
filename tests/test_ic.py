"""Incomplete Cholesky IC(0): defining property, diagonal-shift breakdown path, PCG benefit, integration."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from sdos.generalized import _make_precond, pcg
from sdos.ic import incomplete_cholesky
from conftest import random_hermitian
from test_generalized import spd_B


@pytest.mark.parametrize("cplx", [False, True])
def test_ic0_reproduces_B_on_the_pattern(cplx):
    B = spd_B(300, 200, cplx, seed=6)
    L, shift = incomplete_cholesky(B)
    pat = (abs(sp.tril(B)) > 0).astype(float)
    err = (sp.tril((L @ L.conj().T).tocsr()) - sp.tril(B)).multiply(pat)
    assert abs(err).max() < 1e-10 and shift == 0.0
    assert L.nnz == sp.tril(B).nnz                                   # no fill: the pattern of tril(B)


def test_breakdown_triggers_shift_and_pcg_still_converges():
    n = 400
    G = sp.random(n, n, density=3 / n, random_state=1, format="csr")
    B = (G.T @ G + 1e-3 * sp.identity(n)).tocsr()
    assert np.linalg.eigvalsh(B.toarray())[0] > 0
    L, shift = incomplete_cholesky(B)
    assert shift > 0                                                 # IC(0) broke down without a shift (not an M-matrix)
    rhs = np.random.default_rng(0).normal(size=(n, 1))
    _, bj = pcg(B, rhs, _make_precond(B, "block-jacobi", 1), 1e-10, 20000)
    X, ic = pcg(B, rhs, _make_precond(B, "ic", 1), 1e-10, 20000)
    assert ic["max_res"] < 1e-10 and ic["iters"] < bj["iters"]
    assert np.linalg.norm(B @ X - rhs) < 1e-8 * np.linalg.norm(rhs)


@pytest.mark.parametrize("cplx", [False, True])
def test_ic_reduces_iterations_and_is_symmetric_positive(cplx):
    B = spd_B(200, 200, cplx, seed=6)
    n = B.shape[0]
    M = _make_precond(B, "ic", 1)
    rng = np.random.default_rng(2)
    u = (rng.normal(size=(n, 1)) + (1j * rng.normal(size=(n, 1)) if cplx else 0)).astype(B.dtype)
    v = (rng.normal(size=(n, 1)) + (1j * rng.normal(size=(n, 1)) if cplx else 0)).astype(B.dtype)
    assert abs(np.vdot(u, M(v)) - np.conj(np.vdot(v, M(u)))) < 1e-10 * abs(np.vdot(u, M(v)))    # Hermitian
    assert np.real(np.vdot(u, M(u))) > 0                                                        # positive
    _, bj = pcg(B, u, _make_precond(B, "block-jacobi", 1), 1e-10)
    _, ic = pcg(B, u, M, 1e-10)
    assert ic["iters"] < 0.7 * bj["iters"]


def test_ic_in_estimate_dos():
    n = 200
    A = random_hermitian(n, True, seed=3)
    B = spd_B(n, 200, True, seed=6)
    ev = sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    kw = dict(M=24, R=2, seed=5, verbose=False, bounds=(ev[0], ev[-1]), diagonal_shortcut=False)
    ref = sdos.estimate_dos(A, B=B, **kw)
    r = sdos.estimate_dos(A, B=B, precond="ic", **kw)
    np.testing.assert_allclose(r.mu_samples, ref.mu_samples, atol=1e-8)
    assert r.binv_info["iterations"] < ref.binv_info["iterations"]
