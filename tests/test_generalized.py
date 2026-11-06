"""Generalized problem A x = lambda B x (tests 4 and 6 of the directive) and the diagonal-mass special case."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from conftest import exact_moments, random_hermitian


def spd_B(n, kappa, cplx, seed=3):
    """Sparse Hermitian positive-definite B with a tunable condition number (about kappa)."""
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=4 / n, random_state=seed, format="csr")
    if cplx:
        A = A + 1j * sp.random(n, n, density=4 / n, random_state=seed + 1, format="csr")
    A = (A + A.conj().T).tocsr()
    ev = sla.eigvalsh(A.toarray())
    c = 0.9 / abs(ev[0]) if ev[0] < 0 else 0.5
    d = sp.diags(rng.uniform(1.0, np.sqrt(kappa), n))
    return (d @ (sp.identity(n) + c * A) @ d).tocsr()


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("kappa", [10, 1e3])
@pytest.mark.parametrize("binv", ["pcg", "chebyshev"])
def test_moments_vs_exact_eigh(cplx, kappa, binv):
    n = 250
    A = random_hermitian(n, cplx, seed=5)
    B = spd_B(n, kappa, cplx)
    ev = sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    r = sdos.estimate_dos(A, B=B, M=40, R=30, seed=2, verbose=False, binv=binv, bounds=(ev[0], ev[-1]), diagonal_shortcut=False)
    mex = exact_moments(ev, r.a, r.b, 40)
    se = sdos.moment_errors(r.mu_samples)
    z = np.abs(r.mu - mex)[2:] / se[2:]
    assert z.max() < 5.0, z.max()


def exact_trace_moments(A, B, M, bounds, **opt):
    """Exact (n-vector) Tr T_m(Y)/n of Y = (B^-1 A - b)/a by applying the operator to the full identity basis: no sampling."""
    n = A.shape[0]
    op = sdos.GeneralizedOperator(A, B, **opt)
    a, b = 0.5 * (bounds[1] - bounds[0]), 0.5 * (bounds[1] + bounds[0])
    I = np.eye(n, dtype=op.dtype)
    return sdos.chebyshev_moments_plain(lambda V: (op.apply(V) - b * V) / a, I, M) / n, op


@pytest.mark.parametrize("cplx", [False, True])
def test_both_binv_modes_agree(cplx):
    """Test 6 of the directive: both B^-1 modes give the same moments to tolerance.  (Per-vector estimates differ between the
    modes because the Chebyshev mode works in block-scaled coordinates, a similarity transform; the TRACES agree, so we compare
    the exact traces from the full identity basis, and both against exact diagonalization.)"""
    n = 120
    A = random_hermitian(n, cplx, seed=7)
    B = spd_B(n, 300, cplx, seed=4)
    ev = sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    bd = (ev[0] - 1e-6, ev[-1] + 1e-6)
    mp, _ = exact_trace_moments(A, B, 40, bd, binv="pcg", tol=1e-13)
    mc, op = exact_trace_moments(A, B, 40, bd, binv="chebyshev", cheb_tol=1e-13)
    mex = exact_moments(ev, 0.5 * (bd[1] - bd[0]), 0.5 * (bd[1] + bd[0]), 40)
    np.testing.assert_allclose(mp, mex, atol=1e-8)
    np.testing.assert_allclose(mc, mex, atol=1e-8)
    np.testing.assert_allclose(mp, mc, atol=1e-8)
    assert op.info["degree"] > 5 and op.info["poly_err"] <= 1e-12


def test_generalized_bounds_without_hints():
    """Bounds from the B-inner-product Lanczos (no bounds supplied) contain the whole spectrum with the default margin."""
    n = 250
    A = random_hermitian(n, True, seed=9)
    B = spd_B(n, 100, True, seed=2)
    ev = sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    r = sdos.estimate_dos(A, B=B, M=32, R=2, seed=1, verbose=False, diagonal_shortcut=False)
    assert r.b - r.a < ev[0] and r.b + r.a > ev[-1]


def test_preconditioners():
    n = 200
    A = random_hermitian(n, False, seed=3)
    B = spd_B(n, 200, False, seed=6)
    ev = sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)
    kw = dict(M=24, R=2, seed=5, verbose=False, bounds=(ev[0], ev[-1]), diagonal_shortcut=False)
    ref = sdos.estimate_dos(A, B=B, **kw)
    r = sdos.estimate_dos(A, B=B, precond="none", **kw)
    np.testing.assert_allclose(r.mu_samples, ref.mu_samples, atol=1e-8)
    # ILU is experimental (non-symmetric preconditioner for CG): it warns, and may stall on this matrix (see docs)
    with pytest.warns(UserWarning):
        r = sdos.estimate_dos(A, B=B, precond="ilu", **kw)
    np.testing.assert_allclose(r.mu_samples, ref.mu_samples, atol=1e-4)
    pytest.importorskip("pyamg")
    r = sdos.estimate_dos(A, B=B, precond="amg", **kw)
    np.testing.assert_allclose(r.mu_samples, ref.mu_samples, atol=1e-8)
    assert r.binv_info["iterations"] < ref.binv_info["iterations"]            # AMG needs fewer iterations than block-Jacobi here


def test_diagonal_B_shortcut_and_1d_B():
    n = 300
    A = random_hermitian(n, False, seed=2)
    m = np.random.default_rng(1).uniform(1.0, 5.0, n)
    ev = sla.eigh(A.toarray(), np.diag(m), eigvals_only=True)
    a = sdos.estimate_dos(A, B=m, M=32, R=2, seed=1, verbose=False, bounds=(ev[0], ev[-1]))
    b = sdos.estimate_dos(A, B=sp.diags(m).tocsr(), M=32, R=2, seed=1, verbose=False, bounds=(ev[0], ev[-1]))
    np.testing.assert_allclose(a.mu_samples, b.mu_samples)           # 1-D masses and a diagonal sparse B are the same path
    assert "diagonal" in a.B_mode
    # the full generalized machinery on the same diagonal B gives the same traces (not the same per-vector estimates)
    bd = (ev[0] - 1e-6, ev[-1] + 1e-6)
    mfull, _ = exact_trace_moments(A, sp.diags(m).tocsr(), 32, bd, binv="pcg")
    mex = exact_moments(ev, 0.5 * (bd[1] - bd[0]), 0.5 * (bd[1] + bd[0]), 32)
    np.testing.assert_allclose(mfull, mex, atol=1e-8)


def test_errors():
    A = random_hermitian(100, False, seed=1)
    B = spd_B(100, 10, False)
    with pytest.raises(ValueError):
        sdos.estimate_dos(A, B=B, M=16, R=1, probe=2, verbose=False, diagonal_shortcut=False)
    with pytest.raises(ValueError):
        sdos.estimate_dos(A, B=B, M=16, R=1, vectors="impartial", verbose=False, diagonal_shortcut=False)
    with pytest.raises(NotImplementedError):
        sdos.estimate_dos(A, B=B, M=16, R=1, exact_moments=3, verbose=False, diagonal_shortcut=False)
    bad = (B - 3 * sp.identity(100) * B.diagonal().max()).tocsr()        # indefinite
    with pytest.raises(ValueError):
        sdos.estimate_dos(A, B=bad, M=16, R=1, verbose=False)


def test_too_tight_bounds_warn():
    H = random_hermitian(300, False, seed=2)
    ev = sla.eigvalsh(H.toarray())
    with pytest.warns(UserWarning, match="too tight"):
        sdos.estimate_dos(H, M=64, R=2, bounds=(ev[0] * 0.5, ev[-1] * 0.5), margin=0.0, seed=1, verbose=False)
