"""Comparison with exact diagonalization: cubic Anderson (real) and with Peierls phases (complex Hermitian)."""
import numpy as np
import pytest
import scipy.linalg as sla

import sdos
from conftest import cubic, exact_moments, random_hermitian


@pytest.mark.parametrize("phases", [False, True])
@pytest.mark.parametrize("kind", ["auto", "phase", "rademacher", "gaussian", "impartial"])
def test_moments_vs_exact(phases, kind):
    H = cubic(7, W=3.0, seed=1, phases=phases)
    ev = sla.eigvalsh(H.toarray())
    res = sdos.estimate_dos(H, M=48, R=40, bounds=(ev[0], ev[-1]), margin=0.01, seed=7, verbose=False, vectors=kind)
    mu_ex = exact_moments(ev, res.a, res.b, 48)
    se = sdos.moment_errors(res.mu_samples)
    dev = np.abs(res.mu - mu_ex)
    assert np.all(dev[1:] < 5 * se[1:] + 1e-12), (dev[1:] / se[1:]).max()


@pytest.mark.parametrize("phases", [False, True])
def test_dos_vs_exact_kernel(phases):
    H = cubic(7, W=3.0, seed=1, phases=phases)
    ev = sla.eigvalsh(H.toarray())
    res = sdos.estimate_dos(H, M=64, R=40, bounds=(ev[0], ev[-1]), seed=9, verbose=False)
    mu_ex = exact_moments(ev, res.a, res.b, 64)
    _, rho_ex, idos_ex = sdos.reconstruct(mu_ex, res.a, res.b, len(res.E))
    z = np.abs(res.rho - rho_ex) / (res.rho_err + 1e-12)
    assert np.mean(z > 4) < 0.02
    assert abs(res.idos[-1] - 1.0) < 1e-2 and abs(idos_ex[-1] - 1.0) < 1e-2


def test_lorentz_kernel_normalised_and_consistent():
    H = cubic(6, W=2.0, seed=2)
    ev = sla.eigvalsh(H.toarray())
    res = sdos.estimate_dos(H, M=96, R=30, bounds=(ev[0], ev[-1]), seed=3, verbose=False, kernel="lorentz")
    dE = np.gradient(res.E)
    assert abs(np.sum(res.rho * dE) - 1.0) < 2e-2 and res.idos[-1] == pytest.approx(1.0, abs=2e-2)
    mu_ex = exact_moments(ev, res.a, res.b, 96)
    _, rho_ex, _ = sdos.reconstruct(mu_ex, res.a, res.b, len(res.E), kernel="lorentz")
    assert np.mean(np.abs(res.rho - rho_ex) / (res.rho_err + 1e-12) > 4) < 0.03


def test_probing_makes_low_moments_exact():
    H = cubic(6, W=2.0, seed=4)
    ev = sla.eigvalsh(H.toarray())
    res = sdos.estimate_dos(H, M=24, R=1, probe=2, bounds=(ev[0], ev[-1]), seed=1, verbose=False)
    mu_ex = exact_moments(ev, res.a, res.b, 24)
    np.testing.assert_allclose(res.mu[:3], mu_ex[:3], atol=1e-10)


def test_matrix_free_equals_explicit():
    import scipy.sparse.linalg as spla
    H = random_hermitian(400, True, seed=8)
    b = sdos.spectral_bounds(H, "gershgorin", verbose=False)
    op = spla.LinearOperator(H.shape, matvec=lambda x: H @ x, matmat=lambda X: H @ X, dtype=H.dtype)
    a = sdos.estimate_dos(H, M=48, R=3, bounds=b, seed=2, verbose=False)
    c = sdos.estimate_dos(op, M=48, R=3, bounds=b, seed=2, verbose=False)
    np.testing.assert_allclose(c.mu_samples, a.mu_samples, rtol=1e-10, atol=1e-12)


def test_hermitian_check():
    H = random_hermitian(200, True, seed=9).tolil()
    H[0, 1] = H[0, 1] + 0.5          # break Hermiticity
    with pytest.raises(sdos.HermitianError):
        sdos.estimate_dos(H.tocsr(), M=16, R=1, bounds=(-5, 5), verbose=False)
    import scipy.sparse.linalg as spla
    A = random_hermitian(200, True, seed=9)
    bad = spla.LinearOperator(A.shape, matvec=lambda x: A @ x + 0.3 * np.roll(x, 1), dtype=A.dtype)
    with pytest.raises(sdos.HermitianError):
        sdos.check_hermitian(bad)
    assert sdos.check_hermitian(spla.LinearOperator(A.shape, matvec=lambda x: A @ x, dtype=A.dtype)) < 1e-12
