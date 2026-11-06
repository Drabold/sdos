"""Drabold-Sankey impartial vectors (PRL 70, 3631 (1993)): first three moments exact, large band-energy variance reduction."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse.linalg as spla

import sdos
from conftest import cubic, exact_moments


@pytest.mark.parametrize("phases", [False, True])
def test_first_three_moments_exact_every_vector(phases):
    H = cubic(7, W=3.0, seed=1, phases=phases)
    ev = sla.eigvalsh(H.toarray())
    res = sdos.estimate_dos(H, M=24, R=5, bounds=(ev[0], ev[-1]), seed=3, verbose=False, vectors="impartial")
    mex = exact_moments(ev, res.a, res.b, 24)
    np.testing.assert_allclose(res.mu_samples[:, :3], np.tile(mex[:3], (5, 1)), atol=1e-9)
    assert all(i["P"] < 1e-18 for i in res.impartial_info)


def test_unbiased_within_statistics():
    """Gross-bias check only (60 vectors, low power).  With 4000 vectors the K=2 vectors show rms z = 1.2 (real), 0.9 (complex),
    like unbiased ensembles; constraints on more than 2 moments (K >= 4) are measurably biased (see the status note)."""
    H = cubic(7, W=3.0, seed=1)
    ev = sla.eigvalsh(H.toarray())
    res = sdos.estimate_dos(H, M=40, R=60, bounds=(ev[0], ev[-1]), seed=5, verbose=False, vectors="impartial")
    mex = exact_moments(ev, res.a, res.b, 40)
    se = sdos.moment_errors(res.mu_samples)
    assert np.all(np.abs(res.mu - mex)[3:] < 5 * se[3:])


def test_band_energy_spread_reduced():
    """The paper's headline: importance sampling narrows the distribution of the band energy (we see ~10x)."""
    H = cubic(8, W=3.0, seed=1)
    n = H.shape[0]
    b = sdos.spectral_bounds(H, "gershgorin", verbose=False)
    sd = {}
    for kind in ("gaussian", "rademacher", "impartial"):
        r = sdos.estimate_dos(H, M=30, R=60, bounds=b, seed=7, verbose=False, vectors=kind)
        e = [sdos.band_quantities(m, r.a, r.b, n, n, 2.0)[1] / n for m in r.mu_samples]
        sd[kind] = np.std(e, ddof=1)
    assert sd["impartial"] < sd["gaussian"] / 3 and sd["impartial"] < sd["rademacher"] / 3, sd


def test_errors_and_matrix_free():
    H = cubic(6, W=2.0, seed=2)
    b = sdos.spectral_bounds(H, "gershgorin", verbose=False)
    with pytest.raises(ValueError):
        sdos.estimate_dos(H, M=16, R=1, probe=2, bounds=b, verbose=False, vectors="impartial")
    op = spla.LinearOperator(H.shape, matvec=lambda x: H @ x, matmat=lambda X: H @ X, dtype=H.dtype)
    with pytest.raises(ValueError):          # matrix-free: exact low moments must be supplied
        sdos.estimate_dos(op, M=16, R=1, bounds=b, verbose=False, vectors="impartial")
    a = sdos.ScaledOperator(H, 0.5 * (b[1] - b[0]) * 1.02, 0.5 * (b[0] + b[1]))   # a = (w + 2*margin*w)/2, margin 1%
    tr = a.traces()
    r1 = sdos.estimate_dos(H, M=16, R=2, bounds=b, seed=4, verbose=False, vectors="impartial")
    r2 = sdos.estimate_dos(op, M=16, R=2, bounds=b, seed=4, verbose=False, vectors="impartial", traces=tr)
    np.testing.assert_allclose(r2.mu_samples, r1.mu_samples, rtol=1e-6, atol=1e-8)
