"""Keating dynamical matrix with unequal masses, in both forms (test 5); lambda = omega^2 handling."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from conftest import exact_moments
from keating import keating_hessian


@pytest.fixture(scope="module")
def phi_and_masses():
    Phi = keating_hessian(nc=2)                 # 64 atoms, 192 degrees of freedom
    rng = np.random.default_rng(4)
    m_atom = rng.choice([28.0855, 72.63], size=Phi.shape[0] // 3)
    return Phi, np.repeat(m_atom, 3)


def test_keating_is_stable_with_three_zero_modes(phi_and_masses):
    Phi, m = phi_and_masses
    lam = sla.eigh(Phi.toarray(), np.diag(m), eigvals_only=True)
    assert lam[0] > -1e-9 and np.sum(np.abs(lam) < 1e-8) == 3


@pytest.mark.parametrize("form", ["mass-weighted", "generalized-pcg", "generalized-chebyshev"])
def test_mass_weighted_and_generalized_agree(phi_and_masses, form):
    Phi, m = phi_and_masses
    lam = sla.eigh(Phi.toarray(), np.diag(m), eigvals_only=True)
    b = (lam[0] - 1e-9, lam[-1])
    kw = dict(M=40, R=40, seed=3, verbose=False, bounds=b)
    if form == "mass-weighted":
        s = sp.diags(1 / np.sqrt(m))
        r = sdos.estimate_dos((s @ Phi @ s).tocsr(), **kw)
    elif form == "generalized-pcg":
        r = sdos.estimate_dos(Phi, B=sp.diags(m).tocsr(), diagonal_shortcut=False, binv="pcg", **kw)
    else:
        r = sdos.estimate_dos(Phi, B=sp.diags(m).tocsr(), diagonal_shortcut=False, binv="chebyshev", **kw)
    mex = exact_moments(lam, r.a, r.b, 40)
    se = sdos.moment_errors(r.mu_samples)
    assert np.all(np.abs(r.mu - mex)[2:] < 5 * se[2:] + 1e-12)


def test_vdos_in_omega_normalised_and_consistent(phi_and_masses):
    Phi, m = phi_and_masses
    lam = sla.eigh(Phi.toarray(), np.diag(m), eigvals_only=True)
    res = sdos.vdos(Phi, mass=m, M=96, R=20, seed=1, bounds=(lam[0] - 1e-9, lam[-1]))
    om, g = res["omega"], res["g"]
    area = np.trapezoid(g, om) if hasattr(np, "trapezoid") else np.trapz(g, om)
    assert abs(area - (1.0 - res["neg_weight"] / res["N"])) < 0.05
    assert not res["negative_indicated"]


def test_negative_eigenvalues_are_reported_not_clipped():
    n = 200
    rng = np.random.default_rng(2)
    A = sp.random(n, n, density=5 / n, random_state=3, format="csr")
    A = (A + A.T).tocsr()
    lam0 = sla.eigvalsh(A.toarray())
    D = (A - (lam0[0] - 0.5) * sp.identity(n)).tocsr()        # PSD
    D = (D - 1.0 * sp.identity(n)).tocsr()                     # shift so that several eigenvalues are negative
    ev = sla.eigvalsh(D.toarray())
    assert ev[0] < -0.4                                         # exactly -0.5 by construction, up to rounding
    with pytest.warns(UserWarning, match="negative eigenvalues indicated"):
        res = sdos.vdos(D, M=64, R=10, seed=1, bounds=(ev[0], ev[-1]))
    assert res["negative_indicated"] and res["lambda_min"] < 0 and res["E"].min() < 0       # lambda < 0 kept in rho(lambda)
    n_neg, n_below = int(np.sum(ev < 0)), int(np.sum(ev < -res['resolution']))
    # documented limit: only weight below -resolution is attributed to negative modes (broadening cannot separate the rest from 0)
    assert 0.3 * n_below < res["neg_weight"] <= n_neg + 1, (res["neg_weight"], n_below, n_neg)
