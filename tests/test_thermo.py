"""Harmonic thermodynamics: analytic limits, the Einstein solid, and the Keating crystal against the exact mode sum."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from sdos.thermo import HBAR_OMEGA_METAL, KB_EV
from keating import keating_hessian


def exact_thermo(lam, T, dof_per_atom=3):
    hw = HBAR_OMEGA_METAL * np.sqrt(lam)
    out = {}
    x = hw[None, :] / (KB_EV * np.asarray(T)[:, None])
    occ = 1.0 / np.expm1(x)
    out["U"] = dof_per_atom * np.mean(hw * (0.5 + occ), axis=1) if False else dof_per_atom * (hw[None] * (0.5 + occ)).mean(axis=1)
    out["S"] = dof_per_atom * (x * occ - np.log1p(-np.exp(-x))).mean(axis=1)
    out["Cv"] = dof_per_atom * (x ** 2 * np.exp(x) / np.expm1(x) ** 2).mean(axis=1)
    out["F"] = dof_per_atom * (0.5 * hw[None] + KB_EV * np.asarray(T)[:, None] * np.log1p(-np.exp(-x))).mean(axis=1)
    return out


def fake_result(omega, g):
    return dict(omega=omega, g=g, N=1000, neg_weight=0.0, negative_indicated=False, resolution=np.nan)


def test_einstein_solid_analytic():
    """A narrow line at hbar*omega0: Cv, S, U, F equal the Einstein-model closed forms."""
    w0 = 2.0                                                          # sqrt(lambda); hbar omega0 = 2 x 64.65 meV
    w = np.linspace(w0 - 1e-3, w0 + 1e-3, 401)
    g = np.exp(-0.5 * ((w - w0) / 2e-4) ** 2)
    g /= np.trapezoid(g, w) if hasattr(np, "trapezoid") else np.trapz(g, w)
    T = np.array([50.0, 150.0, 300.0, 900.0])
    r = sdos.harmonic_thermo(fake_result(w, g), T)
    hw = HBAR_OMEGA_METAL * w0
    x = hw / (KB_EV * T)
    np.testing.assert_allclose(r.Cv, 3 * x ** 2 * np.exp(x) / np.expm1(x) ** 2, rtol=1e-3)
    np.testing.assert_allclose(r.S, 3 * (x / np.expm1(x) - np.log1p(-np.exp(-x))), rtol=1e-3)
    np.testing.assert_allclose(r.U, 3 * hw * (0.5 + 1 / np.expm1(x)), rtol=1e-3)
    np.testing.assert_allclose(r.F, 3 * (0.5 * hw + KB_EV * T * np.log1p(-np.exp(-x))), rtol=1e-3)
    np.testing.assert_allclose(r.F, r.U - T * KB_EV * r.S, rtol=1e-6, atol=1e-9)           # F = U - T S
    assert r.zpe == pytest.approx(1.5 * hw, rel=1e-3)


def test_dulong_petit_and_thermodynamic_consistency():
    w = np.linspace(0.05, 3.0, 3000)
    g = w ** 2
    g /= np.trapezoid(g, w) if hasattr(np, "trapezoid") else np.trapz(g, w)      # Debye-like
    T = np.array([1.0, 100.0, 1e3, 1e5])
    r = sdos.harmonic_thermo(fake_result(w, g), T)
    assert r.Cv[-1] == pytest.approx(3.0, rel=2e-3) and r.Cv[0] < 1e-3
    dT = 1e-3 * T[1:3]
    r2 = sdos.harmonic_thermo(fake_result(w, g), T[1:3] + dT)
    np.testing.assert_allclose((r2.U - r.U[1:3]) / dT / KB_EV, r.Cv[1:3], rtol=2e-3)      # Cv = dU/dT
    np.testing.assert_allclose(-(r2.F - r.F[1:3]) / dT / KB_EV, r.S[1:3], rtol=2e-3)      # S = -dF/dT
    with pytest.raises(ValueError):
        sdos.harmonic_thermo({"E": 1}, 300)


def test_keating_crystal_vs_exact_mode_sum_and_negative_modes_warn():
    Phi = keating_hessian(nc=2)
    m = np.random.default_rng(4).choice([28.0855, 72.63], size=Phi.shape[0] // 3)
    lam = sla.eigh(Phi.toarray(), np.diag(np.repeat(m, 3)), eigvals_only=True)
    lam_pos = lam[3:]                                                 # drop the three translations
    bd = (lam[0] - 1e-9, lam[-1])
    res = sdos.vdos(Phi, mass=np.repeat(m, 3), M=200, R=40, seed=2, bounds=bd, zero_modes=3)
    T = np.array([150.0, 300.0, 600.0, 1500.0])
    r = sdos.harmonic_thermo(res, T)
    ex = exact_thermo(lam_pos, T)
    np.testing.assert_allclose(r.Cv, ex["Cv"], rtol=3e-3)
    np.testing.assert_allclose(r.S, ex["S"], rtol=1e-2)
    np.testing.assert_allclose(r.U, ex["U"], rtol=2e-3)
    # without removing the translations the smeared zero-mode weight inflates the entropy (the reason for zero_modes=3)
    r0 = sdos.harmonic_thermo(sdos.vdos(Phi, mass=np.repeat(m, 3), M=200, R=40, seed=2, bounds=bd), T)
    assert np.all(r0.S / ex["S"] - 1 > 3 * np.abs(r.S / ex["S"] - 1))
    assert r.info["T_reliable"] > 0 and r.info["neglected_weight"] < 0.05
    res["negative_indicated"] = True
    with pytest.warns(UserWarning, match="imaginary modes"):
        sdos.harmonic_thermo(res, 300.0)


def test_known_eigenvalues_are_removed_exactly_from_the_moments():
    n = 150
    from conftest import random_hermitian, exact_moments
    A = random_hermitian(n, False, seed=3).toarray()
    w, V = np.linalg.eigh(A)
    A[:] = (V * np.concatenate([[0.0, 0.0, 0.0], w[3:]])) @ V.T          # force three exact zero eigenvalues
    H = sp.csr_matrix(A)
    ev = np.sort(np.linalg.eigvalsh(A))
    bd = (ev[0] - 1e-6, ev[-1] + 1e-6)
    r = sdos.estimate_dos(H, M=40, R=60, seed=1, bounds=bd, verbose=False, exact_moments=0, known_eigenvalues=[(0.0, 3)])
    keep = ev[np.argsort(np.abs(ev))[3:]]                                # the 147 non-zero eigenvalues
    mex = exact_moments(keep, r.a, r.b, 40)
    se = sdos.moment_errors(r.mu_samples)
    assert abs(r.mu[0] - 1.0) < 1e-12 and r.known_removed == 3
    assert np.all(np.abs(r.mu - mex)[2:] < 5 * se[2:] + 1e-12)
    assert abs(r.idos[-1] - 1.0) < 1e-2
