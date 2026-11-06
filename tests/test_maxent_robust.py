"""Maxent robustness to the spectral window and to the choice of default model (measured results of docs/theory.md)."""
import numpy as np
import pytest

import sdos


def hard_edged_density(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(-1, 0.15, 60000) * 18, 0.9 * np.sqrt(rng.uniform(0, 1, 40000)) * 8 + 4])


@pytest.mark.parametrize("f", [0.0, 0.3])
def test_maxent_beats_kpm_and_stays_valid_for_wide_windows(f):
    E = hard_edged_density()
    lo, hi = E.min(), E.max()
    W = hi - lo
    l, h = lo - f * W, hi + f * W
    A, B = 0.5 * (h - l), 0.5 * (h + l)
    th = np.arccos(np.clip((E - B) / A, -1, 1))
    K = 60
    mu = np.array([np.cos(n * th).mean() for n in range(K)])
    Eg = np.linspace(lo + 0.4, hi - 0.4, 800)
    ref = np.zeros_like(Eg)
    for ch in np.array_split(E, 100):
        ref += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.1) ** 2).sum(1)
    ref /= len(E) * 0.1 * np.sqrt(2 * np.pi)
    Ek, rk, _ = sdos.reconstruct(mu, A, B, 2000)
    l1_kpm = np.trapezoid(np.abs(np.interp(Eg, Ek, rk) - ref), Eg)
    errs = {}
    for d in ("chebyshev", "flat", "kpm", "kpm:12"):
        lam, info = sdos.maxent(mu, default=d)
        Em, rm, idos = sdos.maxent_density(lam, A, B, d, 3000, mu=mu)
        assert rm.min() >= 0 and abs(idos[-1] - 1.0) < 1e-3                       # a valid density whatever the window / solver status
        assert info["max_moment_residual"] < 1e-5                                 # moments reproduced even when "converged" is False
        errs[d] = np.trapezoid(np.abs(np.interp(Eg, Em, rm) - ref), Eg)
    assert max(errs.values()) < l1_kpm                                            # better than Jackson-KPM at the same moments
    assert max(errs.values()) < 1.15 * min(errs.values())                         # the default model does not matter


def test_stochastic_maxent_converges_with_a_window_twice_too_wide():
    from conftest import cubic
    H = cubic(8, W=3.0, seed=1)
    r = sdos.estimate_dos(H, M=60, R=6, seed=2, method="maxent", maxent_default="flat", margin=0.5, exact_moments=0, verbose=False)
    assert r.maxent_info["converged"] and r.maxent_info["max_abs_lambda"] < 100
    assert abs(r.idos[-1] - 1.0) < 1e-3 and r.rho.min() >= 0


def test_dos_front_door_defaults_to_the_uniform_prior():
    from conftest import cubic
    r = sdos.dos(cubic(6, W=2.0, seed=1), method="maxent", M=40, R=4, seed=1, exact_moments=0)
    assert r.maxent_default == "flat"
