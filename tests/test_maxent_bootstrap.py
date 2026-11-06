import numpy as np
import sdos


def _mu(K):
    th = np.arccos(np.clip(np.random.default_rng(1).uniform(-0.8, 0.8, 200000), -1, 1))
    return np.array([np.cos(n * th).mean() for n in range(K)])


def test_bootstrap_matches_plain_and_reports_edges():
    mu = _mu(60)
    x, rho, ids, info = sdos.maxent_bootstrap(mu, sigma_floor=0.0)      # exact-moment problem
    assert info["converged"] and info["max_moment_residual"] < 1e-8
    assert -0.95 < info["x_lo"] < -0.75 and 0.75 < info["x_hi"] < 0.95
    assert abs(np.trapezoid(rho, x) - 1.0) < 1e-6


def test_bootstrap_failure_falls_back():
    mu = _mu(60).copy()
    mu[50:] += 0.3 * np.random.default_rng(2).normal(size=10)      # inconsistent moments: no positive density exists
    x, rho, ids, info = sdos.maxent_bootstrap(mu)
    assert np.all(np.isfinite(rho)) and info["moments_used"] < 60


def test_sigma_floor_makes_exact_moments_of_a_singular_density_converge():
    """Exact moments of a discrete, gapped spectrum have no exp-polynomial solution at a few hundred orders: the exact-moment problem fails, a floor of 1e-4 works."""
    rng = np.random.default_rng(5)
    ev = np.sort(np.concatenate([rng.normal(-0.5, 0.12, 1500), rng.normal(0.55, 0.1, 1500)]))
    a, b = 0.5 * (ev[-1] - ev[0]) * 1.01, 0.5 * (ev[-1] + ev[0])
    th = np.arccos(np.clip((ev - b) / a, -1, 1))
    mu = np.array([np.cos(n * th).mean() for n in range(300)])
    _, _, i0 = sdos.maxent_fit(mu[None], a, b, "flat", sigma_floor=0.0, jackknife=False, verbose=False)
    lam, _, i1 = sdos.maxent_fit(mu[None], a, b, "flat", jackknife=False, verbose=False)        # default floor 1e-4
    assert not i0["converged"] and i1["converged"] and i1["iterations"] < 400
    E, rho, idos = sdos.maxent_density(lam, a, b, "flat", mu=mu)
    for e in (-0.3, 0.0, 0.3):
        assert abs(np.interp(e, E, idos) - (ev < e).mean()) < 0.02
    x, r, ids, info = sdos.maxent_bootstrap(mu)
    assert info["converged"] and info["moments_used"] == 300
