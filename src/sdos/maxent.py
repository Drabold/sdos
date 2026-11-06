"""Maximum-entropy reconstruction from Chebyshev moments, with moment uncertainties.

rho(x) = w(x) exp( sum_n lambda_n T_n(x) ) maximises the entropy relative to the default model w
subject to the moments; lambda is found by Newton's method on the convex dual (Mead & Papanicolaou;
Turek; Drabold & Sankey; Silver & Roeder).  Ported unchanged from sdos.py (regression-tested).
"""

import sys

import numpy as np

def _kpm_default_q(mu, ngrid, floor=1e-8):
    """Default model per d(theta): the Jackson-damped KPM density (positive, vanishing where there are no states), floored at ``floor`` x its maximum."""
    from .reconstruct import jackson
    K = len(mu)
    theta = np.pi * (np.arange(ngrid) + 0.5) / ngrid
    c = jackson(K) * np.asarray(mu, float)
    series = c[0] + 2.0 * (c[1:] @ np.cos(np.outer(np.arange(1, K), theta)))
    q = series / np.pi
    return np.maximum(q, floor * q.max())


def _maxent_grid(K, default, ngrid=None, mu=None):
    if ngrid is None:
        ngrid = max(8192, 64 * K)
    theta = np.pi * (np.arange(ngrid) + 0.5) / ngrid
    dth = np.pi / ngrid
    if default == "chebyshev":      # w(x) = 1 / (pi sqrt(1 - x^2))
        q = np.full(ngrid, 1.0 / np.pi)
    elif default == "flat":         # w(x) = 1/2
        q = 0.5 * np.sin(theta)
    elif default == "kpm" or default.startswith("kpm:"):   # Jackson-KPM density of the same moments ("kpm:K0": first K0 only). Offered for experiments: it measured NO better than "flat"
        if mu is None:
            raise ValueError("default='kpm' needs the moments")
        K0 = int(default.split(":")[1]) if ":" in default else len(mu)
        q = _kpm_default_q(np.asarray(mu)[:K0], ngrid)
    else:
        raise ValueError("default must be 'chebyshev', 'flat' or 'kpm'")
    C = np.cos(np.outer(np.arange(K), theta))
    return theta, dth, q, C


def maxent(mu, sigma=None, default="chebyshev", tol=1e-10, maxiter=500,
           ngrid=None, grid=None, lam0=None):
    """Solve for Maxent Lagrange multipliers. Returns (lam, info).  ``lam0``: optional starting multipliers (zero-padded to len(mu))."""
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    s2 = np.zeros(K) if sigma is None else np.asarray(sigma, dtype=float) ** 2
    theta, dth, q, C = grid if grid is not None else _maxent_grid(K, default, ngrid, mu)
    lam = np.zeros(K)
    lam[0] = np.log(mu[0] / (q.sum() * dth))
    if lam0 is not None:
        lam0 = np.asarray(lam0, float)
        lam[:] = 0.0
        lam[:min(K, len(lam0))] = lam0[:K]

    def dual(l):
        expo = np.clip(l @ C, -700.0, 700.0)
        f = q * np.exp(expo)
        return f.sum() * dth - l @ mu + 0.5 * np.sum(s2 * l * l), f

    G, f = dual(lam)
    converged = False
    for it in range(maxiter):
        m = (C @ f) * dth
        g = m - mu + s2 * lam
        gnorm = np.max(np.abs(g))
        if gnorm < tol:
            converged = True
            break
        Hm = (C * f) @ C.T * dth + np.diag(s2)
        # robust Newton step via eigen-decomposition (floor tiny eigenvalues)
        w, V = np.linalg.eigh(Hm)
        w = np.maximum(w, 1e-14 * w.max())
        step = V @ ((V.T @ g) / w)
        t = 1.0
        while True:
            Gn, fn = dual(lam - t * step)
            if Gn <= G - 1e-4 * t * (g @ step) or t < 1e-10:
                break
            t *= 0.5
        lam = lam - t * step
        G, f = Gn, fn
    m = (C @ f) * dth
    info = dict(converged=converged, iterations=it,
                max_moment_residual=float(np.max(np.abs(m - mu))), max_abs_lambda=float(np.max(np.abs(lam))))
    return lam, info


def maxent_density(lam, a, b, default="chebyshev", ngrid=None, mu=None):
    """DOS per orbital and IDOS on an ascending energy grid from multipliers (``mu`` is needed for default='kpm').

    The density is always evaluated on the solver's own fine grid (the multipliers can be huge when the spectral window is wider than the support, and a coarser
    grid then mis-integrates); if ``ngrid`` is given the result is interpolated onto the Chebyshev nodes of that size."""
    K = len(lam)
    theta, dth, q, C = _maxent_grid(K, default, None, mu)
    theta, q, C = theta[::-1], q[::-1], C[:, ::-1]        # ascending x
    f = q * np.exp(np.clip(lam @ C, -700.0, 700.0))       # density per d(theta)
    x = np.cos(theta)
    rho = f / np.sin(theta) / a                           # per unit energy
    idos = np.cumsum(f) * dth - 0.5 * f * dth             # midpoint cumulative
    if ngrid is not None and ngrid != len(theta):
        thr = np.pi * (np.arange(ngrid) + 0.5) / ngrid
        xr = np.cos(thr)[::-1]
        return a * xr + b, np.interp(xr, x, rho), np.interp(xr, x, idos)
    return a * x + b, rho, idos


def band_quantities_grid(E, rho, idos, n_orb, n_elec, spin):
    """Fermi level and band energy from a DOS tabulated on an ascending grid."""
    count = spin * n_orb * idos
    k = np.searchsorted(count, n_elec)
    if k == 0 or k >= len(E):
        raise ValueError("electron count outside spectrum")
    t = (n_elec - count[k - 1]) / (count[k] - count[k - 1])
    EF = E[k - 1] + t * (E[k] - E[k - 1])
    # band energy: trapezoid-in-energy over IDOS increments (second order)
    Ecell = 0.5 * (E[1:k] + E[:k - 1])
    Ebs = E[0] * idos[0] + np.sum(Ecell * np.diff(idos[:k]))
    Ebs += 0.5 * (E[k - 1] + EF) * (n_elec / (spin * n_orb) - idos[k - 1])
    return EF, spin * n_orb * Ebs


def maxent_fit(mus, a, b, default="chebyshev", noise=1.0, sigma=None,
               jackknife=True, verbose=True, sigma_floor=1e-4):
    """Maxent on averaged moments; optional jackknife over samples.

    mus: (R, K) per-sample moments. noise scales the per-moment standard error
    used in the dual (0 = exact-moment Maxent).  ``sigma_floor``: lower bound on the uncertainty of every moment but mu_0 (moments are
    normalised, |mu_n| <= 1).  Exact or nearly exact moments of a singular density (e.g. the exact low moments, or a single sample) have no
    exponential-polynomial solution beyond a few hundred orders and Newton's method then stalls; a floor of 1e-4 restores convergence
    (measured on DTW4096 with exact moments up to M = 400: 17-24 Newton steps instead of failure, band-energy bias <= 0.02 meV/atom) and is
    at or below the statistical error of stochastic moments of large systems (the floor only acts where the sampled uncertainty is smaller).  Set 0 for the exact-moment problem.
    """
    R, K = mus.shape
    mu = mus.mean(axis=0)
    if sigma is None:
        if R > 1 and noise > 0:
            sigma = noise * mus.std(axis=0, ddof=1) / np.sqrt(R)
        else:
            sigma = np.zeros(K)
    sigma = np.maximum(np.asarray(sigma, dtype=float), float(sigma_floor))
    sigma[0] = 0.0                       # mu_0 = 1 is exact
    grid = _maxent_grid(K, default, None, mu)
    lam, info = maxent(mu, sigma, default, grid=grid)
    if verbose:
        flag = "converged" if info["converged"] else "NOT converged"
        print(f"Maxent ({default} default, K = {K}): {flag} in {info['iterations']} "
              f"Newton steps, max moment residual {info['max_moment_residual']:.2e}, max |lambda| {info.get('max_abs_lambda', float('nan')):.3g}")
        if not info["converged"] and info["max_moment_residual"] > 1e-6:
            print("  hint: use fewer moments (-M), a larger --maxent-noise, or --maxent-default kpm (robust to a too-wide spectral window)", file=sys.stderr)
    lams = None
    if jackknife and R > 1:
        lams = []
        for r in range(R):
            mloo = np.delete(mus, r, axis=0).mean(axis=0)
            lams.append(maxent(mloo, sigma, default, grid=grid)[0])
        lams = np.array(lams)
    return lam, lams, info


def maxent_bootstrap(mu, schedule=None, default="flat", tol=1e-9, thresh=1e-4, margin=2.0, maxiter=300, verbose=False, sigma_floor=1e-4):
    """Maxent with bootstrapped support and warm starts (the practice of D. A. Drabold: climb to many moments using the band edges found from fewer).

    The moment count is raised along ``schedule`` (default: 16, 32, 64, ... up to len(mu)).  At each stage the multipliers of the previous stage start Newton's
    method, and after the stage the *support* of the density is shrunk to the interval where the stage's solution exceeds ``thresh`` x its maximum, padded by
    ``margin`` resolution lengths (pi/K in the angle).  The next stage integrates the dual only over that interval: no states are allowed outside the edges, so the
    multipliers no longer have to build huge exponentials to suppress the density in the empty margin of the Chebyshev window (the ill-conditioning seen when
    the window is wider than the spectrum), and the points of the grid are spent where the states are.  The edges are a modelling statement: the final residual of
    the moments is returned (``info['max_moment_residual']``); if the support was cut too tight the stage fails to converge and its support is widened and retried.

    ``sigma_floor`` is the uncertainty assigned to every moment (see :func:`maxent_fit`); 0 gives the exact-moment problem, which stops converging for
    a few hundred moments of a singular density.

    Returns (x, rho_x, idos, info): ``x`` the ascending abscissae in [-1, 1] on which the density is tabulated (zero outside the support is not tabulated:
    ``info['x_lo'], info['x_hi']`` are the edges), ``rho_x`` the density per unit x, ``idos`` the cumulative integral; map to energy with E = a x + b, rho_E = rho_x / a.
    """
    mu = np.asarray(mu, float)
    K = len(mu)
    if schedule is None:
        schedule = []
        k = 16
        while k < K:
            schedule.append(k)
            k *= 2
        schedule.append(K)
    schedule = [int(k) for k in schedule if k <= K]
    th_lo, th_hi = 0.0, np.pi
    lam = None
    hist = []
    good = None
    for k in schedule:
        for attempt in range(6):
            ng = max(4096, 40 * k)
            theta = th_lo + (th_hi - th_lo) * (np.arange(ng) + 0.5) / ng
            dth = (th_hi - th_lo) / ng
            q = np.full(ng, 1.0 / np.pi) if default == "chebyshev" else 0.5 * np.sin(theta)
            C = np.cos(np.outer(np.arange(k), theta))
            sg = None
            if sigma_floor > 0:
                sg = np.full(k, float(sigma_floor)); sg[0] = 0.0
            lam_k, info = maxent(mu[:k], sg, default, tol=tol, maxiter=maxiter, grid=(theta, dth, q, C), lam0=lam)
            if info["converged"] or k == schedule[-1] and attempt == 5:
                break
            pad = 0.5 * (th_hi - th_lo) * 0.25 + margin * np.pi / k             # cut too tight: widen and retry
            th_lo, th_hi = max(0.0, th_lo - pad), min(np.pi, th_hi + pad)
        if not info["converged"] and good is not None:           # stage failed (the exact moments admit no exp-polynomial of this size): keep the last good stage
            theta, dth, f, info = good
            info = dict(info, converged=False, failed_at=k, moments_used=info["moments_used"])
            hist.append((k, th_lo, th_hi, False, -1, np.nan))
            break
        lam = lam_k
        f = q * np.exp(np.clip(lam @ C, -700, 700))
        info = dict(info, moments_used=k)
        good = (theta, dth, f, info)
        hist.append((k, th_lo, th_hi, info["converged"], info["iterations"], info["max_abs_lambda"]))
        if verbose:
            print(f"  K={k:5d} support x in [{np.cos(th_hi):+.4f}, {np.cos(th_lo):+.4f}]  converged={info['converged']} its={info['iterations']} max|lam|={info['max_abs_lambda']:.3g}")
        if k != schedule[-1]:
            rho_th = f / dth
            keep = np.where(rho_th > thresh * rho_th.max())[0]
            pad = margin * np.pi / k
            new_lo = max(th_lo, theta[keep[0]] - pad)
            new_hi = min(th_hi, theta[keep[-1]] + pad)
            th_lo, th_hi = new_lo, new_hi
    order = np.argsort(-theta)                                   # ascending x
    x = np.cos(theta)[order]
    rho_x = (f / np.sin(theta))[order]
    idos = np.cumsum(f[::-1] * dth) - 0.5 * f[::-1] * dth
    info = dict(info, stages=hist, x_lo=float(np.cos(theta[-1] + 0.5 * dth)), x_hi=float(np.cos(theta[0] - 0.5 * dth)))
    return x, rho_x, idos, info
