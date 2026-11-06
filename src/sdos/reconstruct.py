"""Reconstruction of the DOS from Chebyshev moments (KPM kernels), IDOS, band energy and Fermi level."""

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def jackson(M):
    """Jackson damping factors g_n, n = 0..M-1."""
    n = np.arange(M)
    q = np.pi / (M + 1)
    return ((M - n + 1) * np.cos(q * n) + np.sin(q * n) / np.tan(q)) / (M + 1)


def lorentz(M, lam=4.0):
    """Lorentz damping factors g_n = sinh(lam (1 - n/M)) / sinh(lam) (Weisse et al., RMP 78, 275)."""
    n = np.arange(M)
    return np.sinh(lam * (1.0 - n / M)) / np.sinh(lam)


def kernel_factors(M, kernel="jackson", lam=4.0):
    if kernel == "jackson":
        return jackson(M)
    if kernel == "lorentz":
        return lorentz(M, lam)
    raise ValueError("kernel must be 'jackson' or 'lorentz'")


def reconstruct(mu, a, b, npts, kernel="jackson", lam=4.0):
    """DOS (normalised to 1) and integrated DOS on an energy grid.

    mu: (..., M) moments normalised so mu[..., 0] = 1.  Returns E (npts,), rho (..., npts),
    idos (..., npts).  The IDOS is the analytic integral of the kernel-damped series.
    """
    M = mu.shape[-1]
    g = kernel_factors(M, kernel, lam)
    c = g * mu
    theta = np.pi * (np.arange(npts) + 0.5) / npts     # Chebyshev-Gauss nodes
    x = np.cos(theta)[::-1]
    theta = theta[::-1]
    n = np.arange(1, M)
    cosmat = np.cos(np.outer(n, theta))                # (M-1, npts)
    sinmat = np.sin(np.outer(n, theta)) / n[:, None]
    series = c[..., :1] + 2.0 * (c[..., 1:] @ cosmat)
    rho_x = series / (np.pi * np.sqrt(1.0 - x ** 2))
    idos = c[..., :1] * (1.0 - theta / np.pi) - 2.0 * (c[..., 1:] @ sinmat) / np.pi
    E = a * x + b
    return E, rho_x / a, idos


def band_quantities(mu, a, b, n_orb, n_elec, spin, ngrid=20000, kernel="jackson", lam=4.0):
    """Fermi level and band-structure energy from the kernel-damped DOS."""
    M = mu.shape[-1]
    g = kernel_factors(M, kernel, lam)
    c = g * mu
    theta = np.linspace(np.pi, 0.0, ngrid)             # x from -1 to 1
    x = np.cos(theta)
    n = np.arange(1, M)
    idos = c[0] * (1.0 - theta / np.pi) - 2.0 * (np.sin(np.outer(theta, n)) / n) @ c[1:] / np.pi
    count = spin * n_orb * idos
    k = np.searchsorted(count, n_elec)
    if k == 0 or k >= ngrid:
        raise ValueError("electron count outside spectrum")
    t = (n_elec - count[k - 1]) / (count[k] - count[k - 1])
    thF = theta[k - 1] + t * (theta[k] - theta[k - 1])
    EF = a * np.cos(thF) + b
    th = np.linspace(np.pi, thF, ngrid)
    xs = np.cos(th)
    series = c[0] + 2.0 * np.cos(np.outer(th, n)) @ c[1:]
    integrand = (a * xs + b) * series / np.pi
    Ebs = spin * n_orb * _trapezoid(integrand, -th)
    return EF, Ebs
