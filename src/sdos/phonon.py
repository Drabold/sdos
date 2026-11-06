"""Vibrational density of states.  The spectral variable is lambda = omega^2; ``vdos`` returns g(omega) = 2 omega rho(omega^2).

Negative eigenvalues (instabilities, or numerical noise on the three/six translational and rotational zero modes) are never
clipped: they are detected, reported in the result and warned about.  Kernel broadening limits the resolution near lambda = 0
(Jackson width ~ pi a / M); to count or locate modes at or below zero exactly use ``shift_invert_dos`` around 0.
"""

import warnings

import numpy as np
import scipy.sparse as sp

from .api import dos


def vdos(D, mass=None, B=None, dof=3, neg_tol=1e-6, zero_modes=0, **options):
    """DOS in omega of a force-constant matrix D (explicit sparse, or matrix-free with ``B=None`` and no ``mass``).

    ``mass``: masses per degree of freedom (length N) or per atom (length N/dof, repeated ``dof`` times); this is the
    diagonal-B case: mass-weighting is done once, no solves.  Or pass a general ``B`` (mass matrix) for D x = lambda B x.
    Returns the lambda-space :class:`~sdos.DOSResult` plus ``omega``, ``g``, ``g_err`` (omega > 0 part, g = 2 omega rho(omega^2)),
    ``neg_weight`` (N x IDOS(-resolution): eigenvalue count below the resolution), ``lambda_min`` (estimated extreme from the
    bounds) and a warning if negative eigenvalues are indicated.
    ``zero_modes=3`` removes the three rigid translations (known eigenvalues at lambda = 0) from the moments exactly, which matters for the
    entropy and low-temperature C_v (the kernel would otherwise smear their weight into small positive omega).
    """
    if mass is not None:
        m = np.asarray(mass, float)
        n = D.shape[0]
        if len(m) == n // dof and n % dof == 0 and len(m) != n:
            m = np.repeat(m, dof)
        if len(m) != n:
            raise ValueError("mass must have length N or N/dof")
        if B is not None:
            raise ValueError("give either mass or B")
        B = m
    if zero_modes:
        options = dict(options, known_eigenvalues=[(0.0, int(zero_modes))])
    res = dos(D, B=B, **options)
    lam = res["E"]
    lam_min = res["bounds_raw"][0]
    lam_max = res["bounds_raw"][1]
    resol = res.get("resolution", np.nan)
    if np.isfinite(resol):
        neg_weight = float(res["N"] * np.interp(-resol, lam, res["idos"], left=0.0))
    else:
        neg_weight = float(res["N"] * np.interp(0.0, lam, res["idos"], left=0.0))
    pos = lam > 0
    omega = np.sqrt(lam[pos])
    g = 2.0 * omega * res["rho"][pos]
    g_err = 2.0 * omega * res["rho_err"][pos]
    negative = lam_min < -neg_tol * lam_max
    res.update(omega=omega, g=g, g_err=g_err, neg_weight=neg_weight, lambda_min=lam_min, negative_indicated=bool(negative))
    if negative:
        warnings.warn(f"negative eigenvalues indicated: estimated lambda_min = {lam_min:.4g} (= {lam_min / lam_max:.2e} of lambda_max); "
                      f"DOS weight below -resolution corresponds to ~{neg_weight:.1f} states. They are NOT clipped: g(omega) covers "
                      "lambda > 0 only; use shift_invert_dos around 0 to count/locate them", stacklevel=2)
    return res
