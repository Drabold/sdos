"""Harmonic vibrational thermodynamics from a vibrational DOS g(omega): zero-point energy, U, F, S and C_v versus temperature.

For every mode, x = hbar omega / (k_B T):
    u = hbar omega (1/2 + 1/(e^x - 1)),   f = hbar omega / 2 + k_B T ln(1 - e^-x),
    s / k_B = x/(e^x - 1) - ln(1 - e^-x),   c / k_B = x^2 e^x / (e^x - 1)^2,
and the totals per atom are ``dof_per_atom`` x the average over the normalised g (the rigid translations carry no weight in the average;
their contribution is O(1/N) per atom and is not included).

Units: ``energy_per_sqrt_lambda`` converts sqrt(lambda) to hbar*omega in eV.  The default is for LAMMPS-style metal units (lambda in eV/(A^2 amu)):
hbar sqrt(eV / (A^2 amu)) = 64.654 meV.  Pass your own factor for other units.

Limits (reported in ``info``): (1) g(omega) from KPM is smoothed over a resolution delta_omega ~ sqrt(pi a / M) near omega = 0 (where
lambda = omega^2 -> 0), so low-temperature values (k_B T below a few times hbar delta_omega) are not reliable (``T_reliable``); (2) imaginary modes
(lambda < 0) have no harmonic thermodynamics: only lambda > 0 is used and the neglected weight is reported and warned about.
"""

import warnings

import numpy as np

HBAR = 1.054571817e-34
EV = 1.602176634e-19
AMU = 1.66053906660e-27
KB_EV = 8.617333262e-5
HBAR_OMEGA_METAL = HBAR * np.sqrt(EV / (1e-20 * AMU)) / EV      # eV per sqrt(eV/(A^2 amu)) = 0.0646538


class ThermoResult(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc


def _trapz(y, x):
    return (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(y, x)


def harmonic_thermo(res, T, dof_per_atom=3, energy_per_sqrt_lambda=HBAR_OMEGA_METAL, renormalize=True):
    """Thermodynamic functions per atom versus T (K) from the result of :func:`sdos.vdos`.

    Returns a :class:`ThermoResult`: ``T``, ``zpe`` (eV/atom), ``U``, ``F`` (eV/atom, including the zero-point energy), ``S`` (k_B/atom),
    ``Cv`` (k_B/atom), and ``info`` (neglected weight, reliability limit).  Dulong-Petit check: Cv -> dof_per_atom at high T.
    """
    if "omega" not in res:
        raise ValueError("pass the result of sdos.vdos (it carries omega and g)")
    T = np.atleast_1d(np.asarray(T, float))
    w = np.asarray(res["omega"], float)
    g = np.clip(np.asarray(res["g"], float), 0.0, None)
    norm = _trapz(g, w)
    if norm <= 0:
        raise ValueError("empty vibrational spectrum")
    neglected = max(0.0, 1.0 - norm) if not renormalize else max(0.0, 1.0 - norm)
    if res.get("negative_indicated") or res.get("neg_weight", 0.0) > 0.01 * res["N"]:
        warnings.warn(f"imaginary modes indicated (weight ~ {res.get('neg_weight', 0.0) / res['N']:.3g} of the spectrum): the harmonic "
                      "thermodynamics uses lambda > 0 only", stacklevel=2)
    gn = g / norm
    hw = energy_per_sqrt_lambda * w                                   # eV
    out = {k: np.zeros_like(T) for k in ("zpe", "U", "F", "S", "Cv")}
    zpe = dof_per_atom * _trapz(gn * 0.5 * hw, w)
    for i, t in enumerate(T):
        if t <= 0:
            out["U"][i] = out["F"][i] = zpe
            continue
        x = hw / (KB_EV * t)
        big = x > 700
        xs = np.where(big, 1.0, x)
        em = np.where(big, 0.0, np.exp(-xs))                          # e^-x
        occ = em / (1.0 - em)                                         # 1/(e^x - 1)
        lnterm = np.where(big, 0.0, np.log1p(-em))                    # ln(1 - e^-x)
        u = hw * (0.5 + occ)
        f = 0.5 * hw + KB_EV * t * lnterm
        s = xs * occ - lnterm
        c = xs ** 2 * em / (1.0 - em) ** 2
        out["U"][i] = dof_per_atom * _trapz(gn * u, w)
        out["F"][i] = dof_per_atom * _trapz(gn * f, w)
        out["S"][i] = dof_per_atom * _trapz(gn * np.where(big, 0.0, s), w)
        out["Cv"][i] = dof_per_atom * _trapz(gn * np.where(big, 0.0, c), w)
    out["zpe"][:] = zpe
    resol = res.get("resolution", np.nan)
    t_rel = (energy_per_sqrt_lambda * np.sqrt(resol) / KB_EV * 3.0) if np.isfinite(resol) else np.nan
    return ThermoResult(T=T, zpe=zpe, U=out["U"], F=out["F"], S=out["S"], Cv=out["Cv"],
                        info=dict(neglected_weight=neglected, T_reliable=float(t_rel), dof_per_atom=dof_per_atom,
                                  units="U, F, zpe in eV/atom; S, Cv in k_B/atom"))
