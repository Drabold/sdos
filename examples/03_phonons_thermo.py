"""Vibrational DOS of a Keating Si/Ge crystal with unequal masses, then harmonic thermodynamics."""
import sys
sys.path.insert(0, "../tests")
import numpy as np
import sdos
from keating import keating_hessian

Phi = keating_hessian(nc=2)                                          # force constants, eV/A^2 (64 atoms)
mass = np.repeat(np.random.default_rng(4).choice([28.0855, 72.63], size=Phi.shape[0] // 3), 3)
res = sdos.vdos(Phi, mass=mass, M=200, R=40, seed=2, zero_modes=3)   # lambda = omega^2; remove the 3 rigid translations exactly
print("omega grid:", res.omega.min().round(3), "-", res.omega.max().round(3), " negative modes indicated:", res.negative_indicated)
T = np.array([100.0, 300.0, 600.0, 1200.0])
th = sdos.harmonic_thermo(res, T)
print("zero-point energy (eV/atom):", round(th.zpe, 4), "; reliable for T >~", round(th.info["T_reliable"]), "K")
for t, u, f, s, c in zip(th.T, th.U, th.F, th.S, th.Cv):
    print(f"T = {t:6.0f} K  U = {u:.4f} eV  F = {f:.4f} eV  S = {s:.3f} kB  Cv = {c:.3f} kB  (Dulong-Petit 3)")
