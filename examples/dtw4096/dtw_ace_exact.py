"""Exact ACE vibrational spectrum of DTW4096 from LAMMPS' dynamical_matrix (lmp_kk on the GPU, pair_style pace product) against the sdos matrix-free result (dtw_phonons.py ace)."""
import json, warnings
import numpy as np
import scipy.linalg as sla
from sdos.thermo import HBAR_OMEGA_METAL
warnings.filterwarnings("ignore")
n = 4096
D = np.fromfile("/home/drabold/lammps-runs/topoMC/sdos_dtw/dtw4096_ace_dynmat.dat", dtype=np.float64)
assert D.size == (3 * n) ** 2, D.size
D = D.reshape(3 * n, 3 * n)
print("dynmat: symmetric defect", np.abs(D - D.T).max(), " max element", np.abs(D).max(), flush=True)
D = 0.5 * (D + D.T)
lam = np.linalg.eigvalsh(D)
print(f"exact eigenvalues: {int((lam < -1e-6).sum())} negative, {int((np.abs(lam) <= 1e-6).sum())} within 1e-6 of zero; lambda range [{lam[0]:.3e}, {lam[-1]:.3f}] eV/A^2/amu")
lp = lam[lam > 1e-6]
om = np.sqrt(lp) * HBAR_OMEGA_METAL
kb = 8.617333262e-5
T = [100.0, 300.0, 600.0]
zpe = 0.5 * om.sum() / n
cv = [float(np.sum((om / (kb * t)) ** 2 * np.exp(om / (kb * t)) / (np.exp(om / (kb * t)) - 1) ** 2) / n) for t in T]
print(f"exact: ZPE {zpe * 1e3:.3f} meV/atom; Cv(100,300,600 K) = {np.round(cv, 4)}")
s = json.load(open("dtw_phonons_ace.json"))
for m in ("kpm", "maxent"):
    print(f"sdos {m}: ZPE {s[m]['zpe'] * 1e3:.3f} ({100 * (s[m]['zpe'] / zpe - 1):+.2f}%), Cv = {np.round(s[m]['Cv'], 4)} ({', '.join(f'{100 * (a / b - 1):+.2f}%' for a, b in zip(s[m]['Cv'], cv))})")
np.save("dtw4096_ace_lambda.npy", lam)
json.dump(dict(zpe=float(zpe), Cv=cv, n_neg=int((lam < -1e-6).sum())), open("dtw_ace_exact.json", "w"))
