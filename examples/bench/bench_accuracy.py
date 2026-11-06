"""Accuracy of KPM and Maxent against exact results on DTW4096 (Kwon TB, exact dense spectrum cached by examples/dtw4096/dtw_electrons.py).
(1) truncation error vs M, free of sampling noise: moments computed EXACTLY from the eigenvalues, reconstruction vs the exact IDOS-based observables;
(2) stochastic error vs R at fixed M (spread over seeds)."""
import json, warnings
import numpy as np
import sdos
from sdos.interfaces import kwon_si as K
from sdos.reconstruct import reconstruct
from sdos.maxent import maxent, maxent_density, band_quantities_grid
warnings.filterwarnings("ignore")
ev = np.load("../dtw4096/dtw4096_kwon_ev.npy")
n = len(ev); nocc = n // 2
a_, b_ = 0.5 * (ev[-1] - ev[0]) * 1.01, 0.5 * (ev[-1] + ev[0])
th = np.arccos(np.clip((ev - b_) / a_, -1, 1))
Mmax = 400
mu = np.array([np.cos(m * th).mean() for m in range(Mmax)])             # exact moments of the DOS
ebs_ex = 2 * ev[:nocc].sum() / 4096
# exact integrated quantities on the same footing: band energy per atom and N(E) at several energies
Es = np.array([-10.0, -6.0, -3.0, 0.0, 2.0])
exact_N = np.array([(ev < e).mean() for e in Es])
out = dict(M=[], kpm=[], maxent=[])
print("truncation error with exact moments (per-atom band energy error in meV; max |IDOS error| at 5 energies; pointwise L1 vs the exact spectrum broadened to 0.05 eV)")
Eg = np.linspace(ev[0] - 0.5, ev[-1] + 0.5, 3000)
ref = np.zeros_like(Eg)
for ch in np.array_split(ev, 40): ref += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.05) ** 2).sum(1)
ref /= n * 0.05 * np.sqrt(2 * np.pi)
for M in (20, 40, 70, 100, 170, 300, 400):
    row = {}
    E, rs, ids = reconstruct(mu[None, :M], a_, b_, 4000)
    rho, idos = rs[0], ids[0]
    ef, eb = None, None
    from sdos.reconstruct import band_quantities
    ef_, eb_ = band_quantities(mu[:M], a_, b_, n, n, 2.0)
    kd = dict(M=M, dEBS=(eb_ / 4096 - ebs_ex) * 1e3, dN=float(np.max(np.abs(np.interp(Es, E, idos) - exact_N))), L1=float(np.trapezoid(np.abs(np.interp(Eg, E, rho) - ref), Eg)))
    md = None
    if M <= 170:
        lam, info = sdos.maxent(mu[:M], default="flat", tol=1e-9, maxiter=300)
        Em, rm, im = maxent_density(lam, a_, b_, "flat", 4000, mu=mu[:M])
        ef2, eb2 = band_quantities_grid(Em, rm, im, n, n, 2.0)
        md = dict(M=M, dEBS=(eb2 / 4096 - ebs_ex) * 1e3, dN=float(np.max(np.abs(np.interp(Es, Em, im) - exact_N))), L1=float(np.trapezoid(np.abs(np.interp(Eg, Em, rm) - ref), Eg)), conv=bool(info["converged"]))
    out["M"].append(M); out["kpm"].append(kd); out["maxent"].append(md)
    print(f"M={M:4d}  KPM  dE_BS {kd['dEBS']:+9.3f} meV/atom  max|dN| {kd['dN']:.2e}  L1 {kd['L1']:.4f}" + (f" | Maxent dE_BS {md['dEBS']:+9.3f}  max|dN| {md['dN']:.2e}  L1 {md['L1']:.4f} conv={md['conv']}" if md else ""), flush=True)
# stochastic error vs R at M = 100 (band energy per atom, KPM Jackson, plain Rademacher vectors, no exact-moment substitution)
_d = np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz")
H = K.kwon_hamiltonian(_d["X"], _d["L"])
out["R"] = []
print("stochastic error of E_BS (meV/atom) vs R at M = 100: std over 16 seeds (KPM)")
for R in (1, 2, 4, 8, 16, 32):
    vals = []
    for s in range(16):
        r = sdos.dos(H, M=100, R=R, method="kpm", seed=1000 + s, exact_moments=0)
        vals.append((sdos.fermi_and_band_energy(r, n)[1] / 4096 - ebs_ex) * 1e3)
    vals = np.array(vals)
    out["R"].append(dict(R=R, mean=float(vals.mean()), std=float(vals.std(ddof=1))))
    print(f"R={R:3d}  mean {vals.mean():+8.2f}  std {vals.std(ddof=1):7.2f}   (std*sqrt(R) = {vals.std(ddof=1) * np.sqrt(R):.1f})", flush=True)
json.dump(out, open("bench_accuracy.json", "w"), indent=1)
