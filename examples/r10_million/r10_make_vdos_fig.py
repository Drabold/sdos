"""VDOS of the 17^3 R10 replica: matrix-free ACE KPM/Maxent vs exact Bloch levels. (a) full range, (b) low-frequency region as g/omega^2 (Debye)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sdos.results import _rebuild
from sdos.thermo import HBAR_OMEGA_METAL
HW = HBAR_OMEGA_METAL * 1e3  # meV per sqrt(lambda)
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
r = sdos.load("r10_vib_result_n17_M600_R2.npz"); lam = np.sort(np.load("vib_r10_exact_lambda_17.npy")); N = len(lam)
om_ex = HW * np.sqrt(lam[lam > 1e-6]) * 1.0
cur = {"KPM $M{=}600$": (r, "#0072B2", "-"), "Maxent $M{=}300$": (_rebuild(r, 300, "maxent", "flat", None, 8000), "#009E73", "--"), "Maxent $M{=}600$": (_rebuild(r, 600, "maxent", "flat", None, 8000), "#D55E00", "-")}
def g_of(d):
    E = np.asarray(d["E"]); rho = np.asarray(d["rho"]); m = E > 1e-7; om = HW * np.sqrt(E[m]); return om, 2 * om * rho[m] / HW**2  # per meV, per mode
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
a = ax[0]; w = 0.5; bins = np.arange(0, om_ex.max() + w, w); h, e = np.histogram(om_ex, bins=bins); c = 0.5 * (e[1:] + e[:-1])
a.fill_between(c, h / N / w, step="mid", color="0.85", label="exact (0.5 meV bins)")
for k, (d, col, ls) in cur.items():
    if "600" in k and "Maxent" in k: continue
    om, g = g_of(d); a.plot(om, g, color=col, ls=ls, lw=0.9, label=k)
a.set_xlim(0, 70); a.set_ylim(0, 0.06); a.set_xlabel(r"$\hbar\omega$ (meV)"); a.set_ylabel("VDOS per mode (meV$^{-1}$)"); a.legend(frameon=False, fontsize=6.5, loc="upper left"); a.text(0.93, 0.04, "(a)", transform=a.transAxes)
a = ax[1]; w = 1.0; bins = np.arange(0, 16 + w, w); h, e = np.histogram(om_ex, bins=bins); c = 0.5 * (e[1:] + e[:-1])
a.step(c, h / N / w / c**2 * 1e3, where="mid", color="0.5", lw=1.2, label="exact (1 meV bins)")
for k, (d, col, ls) in cur.items():
    om, g = g_of(d); m = (om > 0.5) & (om < 16); a.plot(om[m], g[m] / om[m] ** 2 * 1e3, color=col, ls=ls, lw=0.9, label=k)
a.axvline(7, color="0.5", lw=0.6, ls=":"); a.set_xlim(0, 16); a.set_ylim(0, None); a.set_xlabel(r"$\hbar\omega$ (meV)"); a.set_ylabel(r"VDOS$/\omega^2$ ($10^{-3}$ meV$^{-3}$)"); a.legend(frameon=False, fontsize=6.5, loc="upper left"); a.text(0.93, 0.04, "(b)", transform=a.transAxes)
fig.tight_layout(); fig.savefig("fig_r10_vdos.pdf"); fig.savefig("fig_r10_vdos.png", dpi=150)
for k, (d, col, ls) in cur.items():
    E = np.asarray(d["E"]); idos = np.asarray(d["idos"]); Em = (65.0 / HW) ** 2
    print(k, "fraction of modes above 65 meV:", float(1 - np.interp(Em, E, idos)), "exact:", float((om_ex > 65).mean()))
