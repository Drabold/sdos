"""Figure: icosahedral fullerenes (SIESTA Harris/SZ H and S, non-orthogonal) -> graphene, from the generalized sdos DOS.  Reads fullerene_dos_results.npz/json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = np.load("fullerene_dos_results.npz")
I = json.load(open("fullerene_dos_results.json"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a8a2"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]           # ordinal blue (validated), light = small cluster
SIZES = [60, 240, 540, 960, 2160]

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, ax = plt.subplots(2, 3, figsize=(17.5, 9.2))
for a in ax.ravel():
    a.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
Eg, rg = R["Egra"], R["rho_gra"]

# (a) series -> graphene
a = ax[0, 0]
a.fill_between(Eg, rg, color="#d9d8d2", lw=0, label="graphene (same H, S; k-mesh)")
a.plot(Eg, rg, color=INK2, lw=0.9)
for n, c in zip(SIZES, RAMP):
    a.plot(R[f"E{n}"], R[f"rho{n}"], color=c, lw=1.2, label=f"C$_{{{n}}}$")
a.set_xlim(-22, 20); a.set_ylim(0, 0.13)
a.set_xlabel("E - E$_F$ (eV)"); a.set_ylabel("DOS (states / orbital / eV)")
a.set_title("DOS of C$_{60}$ ... C$_{2160}$ vs graphene", loc="left", fontsize=11, color=INK)
a.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")

# (b) near the Fermi level
z = ax[0, 1]
z.fill_between(Eg, rg, color="#d9d8d2", lw=0); z.plot(Eg, rg, color=INK2, lw=0.9, label="graphene")
for n, c in zip(SIZES, RAMP):
    z.plot(R[f"E{n}"], R[f"rho{n}"], color=c, lw=1.3, label=f"C$_{{{n}}}$")
z.set_xlim(-3.5, 3.5); z.set_ylim(0, 0.04)
z.set_xlabel("E - E$_F$ (eV)"); z.set_ylabel("DOS (states / orbital / eV)")
z.set_title("Near E$_F$: gap closes, Dirac-like dip emerges", loc="left", fontsize=11, color=INK)

# (c) non-orthogonality
b = ax[0, 2]
b.plot(Eg, R["rho960_exact"], color=GRAY, lw=3.0, alpha=0.8, label="exact: eigh(H, S)")
b.plot(R["E960"], R["rho960"], color=BLUE, lw=1.2, label="sdos, generalized (H, S)")
b.plot(R["E960_Signored"], R["rho960_Signored"], color=ORANGE, lw=1.2, label="sdos, S ignored (H only)")
lo = min(R["E960_Signored"][R["rho960_Signored"] > 1e-4].min(), -24); hi = max(R["E960_Signored"][R["rho960_Signored"] > 1e-4].max(), 24)
b.set_xlim(lo - 1, hi + 1); b.set_ylim(0, None)
b.set_xlabel("E - E$_F$ (eV)"); b.set_ylabel("DOS (states / orbital / eV)")
b.set_title("C$_{960}$: ignoring S gives the wrong spectrum", loc="left", fontsize=11, color=INK)
b.legend(frameon=False, fontsize=8.5, loc="upper left")
s_ = I["S960"]
b.text(0.98, 0.97, f"cond(S) = {s_['cond']:.1f}   (S 3.8% dense)\nbandwidth: {s_['bandwidth_generalized']:.1f} eV with S, {s_['bandwidth_S_ignored']:.1f} eV without",
       transform=b.transAxes, ha="right", va="top", fontsize=8.3, color=INK2)

# (d) convergence to graphene: L1 distance
d = ax[1, 0]
L1 = []
msk = (Eg > -20) & (Eg < 20)
for n in SIZES:
    rn = np.interp(Eg, R[f"E{n}"], R[f"rho{n}"])
    L1.append(np.trapezoid(np.abs(rn - rg)[msk], Eg[msk]) if hasattr(np, "trapezoid") else np.trapz(np.abs(rn - rg)[msk], Eg[msk]))
L1 = np.array(L1)
Na = np.array(SIZES, float)
d.loglog(Na, L1, "o-", color=BLUE, lw=1.4, ms=6, mec=SURF, mew=1.5)
fit = np.polyfit(np.log(Na[1:]), np.log(L1[1:]), 1)
d.loglog(Na, np.exp(fit[1]) * Na ** fit[0], "--", color=GRAY, lw=1.0)
d.text(Na[1], np.exp(fit[1]) * Na[1] ** fit[0] * 0.62, f"~ N$^{{{fit[0]:.2f}}}$ (C$_{{240}}$ to C$_{{2160}}$)", fontsize=8.5, color=INK2)
d.set_xlabel("atoms N"); d.set_ylabel(r"$\int |\rho_N - \rho_{graphene}|\, dE$  (-20 to 20 eV)")
d.set_title("Cluster to crystal: distance to graphene vs size", loc="left", fontsize=11, color=INK)

# (e) pentagon vs hexagon atoms
c = ax[1, 1]
c.plot(R["Epdos"], R["pdos_hex"], color=BLUE, lw=1.3, label=f"hexagon-only atoms ({I['pdos']['hexagon_atoms']})")
c.plot(R["Epdos"], R["pdos_pent"], color=ORANGE, lw=1.3, label=f"pentagon atoms ({I['pdos']['pentagon_atoms']})")
c.set_xlim(-22, 20); c.set_ylim(0, None)
c.set_xlabel("E - E$_F$ (eV)"); c.set_ylabel("projected DOS (states / orbital / eV)")
c.set_title("C$_{2160}$: Mulliken-projected DOS", loc="left", fontsize=11, color=INK)
c.legend(frameon=False, fontsize=8.5, loc="upper left")

# (f) cost scaling
f = ax[1, 2]
no = np.array([I[str(n)]["orbitals"] for n in SIZES], float); t = np.array([I[str(n)]["seconds"] for n in SIZES])
f.loglog(no, t, "o-", color=BLUE, lw=1.4, ms=6, mec=SURF, mew=1.5)
ref = t[-1] * no / no[-1]
f.loglog(no, ref, "--", color=GRAY, lw=1.0)
f.text(no[2] * 1.15, ref[2] * 0.62, "linear in N", color=INK2, fontsize=8.5)
for n, x, y in zip(SIZES, no, t):
    f.annotate(f"C$_{{{n}}}$", (x, y), textcoords="offset points", xytext=(-4, 9), fontsize=8, color=INK2)
f.set_xlabel("orbitals N"); f.set_ylabel("seconds (500 moments, 2-3 vectors)")
f.set_title(f"Cost linear in N; {I['2160']['iters_per_solve']:.0f} PCG iterations per solve at every size", loc="left", fontsize=11, color=INK)
bm = I["binv_benchmark_C960"]
fig.text(0.5, 0.012, "B$^{-1}$ on C$_{960}$ (tol 1e-12): " + ", ".join(f"{k} {v['iterations']}" for k, v in bm.items() if v and "iterations" in v)
         + f" PCG iterations; Chebyshev polynomial degree {bm['chebyshev']['degree']}", ha="center", fontsize=9, color=INK2)
fig.suptitle("Icosahedral fullerenes C$_{60n^2}$: SIESTA Harris / single-zeta (H, S), non-orthogonal, via sdos", x=0.05, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig("fullerene_dos.png", dpi=140)
print("wrote fullerene_dos.png; L1 distances to graphene:", dict(zip(SIZES, np.round(L1, 3))), "exponent", round(fit[0], 2))
