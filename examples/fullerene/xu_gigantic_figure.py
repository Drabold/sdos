"""Figure for the Xu orthogonal-TB gigantic fullerene series (standard sdos KPM): DOS vs graphene, exact gap, convergence, band alignment, cost.
Reads xu_smooth.* (400 moments, 12 vectors: DOS curves), xu_gigantic_results.json (1500 moments: gaps/timings) and, if present, xu_smooth_big.* (relaxed C24000+)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = dict(np.load("xu_smooth.npz")); IS = json.load(open("xu_smooth.json")); I = json.load(open("xu_gigantic_results.json"))
import glob
for fj in sorted(glob.glob("xu_smooth_big_*.json")):                # relaxed large shells (one pass per size) replace the unrelaxed ones
    Rb = np.load(fj.replace(".json", ".npz")); Ib = json.load(open(fj))
    for k in Ib:
        if k.isdigit():
            IS[k] = Ib[k]; R[f"E{k}"], R[f"rho{k}"] = Rb[f"E{k}"], Rb[f"rho{k}"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a8a2"
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
ns = sorted(int(k) for k in IS if k.isdigit())
relaxed = {n: IS[str(n)]["geometry"].startswith("relaxed") for n in ns}
nr = [n for n in ns if relaxed[n]]
atoms = {n: IS[str(n)]["atoms"] for n in ns}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, ax = plt.subplots(2, 3, figsize=(17.5, 9.2))
for a in ax.ravel():
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
Eg = R["E"] - IS["graphene_EF"]; rg = R["rho_gra"]
pick = [nr[i] for i in np.unique(np.linspace(1, len(nr) - 1, 5).round().astype(int))]
lab = lambda n: f"C$_{{{60 * n * n}}}$"

a = ax[0, 0]
a.fill_between(Eg, rg, color="#d9d8d2", lw=0, label="graphene (Xu H, k-mesh)"); a.plot(Eg, rg, color=INK2, lw=0.9)
for n, c in zip(pick, RAMP5):
    a.plot(R[f"E{n}"], R[f"rho{n}"], color=c, lw=1.1, label=lab(n))
a.set_xlim(-21, 18); a.set_ylim(0, 0.12); a.set_xlabel("E - E$_F$ (eV)"); a.set_ylabel("DOS (states / orbital / eV)")
a.set_title("Xu tight-binding DOS vs graphene (relaxed geometries)", loc="left", fontsize=11, color=INK); a.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")

z = ax[0, 1]
z.fill_between(Eg, rg, color="#d9d8d2", lw=0); z.plot(Eg, rg, color=INK2, lw=0.9)
for n, c in zip(pick, RAMP5):
    z.plot(R[f"E{n}"], R[f"rho{n}"], color=c, lw=1.2)
z.set_xlim(-3.5, 3.5); z.set_ylim(0, 0.04); z.set_xlabel("E - E$_F$ (eV)"); z.set_ylabel("DOS (states / orbital / eV)")
z.set_title("Near E$_F$", loc="left", fontsize=11, color=INK)

g = ax[0, 2]
ok = [n for n in ns if I.get(str(n), {}).get("EF_exact")]
at = np.array([60 * n * n for n in ok], float); gp = np.array([I[str(n)]["gap"] for n in ok])
g.loglog(at, gp, "o-", color=BLUE, lw=1.4, ms=6, mec=SURF, mew=1.5)
sel = at >= 240; fit = np.polyfit(np.log(at[sel]), np.log(gp[sel]), 1)
g.loglog(at, np.exp(fit[1]) * at ** fit[0], "--", color=GRAY, lw=1.0)
g.text(at[2], np.exp(fit[1]) * at[2] ** fit[0] * 0.55, f"~ N$^{{{fit[0]:.2f}}}$", fontsize=9, color=INK2)
g.set_xlabel("atoms N"); g.set_ylabel("HOMO-LUMO gap (eV), exact"); g.set_title("Gap closes slowly toward graphene's zero gap", loc="left", fontsize=11, color=INK)

# convergence with an optimal rigid shift
msk = (Eg > -20) & (Eg < 18)
trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
def l1_shift(n):
    best = (1e9, 0.0)
    for dlt in np.arange(-1.2, 1.2001, 0.01):
        rn = np.interp(Eg, R[f"E{n}"] + dlt, R[f"rho{n}"])
        v = trap(np.abs(rn - rg)[msk], Eg[msk])
        if v < best[0]:
            best = (v, dlt)
    return best
res = {n: l1_shift(n) for n in nr}
A_ = np.array([atoms[n] for n in nr], float); L1 = np.array([res[n][0] for n in nr]); DL = np.array([res[n][1] for n in nr])
d = ax[1, 0]
d.loglog(A_, L1, "o-", color=BLUE, lw=1.4, ms=6, mec=SURF, mew=1.5)
sel = A_ >= 240; f2 = np.polyfit(np.log(A_[sel]), np.log(L1[sel]), 1)
d.loglog(A_, np.exp(f2[1]) * A_ ** f2[0], "--", color=GRAY, lw=1.0)
d.text(A_[3], np.exp(f2[1]) * A_[3] ** f2[0] * 0.55, f"~ N$^{{{f2[0]:.2f}}}$", fontsize=9, color=INK2)
d.set_xlabel("atoms N"); d.set_ylabel(r"min over rigid shift of $\int |\rho_N - \rho_{graphene}|\, dE$")
d.set_title("Cluster to crystal: distance to graphene", loc="left", fontsize=11, color=INK)

e = ax[1, 1]
orb = np.array([IS[str(n)]["orbitals"] for n in ns], float)
tt = np.array([(I.get(str(n)) or IS[str(n)])["seconds_dos"] for n in ns])
e.loglog(orb, tt, "o-", color=BLUE, lw=1.4, ms=6, mec=SURF, mew=1.5)
big = orb > 5000; ref = tt[big][-1] * orb / orb[big][-1]
e.loglog(orb, ref, "--", color=GRAY, lw=1.0); e.text(orb[big][0], ref[big][0] * 0.5, "linear in N", fontsize=9, color=INK2)
e.annotate(f"{int(orb[-1] / 4):,} atoms\n{tt[-1]:.0f} s", (orb[-1], tt[-1]), textcoords="offset points", xytext=(-70, -32), fontsize=8.5, color=INK2)
e.set_xlabel("orbitals"); e.set_ylabel("seconds (1500 moments, 2 vectors)"); e.set_title("DOS cost: about linear in N", loc="left", fontsize=11, color=INK)

f = ax[1, 2]
ef_off = np.array([IS[str(n)]["E_F"] - IS["graphene_EF"] for n in ok])
f.semilogx(at, ef_off, "s-", color=ORANGE, lw=1.3, ms=6, mec=SURF, mew=1.5, label="E$_F$(cluster) - E$_F$(graphene), exact")
f.semilogx(A_, -DL, "o-", color=BLUE, lw=1.3, ms=6, mec=SURF, mew=1.5, label="rigid shift that best matches graphene")
f.axhline(0, color=INK2, lw=0.8)
f.set_xlabel("atoms N"); f.set_ylabel("energy offset (eV)"); f.legend(frameon=False, fontsize=8.5, loc="lower right")
f.set_title("Band alignment converges to graphene's", loc="left", fontsize=11, color=INK)
fig.suptitle("Gigantic icosahedral fullerenes C$_{60n^2}$ with the Xu et al. orthogonal tight-binding Hamiltonian, via sdos", x=0.05, ha="left", fontsize=13, color=INK)
fig.text(0.5, 0.012, f"largest: {int(orb[-1] / 4):,} atoms ({int(orb[-1]):,} orbitals, peak memory {max(IS[str(n)]['peak_rss_gb'] for n in ns):.2f} GB); exact gaps and E$_F$ by LDL$^T$ inertia counting up to {max(60 * n * n for n in ok):,} atoms; "
         "curves: 400 moments (0.16 eV), 12 vectors; geometries: LCBOP-relaxed" + ("" if all(relaxed.values()) else "; unrelaxed sizes omitted from the curves and the convergence panel"), ha="center", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); fig.savefig("xu_gigantic.png", dpi=140)
print("wrote xu_gigantic.png | gap exponent", round(fit[0], 2), "| shifted-L1 exponent", round(f2[0], 2), "| shifted L1:", dict(zip([60*n*n for n in nr], np.round(L1, 3))), "| shifts:", dict(zip([60*n*n for n in nr], np.round(DL, 2))))
