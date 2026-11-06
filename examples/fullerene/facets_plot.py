import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("facets_ef.json"))
SURF, INK, INK2, GRID, BLUE, ORANGE, GREEN = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834", "#1a9a6a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for a_, N in zip(ax, ("960", "2160", "3840")):
    r = d[N]; E = np.array(r["E"]); e = np.array(r["enh"])
    s = 0; M = []
    while s < len(E):
        t = s
        while t + 1 < len(E) and abs(E[t + 1] - E[s]) < 2e-4: t += 1
        M.append((E[s:t + 1].mean(), t - s + 1, e[s:t + 1].mean(0))); s = t + 1
    a_.axhline(1, color=INK2, lw=0.8, ls=":"); a_.grid(True, color=GRID, lw=0.6); a_.set_axisbelow(True)
    for k, (c, lab, off) in enumerate(((BLUE, "pentagon neighbourhood", 0.0), (ORANGE, "edge strips", 0.0), (GREEN, "facet interior", 0.0))):
        a_.scatter([m[0] for m in M], [m[2][k] for m in M], s=[18 * m[1] for m in M], color=c, alpha=0.85, edgecolor=SURF, lw=0.8, label=lab)
    a_.axvspan(-r["gap"] / 2, r["gap"] / 2, color="#d9d8d2", alpha=0.6, lw=0)
    a_.set_title(f"C{N}   (gap {r['gap']:.2f} eV)", loc="left", fontsize=10, color=INK2); a_.set_xlabel("E - E_F (eV)")
ax[0].set_ylabel("weight in class / fraction of atoms in class"); ax[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("States nearest E_F: where each Ih multiplet puts its weight (marker area = multiplet size; shaded = HOMO-LUMO gap)", x=0.05, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig("facets_ef.png", dpi=150)
