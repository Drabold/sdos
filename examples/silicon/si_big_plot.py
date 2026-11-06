import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, GRID, BLUE, ORANGE, GREEN = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834", "#1a9a6a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
D = {n: json.load(open(f"si_big_n{n}.json")) for n in (32, 64)}
d = D[64]; ed = np.array(d["exact_hist"]["edges"]); ct = np.array(d["exact_hist"]["counts"]); Ec = 0.5 * (ed[1:] + ed[:-1])
print("exact histogram orbitals:", d["exact_hist"]["n_orb"], " expected", 32 * 64 ** 3)
rho_ex = ct / ct.sum() / np.diff(ed)
def smooth(E, rho, eta):
    out = np.zeros_like(E); w = np.diff(ed)
    for i in range(0, len(Ec), 400):
        out += (np.exp(-0.5 * ((E[:, None] - Ec[None, i:i + 400]) / eta) ** 2) * (rho_ex * w)[None, i:i + 400]).sum(1)
    return out / (eta * np.sqrt(2 * np.pi))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a_ in ax: a_.grid(True, color=GRID, lw=0.6); a_.set_axisbelow(True)
for k, (meth, c) in enumerate((("kpm", BLUE), ("maxent", ORANGE))):
    E = np.array(d[meth]["E"]); r = np.array(d[meth]["rho"])
    ref = smooth(E, rho_ex, 0.1); sel = (E > -14.5) & (E < 7.5)
    l1 = np.trapezoid(np.abs(r[sel] - ref[sel]), E[sel])
    print(f"n=64 {meth}: L1 to exact (0.1 eV Gaussian) {l1:.4f}; {d[meth]['seconds']:.0f} s")
    ax[0].plot(E, r, color=c, lw=1.3, label=f"{meth.upper()} 170 moments, 4 vectors")
ax[0].fill_between(Ec, rho_ex * 0 + smooth(Ec, rho_ex, 0.1), color="#d9d8d2", lw=0, label="exact k-sum (0.1 eV Gaussian)")
ax[0].set_xlim(-15, 8); ax[0].set_xlabel("E (eV)"); ax[0].set_ylabel("DOS per orbital"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("2,097,152 atoms (8.4M orbitals), Kwon TB, diamond Si", loc="left", fontsize=10, color=INK2)
# gap / band-edge zoom
for meth, c in (("kpm", BLUE), ("maxent", ORANGE)):
    ax[1].semilogy(d[meth]["E"], np.maximum(d[meth]["rho"], 1e-6), color=c, lw=1.2, label=meth.upper())
ax[1].semilogy(Ec, np.maximum(smooth(Ec, rho_ex, 0.05), 1e-6), color=INK2, lw=0.8, ls="--", label="exact (0.05 eV)")
ax[1].set_xlim(-1.5, 2.5); ax[1].set_ylim(1e-5, 0.3); ax[1].set_xlabel("E (eV)"); ax[1].set_title("gap region (log scale)", loc="left", fontsize=10, color=INK2); ax[1].legend(frameon=False, fontsize=8)
t = {n: (D[n]["kpm"]["seconds"], D[n]["maxent"]["seconds"]) for n in D}
ax[2].bar([0, 1, 3, 4], [t[32][0], t[32][1], t[64][0], t[64][1]], color=[BLUE, ORANGE, BLUE, ORANGE])
ax[2].set_xticks([0.5, 3.5]); ax[2].set_xticklabels(["262,144 atoms", "2,097,152 atoms"]); ax[2].set_ylabel("seconds (4 threads, 4 vectors, M=170)")
ax[2].set_title("time (blue KPM, orange Maxent)", loc="left", fontsize=10, color=INK2)
fig.tight_layout(); fig.savefig("si_big_dos.png", dpi=150)
