"""Figure for the paper: R10 17^3 replica. (a) Kwon DOS in the gap/tails, exact vs KPM vs Maxent; (b) sound-wave weight and in-cell IPR of the exact low-frequency modes."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
d = np.load("r10_kwon_curves.npz"); ev = d["ev"]; N = len(ev); b = json.load(open("r10_character.json"))["bins"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
a = ax[0]; lo, hi = -1.2, 2.2; w = 0.02; bins = np.arange(lo, hi + w, w); h, e = np.histogram(ev, bins=bins); c = 0.5 * (e[1:] + e[:-1])
a.fill_between(c, np.maximum(h / N / w, 1e-9), step="mid", color="0.85", label="exact")
for k, col, ls, lab in (("kpm600", "#0072B2", "-", "KPM $M{=}600$"), ("maxent300", "#009E73", "--", "Maxent $M{=}300$"), ("maxent600", "#D55E00", "-", "Maxent $M{=}600$")):
    m = (d[k + "_E"] > lo) & (d[k + "_E"] < hi); a.plot(d[k + "_E"][m], np.maximum(d[k + "_rho"][m], 1e-9), color=col, ls=ls, lw=1.0, label=lab)
a.set_yscale("log"); a.set_ylim(1e-6, 0.3); a.set_xlim(lo, hi); a.set_xlabel("$E$ (eV)"); a.set_ylabel("DOS per orbital"); a.legend(frameon=False, fontsize=6.5, loc="upper center"); a.text(0.03, 0.04, "(a)", transform=a.transAxes)
a = ax[1]; x = [0.5 * (q["hw_lo"] + q["hw_hi"]) for q in b if q["hw_hi"] <= 39]
a.plot(x, [q["s_median"] for q in b if q["hw_hi"] <= 39], "o-", ms=2.5, lw=0.9, color="#0072B2", label="sound-wave weight (median)")
a.plot(x, [q["s_p10"] for q in b if q["hw_hi"] <= 39], "-", lw=0.7, color="#0072B2", alpha=0.5, label="10th percentile")
a.set_xlabel(r"$\hbar\omega$ (meV)"); a.set_ylabel("sound-wave weight"); a.set_ylim(-0.03, 1.5); a.axvline(7, color="0.5", lw=0.6, ls=":")
a2 = a.twinx(); a2.plot(x, [q["ipr_median"] * 216 for q in b if q["hw_hi"] <= 39], "s-", ms=2.2, lw=0.8, color="#D55E00", label="in-cell IPR $\\times216$"); a2.set_ylim(0.9, 2.1); a2.set_ylabel("in-cell IPR $\\times216$")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper right"); a.text(0.03, 0.04, "(b)", transform=a.transAxes)
fig.tight_layout(); fig.savefig("fig_r10.pdf"); fig.savefig("fig_r10.png", dpi=150)
