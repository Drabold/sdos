import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = np.load("r10_kwon_curves.npz"); ev = d["ev"]; N = len(ev)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (lo, hi) in zip(ax, ((-14, 8), (-1.2, 2.2))):
    bins = np.linspace(lo, hi, int((hi - lo) / 0.02) + 1); h, b = np.histogram(ev, bins=bins); c = 0.5 * (b[1:] + b[:-1])
    a.fill_between(c, h / N / 0.02, step="mid", color="0.8", label="exact (0.02 eV bins)")
    for k, col in (("kpm600", "#0072B2"), ("maxent600", "#D55E00"), ("maxent300", "#009E73")):
        m = (d[k + "_E"] > lo) & (d[k + "_E"] < hi); a.plot(d[k + "_E"][m], d[k + "_rho"][m], color=col, lw=1.3, label=k.replace("kpm", "KPM M=").replace("maxent", "Maxent M="))
    a.set_xlabel("E (eV)"); a.set_ylabel("DOS per orbital")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-6, 0.3); ax[0].legend(frameon=False, fontsize=8); ax[1].set_title("gap and band tails")
fig.tight_layout(); fig.savefig("fig_r10_kwon_tails.png", dpi=130)
