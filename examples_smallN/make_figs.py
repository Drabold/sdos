import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = {"exact": "#000000", "rad R=1": "#0072B2", "splice K=4 R=1": "#D55E00", "splice K=16 R=1": "#CC79A7", "impartial(3) R=1": "#009E73", "rad R=16": "#56B4E9", "splice K=4 R=16": "#E69F00"}
# ---- E1
fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
for ax, (nm, N) in zip(axs.ravel(), (("chain", 64), ("chain", 1024), ("ssh", 256), ("kwon", 256), ("kwon", 2048), ("anderson", 4096))):
    r = [x for x in json.load(open(f"results/e1_{nm}.json")) if x["N"] == N][0]; m = np.array(r["mgrid"]) / r["N"]
    for lab, d in r["lists"].items():
        if "profile_median" in d and lab in C and "R=16" not in lab:
            ax.plot(m, np.array(d["profile_median"]), color=C[lab], lw=1.6, label=lab)
    ax.axhline(0, color="0.5", lw=.6); ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-8); ax.set_title(f"{nm.replace('kwon','Kwon TB')}, N={r['N']}", fontsize=10); ax.set_xlabel("m / N")
for ax in axs[:, 0]: ax.set_ylabel(r"median $\lambda_{\min}(T_m)$")
axs[0, 0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig("fig_e1_cone.pdf"); fig.savefig("fig_e1_cone.png", dpi=130)
# ---- E2 (KPM)
rows = []
for nm in ("chain", "ssh", "kwon", "anderson"):
    for blk in json.load(open(f"results/e2kpm_{nm}.json")): rows += blk["rows"]
cols = {"rad": "#0072B2", "splice K=4": "#D55E00", "splice K=16": "#CC79A7", "imp": "#009E73", "imp+K4": "#E69F00"}
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, nm in zip(axs, ("chain", "ssh", "kwon", "anderson")):
    for e, c in cols.items():
        pts = sorted((r["N"], r["rmse"] / r["W"]) for r in rows if r["system"] == nm and r["M"] == 100 and r["R"] == 4 and r["est"] == e)
        ax.loglog(*zip(*pts), "o-", color=c, ms=3.5, lw=1.4, label=e)
    ax.set_title(nm.replace("kwon","Kwon TB"), fontsize=10); ax.set_xlabel("N (orbitals)")
axs[0].set_ylabel(r"RMSE of $E_{BS}$ per orbital / bandwidth"); axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig("fig_e2_rmse.pdf"); fig.savefig("fig_e2_rmse.png", dpi=130)
