"""Cubic Anderson model (14^3 = 2744 sites, W = 6): KPM (Jackson, Lorentz) and Maxent vs exact diagonalization."""
import sys, time
import numpy as np, scipy.linalg as sla, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "../tests"); from conftest import cubic
import sdos
H = cubic(14, W=6.0, seed=0); n = H.shape[0]
t = time.time(); ev = sla.eigvalsh(H.toarray()); print(f"exact diagonalization: {time.time()-t:.1f} s")
b = (ev[0], ev[-1]); kw = dict(bounds=b, R=10, seed=1, verbose=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
hist, edges = np.histogram(ev, bins=120, density=True); c = 0.5 * (edges[1:] + edges[:-1])
for a_ in ax: a_.step(c, hist, where="mid", color="0.7", lw=1, label="exact (histogram)")
for lab, kk in (("KPM Jackson, M=128", dict(M=128)), ("KPM Lorentz, M=128", dict(M=128, kernel="lorentz")), ("Maxent, M=40", dict(M=40, method="maxent"))):
    t = time.time(); r = sdos.estimate_dos(H, **kw, **kk); print(f"{lab}: {time.time()-t:.2f} s")
    ax[0].plot(r.E, r.rho, lw=1.2, label=lab)
    if kk.get("method") != "maxent": ax[0].fill_between(r.E, r.rho - 2 * r.rho_err, r.rho + 2 * r.rho_err, alpha=0.25, lw=0)
    ax[1].plot(r.E, r.rho, lw=1.2)
ax[1].set_xlim(b[0] * 0.98, b[0] * 0.55); ax[1].set_ylim(0, 0.05); ax[1].set_title("band edge (tail)")
ax[0].set_ylim(0, 0.25); ax[0].set_title(f"cubic Anderson, N={n}, W=6: 10 random vectors"); ax[0].legend(fontsize=8); ax[0].set_xlabel("E"); ax[1].set_xlabel("E"); ax[0].set_ylabel("DOS per site")
fig.tight_layout(); fig.savefig("anderson_demo.png", dpi=140)
