"""KPM (Jackson) vs maximum entropy from IDENTICAL exact moments of graphene's Xu-Hamiltonian DOS (a smooth DOS with van Hove singularities and sharp band edges)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sdos
from sdos.interfaces import xu_carbon as X

a = 1.42 * 3 ** 0.5
a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * 3 ** 0.5 / 2, 0])
tau = [np.zeros(3), np.array([a / 2, a / (2 * 3 ** 0.5), 0])]
terms = []
for al in range(2):
    for be in range(2):
        for n1 in range(-4, 5):
            for n2 in range(-4, 5):
                Rv = n1 * a1 + n2 * a2
                d = tau[be] + Rv - tau[al]
                if 1e-9 < np.linalg.norm(d) < X.RM:
                    terms.append((al, be, Rv, X.hopping_blocks(d[None])[0]))
b1 = 2 * np.pi * np.array([1 / a, -1 / (a * 3 ** 0.5), 0]); b2 = 2 * np.pi * np.array([0, 2 / (a * 3 ** 0.5), 0])
nk = 500
fr = (np.arange(nk) + 0.5) / nk
K1, K2 = np.meshgrid(fr, fr, indexing="ij")
ks = K1.ravel()[:, None] * b1 + K2.ravel()[:, None] * b2
Hk = np.zeros((len(ks), 8, 8), complex)
for al in range(2):
    Hk[:, 4 * al:4 * al + 4, 4 * al:4 * al + 4] += np.diag([X.EPS_S, X.EPS_P, X.EPS_P, X.EPS_P])
for al, be, Rv, B in terms:
    Hk[:, 4 * al:4 * al + 4, 4 * be:4 * be + 4] += B[None] * np.exp(1j * ks @ Rv)[:, None, None]
ev = np.linalg.eigvalsh(Hk).ravel()
ef = 0.5 * (np.sort(np.linalg.eigvalsh(Hk)[:, 3]).max() + np.linalg.eigvalsh(Hk)[:, 4].min())
lo, hi = ev.min() - 0.2, ev.max() + 0.2
A, Bc = 0.5 * (hi - lo), 0.5 * (hi + lo)
Eg = np.linspace(lo + 0.3, hi - 0.3, 1800)
eta = 0.04
ref = np.zeros_like(Eg)
for ch in np.array_split(ev, 300):
    ref += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / eta) ** 2).sum(1)
ref /= len(ev) * eta * np.sqrt(2 * np.pi)
Kmax = 400
th = np.arccos(np.clip((ev - Bc) / A, -1, 1))
mu = np.array([np.cos(n * th).mean() for n in range(Kmax)])
Ks = (20, 40, 60, 100, 150, 200, 300)
err = {"kpm": [], "me": []}
rec = {}
for K in Ks:
    Ek, rk, _ = sdos.reconstruct(mu[:K], A, Bc, 3000)
    lam, info = sdos.maxent(mu[:K])
    Em, rm, _ = sdos.maxent_density(lam, A, Bc, "chebyshev", 4000)
    for name, E_, r_ in (("kpm", Ek, rk), ("me", Em, rm)):
        d = np.interp(Eg, E_, r_) - ref
        err[name].append(np.trapezoid(np.abs(d), Eg))
        rec[(name, K)] = (E_, r_)
SURF, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
for a_ in ax:
    a_.grid(True, color=GRID, lw=0.6); a_.set_axisbelow(True)
for a_, (x0, x1, y1, ttl) in zip(ax[:2], ((ef - 4, ef + 6, 0.075, "van Hove peaks and the Dirac dip: KPM broadens, Maxent sharpens"), (ev.min() - 0.4, ev.min() + 6, 0.06, "band edge: KPM smears, Maxent ripples"))):
    a_.fill_between(Eg - ef, ref, color="#d9d8d2", lw=0, label="exact (4 meV-resolution k-mesh)")
    a_.plot(rec[("kpm", 100)][0] - ef, rec[("kpm", 100)][1], color=BLUE, lw=1.3, label="KPM Jackson, K = 100")
    a_.plot(rec[("me", 100)][0] - ef, rec[("me", 100)][1], color=ORANGE, lw=1.3, label="Maxent, K = 100")
    a_.set_xlim(x0 - ef if "edge" in ttl else x0 - ef, x1 - ef); a_.set_ylim(0, y1)
    a_.set_xlabel("E - E$_F$ (eV)"); a_.set_ylabel("DOS (states / orbital / eV)"); a_.set_title(ttl, loc="left", fontsize=11, color=INK)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
ax[0].set_xlim(-4, 6); ax[1].set_xlim(ev.min() - ef - 0.4, ev.min() - ef + 6)
c = ax[2]
c.loglog(Ks, err["kpm"], "o-", color=BLUE, lw=1.4, ms=6, mec=SURF, mew=1.5, label="KPM Jackson")
c.loglog(Ks, err["me"], "o-", color=ORANGE, lw=1.4, ms=6, mec=SURF, mew=1.5, label="Maxent")
c.set_xlabel("number of moments K"); c.set_ylabel(r"$\int |\rho - \rho_{exact}|\, dE$"); c.legend(frameon=False, fontsize=9)
c.set_title("L1 error: Maxent smaller at every K", loc="left", fontsize=11, color=INK)
fig.suptitle("Graphene (Xu H), identical exact moments: Jackson-KPM is smooth but broad; Maxent is sharper (lower error) but ripples at hard edges", x=0.05, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("maxent_vs_kpm.png", dpi=140)
print({K: (round(e1, 4), round(e2, 4), round(e1 / e2, 2)) for K, e1, e2 in zip(Ks, err["kpm"], err["me"])})
