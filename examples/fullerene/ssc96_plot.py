import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sdos
import goldberg as G
from sdos.interfaces import xu_carbon as X
d = json.load(open("ssc96_C3840.json")); ev = np.array(d["ev"]); ef = d["ef"]
# graphene: exact moments of the Xu Hamiltonian, K=150, same window rule as C3840
a = 1.42 * 3 ** 0.5
a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * 3 ** 0.5 / 2, 0]); tau = [np.zeros(3), np.array([a / 2, a / (2 * 3 ** 0.5), 0])]
terms = []
for al in range(2):
    for be in range(2):
        for n1 in range(-4, 5):
            for n2 in range(-4, 5):
                Rv = n1 * a1 + n2 * a2; dd = tau[be] + Rv - tau[al]
                if 1e-9 < np.linalg.norm(dd) < X.RM: terms.append((al, be, Rv, X.hopping_blocks(dd[None])[0]))
b1 = 2 * np.pi * np.array([1 / a, -1 / (a * 3 ** 0.5), 0]); b2 = 2 * np.pi * np.array([0, 2 / (a * 3 ** 0.5), 0])
fr = (np.arange(400) + 0.5) / 400; K1, K2 = np.meshgrid(fr, fr, indexing="ij"); ks = K1.ravel()[:, None] * b1 + K2.ravel()[:, None] * b2
Hk = np.zeros((len(ks), 8, 8), complex)
for al in range(2): Hk[:, 4*al:4*al+4, 4*al:4*al+4] += np.diag([X.EPS_S, X.EPS_P, X.EPS_P, X.EPS_P])
for al, be, Rv, B in terms: Hk[:, 4*be:4*be+4, 4*al:4*al+4] += 0  # placeholder to keep indices explicit
for al, be, Rv, B in terms: Hk[:, 4*al:4*al+4, 4*be:4*be+4] += B[None] * np.exp(1j * ks @ Rv)[:, None, None]
evg = np.linalg.eigvalsh(Hk); efg = 0.5 * (evg[:, 3].max() + evg[:, 4].min()); evg = evg.ravel()
lo, hi = d["maxent"]["b"] - d["maxent"]["a"], d["maxent"]["b"] + d["maxent"]["a"]
A_, B_ = 0.5 * (hi - lo), 0.5 * (hi + lo)
th = np.arccos(np.clip((evg - B_) / A_, -1, 1)); mu = np.array([np.cos(n * th).mean() for n in range(150)])
Eg_, rg, _ = sdos.reconstruct(mu, A_, B_, 3000)
lam, _ = sdos.maxent(mu, default="flat"); Eg_m, rgm, _ = sdos.maxent_density(lam, A_, B_, "flat", 3000, mu=mu)
# exact eigenvalues, Gaussian 0.1 eV
E = np.linspace(lo, hi, 2500); ex = np.zeros_like(E)
for ch in np.array_split(ev, 60): ex += np.exp(-0.5 * ((E[:, None] - ch[None]) / 0.1) ** 2).sum(1)
ex /= len(ev) * 0.1 * np.sqrt(2 * np.pi)
# geometry sensitivity of the gap
pos_i, _, _ = G.fullerene(8, 0)
gaps = {}
for name, pos in (("ideal sphere", pos_i), ("LCBOP", np.load("C3840_lcbop.npy"))):
    r = sdos.homo_lumo(X.xu_hamiltonian(pos), len(pos) * 2)
    gaps[name] = r["gap"]
print("gaps", gaps)
SURF, INK, INK2, GRID, BLUE, ORANGE, GREEN = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834", "#1a9a6a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a_ in ax: a_.grid(True, color=GRID, lw=0.6); a_.set_axisbelow(True)
for a_, (xl, xh), ttl in ((ax[0], (lo, hi), "full band"), (ax[1], (ef - 3, ef + 3), "near $E_F$")):
    a_.fill_between(E - ef, ex, color="#d9d8d2", lw=0, label="exact eigenvalues (0.1 eV Gaussian)")
    a_.plot(np.array(d["kpm"]["E"]) - ef, d["kpm"]["rho"], color=BLUE, lw=1.2, label="C3840 KPM, K=150")
    a_.plot(np.array(d["maxent"]["E"]) - ef, d["maxent"]["rho"], color=ORANGE, lw=1.4, label="C3840 Maxent, K=150")
    a_.plot(Eg_m - efg, rgm, color=GREEN, lw=1.0, ls="--", label="graphene Maxent, K=150")
    a_.set_xlim(xl - ef, xh - ef); a_.set_xlabel("E - E_F (eV)"); a_.set_title(ttl, loc="left", fontsize=10, color=INK2)
ax[0].set_ylabel("DOS per orbital"); ax[0].legend(frameon=False, fontsize=8)
m = (np.abs(E - ef) < 3); ymax = max(np.interp(0, np.array(d["maxent"]["E"]) - ef, d["maxent"]["rho"]), 1e-3)
ax[1].set_ylim(0, 1.6 * max(ex[m].max(), 0.05))
c = ax[2]; c.bar(range(len(gaps)), list(gaps.values()), color=[GRID, BLUE]); c.set_xticks(range(len(gaps))); c.set_xticklabels(list(gaps)); c.axhline(0.10, color=ORANGE, lw=1); c.text(0.02, 0.105, "1995 paper: 0.10 eV", color=ORANGE, fontsize=9)
c.set_ylabel("C3840 HOMO-LUMO gap (eV), Xu H")
fig.suptitle("C3840 from 150 moments (cf. SSC 96, 833, Fig. 2) with the Xu Hamiltonian", x=0.05, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig("ssc96_C3840_repro.png", dpi=150)
