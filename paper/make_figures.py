"""Figures of the sdos report from the saved benchmark data (all paths relative to ../examples)."""
import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sdos
warnings.filterwarnings("ignore")
EX = "../examples/"
BL, OR, GR, VM, GY, PK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#555555", "#CC79A7"       # Okabe-Ito
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.1, "figure.dpi": 150, "savefig.bbox": "tight"})
W2 = 7.0

# ---- Fig 1: accuracy
A = json.load(open(EX + "bench/bench_accuracy.json"))
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
M = np.array(A["M"]); kp = A["kpm"]; me = A["maxent"]
ax[0].semilogy(M, [abs(k["dEBS"]) for k in kp], "o-", color=BL, ms=3, label="KPM (Jackson)")
mm = [(m, x["dEBS"]) for m, x in zip(M, me) if x and x["conv"]]
ax[0].semilogy([a for a, b in mm], [abs(b) for a, b in mm], "s-", color=VM, ms=3, label="Maxent, exact moments")
FL = [r for r in json.load(open(EX + "bench/bench_accuracy_floor.json")) if r["s0"] == 1e-4]
ax[0].semilogy([r["M"] for r in FL], [abs(r["dEBS"]) for r in FL], "^-", color=GR, ms=3, label=r"Maxent, $s_0=10^{-4}$")
ax[0].set_xlabel("moments $M$"); ax[0].set_ylabel(r"$|\Delta E_{\rm BS}|$ (meV/atom)"); ax[0].legend(); ax[0].set_title("(a) band energy, exact moments", loc="left", fontsize=8)
ax[1].semilogy(M, [k["L1"] for k in kp], "o-", color=BL, ms=3)
mm = [(m, x["L1"], x["conv"]) for m, x in zip(M, me) if x and x["conv"]]
ax[1].semilogy([a for a, b, c in mm], [b for a, b, c in mm], "s-", color=VM, ms=3)
ax[1].semilogy([r["M"] for r in FL], [r["L1"] for r in FL], "^-", color=GR, ms=3)
ax[1].set_xlabel("moments $M$"); ax[1].set_ylabel(r"$\int|\rho-\rho_{\rm exact}|\,dE$"); ax[1].set_title("(b) pointwise", loc="left", fontsize=8)
R = np.array([r["R"] for r in A["R"]]); sd = np.array([r["std"] for r in A["R"]])
ax[2].loglog(R, sd, "o", color=BL, ms=3.5, label="16 seeds each")
ax[2].loglog(R, sd[0] / np.sqrt(R / R[0]) * 0.9, "-", color=GY, lw=0.8, label=r"$\propto R^{-1/2}$")
ax[2].set_xlabel("random vectors $R$"); ax[2].set_ylabel(r"std of $E_{\rm BS}$ (meV/atom)"); ax[2].legend(); ax[2].set_title("(c) sampling, $M=100$", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_accuracy.pdf"); plt.close(fig)

# ---- Fig 2: scaling
Kw = json.load(open(EX + "bench/bench_kwon.json")); Sk = json.load(open(EX + "bench/bench_skf.json"))
for r_ in json.load(open(EX + "bench/bench_skf_cheb_rerun.json")):
    for q_ in Sk:
        if q_["n"] == r_["n"]: q_["chebyshev"] = r_["chebyshev"]
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
o = [r["orbitals"] for r in Kw]
ax[0].loglog(o, [r["t_build"] + r["t_bounds"] + r["t_moments"] for r in Kw], "o-", color=BL, ms=3, label="Kwon (orthogonal): total")
ax[0].loglog(o, [r["t_moments"] for r in Kw], "o--", color=BL, ms=2.5, alpha=0.6, label="  moments only")
for lab, key, c, mk in (("DFTB, PCG block-Jacobi", "pcg_bj", GR, "s"), ("DFTB, Chebyshev $B^{-1}$", "chebyshev", OR, "^")):
    xs = [r["orbitals"] for r in Sk if key in r and "sec" in r[key]]; ys = [r[key]["sec"] for r in Sk if key in r and "sec" in r[key]]
    ax[0].loglog(xs, ys, mk + "-", color=c, ms=3.5, label=lab)
xx = np.array([4e4, 4e6]); ax[0].loglog(xx, 3.5e-5 * xx * 0.55, ":", color=GY, lw=0.8, label=r"$\propto N$")
ax[0].set_xlabel("orbitals $N$"); ax[0].set_ylabel("wall time (s), $M=100$, $R=2$"); ax[0].set_ylim(top=1e4); ax[0].legend(loc="upper left", fontsize=6.5); ax[0].set_title("(a) time", loc="left", fontsize=8)
ax[1].loglog(o, [r["peak_rss_gb"] for r in Kw], "o-", color=BL, ms=3, label="Kwon")
ax[1].loglog([r["orbitals"] for r in Sk if "pcg_bj" in r], [r["pcg_bj"]["peak_rss_gb"] for r in Sk if "pcg_bj" in r], "^-", color=OR, ms=3.5, label="DFTB ($H$, $S$), PCG")
ax[1].loglog([r["orbitals"] for r in Sk if "chebyshev" in r and "peak_rss_gb" in r["chebyshev"]], [r["chebyshev"]["peak_rss_gb"] for r in Sk if "chebyshev" in r and "peak_rss_gb" in r["chebyshev"]], "s--", color=GR, ms=3, label="DFTB, Chebyshev $B^{-1}$")
ax[1].set_xlabel("orbitals $N$"); ax[1].set_ylabel("peak memory (GB)"); ax[1].legend(); ax[1].set_title("(b) memory", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_scaling.pdf"); plt.close(fig)

# ---- Fig 3: 2,097,152-atom diamond Si
D = {n: json.load(open(EX + f"silicon/si_big_n{n}.json")) for n in (32, 64)}
d = D[64]; ed = np.array(d["exact_hist"]["edges"]); ct = np.array(d["exact_hist"]["counts"]); Ec = 0.5 * (ed[1:] + ed[:-1]); rho_ex = ct / ct.sum() / np.diff(ed)
def smooth(E, eta):
    out = np.zeros_like(E); w = np.diff(ed)
    for i in range(0, len(Ec), 400):
        out += (np.exp(-0.5 * ((E[:, None] - Ec[None, i:i + 400]) / eta) ** 2) * (rho_ex * w)[None, i:i + 400]).sum(1)
    return out / (eta * np.sqrt(2 * np.pi))
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
Eg = np.linspace(-14.5, 7.5, 1500)
ax[0].fill_between(Eg, smooth(Eg, 0.1), color="#dddddd", lw=0, label="exact $k$-sum (0.1 eV)")
ax[0].plot(d["kpm"]["E"], d["kpm"]["rho"], color=BL, lw=0.9, label="KPM, $M=170$")
ax[0].plot(d["maxent"]["E"], d["maxent"]["rho"], color=VM, lw=0.9, label="Maxent, $M=170$")
ax[0].set_xlim(-15, 8); ax[0].set_xlabel("$E$ (eV)"); ax[0].set_ylabel("DOS per orbital"); ax[0].legend(loc="upper left"); ax[0].set_title("(a) 2,097,152 atoms, 8.4M orbitals", loc="left", fontsize=8)
Eh = np.linspace(-1.5, 2.5, 1200)
ax[1].semilogy(Eh, np.maximum(smooth(Eh, 0.05), 1e-6), "--", color=GY, lw=0.8, label="exact (0.05 eV)")
ax[1].semilogy(d["kpm"]["E"], np.maximum(d["kpm"]["rho"], 1e-6), color=BL, label="KPM"); ax[1].semilogy(d["maxent"]["E"], np.maximum(d["maxent"]["rho"], 1e-6), color=VM, label="Maxent")
ax[1].set_xlim(-1.5, 2.5); ax[1].set_ylim(1e-5, 0.3); ax[1].set_xlabel("$E$ (eV)"); ax[1].legend(); ax[1].set_title("(b) gap region", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_si2M.pdf"); plt.close(fig)

# ---- Fig 4: C3840 with 150 moments + Dirac scaling
S = json.load(open(EX + "fullerene/ssc96_C3840.json")); ev = np.array(S["ev"]); ef = S["ef"]
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
lo, hi = S["maxent"]["b"] - S["maxent"]["a"], S["maxent"]["b"] + S["maxent"]["a"]
E = np.linspace(lo, hi, 2500); ex = np.zeros_like(E)
for ch in np.array_split(ev, 60): ex += np.exp(-0.5 * ((E[:, None] - ch[None]) / 0.1) ** 2).sum(1)
ex /= len(ev) * 0.1 * np.sqrt(2 * np.pi)
for a_, xl in ((ax[0], (lo, hi)), (ax[1], (ef - 3, ef + 3))):
    a_.fill_between(E - ef, ex, color="#dddddd", lw=0)
    a_.plot(np.array(S["kpm"]["E"]) - ef, S["kpm"]["rho"], color=BL, lw=0.9); a_.plot(np.array(S["maxent"]["E"]) - ef, S["maxent"]["rho"], color=VM, lw=0.9)
    a_.set_xlim(xl[0] - ef, xl[1] - ef); a_.set_xlabel("$E-E_F$ (eV)")
Xm = np.load(EX + "fullerene/c3840_maxent_exact_s0_1e-4.npz")    # Maxent s0 = 1e-4 from the exact moments (M = 150) of the exact spectrum
for a_ in (ax[0], ax[1]): a_.plot(Xm["E"] - ef, Xm["rho"], color=GR, lw=0.8, ls="--")
ax[0].set_ylabel("DOS per orbital"); ax[0].set_title("(a) C$_{3840}$, $M=150$", loc="left", fontsize=8)
ax[1].set_ylim(0, 0.08); ax[1].set_title("(b) near $E_F$", loc="left", fontsize=8)
N = np.array([60, 240, 540, 960, 1500, 2160, 2940, 3840, 4860, 6000]); gap = np.array([1.537, 1.015, 0.713, 0.549, 0.446, 0.374, 0.323, 0.284, 0.253, 0.228])
hv = 4.196; Req = np.sqrt(N * 2.6196 / (4 * np.pi))     # graphene area per atom 3 sqrt(3)/4 * 1.42^2
ax[2].plot(N, gap * Req / hv, "o-", color=GR, ms=3); ax[2].axhline(2.0, color=GY, lw=0.8, ls="--"); ax[2].text(300, 2.03, "smooth sphere: 2", fontsize=7, color=GY)
ax[2].set_xscale("log"); ax[2].set_ylim(1.2, 2.2); ax[2].set_xlabel("atoms $N_{\\rm at}$"); ax[2].set_ylabel(r"gap$\,R/\hbar v_F$"); ax[2].set_title("(c) Dirac scaling", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_fullerene.pdf"); plt.close(fig)

# ---- Fig 4b: relaxed (LCBOP) against the unrelaxed sphere, exact spectra, C240 ... C3840
Q = json.load(open(EX + "fullerene/c3840_extra.json")); SS = json.load(open(EX + "fullerene/sphere_series.json"))
SS["3840"] = dict(sphere=dict(ev=Q["ev_sphere"], ef=Q["out"]["sphere"]["ef"], gap=Q["out"]["sphere"]["gap"]), relaxed=dict(ev=list(ev), ef=ef, gap=S["gap"]))
def smo(Eq, e, eta=0.1):
    o = np.zeros_like(Eq)
    for ch in np.array_split(np.asarray(e), 60): o += np.exp(-0.5 * ((Eq[:, None] - ch[None]) / eta) ** 2).sum(1)
    return o / (len(e) * eta * np.sqrt(2 * np.pi))
fig, ax = plt.subplots(2, 2, figsize=(W2, 3.8), sharex=True)
Eq = np.linspace(-30, 20, 2500)
for a_, n in zip(ax.ravel(), ("240", "960", "2160", "3840")):
    r = SS[n]
    a_.plot(Eq, smo(Eq + r["sphere"]["ef"], r["sphere"]["ev"]), color=OR, lw=0.8, label="unrelaxed sphere")
    a_.plot(Eq, smo(Eq + r["relaxed"]["ef"], r["relaxed"]["ev"]), color=GY, lw=0.8, label="relaxed (LCBOP)")
    a_.set_title(f"C$_{{{n}}}$: gap {r['sphere']['gap']:.3f} / {r['relaxed']['gap']:.3f} eV", loc="left", fontsize=8); a_.set_xlim(-30, 20)
ax[0, 0].legend(loc="upper left", fontsize=6.5)
for a_ in ax[1]: a_.set_xlabel("$E-E_F$ (eV)")
for a_ in ax[:, 0]: a_.set_ylabel("DOS per orbital")
fig.tight_layout(); fig.savefig("fig_c3840_sphere.pdf"); plt.close(fig)

# ---- Fig 4c: DOS against fullerene size, relaxed LCBOP geometries, KPM (M = 1500), C60 ... C150000, graphene
Jx = json.load(open(EX + "fullerene/xu_smooth.json")); Dx = np.load(EX + "fullerene/xu_smooth.npz")
big = {k: np.load(EX + f"fullerene/xu_smooth_big_{k}.npz") for k in (20, 50)}
sizes = [(1, 60), (2, 240), (4, 960), (8, 3840), (20, 24000), (50, 150000)]
fig, ax = plt.subplots(1, 2, figsize=(W2, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
cols = [BL, GR, OR, VM, PK, "#000000"]
off = 0.0
for (k, n), c in zip(sizes, cols):
    src = Dx if k in (1, 2, 4, 8) else big[k]; Ek, rk = src[f"E{k}"], src[f"rho{k}"]
    ax[0].plot(Ek, rk + off, color=c, lw=0.7); ax[0].text(14.0, off + 0.005, f"C$_{{{n}}}$", fontsize=7, color=c, va="bottom")
    ax[1].plot(Ek, rk, color=c, lw=0.8, label=f"C$_{{{n}}}$"); off += 0.14
ax[1].plot(Dx["E"] - Jx["graphene_EF"], Dx["rho_gra"], "k--", lw=0.7, label="graphene")
ax[0].set_xlim(-26, 19); ax[0].set_ylim(-0.02, 0.9); ax[0].set_xlabel("$E-E_F$ (eV)"); ax[0].set_ylabel("DOS per orbital (offset)"); ax[0].set_title("(a) full band", loc="left", fontsize=8)
ax[1].set_xlim(-4, 4); ax[1].set_ylim(0, 0.1); ax[1].set_xlabel("$E-E_F$ (eV)"); ax[1].legend(fontsize=6.5, ncol=2); ax[1].set_title("(b) near $E_F$", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_fu_series.pdf"); plt.close(fig)

# ---- Fig 5: DTW4096 electrons (Kwon) and phonons (ACE), exact vs sdos
H = None
from sdos.interfaces import kwon_si as K
dd = np.load(EX + "../../lammps-runs/topoMC/voids/host4096_ref.npz") if False else np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz")
Hk = K.kwon_hamiltonian(dd["X"], dd["L"]); evk = np.load(EX + "dtw4096/dtw4096_kwon_ev.npy")
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
rk = sdos.dos(Hk, M=170, R=12, method="kpm", seed=5); rm = sdos.dos(Hk, M=170, R=12, method="maxent", seed=5)
Eg = np.linspace(evk[0] - 1, evk[-1] + 1, 2500); ex = np.zeros_like(Eg)
for ch in np.array_split(evk, 40): ex += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.08) ** 2).sum(1)
ex /= len(evk) * 0.08 * np.sqrt(2 * np.pi)
ax[0].fill_between(Eg, ex, color="#dddddd", lw=0, label="exact (0.08 eV)"); ax[0].plot(rk.E, rk.rho, color=BL, lw=0.9, label="KPM"); ax[0].plot(rm.E, rm.rho, color=VM, lw=0.9, label="Maxent")
ax[0].set_xlabel("$E$ (eV)"); ax[0].set_ylabel("DOS per orbital"); ax[0].legend(loc="upper left"); ax[0].set_title("(a) DTW4096, Kwon TB, $\\Gamma$", loc="left", fontsize=8)
P = json.load(open(EX + "dtw4096/dtw_phonons_ace.json")); lam = np.load(EX + "dtw4096/dtw4096_ace_lambda.npy")
from sdos.thermo import HBAR_OMEGA_METAL
om = np.sqrt(lam[lam > 1e-6]) * HBAR_OMEGA_METAL * 241.7989          # hbar omega (eV) -> THz
og = np.linspace(0, om.max() * 1.02, 1200); g = np.zeros_like(og)
for ch in np.array_split(om, 40): g += np.exp(-0.5 * ((og[:, None] - ch[None]) / 0.25) ** 2).sum(1)
g /= len(lam) * 0.25 * np.sqrt(2 * np.pi)
sc = 241.7989
ax[1].fill_between(og, g, color="#dddddd", lw=0, label="exact (GPU dynmat, 0.25 THz)")
ax[1].plot(np.array(P["kpm"]["omega"]) * HBAR_OMEGA_METAL * sc, np.array(P["kpm"]["g"]) / (HBAR_OMEGA_METAL * sc) * (3 * 4096 - 3) / (3 * 4096) , color=BL, lw=0.9, label="KPM, matrix-free ACE")
ax[1].set_xlabel("frequency (THz)"); ax[1].set_ylabel("VDOS per mode"); ax[1].legend(loc="upper left"); ax[1].set_title("(b) DTW4096, ACE", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_dtw4096.pdf"); plt.close(fig)

# ---- Fig: million-atom amorphous Si (Morrow et al. model), Kwon TB, Gamma
c = np.load(EX + "morrow1M/morrow1M_curves.npz"); ev4 = np.load(EX + "dtw4096/dtw4096_kwon_ev.npy")
Eg = np.linspace(-15, 8, 3000); ref = np.zeros_like(Eg)
for ch in np.array_split(ev4, 40): ref += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.08) ** 2).sum(1)
ref /= len(ev4) * 0.08 * np.sqrt(2 * np.pi)
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
ax[0].fill_between(Eg, ref, color="#dddddd", lw=0, label="DTW4096, exact (0.08 eV)")
ax[0].plot(c["kpm300_E"], c["kpm300_rho"], color=BL, lw=0.8, label="$10^6$ atoms, KPM $M=300$")
ax[0].plot(c["maxent300_E"], c["maxent300_rho"], color=VM, lw=0.8, label="$10^6$ atoms, Maxent $M=300$")
ax[0].set_xlim(-15, 8); ax[0].set_xlabel("$E$ (eV)"); ax[0].set_ylabel("DOS per orbital"); ax[0].legend(loc="upper left"); ax[0].set_title("(a) whole band", loc="left", fontsize=8)
ax[1].semilogy(c["kpm300_E"], np.clip(c["kpm300_rho"], 1e-6, None), color=BL, lw=0.8, label="KPM $M=300$")
for M, col in ((60, PK), (300, VM)):
    ax[1].semilogy(c[f"maxent{M}_E"], np.clip(c[f"maxent{M}_rho"], 1e-6, None), color=col, lw=0.8, label=f"Maxent $M={M}$, $R=2$")
c2 = np.load(EX + "morrow1M/run2/morrow1M_curves.npz")
ax[1].semilogy(c2["maxent600_E"], np.clip(c2["maxent600_rho"], 1e-6, None), color=OR, lw=0.8, label="Maxent $M=600$, $R=4$")
ax[1].semilogy(Eg, np.clip(ref, 1e-6, None), color=GY, lw=0.6, ls="--", label="DTW4096 exact")
ax[1].set_xlim(-1.5, 3); ax[1].set_ylim(1e-5, 0.3); ax[1].set_xlabel("$E$ (eV)"); ax[1].legend(); ax[1].set_title("(b) gap region", loc="left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_morrow1M.pdf"); plt.close(fig)
print("figures written")
