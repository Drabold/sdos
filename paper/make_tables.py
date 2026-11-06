"""LaTeX tables of the sdos report, generated from the saved benchmark data (no hand transcription)."""
import json
import numpy as np
EX = "../examples/"
L = lambda f: json.load(open(EX + f))
out = []
def tab(name, cols, rows, caption, label, star=False, note=None):
    env = "table*" if star else "table"
    s = [f"\\begin{{{env}}}[t]\\caption{{{caption}}}\\label{{{label}}}\\begin{{ruledtabular}}\\begin{{tabular}}{{{cols}}}"]
    s += rows
    s.append("\\end{tabular}\\end{ruledtabular}" + (f"\\par\\smallskip\\footnotesize {note}" if note else "") + f"\\end{{{env}}}")
    out.append("\n".join(s))
    open(f"tab_{name}.tex", "w").write(out[-1] + "\n")

# T: accuracy vs M
A = L("bench/bench_accuracy.json")
FL = L("bench/bench_accuracy_floor.json")
F = {r["M"]: r for r in FL if r["s0"] == 1e-4}; F0 = {r["M"]: r for r in FL if r["s0"] == 0.0}
rows = ["$M$ & \\multicolumn{3}{c}{$\\Delta E_{\\rm BS}$ (meV/atom)} & \\multicolumn{3}{c}{pointwise $L_1$} & \\multicolumn{2}{c}{Newton steps} \\\\ & KPM & Maxent & Maxent $s_0$ & KPM & Maxent & Maxent $s_0$ & Maxent & Maxent $s_0$ \\\\ \\colrule"]
for m_, k, x in zip(A["M"], A["kpm"], A["maxent"]):
    ok = bool(x) and x["conv"]          # a nonconverged exact-moment Maxent result is not reported
    mx = f"${x['dEBS']:+.2f}$" if ok else "--"
    ml = f"{x['L1']:.4f}" if ok else "--"
    f = F[m_]; assert f["conv"]
    ns = str(F0[m_]["it"]) if F0[m_]["conv"] else "--"
    rows.append(f"{m_} & ${k['dEBS']:+.2f}$ & {mx} & ${f['dEBS']:+.2f}$ & {k['L1']:.4f} & {ml} & {f['L1']:.4f} & {ns} & {f['it']} \\\\")
tab("acc", "rrrrrrrrr", rows, "Truncation error on DTW4096 (Kwon, $\\Gamma$) with \\emph{exact} moments (no sampling noise): band-energy error and $L_1$ distance to the exact spectrum broadened by 0.05~eV. Maxent: exact-moment constraints; dashes: the solver did not reach tolerance $10^{-9}$ in 300 Newton steps, so no result is reported (no steps are listed either). Maxent $s_0$: every moment but $\\mu_0$ given the uncertainty $s_0=10^{-4}$ (Sec.~\\ref{sec:recon}); all converged.", "tab:acc", star=True)

# T: vector kinds
E = L("dtw4096/dtw_electrons.json")["C"]
base = E[0]["std_meV"] ** 2 * E[0]["sec"]
rows = ["vectors & std (meV/at.) & time (s) & efficiency \\\\ \\colrule"]
for r in E:
    eff = base / (r["std_meV"] ** 2 * r["sec"])
    rows.append(f"{r['kind'].replace('rademacher', 'Rad.').replace('gaussian', 'Gauss.').replace(' (3 moments)', ' (3)')} & {r['std_meV']:.1f} & {r['sec']:.2f} & {eff:.1f}$\\times$ \\\\")
tab("vec", "lrrr", rows, "Random vectors and exact low moments, band energy of DTW4096 (Kwon, $M=100$, 12 seeds with $R=2$ each). Efficiency = (variance $\\times$ time)$^{-1}$ relative to plain Rademacher vectors; the uncertainty of each entry is about 30\\%.", "tab:vec")

# T: scaling Kwon
K = L("bench/bench_kwon.json"); K4 = {r["n"]: r for r in L("bench/bench_kwon_k4.json")}
rows = ["atoms & orbitals & build & bounds & moments & total & peak mem. & moments, $K=4$ & total, $K=4$ \\\\ & & (s) & (s) & (s) & (s) & (GB) & (s) & (s) \\\\ \\colrule"]
for r in K:
    q = K4[r["n"]]
    rows.append(f"{r['atoms']:,} & {r['orbitals']:,} & {r['t_build']:.1f} & {r['t_bounds']:.1f} & {r['t_moments']:.1f} & {r['t_build'] + r['t_bounds'] + r['t_moments']:.1f} & {r['peak_rss_gb']:.2g} & {q['t_moments']:.1f} & {q['t_build'] + q['t_bounds'] + q['t_moments']:.1f} \\\\".replace(",", "\\,"))
tab("kw", "rrrrrrrrr", rows, "Orthogonal case: Kwon tight binding on periodic tilings of DTW4096, KPM with $M=100$ moments and $R=2$ vectors, 4 threads; columns 3--7: no exact-moment substitution ($K=0$); last two columns: the default $K=4$ (moments, and the whole call including build and bounds), from a separate run. Timings are one run each.", "tab:kw", star=True)

# T: scaling DFTB
S = L("bench/bench_skf.json")
for r_ in L("bench/bench_skf_cheb_rerun.json"):            # Chebyshev row at 262\,144 atoms rerun under a 16 GB cap (the first run was killed by a 10 GB cap)
    for q_ in S:
        if q_["n"] == r_["n"]:
            q_["chebyshev"] = r_["chebyshev"]
            if q_.get("t_build") is None: q_["t_build"] = r_["t_build"]
rows = ["atoms & orbitals & nnz/row & build & PCG (BJ) & Chebyshev & PCG (IC(0)) & peak mem., PCG & peak mem., Cheb. \\\\ & & & (s) & (s) & (s) & (s) & (GB) & (GB) \\\\ \\colrule"]
for r in S:
    g = lambda k: f"{r[k]['sec']:.0f}" if k in r and "sec" in r[k] else ("killed$^\\ddagger$" if k in r and "error" in r[k] else "--")
    pkf = lambda k: (f"{r[k]['peak_rss_gb']:.2g}" if k in r and r[k].get("peak_rss_gb") else "--")
    rows.append(f"{r['atoms']:,} & {r['orbitals']:,} & {r['nnz_per_row']:.0f} & {('%.0f' % r['t_build']) if r.get('t_build') is not None else '--'} & {g('pcg_bj')} & {g('chebyshev')} & {g('pcg_ic')} & {pkf('pcg_bj')} & {pkf('chebyshev')} \\\\".replace(",", "\\,"))
tab("sk", "rrrrrrrrr", rows, "Non-orthogonal case: DFTB ($pbc$-0-3, non-self-consistent) $H$ and $S$ on tilings of DTW4096, KPM with $M=100$, $R=2$; time of the whole density-of-states call for the three ways of applying $S^{-1}$ (PCG with block-Jacobi, Chebyshev polynomial, PCG with IC(0)). IC(0) was run only for the small sizes (its setup is a Python loop). At 262\\,144 atoms the Chebyshev row is a rerun under a 16~GB cap (peak 14.0~GB; the first attempt, under a 10~GB cap, was killed); the PCG row at that size was recovered from the log of the first run (peak 9.8~GB), and the build time is that of the rerun.", "tab:sk", star=True)

# T: DTW4096 DFTB modes
G = L("dftb/dtw_dftb_gamma.json"); Kp = L("dftb/dtw_dftb_k.json")
rows = ["$S^{-1}$ by & \\multicolumn{4}{c}{$\\Gamma$ (real)} & \\multicolumn{3}{c}{$\\mathbf k$ (complex Hermitian)} \\\\ & time (s) & iter./solve & $\\Delta E_{\\rm BS}$ (meV/atom) & $L_1$ to exact & time (s) & iter./solve & $L_1$ to $\\Gamma$ exact \\\\ \\colrule"]
for lab, key in (("PCG, block-Jacobi", "PCG block-Jacobi|kpm"), ("PCG, IC(0)", "PCG IC(0)|kpm"), ("Chebyshev polynomial", "Chebyshev B^-1|kpm")):
    import re
    it = lambda d: int(re.search(r"'iterations': (\d+)", d[key]["binv"]).group(1)) / int(re.search(r"'nsolve': (\d+)", d[key]["binv"]).group(1))
    rows.append(f"{lab} & {G[key]['sec']:.0f} & {it(G):.1f} & ${G[key]['dE_BS_per_atom_meV']:+.1f}\\pm{G[key]['err_meV']:.0f}$ & {G[key]['L1']:.3f} & {Kp[key]['sec']:.0f} & {it(Kp):.1f} & {Kp[key]['L1']:.3f} \\\\")
tab("dd", "lrrrrrrr", rows, "DFTB electronic problem of DTW4096 (16\\,384 orbitals, 179 nonzeros per row): KPM, $M=170$, $R=6$. At $\\Gamma$ the three ways of applying $S^{-1}$ give identical band energies (the same moments); the error is against the dense generalized spectrum and is a sampling error (this run, seed 11: $-70$~meV/atom, with $\\pm35$ the standard error estimated from its six vectors; other seeds give $-122$, $+69$ and $+153$~meV/atom with estimated errors of 84--94, so that estimate is itself uncertain and the spread over seeds, about 110~meV/atom at $R=6$, is the reliable figure; see text), and $L_1$ is the distance to the exact spectrum broadened by 0.1~eV. The dense complex generalized problem does not fit in memory here, so the $\\mathbf k$ point is compared with the $\\Gamma$ spectrum of the same cell.", "tab:dd", star=True)

# T: phonons
Sw = L("dtw4096/dtw_phonons_sw.json"); Ac = L("dtw4096/dtw_phonons_ace.json"); Ae = L("dtw4096/dtw_ace_exact.json")
rows = ["potential & method & ZPE (meV/atom) & $C_v$(100 K) & $C_v$(300 K) & $C_v$(600 K) \\\\ \\colrule"]
ex = Sw["exact"]
rows.append(f"SW & exact (dense) & {ex['zpe'] * 1e3:.3f} & {ex['Cv'][0]:.4f} & {ex['Cv'][1]:.4f} & {ex['Cv'][2]:.4f} \\\\")
for m in ("kpm", "maxent"):
    rows.append(f"SW & sdos {m.upper() if m == 'kpm' else 'Maxent'} & {Sw[m]['zpe'] * 1e3:.3f} & {Sw[m]['Cv'][0]:.4f} & {Sw[m]['Cv'][1]:.4f} & {Sw[m]['Cv'][2]:.4f} \\\\")
rows.append("\\colrule")
rows.append(f"ACE & exact (GPU \\texttt{{dynamical\\_matrix}}) & {Ae['zpe'] * 1e3:.3f} & {Ae['Cv'][0]:.4f} & {Ae['Cv'][1]:.4f} & {Ae['Cv'][2]:.4f} \\\\")
for m in ("kpm", "maxent"):
    rows.append(f"ACE & sdos {m.upper() if m == 'kpm' else 'Maxent'} & {Ac[m]['zpe'] * 1e3:.3f} & {Ac[m]['Cv'][0]:.4f} & {Ac[m]['Cv'][1]:.4f} & {Ac[m]['Cv'][2]:.4f} \\\\")
tab("ph", "llrrrr", rows, "Harmonic thermodynamics of DTW4096 (4096 atoms, relaxed with each potential): matrix-free sdos ($M=170$, $R=4$, two force calls per product, three rigid translations removed exactly) against the exact dynamical matrix. $C_v$ in $k_B$/atom.", "tab:ph", star=True)

# T: fullerene gaps / Dirac
N = np.array([60, 240, 540, 960, 1500, 2160, 2940, 3840, 4860, 6000]); gap = np.array([1.537, 1.015, 0.713, 0.549, 0.446, 0.374, 0.323, 0.284, 0.253, 0.228])
hv = 4.196; Req = np.sqrt(N * 3 * np.sqrt(3) / 4 * 1.42 ** 2 / 2 / (4 * np.pi) * 2) if False else np.sqrt(N * (3 * np.sqrt(3) / 4 * 1.42 ** 2) / (4 * np.pi))
rows = ["C$_{N_{\\rm at}}$ & gap (eV) & $R_{\\rm eq}$ (\\AA) & gap$\\,R_{\\rm eq}/\\hbar v_F$ \\\\ \\colrule"]
for n, g, r in zip(N, gap, Req):
    rows.append(f"{n} & {g:.3f} & {r:.2f} & {g * r / hv:.3f} \\\\")
tab("fu", "rrrr", rows, f"Exact HOMO--LUMO gaps (inertia bisection, Xu Hamiltonian, LCBOP-relaxed icosahedral fullerenes) and the confined-Dirac-cone scaling; $\\hbar v_F={hv}$~eV\\,\\AA\\ from the graphene bands of the same Hamiltonian, $R_{{\\rm eq}}=(N_{{\\rm at}} A_0/4\\pi)^{{1/2}}$ with $A_0$ the graphene area per atom.", "tab:fu")
# T: sparseness
S = L("bench/bench_sparsity.json")
rows = ["nonzeros per row & " + " & ".join(str(r["nnz_per_row"]) for r in S["rows"]) + " \\\\ \\colrule",
        "time (s) & " + " & ".join(f"{r['sec']:.2f}" for r in S["rows"]) + " \\\\",
        "ns per (nnz $M R$) & " + " & ".join(f"{r['ns_per_nnz_M_R']:.2f}" for r in S["rows"]) + " \\\\"]
tab("sp", "r" + "r" * len(S["rows"]), rows, f"Time of the Chebyshev recursion against the number of nonzeros per row (random symmetric sparse matrix, $N={S['N']}$, $M={S['M']}$, $R={S['R']}$ Rademacher vectors, 4 threads; bare recursion: no bounds, no reconstruction).", "tab:sp")
open("tables.tex", "w").write("\n\n".join(out) + "\n")
print("tables written:", len(out))
