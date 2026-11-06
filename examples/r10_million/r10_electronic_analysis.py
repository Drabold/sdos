"""KPM and Maxent (from the saved moments) vs the exact spectrum of the 17^3 replica of R10 (Kwon TB), whole band and tails.  Cumulative counts N(E)/N need no broadening."""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from sdos.results import _rebuild
f = sys.argv[1] if len(sys.argv) > 1 else "r10_kwon_result_M600_R4_K0.npz"
r = sdos.load(f); ev = np.load("kwon_r10_exact_ev_17.npy"); N = len(ev); nocc = N // 2
print("levels", N, "VBM", ev[nocc - 1], "CBM", ev[nocc], "gap", ev[nocc] - ev[nocc - 1])
res = {"kpm600": r}
for M in (100, 300, 600):
    m = _rebuild(r, M, "maxent", "flat", None, 8000); res[f"maxent{M}"] = m; print("Maxent", M, m["maxent_info"]["converged"], m["maxent_info"]["iterations"])
res["kpm300"] = _rebuild(r, 300, "kpm", "flat", None, 8000)
out = {}
for name, d in res.items():
    E, idos = d["E"], d["idos"]; exN = np.searchsorted(ev, E, side="right") / N
    sel = (E > ev[0] - 0.5) & (E < ev[-1] + 0.5)
    out[name] = dict(max_abs_Ncount=float(np.abs(idos[sel] - exN[sel]).max() * N),
                     tail_valence=float(np.abs(idos - exN)[(E > ev[nocc - 1] - 1.0) & (E < ev[nocc - 1] + 0.3)].max() * N),
                     tail_conduction=float(np.abs(idos - exN)[(E > ev[nocc] - 0.3) & (E < ev[nocc] + 1.0)].max() * N))
    EF, Ebs, _, _ = sdos.fermi_and_band_energy(d, float(N / 1) * 1.0, spin=1.0) if False else (None, None, None, None)
    print(name, {k: round(v, 1) for k, v in out[name].items()}, "(counts of states, out of", N, ")")
np.savez("r10_kwon_curves.npz", **{f"{k}_{q}": v[q] for k, v in res.items() for q in ("E", "rho", "idos")}, ev=ev)
json.dump(out, open("r10_kwon_analysis.json", "w"), indent=1)
