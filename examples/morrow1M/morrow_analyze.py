"""Analysis of morrow1M_result.npz: KPM and Maxent (from saved moments) at several M, E_F, band energy per atom, gap region, comparison with the 4096-atom DTW cell."""
import json, warnings
import numpy as np, sdos
warnings.filterwarnings("ignore")
r = sdos.load("morrow1M_result.npz"); N = r["N"]; nat = N // 4
ev = np.load("../dtw4096/dtw4096_kwon_ev.npy"); ebs_dtw = 2 * ev[:len(ev) // 2].sum() / 4096
out = {}
EF, Ebs, dEF, dEbs = sdos.fermi_and_band_energy(r, 4.0 * nat)
out["kpm300"] = dict(EF=EF, Ebs_per_atom=Ebs / nat, err=dEbs / nat)
print(f"KPM M=300: E_F {EF:.4f}  E_BS/atom {Ebs / nat:.4f} +- {dEbs / nat:.1e} eV   (DTW4096 exact: {ebs_dtw:.4f})")
from sdos.results import _rebuild
res = {"kpm300": r}
for M in (60, 100, 170, 300):
    m = _rebuild(r, M, "maxent", "flat", None, 4000)
    info = m["maxent_info"]
    EF, Ebs, dEF, dEbs = sdos.fermi_and_band_energy(m, 4.0 * nat)
    out[f"maxent{M}"] = dict(conv=bool(info["converged"]), it=int(info["iterations"]), EF=EF, Ebs_per_atom=Ebs / nat)
    print(f"Maxent M={M}: converged {info['converged']} ({info['iterations']} it)  E_F {EF:.4f}  E_BS/atom {Ebs / nat:.4f}")
    res[f"maxent{M}"] = m
np.savez("morrow1M_curves.npz", **{f"{k}_{q}": v[q] for k, v in res.items() for q in ("E", "rho")})
json.dump(out, open("morrow1M_analysis.json", "w"), indent=1)
