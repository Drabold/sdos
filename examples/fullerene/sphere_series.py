"""Exact Xu spectra of C_N on the LCBOP-relaxed and on the unrelaxed spherical (Caspar-Klug, projected) geometries for N = 240, 960, 2160 (3840 is in c3840_extra.py)."""
import json, numpy as np, scipy.linalg as sla
import goldberg as G
from sdos.interfaces import xu_carbon as X
res = {}
for n, mk in ((240, 2), (960, 4), (2160, 6)):
    out = {}
    for name, pos in (("sphere", G.fullerene(mk, mk, 1.42)[0]), ("relaxed", np.load(f"C{n}_lcbop.npy"))):
        assert len(pos) == n
        H = X.xu_hamiltonian(pos); ev = sla.eigvalsh(H.toarray(), driver="evd", overwrite_a=True, check_finite=False)
        nocc = len(ev) // 2; out[name] = dict(ev=ev.tolist(), gap=float(ev[nocc] - ev[nocc - 1]), ef=float(0.5 * (ev[nocc] + ev[nocc - 1])))
        c = pos - pos.mean(0); r = np.linalg.norm(c, axis=1); out[name]["R_rms"] = float(r.std())
        print(n, name, out[name]["gap"], out[name]["R_rms"], flush=True)
    res[str(n)] = out
json.dump(res, open("sphere_series.json", "w"))
