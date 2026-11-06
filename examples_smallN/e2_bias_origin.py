"""Where does the E_BS bias of the KPM estimators come from?  Compare (i) the canonical E_BS (E_F found from the noisy IDOS, as in the package) with
(ii) the grand-canonical-at-fixed-E_F band energy  int_{-inf}^{E_F^exact} E rho_est dE  (a LINEAR functional of the moments: must be unbiased if the moment
list is unbiased, whatever the splice does).  M=100, R=4, 500 seeds, Rademacher and spliced K=4,16."""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from common import *
out = []
for name, size in (("ssh", 8 * 0 + 256), ("chain", 256), ("kwon", 2), ("kwon", 3), ("anderson", 8)):
    ref = Ref(name, size); N = ref.N; M = 100; R = 4; NS = 2000; rng = np.random.default_rng(1)
    T = ref.cheb(M); mex = T.mean(1); mr = ref.vec_moments(ref.rademacher(rng, NS * R), M, T).reshape(M, NS, R).mean(2)
    E, rho, idos = reconstruct(mex[None], ref.a, ref.b, 8192); EFx = 0.5 * (ref.ev[ref.nocc - 1] + ref.ev[ref.nocc])
    k = np.searchsorted(E, EFx)
    def fixedEF(A):
        E_, rho_, id_ = reconstruct(A, ref.a, ref.b, 8192)
        w = np.where(E_ < EFx, 1.0, 0.0)
        dE = np.gradient(E_); return (rho_ * E_[None] * (dE * w)[None]).sum(1)   # int E rho dE up to EF_exact (per orbital)
    def canon(A):
        E_, rho_, id_ = reconstruct(A, ref.a, ref.b, 8192)
        return np.array([band_quantities_grid(E_, None, id_[s], 1, ref.nocc / ref.N, 1.0)[1] for s in range(A.shape[0])])
    row = dict(system=name, N=N)
    for lab, A in (("rad", mr), ("K=4", splice(mr, mex, 4)), ("K=16", splice(mr, mex, 16))):
        for fn, f in (("canonical", canon), ("fixed_EF", fixedEF)):
            e = f(A.T.copy()) - f(mex[None])[0]
            row[f"{lab}_{fn}"] = dict(bias=float(e.mean()), se=float(e.std(ddof=1) / np.sqrt(NS)), sd=float(e.std(ddof=1)))
    out.append(row); print(name, N, {k: "b/se=%+.1f b/sd=%+.2f" % (v["bias"] / v["se"], v["bias"] / v["sd"]) for k, v in row.items() if isinstance(v, dict)}, flush=True)
json.dump(out, open("results/e2_bias_origin.json", "w"), indent=1)
