"""Does a larger number of random vectors R help Maxent converge? DTW4096 (Kwon, Gamma), M = 100, 300, stochastic moments, no exact-moment substitution,
default floor 1e-4 and floor 0. Reports Newton steps, converged flag, moment residual and the L1 distance to the exact spectrum broadened by 0.05 eV."""
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from sdos.interfaces import kwon_si as K
from sdos.maxent import maxent_fit, maxent_density
d = np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz"); H = K.kwon_hamiltonian(d["X"], d["L"]); ev = np.load("dtw4096_kwon_ev.npy"); N = H.shape[0]
r0 = sdos.dos(H, M=8, R=1, method="kpm", seed=0, exact_moments=0, verbose=False); print([k for k in dir(r0) if not k.startswith("_")], flush=True)
a, b = r0.a, r0.b
Eg = np.linspace(-14.5, 7.5, 3000); ex = np.zeros_like(Eg)
for ch in np.array_split(ev, 40): ex += np.exp(-0.5 * ((Eg[:, None] - ch[None]) / 0.05) ** 2).sum(1)
ex /= N * 0.05 * np.sqrt(2 * np.pi)
out = []
for M in (100, 300):
    for R in (1, 4, 16, 64):
        t = time.time()
        r = sdos.dos(H, M=M, R=R, method="kpm", seed=5, exact_moments=0, verbose=False)
        mus = np.asarray(getattr(r, "moments_samples", None) if getattr(r, "moments_samples", None) is not None else r.mu_samples) if (hasattr(r, "moments_samples") or hasattr(r, "mu_samples")) else None
        print(M, R, "moments obtained", None if mus is None else mus.shape, f"{time.time()-t:.0f}s", flush=True)
        if mus is None: continue
        for sf in (1e-4, 0.0):
            lam, _, info = maxent_fit(mus, r.a, r.b, default="chebyshev", noise=1.0, sigma_floor=sf, jackknife=False, verbose=False)
            E, rho, _ = maxent_density(lam, r.a, r.b, default="chebyshev", mu=mus.mean(0)); rg = np.interp(Eg, E, rho)
            row = dict(M=M, R=R, floor=sf, converged=bool(info["converged"]), iters=int(info["iterations"]), resid=float(info["max_moment_residual"]), L1=float(np.trapezoid(abs(rg - ex), Eg)))
            out.append(row); print(row, flush=True)
json.dump(out, open("random_vectors_test.json", "w"), indent=1)
