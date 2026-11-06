"""Compute for the fullerene figure: generalized KPM DOS of the SIESTA Harris/SZ (H, S) of C60..C2160, graphene reference, S-ignored comparison on C960,
pentagon vs hexagon projected DOS on C2160, and a B^-1 benchmark.  Results -> fullerene_dos_results.npz / .json"""
import json, os, sys, time, warnings
import numpy as np, scipy.linalg as sla, scipy.sparse.linalg as spla
import sisl
import goldberg as G
import graphene_ref as GR
import sdos
from sdos.interfaces.sisl_if import from_sisl
from sdos.generalized import GeneralizedOperator, pcg, _make_precond

warnings.filterwarnings("ignore")
RUN = os.path.expanduser("~/siesta_runs/fullerene")
SIZES = [60, 240, 540, 960, 2160]
M, ETA = 500, 0.10
E = np.linspace(-26, 21, 1500)


def load(n):
    f = f"{RUN}/c60_harris_sz/c60.fdf" if n == 60 else f"{RUN}/c{n}_harris_sz/fuller.fdf"
    return from_sisl(sisl.get_sile(f).read_hamiltonian())


def gauss_dos(ev, E, eta, N):
    d = np.zeros_like(E)
    for ch in np.array_split(ev, max(1, len(ev) // 500)):
        d += np.exp(-0.5 * ((E[:, None] - ch[None]) / eta) ** 2).sum(axis=1)
    return d / (N * eta * np.sqrt(2 * np.pi))


out, info = {}, {}
for n in SIZES:
    H, S = load(n)
    t = time.time()
    r = sdos.dos(H, B=S, M=M, R=3 if n < 2000 else 2, seed=1, binv="pcg", precond="block-jacobi", block_size=4, exact_moments=0, npts=3000)
    out[f"E{n}"], out[f"rho{n}"], out[f"err{n}"] = r.E, r.rho, r.rho_err
    info[n] = dict(orbitals=H.shape[0], seconds=round(time.time() - t, 1), nsolve=r.binv_info["nsolve"], pcg_iterations=r.binv_info["iterations"],
                   iters_per_solve=round(r.binv_info["iterations"] / r.binv_info["nsolve"], 1), resolution=float(r.resolution), window=[float(r.b - r.a), float(r.b + r.a)])
    print(n, info[n], flush=True)

# graphene from the same Hamiltonian
out["Egra"] = E
out["rho_gra"] = GR.graphene_dos(f"{RUN}/graphene", E, eta=ETA, nk=200)
print("graphene done", flush=True)

# C960: exact vs generalized vs S ignored
H, S = load(960)
ev = sla.eigh(H.toarray(), S.toarray(), eigvals_only=True)
out["E960"] = out["E960"]
out["rho960_exact"] = gauss_dos(ev, E, ETA, len(ev))
ri = sdos.dos(H, M=M, R=3, seed=1, exact_moments=0, npts=3000)
out["rho960_Signored"], out["E960_Signored"] = ri.rho, ri.E
sv = sla.eigvalsh(S.toarray())                                   # dense: ARPACK cannot find the smallest eigenvalues of S without a shift
info["S960"] = dict(lam_min=float(sv[0]), lam_max=float(sv[-1]), cond=float(sv[-1] / sv[0]), bandwidth_generalized=float(ev[-1] - ev[0]), bandwidth_S_ignored=float(ri.E[ri.rho > 1e-6].max() - ri.E[ri.rho > 1e-6].min()))
out["ev960"] = ev
print("C960 nonorth done", info["S960"], flush=True)

# B^-1 benchmark on C960
rng = np.random.default_rng(0)
rhs = rng.normal(size=(S.shape[0], 1))
bench = {}
for pre in ("none", "block-jacobi", "ic", "amg"):
    try:
        t = time.time()
        Mp = _make_precond(S, pre, 4)
        _, inf = pcg(S, rhs, Mp, 1e-12, 5000)
        bench[pre] = dict(iterations=inf["iters"], seconds=round(time.time() - t, 2), residual=inf["max_res"])
    except ImportError:
        bench[pre] = None
op = GeneralizedOperator(H, S, binv="chebyshev", block_size=4, tol=1e-12)
bench["chebyshev"] = dict(degree=op.degree, kappa_scaled=op.info["kappa"], poly_err=op.info["poly_err"])
info["binv_benchmark_C960"] = bench
print("benchmark", bench, flush=True)

# pentagon vs hexagon PDOS on C2160
npy = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/C2160_lcbop.npy")
pos, nb, T = G.fullerene(6, 6)
faces = G.ring_atoms(npy, nb)
pent = sorted({a for f in faces if len(f) == 5 for a in f})
hexa = sorted(set(range(len(npy))) - set(pent))
H2, S2 = load(2160)
orb = lambda atoms: np.array([4 * a + k for a in atoms for k in range(4)])
p = sdos.pdos(H2, S2, groups={"pent": orb(pent), "hex": orb(hexa)}, M=300, R=3, mode="separate", seed=2, block_size=4)
no = H2.shape[0]
for g, atoms in (("pent", pent), ("hex", hexa)):
    out[f"pdos_{g}"] = p.rho[g] * no / (4 * len(atoms))                 # per orbital of the group
out["Epdos"] = p.E
info["pdos"] = dict(pentagon_atoms=len(pent), hexagon_atoms=len(hexa))
np.savez("fullerene_dos_results.npz", **out)
json.dump(info, open("fullerene_dos_results.json", "w"), indent=1)
print("saved", flush=True)
