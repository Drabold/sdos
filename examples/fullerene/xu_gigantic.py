"""Gigantic icosahedral fullerenes with the Xu et al. orthogonal tight-binding Hamiltonian: standard sdos KPM (no overlap matrix, no dense memory).
Relaxed (LCBOP) geometries up to C7260-C8640, ideal sphere-projected geometries beyond (flagged).  Graphene reference from the same Hamiltonian."""
import json, os, resource, sys, time, warnings
import numpy as np, scipy.linalg as sla
import goldberg as G
import sdos
from sdos.interfaces import xu_carbon as X

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
E = np.linspace(-26, 22, 2400)
M, R_VEC = int(os.environ.get('XU_M', 1500)), int(os.environ.get('XU_R', 2))
ETA = float(os.environ.get('XU_ETA', 3.1416 * 21.6 / M))      # graphene broadening = the Jackson resolution pi a / M of the KPM curves (a ~ 21.6 eV)
OUT = os.environ.get('XU_OUT', 'xu_gigantic_results')
EXACT_MAX_ORBITALS = int(os.environ.get('XU_EXACT_MAX', 25000))          # beyond this the LDL^T bisection costs more than the whole DOS (fill grows): use graphene's neutral E_F, flagged


def graphene_xu_dos(E, eta, nk=420, a=1.42 * 3 ** 0.5):
    """Exact graphene DOS per orbital from the Xu H(k) (2 atoms, 8 orbitals), Gaussian eta; returns (dos, E_F) with E_F at charge neutrality."""
    a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * 3 ** 0.5 / 2, 0])
    tau = [np.zeros(3), np.array([a / 2, a / (2 * 3 ** 0.5), 0])]
    terms = []
    for al in range(2):
        for be in range(2):
            for n1 in range(-4, 5):
                for n2 in range(-4, 5):
                    Rv = n1 * a1 + n2 * a2
                    d = tau[be] + Rv - tau[al]
                    r = np.linalg.norm(d)
                    if 1e-9 < r < X.RM:
                        terms.append((al, be, Rv, X.hopping_blocks(d[None])[0]))
    b1 = 2 * np.pi * np.array([1 / a, -1 / (a * 3 ** 0.5), 0]); b2 = 2 * np.pi * np.array([0, 2 / (a * 3 ** 0.5), 0])
    fr = (np.arange(nk) + 0.5) / nk
    K1, K2 = np.meshgrid(fr, fr, indexing="ij")
    ks = (K1.ravel()[:, None] * b1 + K2.ravel()[:, None] * b2)
    Hk = np.zeros((len(ks), 8, 8), complex)
    for al in range(2):
        Hk[:, 4 * al:4 * al + 4, 4 * al:4 * al + 4] += np.diag([X.EPS_S, X.EPS_P, X.EPS_P, X.EPS_P])
    for al, be, Rv, B in terms:
        Hk[:, 4 * al:4 * al + 4, 4 * be:4 * be + 4] += B[None] * np.exp(1j * ks @ Rv)[:, None, None]
    assert np.abs(Hk - np.conj(np.transpose(Hk, (0, 2, 1)))).max() < 1e-10
    ev = np.linalg.eigvalsh(Hk)                                     # (nk^2, 8)
    ef = 0.5 * (ev[:, 3].max() + ev[:, 4].min())
    flat = ev.ravel()
    d = np.zeros_like(E)
    for ch in np.array_split(flat, 400):
        d += np.exp(-0.5 * ((E[:, None] - ch[None]) / eta) ** 2).sum(axis=1)
    return d / (len(flat) * eta * np.sqrt(2 * np.pi)), float(ef)


def geometry(n):
    f = os.path.join(HERE, f"C{60 * n * n}_lcbop.npy")
    if os.path.exists(f):
        return np.load(f), "relaxed LCBOP"
    pos, nb, T = G.fullerene(n, n, bond=1.42)
    return pos, "ideal sphere projection (not relaxed)"


if __name__ == "__main__":
    ns = [int(v) for v in sys.argv[1:]]
    out, info = {}, {}
    dg, efg = graphene_xu_dos(E, ETA)
    out["E"], out["rho_gra"] = E, dg
    info["graphene_EF"] = efg
    print("graphene Xu: E_F =", efg, flush=True)
    for n in ns:
        N = 60 * n * n
        t0 = time.time()
        pos, kind = geometry(n)
        t1 = time.time()
        H = X.xu_hamiltonian(pos)
        t2 = time.time()
        r = sdos.dos(H, M=M, R=R_VEC, seed=1, exact_moments=6, npts=4000, verbose=False)
        t3 = time.time()
        t4 = time.time()
        if 4 * N <= EXACT_MAX_ORBITALS:                                             # exact: bisection on the inertia count, then eigsh near the gap
            hl = sdos.homo_lumo(H, n_occupied=2 * N, tol=1e-6)
            ef = hl["fermi"]
        else:                                                           # too large for the factorization here: use graphene's neutral Fermi level
            hl, ef = None, info["graphene_EF"]
        out[f"E{n}"], out[f"rho{n}"] = r.E - ef, r.rho
        info[n] = dict(atoms=N, orbitals=4 * N, geometry=kind, nnz_per_row=round(H.nnz / H.shape[0], 1), seconds_geometry=round(t1 - t0, 1), seconds_build=round(t2 - t1, 1),
                       seconds_dos=round(t3 - t2, 1), seconds_gap=round(time.time() - t4, 1), gap=(hl or {}).get('gap'), homo=(hl or {}).get('homo'), lumo=(hl or {}).get('lumo'), EF_exact=hl is not None, resolution=float(r.resolution), E_F=float(ef), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2))
        print(n, info[n], flush=True)
        np.savez(OUT + ".npz", **out)
        json.dump(info, open(OUT + ".json", "w"), indent=1)
    print("done", flush=True)
