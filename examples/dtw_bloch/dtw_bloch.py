"""B1 test: is the sharp drop of the sound-wave weight at ~7 meV in the R10 replica a zone-folding artifact of the 216-atom cell?
Bloch dynamical matrices D(k) of DTW4096 (4096 atoms, ACE, cell 43.6 x 43.7 x 43.9 A; zone face pi/L = 0.072 A^-1) from the Gamma-point matrix written by LAMMPS' dynamical_matrix
(minimum-image pair phases: valid because the cell is much larger than the ACE cutoff), lowest modes by dense eigh, and for every mode the weight on plane waves e^{i(k+G).r} eps
(summed over the three polarizations): at G = 0 (the 'same k' definition used for R10) and the best G.   usage: python dtw_bloch.py NK [seed]  -> dtw_bloch_nk{NK}.npz"""
import sys, time, numpy as np, scipy.linalg as sla
NK = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
HW = 64.654                                       # meV / sqrt(lambda)
n = 4096; Dg = np.fromfile("/home/drabold/lammps-runs/topoMC/sdos_dtw/dtw4096_ace_dynmat.dat", dtype=np.float64).reshape(3 * n, 3 * n); Dg = 0.5 * (Dg + Dg.T)
txt = open("/home/drabold/lammps-runs/topoMC/sdos_dtw/dtw4096_ace_relaxed.data").read(); Lb = [list(map(float, l.split()[:2])) for l in txt.split("\n") if l.endswith(("xlo xhi", "ylo yhi", "zlo zhi"))]
L = np.array([b[1] - b[0] for b in Lb]); body = txt.split("Atoms # atomic")[1].strip().split("\n")
rows = sorted([l.split() for l in body if len(l.split()) >= 5 and l.split()[0].isdigit()], key=lambda r: int(r[0])); X = np.array([[float(v) for v in r[2:5]] for r in rows])[:n]; assert len(X) == n
d = X[None, :, :] - X[:, None, :]; Rimg = -L * np.round(d / L)                                   # R[i, j] = -L round((x_j - x_i)/L)
mask = (np.abs(Dg.reshape(n, 3, n, 3)).max(axis=(1, 3)) > 0)                                    # atom pairs with a nonzero block
print("cell", L, "pairs with nonzero block per atom", mask.sum(1).mean(), flush=True)
rng = np.random.default_rng(seed); kf = np.pi / L
ks = [np.zeros(3)] + [rng.uniform(-1, 1, 3) * kf for _ in range(NK - 1)]
G = np.array([[a, b, c] for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)]) * (2 * np.pi / L)
out = dict(k=[], lam=[], w0=[], wbest=[], Gbest=[])
for k in ks:
    t = time.time(); ph = np.where(mask, np.exp(1j * np.einsum("ijc,c->ij", Rimg, k)), 0)
    Dk = (Dg.reshape(n, 3, n, 3) * ph[:, None, :, None]).reshape(3 * n, 3 * n); Dk = 0.5 * (Dk + Dk.conj().T)
    lam, V = sla.eigh(Dk, subset_by_index=[0, 560], driver="evr", overwrite_a=True); del Dk
    sel = lam < (16.0 / HW) ** 2; lam = lam[sel]; V = V[:, sel]; print("k", np.round(k, 4), "modes <16 meV:", len(lam), f"{time.time() - t:.0f} s", flush=True)
    for j, lm in enumerate(lam):
        psi = V[:, j].reshape(n, 3); wq = []
        for q in (k + G):
            e = np.exp(-1j * (X @ q)); a = e @ psi / np.sqrt(n); wq.append((np.abs(a) ** 2).sum())
        wq = np.array(wq); i0 = np.argmin(np.abs(G).sum(1)); ib = int(np.argmax(wq))
        out["k"].append(k); out["lam"].append(lm); out["w0"].append(wq[i0]); out["wbest"].append(wq[ib]); out["Gbest"].append(G[ib])
    np.savez(f"dtw_bloch_nk{NK}.npz", **{kk: np.array(v) for kk, v in out.items()}, L=L)
print("DONE", flush=True)
