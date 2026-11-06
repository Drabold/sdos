"""Low-frequency vibrational modes of the R10 replica (ACE).
A) n = 3 replica (17496 dof): shift-invert (sdos.window_eigs) on the explicit Hessian assembled from the force constants, against the exact Bloch eigenvalues.
   (At n = 6 the sparse factorization does not fit: ~900 nonzeros per row, 300 neighbour atoms.)
B) Character of every mode below lam_max for the full 17^3 replica from the exact Bloch eigenvectors of D(k) (648x648): IPR within the 216-atom cell (weights summed over the
   three components; uniform = 1/216) and the sound-wave weight s = sum over the 3 polarisations of |<plane wave e^{ik.r_i} eps/sqrt(216) | v>|^2, v = sqrt(m) u (s -> 1 for a long-wavelength acoustic wave).
usage: python r10_lowmodes.py lam_max"""
import sys, time, json, numpy as np, scipy.sparse as sp, sdos
lmax = float(sys.argv[1]); m = 28.0855
Phi = np.load("ifc_r10.npy").reshape(216, 3, 27, 216, 3); cells = [(c % 3, (c // 3) % 3, c // 9) for c in range(27)]; Rv = np.array([[(v if v < 2 else -1) for v in cc] for cc in cells])
from common_r10 import r10_positions, L0
x0 = r10_positions()
def hessian(n):
    gx, gy, gz = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"); C = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    cid = lambda c: ((c[:, 0] % n) * n + (c[:, 1] % n)) * n + (c[:, 2] % n)
    ii, aa, jj, bb = np.meshgrid(np.arange(216), np.arange(3), np.arange(216), np.arange(3), indexing="ij"); rows = []; cols = []; vals = []
    for c in range(27):
        ph = Phi[:, :, c, :, :]; msk = np.abs(ph) > 1e-6; tgt = cid(C + Rv[c])
        for ci in range(len(C)):
            rows.append(3 * (216 * ci + ii[msk]) + aa[msk]); cols.append(3 * (216 * tgt[ci] + jj[msk]) + bb[msk]); vals.append(ph[msk] / m)
    D = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(648 * n ** 3,) * 2); return 0.5 * (D + D.T), C
out = {}
# ---- A
n = 3; t = time.time(); D, C = hessian(n); lam3 = np.load("vib_r10_exact_lambda_3.npy"); sel = (lam3 > 1e-6) & (lam3 < lmax); nex = int(sel.sum())
print(f"[A] n=3: D {D.shape} nnz {D.nnz} ({time.time() - t:.0f} s); exact states in (0, {lmax}): {nex}", flush=True)
t = time.time(); W = sdos.window_eigs(D, window=(1e-6, lmax), method="shift-invert", k=nex + 10, block_size=3); E = np.sort(np.asarray(W.E))
dE = np.abs(E - np.sort(lam3[sel])[:len(E)]).max() if len(E) == nex else None
print(f"[A] shift-invert found {len(E)} states in {time.time() - t:.0f} s; max residual {float(np.max(W.residual)):.1e}; max |E - exact| {dE}", flush=True)
V = np.asarray(W.V); w = (np.abs(V) ** 2).reshape(len(C), 216, 3, -1).sum(2); wi = w.sum(0); wi /= wi.sum(0)[None]
out["A"] = dict(n_found=len(E), n_exact=nex, max_resid=float(np.max(W.residual)), max_dE=None if dE is None else float(dE), ipr_cell=((wi ** 2).sum(0)).tolist(), E=E.tolist())
# ---- B
n = 17; t = time.time(); rec = []
Dk0 = lambda k: np.einsum("c,iacjb->iajb", np.exp(1j * (Rv @ k)), Phi.reshape(216, 3, 27, 216, 3)).reshape(648, 648) / m
for p in np.ndindex(n, n, n):
    kk = 2 * np.pi * np.array(p) / n; kq = np.where(np.array(p) > n // 2, np.array(p) - n, np.array(p))      # minimum-image wavevector index
    D1 = Dk0(kk); D1 = 0.5 * (D1 + D1.conj().T); w, U = np.linalg.eigh(D1)
    idx = np.nonzero((w > 1e-6) & (w < lmax))[0]
    if len(idx) == 0: continue
    kcart = 2 * np.pi * kq / (n * L0)                                  # physical wavevector (1/A); phase e^{i k.r_i} on the home-cell positions
    ph = np.exp(1j * (x0 @ kcart)) / np.sqrt(216.0)
    for j in idx:
        v = U[:, j].reshape(216, 3); wt = (np.abs(v) ** 2).sum(1); ipr = float((wt ** 2).sum())
        s = float(sum(abs(np.vdot(ph * (np.arange(3) == e)[None, :].astype(float)[:, :1].T[0] if False else np.outer(ph, np.eye(3)[e]).ravel(), U[:, j])) ** 2 for e in range(3)))
        rec.append((float(w[j]), ipr, s, float(np.linalg.norm(kcart))))
rec = np.array(rec); print(f"[B] n=17: {len(rec)} modes below lam_max = {lmax} ({time.time() - t:.0f} s)", flush=True)
out["B"] = dict(lam=rec[:, 0].tolist(), ipr_cell=rec[:, 1].tolist(), sound_weight=rec[:, 2].tolist(), k=rec[:, 3].tolist())
json.dump(out, open("r10_lowmodes.json", "w"))
hw = 64.654 * np.sqrt(rec[:, 0]); 
for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 9)):
    s = (hw >= lo) & (hw < hi); print(f"  hbar omega {lo}-{hi} meV: {s.sum()} modes, median IPR_cell {np.median(rec[s, 1]):.4f}, median sound weight {np.median(rec[s, 2]):.3f}")
