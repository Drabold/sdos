"""Sound-wave weight and in-cell IPR of the exact Bloch modes of the 17^3 replica of R10 (ACE) versus hbar*omega, up to lam_max: where the modes stop being plane-wave-like
(propagon -> diffuson crossover of this structure).  usage: python r10_character_scan.py lam_max -> r10_character.json"""
import sys, time, json, numpy as np
from common_r10 import r10_positions, L0
lmax = float(sys.argv[1]); m = 28.0855; n = 17
Phi = np.load("ifc_r10.npy").reshape(216, 3, 27, 216, 3); cells = [(c % 3, (c // 3) % 3, c // 9) for c in range(27)]; Rv = np.array([[(v if v < 2 else -1) for v in cc] for cc in cells])
x0 = r10_positions(); lam_all = []; ipr_all = []; s_all = []; t = time.time(); seen = set()
for p in np.ndindex(n, n, n):
    pm = tuple((-v) % n for v in p)
    if pm in seen and pm != p: continue                       # k and -k: same eigenvalues, conjugate eigenvectors (same character): use one
    seen.add(p); kk = 2 * np.pi * np.array(p) / n; kq = np.where(np.array(p) > n // 2, np.array(p) - n, np.array(p)); wgt = 1 if pm == p else 2
    D1 = np.einsum("c,iacjb->iajb", np.exp(1j * (Rv @ kk)), Phi).reshape(648, 648) / m; D1 = 0.5 * (D1 + D1.conj().T)
    w, U = np.linalg.eigh(D1); idx = np.nonzero((w > 1e-6) & (w < lmax))[0]
    if len(idx) == 0: continue
    ph = np.exp(1j * (x0 @ (2 * np.pi * kq / (n * L0)))) / np.sqrt(216.0); Uj = U[:, idx].reshape(216, 3, -1)
    s = (np.abs(np.einsum("i,iaj->aj", ph.conj(), Uj)) ** 2).sum(0); ipr = ((np.abs(Uj) ** 2).sum(1) ** 2).sum(0)
    for _ in range(wgt): lam_all.append(w[idx]); ipr_all.append(ipr); s_all.append(s)
lam = np.concatenate(lam_all); ipr = np.concatenate(ipr_all); s = np.concatenate(s_all); hw = 64.654 * np.sqrt(lam)
print(len(lam), "modes", f"{time.time() - t:.0f} s", flush=True)
edges = np.arange(0, hw.max() + 1, 1.0); rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (hw >= lo) & (hw < hi)
    if sel.sum(): rows.append(dict(hw_lo=float(lo), hw_hi=float(hi), n=int(sel.sum()), s_median=float(np.median(s[sel])), s_p10=float(np.percentile(s[sel], 10)), s_mean=float(s[sel].mean()), ipr_median=float(np.median(ipr[sel])), ipr_p90=float(np.percentile(ipr[sel], 90))))
    print(f"{lo:4.0f}-{hi:4.0f} meV: n={sel.sum():6d}  sound weight median {rows[-1]['s_median']:.3f} p10 {rows[-1]['s_p10']:.3f} | IPR_cell median {rows[-1]['ipr_median']:.4f} p90 {rows[-1]['ipr_p90']:.4f}" if sel.sum() else "", flush=True)
json.dump(dict(lam_max=lmax, bins=rows), open("r10_character.json", "w"), indent=1)
