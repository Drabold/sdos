"""DFTB (pbc-0-3, non-SCC) electronic DOS of DTW4096: generalized problem H x = e S x from the sdos .skf builder, Gamma point (real) and a Monkhorst-Pack-type k point (complex Hermitian),
against the exact generalized spectrum (dense).  usage: python dtw_dftb.py gamma|k"""
import sys, time, warnings, json
import numpy as np
import scipy.linalg as sla
import sdos
from sdos.interfaces import skf

warnings.filterwarnings("ignore")
mode = sys.argv[1]
d = np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz")
X, L = d["X"], d["L"]
sk = skf.SKSet("/home/drabold/opt/slako/pbc-0-3", ["Si"])
k = None if mode == "gamma" else 2 * np.pi * np.array([0.25 / L[0], 0.25 / L[1], 0.25 / L[2]])          # a (1/4,1/4,1/4) point of the cell's own reciprocal lattice
t = time.time()
H, S = skf.skf_hamiltonian(X, ["Si"] * len(X), L, sk, kpoint=k)
n = H.shape[0]
print(f"[{mode}] built H,S: {n} orbitals, nnz/row {H.nnz / n:.0f}, complex={np.iscomplexobj(H.data)}, {time.time() - t:.0f} s", flush=True)
nocc = n // 2
out = dict(mode=mode, n=n)
# exact: dense generalized eigenvalues (memory: ~3 dense matrices); complex k-point is done for the same size only if memory allows
import os
cache = f"dtw_dftb_{mode}_ev.npy"
if mode != "gamma":
    ev = np.load("dtw_dftb_gamma_ev.npy")      # dense complex generalized diagonalisation of 16k orbitals does not fit in memory here: compare with the Gamma-point spectrum (same cell)
    print(f"[{mode}] no exact reference at this size (complex, 16384 orbitals); compared with the Gamma-point exact spectrum", flush=True)
elif os.path.exists(cache):
    ev = np.load(cache)
else:
    t = time.time()
    Ld = sla.cholesky(S.toarray(), lower=True, overwrite_a=True, check_finite=False)
    Hd = H.toarray()
    Hd = sla.solve_triangular(Ld, Hd, lower=True, overwrite_b=True, check_finite=False)
    Hd = sla.solve_triangular(Ld, Hd.conj().T, lower=True, overwrite_b=True, check_finite=False)
    del Ld
    ev = np.sort(sla.eigvalsh(Hd, driver="evd", overwrite_a=True, check_finite=False))
    del Hd
    np.save(cache, ev)
    print(f"[{mode}] exact generalized spectrum: {time.time() - t:.0f} s; gap {ev[nocc] - ev[nocc - 1]:.4f} eV", flush=True)
ebs = 2.0 * ev[:nocc].sum()
out.update(gap=float(ev[nocc] - ev[nocc - 1]), ebs=float(ebs))
for label, kw in (("PCG block-Jacobi", dict(binv="pcg", precond="block-jacobi")), ("PCG IC(0)", dict(binv="pcg", precond="ic")), ("Chebyshev B^-1", dict(binv="chebyshev"))):
    for meth in ("kpm",) if label != "PCG block-Jacobi" else ("kpm", "maxent"):
        t = time.time()
        try:
            r = sdos.dos(H, B=S, M=170, R=6, method=meth, seed=11, **kw)
            ef, eb, _, eeb = sdos.fermi_and_band_energy(r, n)
            sec = time.time() - t
            bi = r.get("binv_info", {})
            ref = np.zeros_like(r.E)
            for ch in np.array_split(ev, 40):
                ref += np.exp(-0.5 * ((r.E[:, None] - ch[None]) / 0.1) ** 2).sum(1)
            ref /= n * 0.1 * np.sqrt(2 * np.pi)
            sel = (r.E > ev[0] - 1) & (r.E < ev[-1] + 1)
            l1 = float(np.trapezoid(np.abs(r.rho[sel] - ref[sel]), r.E[sel]))
            out[f"{label}|{meth}"] = dict(sec=sec, dE_BS_per_atom_meV=float((eb - ebs) / 4096 * 1e3), err_meV=float(eeb / 4096 * 1e3), L1=l1, binv=str({k_: v for k_, v in bi.items() if k_ in ("iterations", "nsolve", "degree", "kappa")}))
            print(f"[{mode}] {label:18s} {meth:6s}: {sec:6.0f} s  E_BS error {out[f'{label}|{meth}']['dE_BS_per_atom_meV']:+7.2f} meV/atom (+-{out[f'{label}|{meth}']['err_meV']:.2f}), L1(0.1 eV Gaussian) {l1:.4f}  {out[f'{label}|{meth}']['binv']}", flush=True)
        except Exception as exc:
            print(f"[{mode}] {label} {meth} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
json.dump(out, open(f"dtw_dftb_{mode}.json", "w"), indent=1)
print("done", flush=True)
