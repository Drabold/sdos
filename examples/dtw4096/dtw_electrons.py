"""DTW4096 (host4096_ref.npz, 4096 Si atoms, Kwon TB, Gamma point): sdos against the exact spectrum (ev_gamma, 16384 eigenvalues).
A. the interface reproduces the exact edges; B. integrated quantities (E_F, band energy) vs number of moments, KPM vs Maxent; C. vector kinds at equal M (spread of E_BS over seeds, cost)."""
import json, time, warnings
import numpy as np
import sdos
from sdos.interfaces import kwon_si as K

warnings.filterwarnings("ignore")
d = np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz")
X, L = d["X"], d["L"]
H = K.kwon_hamiltonian(X, L)
import os, scipy.linalg as sla
# NOTE: host4096_ref.npz['ev_gamma'] is NOT the spectrum of this Hamiltonian (mean -3.93 eV vs trace/N -0.41 eV): another model or energy zero. Exact reference computed here.
if os.path.exists("dtw4096_kwon_ev.npy"):
    ev = np.load("dtw4096_kwon_ev.npy")
else:
    t = time.time(); ev = np.sort(sla.eigvalsh(H.toarray(), driver="evd", overwrite_a=True, check_finite=False)); np.save("dtw4096_kwon_ev.npy", ev)
    print(f"dense eigenvalues of the sdos-built H: {time.time() - t:.0f} s", flush=True)
N = H.shape[0]; ne = N // 2 * 2                      # 4 electrons/atom, spin 2 -> N/2 occupied orbitals ... n_elec counts electrons
n_elec = N                                           # 4 per atom x 4096 atoms = 16384 = N electrons (N orbitals)
nocc = n_elec // 2
ebs_exact = 2.0 * ev[:nocc].sum()
print(f"N = {N} orbitals; exact gap {ev[nocc] - ev[nocc - 1]:.4f} eV (HOMO {ev[nocc - 1]:.4f}, LUMO {ev[nocc]:.4f}); E_BS exact {ebs_exact:.3f} eV = {ebs_exact / 4096:.5f} eV/atom", flush=True)
# A. interface check
mid = 0.5 * (ev[nocc] + ev[nocc - 1])
t = time.time()
w = sdos.window_eigs(H, window=(mid - 0.08, mid + 0.08), method="shift-invert")
ref = ev[(ev > mid - 0.08) & (ev < mid + 0.08)]
print(f"A. shift-invert window around the gap (sdos-built H against its own dense spectrum): {len(w.E)} states from the sdos-built H vs {len(ref)} in ev_gamma; max |difference| = "
      f"{np.max(np.abs(w.E - ref)) if len(w.E) == len(ref) else float('nan'):.1e} eV ({time.time() - t:.0f} s)", flush=True)
out = dict(gap=float(ev[nocc] - ev[nocc - 1]), ebs_exact=float(ebs_exact))
# B. moment sweep
rowsB = []
for M in (30, 50, 70, 100, 170, 300):
    row = dict(M=M)
    for meth in ("kpm", "maxent"):
        if meth == "maxent" and M > 170:
            continue
        t = time.time()
        r = sdos.dos(H, M=M, R=10, method=meth, seed=1)
        ef, ebs, eef, eebs = sdos.fermi_and_band_energy(r, n_elec)
        row[meth] = dict(dE_BS_per_atom=(ebs - ebs_exact) / 4096, err=eebs / 4096, EF=ef, sec=time.time() - t)
        print(f"B. M={M:4d} {meth:6s}: E_BS error {row[meth]['dE_BS_per_atom'] * 1e3:+8.3f} meV/atom (+-{eebs / 4096 * 1e3:.3f} stat), E_F {ef:.3f}, {row[meth]['sec']:.1f} s", flush=True)
    rowsB.append(row)
out["B"] = rowsB
# C. vector kinds at M = 100, 12 seeds x R = 2
rowsC = []
for label, kw in (("rademacher", dict(vectors="rademacher", exact_moments=0)), ("gaussian", dict(vectors="gaussian", exact_moments=0)),
                  ("impartial (3 moments)", dict(vectors="impartial", exact_moments=0)),
                  ("rademacher + exact K=4", dict(vectors="rademacher", exact_moments=4)), ("rademacher + exact K=8", dict(vectors="rademacher", exact_moments=8)),
                  ("rademacher + exact K=16", dict(vectors="rademacher", exact_moments=16))):
    vals, ts = [], []
    for s in range(12):
        t = time.time()
        try:
            r = sdos.dos(H, M=100, R=2, method="kpm", seed=100 + s, **kw)
        except Exception as exc:
            print("C.", label, "failed:", exc); break
        ts.append(time.time() - t)
        vals.append((sdos.fermi_and_band_energy(r, n_elec)[1] - ebs_exact) / 4096)
    if vals:
        vals = np.array(vals)
        rowsC.append(dict(kind=label, mean_meV=float(vals.mean() * 1e3), std_meV=float(vals.std(ddof=1) * 1e3), sec=float(np.mean(ts))))
        print(f"C. {label:26s}: E_BS error mean {vals.mean() * 1e3:+8.3f} std {vals.std(ddof=1) * 1e3:7.3f} meV/atom over 12 seeds (R=2 each); {np.mean(ts):.2f} s per run", flush=True)
out["C"] = rowsC
json.dump(out, open("dtw_electrons.json", "w"), indent=1)
