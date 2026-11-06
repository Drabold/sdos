"""DTW4096 phonons with sdos: matrix-free Hessian (finite-difference forces, 2 force calls per product) -> vibrational DOS and harmonic thermodynamics.
usage: python dtw_phonons.py sw|ace M R
 'sw'  : Stillinger-Weber; also builds the exact dense dynamical matrix column by column and compares (an exactness test of the whole chain at 4096 atoms).
 'ace' : SiH-ACE-25 (the potential that relaxed the DTW host); the exactness of the operator is checked at 512 atoms in examples/lammps/real_lammps_test.py."""
import sys, time, json, warnings
import numpy as np
import scipy.linalg as sla
import sdos
from sdos.interfaces.lmp import LammpsForces
from sdos.thermo import HBAR_OMEGA_METAL

warnings.filterwarnings("ignore")
pot, M, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
DIR = "/home/drabold/lammps-runs/topoMC/sdos_dtw"
pair = {"sw": "pair_style sw\npair_coeff * * /home/drabold/src/lammps/potentials/Si.sw Si\n",
        "ace": f"pair_style pace\npair_coeff * * {DIR}/SiH-ACE-25.yaml Si\n"}[pot]
head = "units metal\nboundary p p p\natom_style atomic\natom_modify map array sort 0 0.0\n"
# 1. relax
from lammps import lammps
l = lammps(cmdargs=["-screen", "none", "-log", "none"])
l.commands_string(head + f"read_data {DIR}/dtw4096.data\nmass 1 28.0855\n{pair}neighbor 1.0 bin\nthermo_style custom step pe fnorm fmax\nmin_style cg\nminimize 0 1e-7 3000 30000\nwrite_data {DIR}/dtw4096_{pot}_relaxed.data nocoeff\n")
print(f"[{pot}] relaxed: pe/atom {l.get_thermo('pe') / 4096:.6f}, fmax {l.get_thermo('fmax'):.2e}", flush=True)
l.close()
open(f"{DIR}/in.{pot}", "w").write(head + f"read_data {DIR}/dtw4096_{pot}_relaxed.data\nmass 1 28.0855\n{pair}neighbor 1.0 bin\n")
F = LammpsForces(f"{DIR}/in.{pot}")
op = F.operator(delta=1e-3)
out = dict(pot=pot, M=M, R=R)
T = [100.0, 300.0, 600.0]
res = {}
for meth in ("kpm", "maxent"):
    F.calls = 0
    t = time.time()
    r = sdos.vdos(op, zero_modes=3, M=M, R=R, method=meth, seed=3, hermitian_tol=op.hermitian_tol)
    th = sdos.harmonic_thermo(r, T)
    res[meth] = r
    out[meth] = dict(sec=time.time() - t, force_calls=F.calls, zpe=float(th.zpe), Cv=[float(c) for c in np.atleast_1d(th.Cv)], S=[float(s) for s in np.atleast_1d(th.S)],
                     omega=np.asarray(r["omega"]).tolist(), g=np.asarray(r["g"]).tolist())
    print(f"[{pot}] {meth}: {time.time() - t:.0f} s, {F.calls} force calls; ZPE {float(th.zpe) * 1e3:.3f} meV/atom, Cv(100,300,600 K) = {np.round(np.atleast_1d(th.Cv), 4)} kB/atom", flush=True)
    json.dump(out, open(f"dtw_phonons_{pot}.json", "w"))
if pot == "sw":
    n = F.n; x0 = F.x0.copy(); d = 1e-3
    Phi = np.zeros((3 * n, 3 * n))
    t = time.time()
    for k in range(3 * n):
        xp = x0.copy().reshape(-1); xp[k] += d
        xm = x0.copy().reshape(-1); xm[k] -= d
        Phi[:, k] = -(F(xp.reshape(n, 3)) - F(xm.reshape(n, 3))).reshape(-1) / (2 * d)
        if k % 2000 == 0:
            print(f"  dense column {k}/{3 * n} ({time.time() - t:.0f} s)", flush=True)
    Phi = 0.5 * (Phi + Phi.T)
    mi = np.repeat(F.mass, 3)
    Dm = Phi / np.sqrt(mi[:, None] * mi[None, :])
    lam = np.linalg.eigvalsh(Dm)
    lam_pos = lam[lam > 1e-6]
    om = np.sqrt(lam_pos) * HBAR_OMEGA_METAL           # hbar omega in eV
    ex = dict(n_neg=int((lam < -1e-6).sum()), n_zero=int((np.abs(lam) <= 1e-6).sum()), zpe=float(0.5 * om.sum() / n))
    kb = 8.617333262e-5
    ex["Cv"] = [float(np.sum((om / (kb * T_)) ** 2 * np.exp(om / (kb * T_)) / (np.exp(om / (kb * T_)) - 1) ** 2) / n) for T_ in T]
    out["exact"] = ex; out["exact_lambda"] = lam.tolist()
    print(f"[sw] exact: {ex['n_neg']} negative, {ex['n_zero']} ~zero eigenvalues; ZPE {ex['zpe'] * 1e3:.3f} meV/atom, Cv = {np.round(ex['Cv'], 4)}; dense build {time.time() - t:.0f} s", flush=True)
    json.dump(out, open(f"dtw_phonons_{pot}.json", "w"))
print("done", flush=True)
