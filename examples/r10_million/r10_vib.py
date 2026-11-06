"""Matrix-free ACE vibrational DOS of the n^3 replica of R10 (LAMMPS Kokkos/GPU through its Python module, two force calls per product), KPM and Maxent from the same moments,
against the exact spectrum from the force constants (exact_vib_r10.py).  usage: python r10_vib.py n M R"""
import sys, time, json, resource, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from lammps import lammps
from sdos.interfaces.lmp import HessianOperator
n, M, R = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
lam = np.load(f"vib_r10_exact_lambda_{n}.npy")
class KKForces:
    def __init__(self, data):
        self.lmp = lammps(name="kk", cmdargs=["-screen", "none", "-log", "none", "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"])
        for c in ["units metal", "boundary p p p", "atom_style atomic", "atom_modify map array sort 0 0.0", f"read_data {data}", "mass 1 28.0855",
                  "include /home/drabold/lammps-runs/rosset_external/pot_kk.in", "neighbor 1.0 bin"]: self.lmp.command(c)
        self.n = int(self.lmp.get_natoms()); self.lmp.command("run 0 post no"); self.calls = 0
        self.x0 = np.ctypeslib.as_array(self.lmp.gather_atoms("x", 1, 3)).reshape(self.n, 3).copy(); self.mass = np.full(self.n, 28.0855)
    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        self.lmp.scatter_atoms("x", 1, 3, np.ctypeslib.as_ctypes(x)); self.lmp.command("run 0 post no"); self.calls += 1
        return np.ctypeslib.as_array(self.lmp.gather_atoms("f", 1, 3)).reshape(self.n, 3).copy()
F = KKForces(f"r10_rep{n}.data"); op = HessianOperator(F, F.x0, F.mass, 1e-3)
lo, hi = float(lam.min()), float(lam.max()); log = dict(n=n, atoms=F.n, M=M, R=R)
t = time.time(); r = sdos.vdos(op, zero_modes=3, M=M, R=R, method="kpm", seed=3, hermitian_tol=1e-3, bounds=(lo, hi), verbose=False)
log.update(t_kpm=time.time() - t, force_calls=F.calls, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
r.save(f"r10_vib_result_n{n}_M{M}_R{R}.npz"); json.dump(log, open(f"r10_vib_log_n{n}_M{M}_R{R}.json", "w"), indent=1); print(log, flush=True); print("VIB_DONE", flush=True)
import os; os._exit(0)
