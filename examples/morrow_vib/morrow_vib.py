"""Matrix-free ACE vibrational DOS of the relaxed one-million-atom Morrow et al. a-Si model (main-text structure), LAMMPS Kokkos/GPU through its Python module.
usage: python morrow_vib.py M R [seed=3]   (needs morrow_relaxed.data from relax_morrow.py; spectral bounds by the package's Lanczos estimate)"""
import sys, time, json, resource, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from lammps import lammps
from sdos.interfaces.lmp import HessianOperator
M, R = int(sys.argv[1]), int(sys.argv[2]); SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3; TAG = f"M{M}_R{R}" + (f"_seed{SEED}" if SEED != 3 else "")
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
F = KKForces("morrow_relaxed.data"); DELTA = 1e-4; op = HessianOperator(F, F.x0, F.mass, DELTA); log = dict(atoms=F.n, M=M, R=R, delta=DELTA)
t = time.time(); r = sdos.vdos(op, zero_modes=3, M=M, R=R, method="kpm", seed=SEED, hermitian_tol=1e-3, bounds=(-0.02, 1.40), verbose=True)  # window from a 30-step Lanczos at delta=1e-4 (extreme Ritz values 0.0035 and 1.336); Lanczos at delta=1e-3 diverged (finite-difference operator nonlinear on localized vectors)
log.update(t_kpm=time.time() - t, force_calls=F.calls, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
r.save(f"morrow_vib_result_{TAG}.npz"); json.dump(log, open(f"morrow_vib_log_{TAG}.json", "w"), indent=1); print(log, flush=True); print("VIB_DONE", flush=True)
import os; os._exit(0)
