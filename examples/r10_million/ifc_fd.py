"""Real-space interatomic force constants of the relaxed R10 cell (216 atoms, ACE) by central differences of the ACE forces (LAMMPS Kokkos GPU through its
Python module) on the 3x3x3 replica (5832 atoms): Phi[3i+a, c, 3j+b] = d^2E/du_{i a}(0) du_{j b}(cell c), i, j in the home cell, c = cx + 3 cy + 9 cz.
usage: python ifc_fd.py [h=0.005]  -> ifc_r10.npy (648 x 27 x 648)"""
import sys, time, numpy as np
from lammps import lammps
h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005
l = lammps(name="kk", cmdargs=["-screen", "none", "-log", "none", "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"])
for c in ["units metal", "boundary p p p", "atom_style atomic", "atom_modify map array sort 0 0.0", "read_data r10_rep3.data",
          "include /home/drabold/lammps-runs/rosset_external/pot_kk.in", "neighbor 1.0 bin"]: l.command(c)
n = int(l.get_natoms()); assert n == 5832
x0 = np.ctypeslib.as_array(l.gather_atoms("x", 1, 3)).reshape(n, 3).copy()
def forces(x):
    l.scatter_atoms("x", 1, 3, np.ctypeslib.as_ctypes(np.ascontiguousarray(x.reshape(-1))))
    l.command("run 0 post no")
    return np.ctypeslib.as_array(l.gather_atoms("f", 1, 3)).reshape(n, 3).copy()
f0 = forces(x0); print("max |F| at the reference:", abs(f0).max(), flush=True)
Phi = np.zeros((648, 27, 648)); t = time.time()
for i in range(216):
    for a in range(3):
        xp = x0.copy(); xp[i, a] += h; xm = x0.copy(); xm[i, a] -= h
        dF = -(forces(xp) - forces(xm)) / (2 * h)                  # (n, 3): -dF_j,b / du_i,a = Phi[(i a), (j b)]
        Phi[3 * i + a] = dF.reshape(27, 216, 3).reshape(27, 648)    # atom index = 216 c + j
    if i % 20 == 0: print(i, f"{time.time() - t:.0f} s", flush=True)
np.save("ifc_r10.npy", Phi); print("done", time.time() - t, flush=True)
l.close()
