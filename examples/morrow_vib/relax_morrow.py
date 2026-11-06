"""CG relaxation of the Morrow 1M model with the ACE potential (Kokkos/GPU), positions only (fixed cell)."""
import time
from lammps import lammps
l = lammps(name="kk", cmdargs=["-screen", "none", "-log", "relax_morrow.log", "-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"])
for c in ["units metal", "boundary p p p", "atom_style atomic", "atom_modify map array sort 0 0.0", "read_data morrow_input.data", "mass 1 28.0855",
          "include /home/drabold/lammps-runs/rosset_external/pot_kk.in", "neighbor 1.0 bin", "thermo 200", "thermo_style custom step pe press fnorm fmax", "min_style cg"]: l.command(c)
t = time.time(); l.command("minimize 0.0 1e-6 20000 40000"); print("min time", time.time() - t, "pe/atom", l.get_thermo("pe") / l.get_natoms(), "fmax", l.get_thermo("fmax"), flush=True)
l.command("write_data morrow_relaxed.data nocoeff"); print("RELAX_DONE", flush=True)
import os; os._exit(0)
