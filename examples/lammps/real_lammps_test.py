"""Real LAMMPS test of sdos' matrix-free Hessian (ACE potential): HessianOperator products against the exact matrix written by LAMMPS' dynamical_matrix command."""
import numpy as np
import sdos
from sdos.interfaces.lmp import LammpsForces

T = "/home/drabold/lammps-runs/test"
open("in.si512", "w").write(f"""units metal
boundary p p p
atom_style atomic
atom_modify map array sort 0 0.0
read_data {T}/si512_dm_ref.data
mass 1 28.0855
pair_style pace
pair_coeff * * {T}/SiH-ACE-25.yaml Si
""")
F = LammpsForces("in.si512")
op = F.operator(delta=1e-3)
n = F.n
D = np.loadtxt(f"{T}/si512_dynmat.dat").reshape(3 * n, 3 * n)
print("atoms", n, "dynmat symmetric defect", np.abs(D - D.T).max(), " largest element", np.abs(D).max())
rng = np.random.default_rng(0)
for k in range(3):
    v = rng.normal(size=3 * n)
    a = op @ v
    pv = op.project(v) if hasattr(op, "project") else v
    b = D @ pv
    print(f"  product {k}: |op v|/|D v| = {np.linalg.norm(a) / np.linalg.norm(b):.5f},  relative difference {np.linalg.norm(a - b) / np.linalg.norm(b):.2e}")
print("force calls so far", F.calls)
