"""Projected and local DOS of a two-species system with a non-orthogonal basis (Mulliken convention)."""
import sys
sys.path.insert(0, "../tests")
import numpy as np
import sdos
from conftest import random_hermitian
from test_generalized import spd_B

n = 240                                                              # 60 sites x 4 orbitals
A = random_hermitian(n, False, seed=3)
B = spd_B(n, 20, False, seed=2)
species = ["Si" if i % 3 else "Ge" for i in range(60)]
groups = sdos.site_groups(species, 4)
p = sdos.pdos(A, B, groups=groups, M=128, R=30, seed=1)              # all groups from one propagation ("shared" mode)
print({g: round(float(p.idos[g][-1]), 3) for g in groups}, "-> integrals = fraction of orbitals", {g: round(len(v) / n, 3) for g, v in groups.items()})
l = sdos.ldos(A, B, orbitals=[0, 1, 2, 3], M=128)                    # exact, no sampling noise
print("LDOS of orbitals 0..3 integrate to", [round(float(l.idos[i][-1]), 3) for i in range(4)])
