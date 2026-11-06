"""Convert the Morrow-Deringer 1M-atom a-Si xyz (maintext model) to a LAMMPS data file."""
import numpy as np
f = open("/home/drabold/data/morrow_zenodo/aSi_1Matom_maintext.xyz"); n = int(f.readline()); f.readline()
x = np.array([l.split()[1:4] for l in (f.readline() for _ in range(n))], dtype=float); L = 273.643
with open("morrow_input.data", "w") as o:
    o.write(f"Morrow 1M a-Si\n\n{n} atoms\n1 atom types\n\n0 {L} xlo xhi\n0 {L} ylo yhi\n0 {L} zlo zhi\n\nMasses\n\n1 28.0855\n\nAtoms # atomic\n\n")
    o.write("".join(f"{i+1} 1 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for i, p in enumerate(x)))
print(n, "atoms")
