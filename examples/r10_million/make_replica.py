"""3x3x3 replica of the relaxed R10 cell with explicit ordering: id = 216*c + i, c = ix + 3 iy + 9 iz (home cell c = 0)."""
import numpy as np, sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
L = 16.54808251; X = []
for l in open("r10_relaxed.data").read().split("Atoms # atomic")[1].split("Velocities")[0].strip().split("\n")[1:] if False else []: pass
txt = open("r10_relaxed.data").read(); body = txt.split("Atoms # atomic")[1].strip().split("\n")
rows = sorted([l.split() for l in body if len(l.split()) >= 5 and l.split()[0].isdigit()], key=lambda r: int(r[0]))
x0 = np.array([[float(v) for v in r[2:5]] for r in rows]); assert len(x0) == 216
N = 216 * n ** 3
with open(f"r10_rep{n}.data", "w") as f:
    f.write(f"R10 replica {n}^3\n\n{N} atoms\n1 atom types\n\n0 {n*L:.8f} xlo xhi\n0 {n*L:.8f} ylo yhi\n0 {n*L:.8f} zlo zhi\n\nMasses\n\n1 28.0855\n\nAtoms # atomic\n\n")
    k = 1
    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                for p in x0 + L * np.array([ix, iy, iz]):
                    f.write(f"{k} 1 {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n"); k += 1
print(N, "atoms written")
