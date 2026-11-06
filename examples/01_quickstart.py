"""The simple way: DOS of a sparse matrix in one call (a disordered cubic lattice, 12^3 sites)."""
import sys
sys.path.insert(0, "../tests")
import numpy as np
import sdos
from conftest import cubic

H = cubic(12, W=4.0, seed=1)                  # any SciPy sparse matrix, array, file name or LinearOperator works
res = sdos.dos(H)                              # 256 moments, 10 random vectors, first 4 moments exact
print(res)                                     # what was computed
print("integral of rho:", round(float(np.sum(res.rho * np.gradient(res.E))), 4))
res.save("quickstart_dos.txt")
if "--plot" in sys.argv:
    res.plot().figure.savefig("quickstart.png", dpi=130)
