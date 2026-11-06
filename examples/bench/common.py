import resource, time
import numpy as np

def dtw_tiled(n):
    """DTW4096 (periodic, 4096 atoms) tiled n x n x n: positions and box.  A tiling, not independent structures: used for cost scaling only."""
    d = np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz")
    X, L = d["X"], d["L"]
    X = np.mod(X, L)
    sh = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3) * L
    return (X[None] + sh[:, None, :]).reshape(-1, 3), L * n

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

class Timer:
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): self.s = time.time() - self.t
