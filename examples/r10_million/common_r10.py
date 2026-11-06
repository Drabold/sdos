import numpy as np
L0 = 16.54808251
def r10_positions():
    """Relaxed (ACE) R10 positions (216, 3) in the cubic cell of edge L0."""
    txt = open("/home/drabold/sdos_pkg/examples/r10_million/r10_relaxed.data").read().split("Atoms # atomic")[1].strip().split("\n")
    rows = sorted([l.split() for l in txt if len(l.split()) >= 5 and l.split()[0].isdigit()], key=lambda r: int(r[0]))
    return np.array([[float(v) for v in r[2:5]] for r in rows])
def replica(n):
    x0 = r10_positions(); g = np.arange(n)
    cells = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pos = (x0[None] + L0 * cells[:, None, :]).reshape(-1, 3)
    return pos, np.full(3, n * L0)
