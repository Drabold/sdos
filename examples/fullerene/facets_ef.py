"""States nearest E_F of icosahedral fullerenes (Xu H, LCBOP geometry): weight on the 20 flat facets ('graphene grains'), on the 30 edge strips and near the 12 pentagons."""
import sys, json, itertools
import numpy as np
from scipy.spatial import cKDTree
import sdos
import goldberg as G
from sdos.interfaces import xu_carbon as X

def classes(pos, w_edge=2.5, w_pent=3.0):
    N = len(pos)
    nb = cKDTree(pos).query(pos, k=4)[1][:, 1:]
    rings = G.ring_atoms(pos, nb)
    pent = [r for r in rings if len(r) == 5]
    assert len(pent) == 12, len(pent)
    pc = np.array([pos[r].mean(0) for r in pent]); pc /= np.linalg.norm(pc, axis=1)[:, None]
    dmat = np.linalg.norm(pc[:, None] - pc[None], axis=2); dn = np.sort(dmat[dmat > 1e-6])[0]
    adj = dmat < 1.05 * dn
    faces = [t for t in itertools.combinations(range(12), 3) if adj[t[0], t[1]] and adj[t[1], t[2]] and adj[t[0], t[2]]]
    assert len(faces) == 20, len(faces)
    fn = np.array([pc[list(t)].mean(0) for t in faces]); fn /= np.linalg.norm(fn, axis=1)[:, None]
    u = pos / np.linalg.norm(pos, axis=1)[:, None]
    facet = np.argmax(u @ fn.T, axis=1)
    R = np.linalg.norm(pos, axis=1).mean()
    # distance (arc, Angstrom) to the nearest icosahedron edge: angle from u to the great-circle arc between two adjacent vertices
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if adj[i, j]]
    dedge = np.full(N, np.inf)
    for i, j in edges:
        nrm = np.cross(pc[i], pc[j]); nrm /= np.linalg.norm(nrm)
        proj = u - np.outer(u @ nrm, nrm); proj /= np.linalg.norm(proj, axis=1)[:, None]
        inside = (np.cross(pc[i], proj) @ nrm > 0) & (np.cross(proj, pc[j]) @ nrm > 0)
        dd = np.where(inside, np.arcsin(np.clip(np.abs(u @ nrm), 0, 1)), np.inf)
        dedge = np.minimum(dedge, dd * R)
    pcen = np.array([pos[r].mean(0) for r in pent])               # actual pentagon centres (the faceted cage's vertices sit beyond the mean radius)
    dpent = cKDTree(pcen).query(pos)[0]
    cls = np.where(dpent < w_pent, 0, np.where(dedge < w_edge, 1, 2))     # 0 pentagon, 1 edge strip, 2 facet interior
    return cls, facet, dict(R=float(R), n_edges=len(edges))

def run(N, halfwidth=0.6):
    pos = np.load(f"C{N}_lcbop.npy")
    cls, facet, geo = classes(pos)
    H = X.xu_hamiltonian(pos)
    n = H.shape[0]
    hl = sdos.homo_lumo(H, n // 2)
    ef = hl["fermi"]
    w = sdos.window_eigs(H, window=(ef - halfwidth, ef + halfwidth), method="shift-invert")
    E, V = w.E, w.V
    p = (np.abs(V) ** 2).reshape(len(pos), 4, -1).sum(1)          # per atom, per state
    frac = np.array([(cls == c).mean() for c in range(3)])
    cw = np.array([[p[cls == c, s].sum() for c in range(3)] for s in range(p.shape[1])])
    enh = cw / frac[None]
    fw = np.array([[p[facet == f, s].sum() for f in range(20)] for s in range(p.shape[1])])
    return dict(N=N, gap=hl["gap"], ef=ef, E=(E - ef).tolist(), ipr=np.asarray(w.ipr).tolist(), enh=enh.tolist(), frac=frac.tolist(), facet_w=fw.tolist(), geo=geo)

if __name__ == "__main__":
    out = {}
    for N in map(int, sys.argv[1:]):
        r = run(N); out[N] = r
        e = np.array(r["enh"]); E = np.array(r["E"]); ip = np.array(r["ipr"])
        print(f"C{N}: gap {r['gap']:.3f} eV, {len(E)} states within +-0.6 eV; atom fractions pent/edge/interior = {np.round(r['frac'], 3)}", flush=True)
        s = 0
        while s < len(E):
            t = s
            while t + 1 < len(E) and abs(E[t + 1] - E[s]) < 2e-4: t += 1
            print(f"  E-EF={E[s]:+.3f} x{t - s + 1}  <ipr>={ip[s:t + 1].mean():.2e}  enh(pent,edge,interior)={e[s:t + 1].mean(0).round(2)}")
            s = t + 1
        json.dump(out, open("facets_ef.json", "w"))
