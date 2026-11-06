"""Icosahedral fullerenes C_{20 T}, T = m^2 + m k + k^2, by the Caspar-Klug / Goldberg construction.

(n, n) gives the achiral series C_{60 n^2} (C60, C240, C540, C960, ..., C3840 = n 8 as in Drabold, Ordejon, Dong, Martin, Solid State Commun. 96, 833 (1995));
(n, 0) gives C_{20 n^2} (C20, C80, C180, ...); (m, k) with m != k != 0 are chiral (symmetry I, not Ih).
Method: the triangular lattice points inside the equilateral triangle (0, v, R60 v), v = m a + k b, are mapped barycentrically onto each of the 20 faces of
the icosahedron and projected onto the sphere (the geodesic polyhedron: 10 T + 2 vertices); the fullerene atoms are the centres of the triangles of its
convex hull (the dual), bonded when the triangles share an edge.  The radius is set from the area per atom of graphene for the given bond length.
"""
import numpy as np
from scipy.spatial import ConvexHull

PHI = (1 + 5 ** 0.5) / 2


def icosahedron():
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [(0, s1, s2 * PHI), (s1, s2 * PHI, 0), (s2 * PHI, 0, s1)]
    v = np.array(v, float)
    v /= np.linalg.norm(v[0])
    hull = ConvexHull(v)
    faces = []
    for s in hull.simplices:
        p = v[s]
        if np.linalg.det(p) < 0:                       # counter-clockwise seen from outside
            s = s[[0, 2, 1]]
        faces.append(s)
    return v, np.array(faces)


def geodesic_points(m, k):
    """Unit-sphere vertices of the class (m, k) geodesic polyhedron (10 T + 2 of them)."""
    if m < k:
        m, k = k, m
    T = m * m + m * k + k * k
    a, b = np.array([1.0, 0.0]), np.array([0.5, 3 ** 0.5 / 2])
    v1 = m * a + k * b
    v2 = -k * a + (m + k) * b                          # v1 rotated by +60 degrees
    M = np.array([v1, v2]).T
    rng = range(-(m + k) - 1, (m + k) + 2)
    pts = []
    for i in rng:
        for j in rng:
            p = i * a + j * b
            l1, l2 = np.linalg.solve(M, p)
            l0 = 1 - l1 - l2
            if min(l0, l1, l2) > -1e-9:
                pts.append((l0, l1, l2))
    pts = np.array(pts)
    V, F = icosahedron()
    out = []
    for f in F:
        P = pts @ V[f]
        out.append(P / np.linalg.norm(P, axis=1)[:, None])
    out = np.vstack(out)
    key = np.round(out, 7)
    _, idx = np.unique(key, axis=0, return_index=True)
    out = out[np.sort(idx)]
    assert len(out) == 10 * T + 2, (len(out), 10 * T + 2)
    return out, T


def fullerene(m, k, bond=1.42):
    """(positions (N, 3) in Angstrom, bonds (N, 3) neighbour indices, T) of the Goldberg fullerene (m, k)."""
    P, T = geodesic_points(m, k)
    hull = ConvexHull(P)
    tri = hull.simplices
    cen = P[tri].mean(axis=1)
    cen /= np.linalg.norm(cen, axis=1)[:, None]
    N = len(tri)                                       # 20 T atoms
    edge = {}
    for t, s in enumerate(tri):
        for e in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            edge.setdefault(tuple(sorted(e)), []).append(t)
    nb = [[] for _ in range(N)]
    for ts in edge.values():
        assert len(ts) == 2
        nb[ts[0]].append(ts[1])
        nb[ts[1]].append(ts[0])
    A_atom = 3 * 3 ** 0.5 / 4 * bond ** 2
    R = np.sqrt(N * A_atom / (4 * np.pi))
    return cen * R, np.array(nb), T


def faces_from_geometry(pos, nb):
    """Ring sizes by walking faces using the geometric (counter-clockwise) order of the three neighbours at each atom."""
    N = len(pos)
    order = []
    for i in range(N):
        n = pos[i] / np.linalg.norm(pos[i])
        a = pos[nb[i][0]] - pos[i]
        a -= (a @ n) * n
        ang = []
        for j in nb[i]:
            d = pos[j] - pos[i]
            d -= (d @ n) * n
            ang.append(np.arctan2(n @ np.cross(a, d), a @ d))
        order.append([nb[i][t] for t in np.argsort(ang)])         # counter-clockwise seen from outside
    return _collect(order)


def ring_atoms(pos, nb):
    """Atom index lists of every ring (face) of the fullerene, walking faces by the geometric counter-clockwise order of the three neighbours."""
    N = len(pos)
    order = []
    for i in range(N):
        n = pos[i] / np.linalg.norm(pos[i])
        a = pos[nb[i][0]] - pos[i]
        a -= (a @ n) * n
        ang = []
        for j in nb[i]:
            d = pos[j] - pos[i]
            d -= (d @ n) * n
            ang.append(np.arctan2(n @ np.cross(a, d), a @ d))
        order.append([nb[i][t] for t in np.argsort(ang)])
    seen, faces = set(), []
    for i in range(N):
        for j in order[i]:
            if (i, j) in seen:
                continue
            a, b, face = i, j, []
            while (a, b) not in seen:
                seen.add((a, b))
                face.append(a)
                o = order[b]
                a, b = b, o[(o.index(a) + 1) % 3]
            faces.append(face)
    return faces


def _collect(order):
    N = len(order)
    seen, sizes = set(), []
    for i in range(N):
        for j in order[i]:
            if (i, j) in seen:
                continue
            a, b, cnt = i, j, 0
            while (a, b) not in seen:
                seen.add((a, b))
                cnt += 1
                o = order[b]
                a, b = b, o[(o.index(a) + 1) % 3]
            sizes.append(cnt)
    return sizes


def write_xyz(path, pos, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n{comment}\n")
        for p in pos:
            f.write(f"C {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def write_lammps_data(path, pos, pad=20.0):
    c = pos - pos.mean(axis=0)
    lo = c.min(axis=0) - pad
    hi = c.max(axis=0) + pad
    with open(path, "w") as f:
        f.write(f"C{len(pos)} fullerene\n\n{len(pos)} atoms\n1 atom types\n\n")
        for a, l, h in zip("xyz", lo, hi):
            f.write(f"{l:.6f} {h:.6f} {a}lo {a}hi\n")
        f.write("\nMasses\n\n1 12.011\n\nAtoms # atomic\n\n")
        for i, p in enumerate(c, 1):
            f.write(f"{i} 1 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


if __name__ == "__main__":
    import sys
    m, k = int(sys.argv[1]), int(sys.argv[2])
    pos, nb, T = fullerene(m, k)
    sz = faces_from_geometry(pos, nb)
    print(f"C{len(pos)} (T = {T}): {sz.count(5)} pentagons, {sz.count(6)} hexagons")
    write_xyz(f"C{len(pos)}_ideal.xyz", pos, f"Goldberg ({m},{k}) ideal sphere projection")
