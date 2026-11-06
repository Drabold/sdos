"""Goldberg (Caspar-Klug) fullerene generator: counts, rings, valence, icosahedral symmetry."""
import itertools
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "examples" / "fullerene"))
import goldberg as G


def ring_sizes(pos, nb):
    N = len(pos)
    order = []
    for i in range(N):
        n = pos[i] / np.linalg.norm(pos[i])
        a = pos[nb[i][0]] - pos[i]
        a = a - (a @ n) * n
        ang = []
        for j in nb[i]:
            d = pos[j] - pos[i]
            d = d - (d @ n) * n
            ang.append(np.arctan2(n @ np.cross(a, d), a @ d))
        order.append([nb[i][t] for t in np.argsort(ang)])
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


def icosahedral_rotations():
    """The 60 proper rotations of the icosahedron (generated from a 5-fold and a 3-fold axis)."""
    V, F = G.icosahedron()
    # 5-fold about vertex V[0]; 2-fold about an edge midpoint; build by closure
    def rot(axis, ang):
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    c5 = rot(V[0], 2 * np.pi / 5)
    fc = V[F[0]].mean(axis=0)
    c3 = rot(fc, 2 * np.pi / 3)
    group = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for g in frontier:
            for h in (c5, c3):
                x = h @ g
                if not any(np.allclose(x, y, atol=1e-8) for y in group):
                    group.append(x)
                    new.append(x)
        frontier = new
    return group


@pytest.mark.parametrize("m,k,N", [(1, 0, 20), (1, 1, 60), (2, 0, 80), (2, 1, 140), (2, 2, 240), (3, 0, 180), (3, 3, 540), (4, 4, 960)])
def test_counts_rings_valence(m, k, N):
    pos, nb, T = G.fullerene(m, k)
    assert len(pos) == N == 20 * T
    assert all(len(x) == 3 for x in nb)
    sizes = ring_sizes(pos, nb)
    pent, hexa = sizes.count(5), sizes.count(6)
    assert pent == 12 and pent + hexa == len(sizes)                    # exactly 12 pentagons, the rest hexagons
    assert len(sizes) == N // 2 + 2                                    # Euler: F = N/2 + 2 for a cubic polyhedron
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    bonds = np.array([d[i, j] for i in range(N) for j in nb[i]])
    assert bonds.min() > 0.7 * bonds.max()                             # bond lengths stay comparable on the ideal sphere


@pytest.mark.parametrize("m,k", [(1, 1), (2, 2), (3, 0), (2, 1)])
def test_icosahedral_symmetry(m, k):
    pos, nb, T = G.fullerene(m, k)
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    n_ok = 0
    for g in icosahedral_rotations():
        dist, idx = tree.query(pos @ g.T)
        if dist.max() < 1e-6 and len(set(idx)) == len(pos):
            n_ok += 1
    assert n_ok == 60                                                  # rotational group I (chiral (2,1) has no mirrors)
    if m == k or k == 0:                                               # achiral: inversion is also a symmetry (Ih)
        dist, idx = tree.query(-pos)
        assert dist.max() < 1e-6


def test_c60_is_the_truncated_icosahedron():
    """Two bond classes: 30 hexagon-hexagon (short) and 60 pentagon-hexagon (long) bonds; all atoms on one sphere.  (The sphere-projected
    starting geometry has bond lengths ~1.34 / 1.47 A, to be relaxed; only the topology is tested here.)"""
    pos, nb, T = G.fullerene(1, 1, bond=1.42)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    b = np.array([d[i, j] for i in range(60) for j in nb[i] if i < j])
    assert len(b) == 90
    lens = np.unique(np.round(b, 4))
    assert len(lens) == 2 and (np.sum(np.isclose(b, lens[0], atol=1e-3)), np.sum(np.isclose(b, lens[1], atol=1e-3))) == (30, 60)
    R = np.linalg.norm(pos, axis=1)
    assert np.allclose(R, R[0])
