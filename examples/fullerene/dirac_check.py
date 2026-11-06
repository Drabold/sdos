import numpy as np
from sdos.interfaces import xu_carbon as X
a = 1.42 * 3 ** 0.5
a1, a2 = np.array([a, 0, 0]), np.array([a / 2, a * 3 ** 0.5 / 2, 0]); tau = [np.zeros(3), np.array([a / 2, a / (2 * 3 ** 0.5), 0])]
terms = []
for al in range(2):
    for be in range(2):
        for n1 in range(-4, 5):
            for n2 in range(-4, 5):
                Rv = n1 * a1 + n2 * a2; d = tau[be] + Rv - tau[al]
                if 1e-9 < np.linalg.norm(d) < X.RM: terms.append((al, be, Rv, X.hopping_blocks(d[None])[0]))
b1 = 2 * np.pi * np.array([1 / a, -1 / (a * 3 ** 0.5), 0]); b2 = 2 * np.pi * np.array([0, 2 / (a * 3 ** 0.5), 0])
def bands(k):
    H = np.zeros((8, 8), complex)
    for al in range(2): H[4*al:4*al+4, 4*al:4*al+4] += np.diag([X.EPS_S, X.EPS_P, X.EPS_P, X.EPS_P])
    for al, be, Rv, B in terms: H[4*al:4*al+4, 4*be:4*be+4] += B * np.exp(1j * k @ Rv)
    return np.linalg.eigvalsh(H)
best = min(((bands(f1 * b1 + f2 * b2)[4] - bands(f1 * b1 + f2 * b2)[3], f1, f2) for f1 in (1/3, 2/3) for f2 in (1/3, 2/3)))
g, f1, f2 = best; K = f1 * b1 + f2 * b2
print("K point gap", g)
hv = []
for ang in np.linspace(0, np.pi, 7):
    q = 1e-3 * np.array([np.cos(ang), np.sin(ang), 0]); e = bands(K + q); hv.append((e[4] - e[3]) / 2 / 1e-3)
hv = np.mean(hv); print("hbar v_F = %.3f eV A" % hv, " (v_F = %.3g m/s)" % (hv * 1e-10 / 6.582e-16))
A_at = 3 * 3 ** 0.5 / 2 * 1.42 ** 2 / 2   # graphene area per atom
gaps = {60: 1.537, 240: 1.015, 540: 0.713, 960: 0.549, 1500: 0.446, 2160: 0.374, 2940: 0.323, 3840: 0.284, 4860: 0.253, 6000: 0.228}
print(" N    gap   R_eq(A)  gap*R/hbar v")
for N, g in gaps.items():
    R = np.sqrt(N * A_at / (4 * np.pi)); print(f"{N:5d} {g:.3f} {R:7.2f}   {g * R / hv:.3f}")
