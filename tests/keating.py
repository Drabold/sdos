"""Keating force-constant matrix of a periodic diamond supercell (analytic gradient, central-difference Hessian)."""
import numpy as np
import scipy.sparse as sp


def diamond(nc=3, a=5.43):
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    basis = np.vstack([fcc, fcc + 0.25])
    cells = np.array([[i, j, k] for i in range(nc) for j in range(nc) for k in range(nc)])
    X = ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)) * a
    return X, np.full(3, nc * a)


def keating_model(nc=2, alpha=1.0, beta=0.285, a=5.43):
    """(X0, forces(x)) of the periodic Keating diamond supercell; forces = -gradient."""
    X0, L = diamond(nc, a)
    n = len(X0)
    d = a * np.sqrt(3) / 4
    mic = lambda v: v - L * np.rint(v / L)
    D = np.linalg.norm(mic(X0[:, None] - X0[None]), axis=-1)
    nb = [np.nonzero(np.abs(D[i] - d) < 1e-6 * d)[0] for i in range(n)]
    assert all(len(x) == 4 for x in nb)
    ca, cb = 3 * alpha / (16 * d * d), 3 * beta / (8 * d * d)

    def grad(X):
        g = np.zeros_like(X)
        for i in range(n):
            u = [mic(X[j] - X[i]) for j in nb[i]]
            for jj, j in enumerate(nb[i]):
                if i < j:
                    r = u[jj]
                    f = 4 * ca * (r @ r - d * d) * r
                    g[j] += f
                    g[i] -= f
            for p in range(4):
                for q in range(p + 1, 4):
                    c = 2 * cb * (u[p] @ u[q] + d * d / 3)
                    g[nb[i][p]] += c * u[q]
                    g[nb[i][q]] += c * u[p]
                    g[i] -= c * (u[p] + u[q])
        return g

    return X0, (lambda x: -grad(np.asarray(x).reshape(-1, 3)))


def keating_hessian(nc=2, alpha=1.0, beta=0.285, a=5.43, h=1e-5):
    X0, forces = keating_model(nc, alpha, beta, a)
    n = len(X0)
    grad = lambda x: -forces(x)
    Hs = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        X = X0.copy().ravel()
        X[k] += h
        gp = grad(X.reshape(-1, 3)).ravel()
        X[k] -= 2 * h
        gm = grad(X.reshape(-1, 3)).ravel()
        Hs[:, k] = (gp - gm) / (2 * h)
    Hs = 0.5 * (Hs + Hs.T)
    Hs[np.abs(Hs) < 1e-9] = 0.0
    return sp.csr_matrix(Hs)
