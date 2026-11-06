"""Spectral bounds and probing colorings."""

import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def spectral_bounds(H, method="lanczos", verbose=True, v0_seed=12345):
    """Return (Emin, Emax): guaranteed (Gershgorin, explicit matrices only) or estimated (Lanczos).

    Lanczos extremal Ritz values lie *inside* the spectrum, so callers add a safety margin
    (``estimate_dos(margin=...)``); for matrix-free operators only the Lanczos route exists.
    The ARPACK start vector is generated from ``v0_seed`` so the bounds (and hence the whole DOS for a given ``seed``) are
    reproducible; the original sdos.py let ARPACK pick a random start, so repeated runs differed at the 1e-4 level in the bounds.
    """
    n = H.shape[0]
    explicit = sp.issparse(H)
    if explicit:
        absH = abs(H)
        d = np.real(H.diagonal())
        radius = np.asarray(absH.sum(axis=1)).ravel() - np.abs(d)
        gmin, gmax = float((d - radius).min()), float((d + radius).max())
        if method == "gershgorin" or n < 3:
            return gmin, gmax
    elif method == "gershgorin":
        raise ValueError("Gershgorin bounds need an explicit sparse matrix")
    try:
        v0 = np.random.default_rng(v0_seed).normal(size=n)
        if np.issubdtype(H.dtype, np.complexfloating):
            v0 = v0 + 1j * np.random.default_rng(v0_seed + 1).normal(size=n)
        kw = dict(k=1, tol=1e-3, maxiter=max(300, 10 * int(np.sqrt(n))), v0=v0)
        emax = float(np.real(spla.eigsh(H, which="LA", return_eigenvectors=False, **kw)[0]))
        emin = float(np.real(spla.eigsh(H, which="SA", return_eigenvectors=False, **kw)[0]))
        return emin, emax
    except Exception as exc:  # fall back to the safe bounds when they exist
        if not explicit:
            raise
        if verbose:
            print(f"Lanczos bounds failed ({exc}); using Gershgorin", file=sys.stderr)
        return gmin, gmax


def distance_coloring(H, d):
    """Greedy coloring such that same-colored sites are > d hops apart (explicit matrices only)."""
    n = H.shape[0]
    G = H.copy().tocsr()
    G.data = np.ones_like(G.data, dtype=np.int8)
    G = (G + sp.identity(n, dtype=np.int8, format="csr")).astype(bool).astype(np.int8)
    P = G.copy()
    for _ in range(d - 1):
        P = (P @ G).astype(bool).astype(np.int8)
    P = P.tocsr()
    indptr, indices = P.indptr, P.indices
    colors = -np.ones(n, dtype=np.int64)
    mark = -np.ones(n + 1, dtype=np.int64)   # mark[c] = node that saw color c
    order = np.argsort(-np.diff(indptr), kind="stable")  # high degree first
    for i in order:
        nb = indices[indptr[i]:indptr[i + 1]]
        cs = colors[nb]
        mark[cs[cs >= 0]] = i
        c = 0
        while mark[c] == i:
            c += 1
        colors[i] = c
    return colors, int(colors.max()) + 1
