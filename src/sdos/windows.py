"""Shift-and-invert Lanczos: detailed DOS in a chosen energy window (stochastic Lanczos quadrature).

NOTE: this route is NOT O(N).  It needs a sparse LU factorization of (A - sigma), whose fill grows
superlinearly for 3D connectivity; it is meant for windows (band tails, gap and defect states).
Ported unchanged from sdos.py (regression-tested); the generalized problem is added in a later phase.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _si_lanczos(lu, n, m, R, rng, complex_phases):
    """Block (column-by-column) Lanczos on A^-1 with full reorthogonalization.

    Returns alpha (k, R), beta (k-1, R) of the R tridiagonal matrices.
    """
    dtype = np.complex128 if complex_phases else np.float64
    if m * n * R * np.dtype(dtype).itemsize > 4e9:
        raise MemoryError("shift-invert Lanczos basis would exceed 4 GB; "
                          "reduce --si-steps or -R")
    if complex_phases:
        X = np.exp(2j * np.pi * rng.random((n, R)))
    else:
        X = rng.choice(np.array([-1.0, 1.0]), size=(n, R))
    Q = np.empty((m, n, R), dtype=dtype)
    Q[0] = X / np.sqrt(n)
    alpha = np.zeros((m, R))
    beta = np.zeros((m, R))
    k = m
    for j in range(m):
        w = lu.solve(Q[j])
        alpha[j] = np.real(np.einsum("ir,ir->r", Q[j].conj(), w))
        w = w - alpha[j] * Q[j]
        if j > 0:
            w = w - beta[j - 1] * Q[j - 1]
        for _ in range(2):                       # full reorthogonalization
            for i in range(j + 1):
                w = w - np.einsum("ir,ir->r", Q[i].conj(), w) * Q[i]
        beta[j] = np.sqrt(np.real(np.einsum("ir,ir->r", w.conj(), w)))
        if j + 1 == m:
            break
        if beta[j].min() < 1e-12:                # invariant subspace found
            k = j + 1
            break
        Q[j + 1] = w / beta[j]
    return alpha[:k], beta[:k - 1], k


def si_nodes(H, sigma, m=150, R=4, seed=None, complex_phases=None):
    """Gauss-quadrature nodes/weights of the spectral measure at shift sigma.

    Returns list of R tuples (E_k, w_k); sum_k w_k = 1 for each.
    """
    n = H.shape[0]
    if complex_phases is None:
        complex_phases = np.iscomplexobj(H.data)
    rng = np.random.default_rng(seed)
    A = (H - sigma * sp.identity(n, format="csr", dtype=H.dtype)).tocsc()
    lu = spla.splu(A)
    alpha, beta, k = _si_lanczos(lu, n, min(m, n), R, rng, complex_phases)
    from scipy.linalg import eigh_tridiagonal
    out = []
    for r in range(R):
        nu, V = eigh_tridiagonal(alpha[:, r], beta[:, r])
        keep = np.abs(nu) > 1e-300
        out.append((sigma + 1.0 / nu[keep], (V[0] ** 2)[keep]))
    return out


def shift_invert_dos(H, erange, nshift=1, m=150, R=4, eta=None, npts=None,
                     seed=None, complex_phases=None, verbose=True):
    """DOS on erange = (E1, E2) from shift-and-invert Lanczos.

    nshift shifts are spread evenly over the range; each one is responsible for
    its own (Voronoi) sub-window. A Gaussian of width eta (default: about the
    mean Ritz-node spacing in the sub-window) broadens the quadrature nodes.
    Returns the same dict keys as estimate_dos (rho normalized to 1 per orbital).
    """
    t0 = time.time()
    n = H.shape[0]
    e1, e2 = float(erange[0]), float(erange[1])
    if not e2 > e1:
        raise ValueError("energy range must have E2 > E1")
    if npts is None:
        npts = 1000
    E = np.linspace(e1, e2, npts)
    dE = E[1] - E[0]
    edges = np.linspace(e1, e2, nshift + 1)
    sigmas = 0.5 * (edges[:-1] + edges[1:])
    rho_s = np.zeros((R, npts))
    etas, nin, ms = [], [], []
    for s, sig in enumerate(sigmas):
        nodes = si_nodes(H, sig, m=m, R=R, seed=None if seed is None else seed + s,
                         complex_phases=complex_phases)
        lo, hi = edges[s], edges[s + 1]
        last = s == nshift - 1
        sel = (E >= lo) & ((E <= hi) if last else (E < hi))
        n_in = np.mean([np.sum((Ek >= lo) & (Ek <= hi)) for Ek, _ in nodes])
        eta_s = eta if eta is not None else max((hi - lo) / max(n_in, 1.0), 2 * dE)
        etas.append(eta_s); nin.append(n_in); ms.append(len(nodes[0][0]))
        for r, (Ek, wk) in enumerate(nodes):
            near = np.abs(Ek - sig) < (hi - lo) / 2 + 8 * eta_s
            z = (E[sel][:, None] - Ek[near][None, :]) / eta_s
            rho_s[r, sel] = (np.exp(-0.5 * z * z) @ wk[near]) / (eta_s * np.sqrt(2 * np.pi))
        if verbose:
            print(f"  shift {sigma_fmt(sig)}: window [{lo:.5g}, {hi:.5g}], {len(nodes[0][0])} "
                  f"Lanczos steps, ~{n_in:.1f} Ritz nodes in window, eta = {eta_s:.3g}")
    rho = rho_s.mean(axis=0)
    rho_err = (rho_s.std(axis=0, ddof=1) / np.sqrt(R)) if R > 1 else np.full(npts, np.nan)
    idos_s = np.cumsum(rho_s, axis=1) * dE
    idos = idos_s.mean(axis=0)
    idos_err = (idos_s.std(axis=0, ddof=1) / np.sqrt(R)) if R > 1 else np.full(npts, np.nan)
    if verbose:
        print(f"shift-invert Lanczos: {nshift} LU factorization(s), {R} vector(s), "
              f"states in window = {idos[-1] * n:.1f}"
              + (f" +- {idos_err[-1] * n:.1f}" if R > 1 else "") + f" ({time.time() - t0:.2f} s)")
    return dict(E=E, rho=rho, rho_err=rho_err, idos=idos, idos_err=idos_err, N=n,
                M=ms[0], R=R, ncolors=1, method="si", sigmas=sigmas, eta=np.array(etas),
                resolution=float(np.mean(etas)), erange=(e1, e2))


def sigma_fmt(x):
    return f"{x:.6g}"
