#!/usr/bin/env python3
"""
sdos.py -- Order-N density of states for a large sparse Hermitian matrix.

Method
------
Stochastic trace estimation with impartial-in-expectation random vectors
(Drabold & Sankey, PRL 70, 3631 (1993); Skilling; Silver & Roder), combined
with the Kernel Polynomial Method:

  * H is shifted/scaled to Ht with spectrum inside (-1, 1).
  * Chebyshev moments mu_n = Tr T_n(Ht) / N are estimated from random vectors.
    The doubling identities
        T_{2n}   = 2 T_n^2       - T_0
        T_{2n+1} = 2 T_{n+1} T_n - T_1
    give 2K moments from K sparse matrix-vector products per vector.
  * Optional probing (distance-d graph coloring): within one sample, sites of
    the same color are more than d hops apart, so moments n <= d are EXACT
    and short-range fluctuations vanish at all orders.
  * The DOS is reconstructed either with the Jackson kernel (KPM, resolution
    ~ pi*a/M, integrated DOS analytic) or by maximum entropy (--method maxent):
    rho = w(x) exp(sum lambda_n T_n(x)), solved by Newton's method on the convex
    dual, with optional moment uncertainties so noisy stochastic moments are fit
    to within their errors. Maxent extracts much more from few moments
    (typically 20-60); KPM is linear and robust for hundreds or thousands.

Shift-and-invert Lanczos (--method si --si-range E1 E2) gives sharp, detailed
DOS inside a chosen window (defect levels, band tails, gap states): Lanczos on
(H - sigma)^-1 converges the eigenvalues nearest sigma first (stochastic Lanczos
quadrature, sparse LU of H - sigma). Use several shifts for wider windows.

Cost: O(nnz * M * number_of_vectors). Statistical error per state window
falls like 1/sqrt(N_RV * N * eta * rho(E)), so large systems need few vectors.

Input
-----
Text file of triplets, one per line:   i  j  Re(H_ij)  [Im(H_ij)]
  * 0- or 1-based indices (auto-detected, or --base).
  * Either one triangle of the Hermitian matrix (the other is generated as the
    conjugate transpose) or the full matrix (auto-detected, or --storage).
  * Duplicate entries are summed. Lines starting with '#', '%', '!' are ignored.
  * A Matrix Market file (first line '%%MatrixMarket') is read with scipy.io.

Usage example
-------------
  python sdos.py H.dat -M 512 -R 10 --probe 2 -o dos.dat --plot dos.png \
         --electrons 4096 --spin 2

Run `python sdos.py -h` for all options.
"""

import argparse
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def read_matrix(path, base="auto", storage="auto", verbose=True):
    """Read triplets (or Matrix Market) and return a full Hermitian CSR matrix."""
    with open(path, "r") as f:
        first = f.readline()
    if first.startswith("%%MatrixMarket"):
        import scipy.io
        A = sp.coo_matrix(scipy.io.mmread(path))
        sym = "symmetric" in first.lower() or "hermitian" in first.lower()
        rows, cols, vals = A.row, A.col, A.data
        n = A.shape[0]
        if sym:
            storage = "triangle"
        base_used = 0
    else:
        data = np.loadtxt(path, comments=("#", "%", "!"), ndmin=2)
        if data.shape[1] < 3:
            raise ValueError("need at least 3 columns: i j Re(H_ij) [Im(H_ij)]")
        rows = data[:, 0].astype(np.int64)
        cols = data[:, 1].astype(np.int64)
        vals = data[:, 2].astype(np.float64)
        if data.shape[1] >= 4:
            im = data[:, 3]
            if np.any(im != 0.0):
                vals = vals + 1j * im
        if base == "auto":
            base_used = 0 if min(rows.min(), cols.min()) == 0 else 1
        else:
            base_used = int(base)
        rows = rows - base_used
        cols = cols - base_used
        n = int(max(rows.max(), cols.max())) + 1

    upper = np.any(rows < cols)
    lower = np.any(rows > cols)
    if storage == "auto":
        storage = "full" if (upper and lower) else "triangle"

    T = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    T.sum_duplicates()
    if storage == "triangle":
        if upper and lower:
            raise ValueError("--storage triangle given, but entries found in both triangles")
        d = T.diagonal()
        if np.iscomplexobj(d) and np.any(np.abs(d.imag) > 1e-12 * max(1.0, np.abs(d).max())):
            print("warning: diagonal has imaginary parts; they are discarded", file=sys.stderr)
        H = T + T.conj().T - sp.diags(d)
        H = H.tocsr()
        if np.iscomplexobj(H.data):
            H.setdiag(np.real(d))
    else:
        H = T
        herm_err = abs(H - H.conj().T).max() if H.nnz else 0.0
        scale = abs(H).max() if H.nnz else 1.0
        if herm_err > 1e-10 * scale:
            print(f"warning: full-storage matrix is not Hermitian (max |H-H^+| = {herm_err:.3e}); "
                  "using (H+H^+)/2", file=sys.stderr)
            H = 0.5 * (H + H.conj().T)
    H = H.tocsr()
    H.eliminate_zeros()
    if np.iscomplexobj(H.data) and np.all(H.data.imag == 0):
        H = H.real.tocsr()
    if verbose:
        print(f"read {path}: N = {n}, nnz = {H.nnz}, storage = {storage}, "
              f"index base = {base_used}, {'complex' if np.iscomplexobj(H.data) else 'real'}")
    return H


# ---------------------------------------------------------------------------
# Spectral bounds and scaling
# ---------------------------------------------------------------------------

def spectral_bounds(H, method="lanczos", verbose=True):
    """Return (Emin, Emax), guaranteed (Gershgorin) or estimated (Lanczos)."""
    absH = abs(H)
    d = np.real(H.diagonal())
    radius = np.asarray(absH.sum(axis=1)).ravel() - np.abs(d)
    gmin, gmax = float((d - radius).min()), float((d + radius).max())
    if method == "gershgorin" or H.shape[0] < 3:
        return gmin, gmax
    try:
        kw = dict(k=1, tol=1e-4, maxiter=max(300, 10 * int(np.sqrt(H.shape[0]))))
        emax = float(np.real(spla.eigsh(H, which="LA", return_eigenvectors=False, **kw)[0]))
        emin = float(np.real(spla.eigsh(H, which="SA", return_eigenvectors=False, **kw)[0]))
        return emin, emax
    except Exception as exc:  # fall back to the safe bounds
        if verbose:
            print(f"Lanczos bounds failed ({exc}); using Gershgorin", file=sys.stderr)
        return gmin, gmax


# ---------------------------------------------------------------------------
# Probing: distance-d greedy graph coloring
# ---------------------------------------------------------------------------

def distance_coloring(H, d):
    """Greedy coloring such that same-colored sites are > d hops apart."""
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


# ---------------------------------------------------------------------------
# Chebyshev moments
# ---------------------------------------------------------------------------

def make_vectors(n, sample_colors, ncolors, rng, complex_phases):
    """Random vectors for one sample: one vector per color (one if no probing)."""
    if complex_phases:
        z = np.exp(2j * np.pi * rng.random(n))
    else:
        z = rng.choice(np.array([-1.0, 1.0]), size=n)
    if sample_colors is None:
        return z[:, None]
    X = np.zeros((n, ncolors), dtype=z.dtype)
    X[np.arange(n), sample_colors] = z
    return X


def chebyshev_moments(Ht, X, M):
    """Return mu[0..M-1] = sum over columns of x^+ T_n(Ht) x, via doubling."""
    K = (M + 1) // 2           # need v_0 .. v_K
    mu = np.zeros(2 * K + 2)
    v0 = X
    v1 = Ht @ X
    mu0 = np.real(np.vdot(v0, v0))
    mu1 = np.real(np.vdot(v0, v1))
    mu[0], mu[1] = mu0, mu1
    # n = 1: mu2 = 2<v1,v1> - mu0 ; mu3 = 2<v2,v1> - mu1
    vprev, vcur = v0, v1
    for nn in range(1, K + 1):
        vnext = 2.0 * (Ht @ vcur) - vprev
        mu[2 * nn] = 2.0 * np.real(np.vdot(vcur, vcur)) - mu0
        mu[2 * nn + 1] = 2.0 * np.real(np.vdot(vnext, vcur)) - mu1
        vprev, vcur = vcur, vnext
    return mu[:M]


def jackson(M):
    n = np.arange(M)
    q = np.pi / (M + 1)
    return ((M - n + 1) * np.cos(q * n) + np.sin(q * n) / np.tan(q)) / (M + 1)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct(mu, a, b, npts):
    """DOS (normalized to 1) and integrated DOS on an energy grid.

    mu: (..., M) moments normalized so mu[...,0] = 1.
    Returns E (npts,), rho (..., npts), idos (..., npts).
    """
    M = mu.shape[-1]
    g = jackson(M)
    c = g * mu
    theta = np.pi * (np.arange(npts) + 0.5) / npts     # Chebyshev-Gauss nodes
    x = np.cos(theta)[::-1]
    theta = theta[::-1]
    n = np.arange(1, M)
    cosmat = np.cos(np.outer(n, theta))                # (M-1, npts)
    sinmat = np.sin(np.outer(n, theta)) / n[:, None]
    series = c[..., :1] + 2.0 * (c[..., 1:] @ cosmat)
    rho_x = series / (np.pi * np.sqrt(1.0 - x ** 2))
    idos = c[..., :1] * (1.0 - theta / np.pi) - 2.0 * (c[..., 1:] @ sinmat) / np.pi
    E = a * x + b
    return E, rho_x / a, idos


def band_quantities(mu, a, b, n_orb, n_elec, spin, ngrid=20000):
    """Fermi level and band-structure energy from the Jackson-smoothed DOS."""
    M = mu.shape[-1]
    g = jackson(M)
    c = g * mu
    theta = np.linspace(np.pi, 0.0, ngrid)             # x from -1 to 1
    x = np.cos(theta)
    n = np.arange(1, M)
    idos = c[0] * (1.0 - theta / np.pi) - 2.0 * (np.sin(np.outer(theta, n)) / n) @ c[1:] / np.pi
    count = spin * n_orb * idos
    k = np.searchsorted(count, n_elec)
    if k == 0 or k >= ngrid:
        raise ValueError("electron count outside spectrum")
    t = (n_elec - count[k - 1]) / (count[k] - count[k - 1])
    thF = theta[k - 1] + t * (theta[k] - theta[k - 1])
    EF = a * np.cos(thF) + b
    # integrate E rho(E) dE = int (a x + b) [series/pi] dtheta  (x = cos theta)
    th = np.linspace(np.pi, thF, ngrid)
    xs = np.cos(th)
    series = c[0] + 2.0 * np.cos(np.outer(th, n)) @ c[1:]
    integrand = (a * xs + b) * series / np.pi
    Ebs = spin * n_orb * np.trapezoid(integrand, -th)
    return EF, Ebs


# ---------------------------------------------------------------------------
# Maximum entropy reconstruction
# ---------------------------------------------------------------------------
#
# rho(x) = w(x) exp( sum_n lambda_n T_n(x) ),   x in (-1, 1)
#
# maximizes the entropy relative to the default model w(x) subject to the
# Chebyshev moments. lambda is found by Newton's method on the convex dual
#
#   Gamma(lambda) = int w exp(lambda.T) dx - lambda.mu + (1/2) sum sigma_n^2 lambda_n^2
#
# The sigma term ("noisy-moment" Maxent) lets the fit miss a moment by about
# its statistical error instead of chasing noise. With sigma = 0 it is the
# classic exact-moment Maxent (Mead & Papanicolaou; Turek; Drabold & Sankey).
# Substituting x = cos(theta) makes every integral a smooth periodic integral
# in theta, evaluated to spectral accuracy with the midpoint rule.

def _maxent_grid(K, default, ngrid=None):
    if ngrid is None:
        ngrid = max(8192, 64 * K)
    theta = np.pi * (np.arange(ngrid) + 0.5) / ngrid
    dth = np.pi / ngrid
    if default == "chebyshev":      # w(x) = 1 / (pi sqrt(1 - x^2))
        q = np.full(ngrid, 1.0 / np.pi)
    elif default == "flat":         # w(x) = 1/2
        q = 0.5 * np.sin(theta)
    else:
        raise ValueError("default must be 'chebyshev' or 'flat'")
    C = np.cos(np.outer(np.arange(K), theta))
    return theta, dth, q, C


def maxent(mu, sigma=None, default="chebyshev", tol=1e-10, maxiter=500,
           ngrid=None, grid=None):
    """Solve for Maxent Lagrange multipliers. Returns (lam, info)."""
    mu = np.asarray(mu, dtype=float)
    K = len(mu)
    s2 = np.zeros(K) if sigma is None else np.asarray(sigma, dtype=float) ** 2
    theta, dth, q, C = grid if grid is not None else _maxent_grid(K, default, ngrid)
    lam = np.zeros(K)
    lam[0] = np.log(mu[0] / (q.sum() * dth))

    def dual(l):
        expo = np.clip(l @ C, -700.0, 700.0)
        f = q * np.exp(expo)
        return f.sum() * dth - l @ mu + 0.5 * np.sum(s2 * l * l), f

    G, f = dual(lam)
    converged = False
    for it in range(maxiter):
        m = (C @ f) * dth
        g = m - mu + s2 * lam
        gnorm = np.max(np.abs(g))
        if gnorm < tol:
            converged = True
            break
        Hm = (C * f) @ C.T * dth + np.diag(s2)
        # robust Newton step via eigen-decomposition (floor tiny eigenvalues)
        w, V = np.linalg.eigh(Hm)
        w = np.maximum(w, 1e-14 * w.max())
        step = V @ ((V.T @ g) / w)
        t = 1.0
        while True:
            Gn, fn = dual(lam - t * step)
            if Gn <= G - 1e-4 * t * (g @ step) or t < 1e-10:
                break
            t *= 0.5
        lam = lam - t * step
        G, f = Gn, fn
    m = (C @ f) * dth
    info = dict(converged=converged, iterations=it,
                max_moment_residual=float(np.max(np.abs(m - mu))))
    return lam, info


def maxent_density(lam, a, b, default="chebyshev", ngrid=None):
    """DOS per orbital and IDOS on an ascending energy grid from multipliers."""
    K = len(lam)
    theta, dth, q, C = _maxent_grid(K, default, ngrid)
    theta, q, C = theta[::-1], q[::-1], C[:, ::-1]        # ascending x
    f = q * np.exp(np.clip(lam @ C, -700.0, 700.0))       # density per d(theta)
    x = np.cos(theta)
    rho = f / np.sin(theta) / a                           # per unit energy
    idos = np.cumsum(f) * dth - 0.5 * f * dth             # midpoint cumulative
    return a * x + b, rho, idos


def band_quantities_grid(E, rho, idos, n_orb, n_elec, spin):
    """Fermi level and band energy from a DOS tabulated on an ascending grid."""
    count = spin * n_orb * idos
    k = np.searchsorted(count, n_elec)
    if k == 0 or k >= len(E):
        raise ValueError("electron count outside spectrum")
    t = (n_elec - count[k - 1]) / (count[k] - count[k - 1])
    EF = E[k - 1] + t * (E[k] - E[k - 1])
    # band energy: trapezoid-in-energy over IDOS increments (second order)
    Ecell = 0.5 * (E[1:k] + E[:k - 1])
    Ebs = E[0] * idos[0] + np.sum(Ecell * np.diff(idos[:k]))
    Ebs += 0.5 * (E[k - 1] + EF) * (n_elec / (spin * n_orb) - idos[k - 1])
    return EF, spin * n_orb * Ebs


def maxent_fit(mus, a, b, default="chebyshev", noise=1.0, sigma=None,
               jackknife=True, verbose=True):
    """Maxent on averaged moments; optional jackknife over samples.

    mus: (R, K) per-sample moments. noise scales the per-moment standard error
    used in the dual (0 = exact-moment Maxent).
    """
    R, K = mus.shape
    mu = mus.mean(axis=0)
    if sigma is None:
        if R > 1 and noise > 0:
            sigma = noise * mus.std(axis=0, ddof=1) / np.sqrt(R)
        else:
            sigma = np.zeros(K)
    sigma = np.asarray(sigma, dtype=float).copy()
    sigma[0] = 0.0                       # mu_0 = 1 is exact
    grid = _maxent_grid(K, default)
    lam, info = maxent(mu, sigma, default, grid=grid)
    if verbose:
        flag = "converged" if info["converged"] else "NOT converged"
        print(f"Maxent ({default} default, K = {K}): {flag} in {info['iterations']} "
              f"Newton steps, max moment residual {info['max_moment_residual']:.2e}")
        if not info["converged"]:
            print("  hint: use fewer moments (-M) or a larger --maxent-noise", file=sys.stderr)
    lams = None
    if jackknife and R > 1:
        lams = []
        for r in range(R):
            mloo = np.delete(mus, r, axis=0).mean(axis=0)
            lams.append(maxent(mloo, sigma, default, grid=grid)[0])
        lams = np.array(lams)
    return lam, lams, info


# ---------------------------------------------------------------------------
# Shift-and-invert Lanczos: detailed DOS in a chosen energy window
# ---------------------------------------------------------------------------
#
# Lanczos on (H - sigma)^-1 started from a random vector x (|x| = 1) gives the
# Gauss quadrature of the spectral measure of x:
#     sum_k w_k delta(nu - nu_k),   nu_k = Ritz values, w_k = (first component
#     of the k-th tridiagonal eigenvector)^2.
# The eigenvalues of H closest to sigma are the extremal eigenvalues of the
# inverse, so they converge first and are resolved sharply: E_k = sigma + 1/nu_k.
# Averaging over random vectors gives the stochastic-trace DOS (stochastic
# Lanczos quadrature). Cost: one sparse LU of H - sigma, then m triangular
# solves per vector. This resolves individual defect levels, band tails and
# gap states that a global Chebyshev expansion smears over pi*a/M.

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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def estimate_dos(H, M=256, R=10, probe=0, bounds=None, margin=0.01,
                 complex_phases=None, seed=None, npts=None, verbose=True,
                 method="kpm", maxent_default="chebyshev", maxent_noise=1.0):
    """Main entry point.

    Returns dict with E, rho, rho_err, idos, idos_err, mu, mu_samples, a, b, ...
    rho is normalized to 1 (multiply by N for states per energy).
    """
    t0 = time.time()
    n = H.shape[0]
    if complex_phases is None:
        complex_phases = np.iscomplexobj(H.data)
    if bounds is None:
        emin, emax = spectral_bounds(H, verbose=verbose)
    else:
        emin, emax = bounds
    w = emax - emin
    emin -= margin * w
    emax += margin * w
    a = 0.5 * (emax - emin)
    b = 0.5 * (emax + emin)
    Ht = ((H - b * sp.identity(n, format="csr")) / a).tocsr()
    if verbose:
        print(f"spectral window [{emin:.6g}, {emax:.6g}], a = {a:.6g}, b = {b:.6g}  "
              f"({time.time() - t0:.2f} s)")

    colors, ncolors = (None, 1)
    if probe > 0:
        tc = time.time()
        colors, ncolors = distance_coloring(H, probe)
        if verbose:
            print(f"distance-{probe} coloring: {ncolors} colors ({time.time() - tc:.2f} s); "
                  f"moments n <= {probe} are exact")

    rng = np.random.default_rng(seed)
    mus = np.zeros((R, M))
    tm = time.time()
    for r in range(R):
        X = make_vectors(n, colors, ncolors, rng, complex_phases)
        mus[r] = chebyshev_moments(Ht, X, M) / n
    if verbose:
        nmv = R * ncolors * ((M + 1) // 2 + 1)
        print(f"{R} sample(s) x {ncolors} vector(s), M = {M} moments, "
              f"{nmv} mat-vecs ({time.time() - tm:.2f} s)")

    mu = mus.mean(axis=0)
    out = dict(mu=mu, mu_samples=mus, a=a, b=b, N=n, M=M, R=R, ncolors=ncolors,
               method=method, maxent_default=maxent_default)
    if method == "kpm":
        if npts is None:
            npts = max(2 * M, 1000)
        E, rho_s, idos_s = reconstruct(mus, a, b, npts)
        rho = rho_s.mean(axis=0)
        idos = idos_s.mean(axis=0)
        if R > 1:
            rho_err = rho_s.std(axis=0, ddof=1) / np.sqrt(R)
            idos_err = idos_s.std(axis=0, ddof=1) / np.sqrt(R)
        else:
            rho_err = np.full_like(rho, np.nan)
            idos_err = np.full_like(idos, np.nan)
        out["resolution"] = np.pi * a / M
    elif method == "maxent":
        lam, lams, info = maxent_fit(mus, a, b, maxent_default, maxent_noise,
                                     verbose=verbose)
        E, rho, idos = maxent_density(lam, a, b, maxent_default, npts)
        if lams is not None:
            jk = [maxent_density(l, a, b, maxent_default, npts) for l in lams]
            rj = np.array([j[1] for j in jk])
            ij = np.array([j[2] for j in jk])
            fac = np.sqrt((R - 1) / R) * np.sqrt(R)   # jackknife: sqrt((R-1)/R * sum)
            rho_err = rj.std(axis=0) * fac
            idos_err = ij.std(axis=0) * fac
        else:
            rho_err = np.full_like(rho, np.nan)
            idos_err = np.full_like(idos, np.nan)
        out.update(lam=lam, lam_jackknife=lams, maxent_info=info, resolution=np.nan)
    else:
        raise ValueError("method must be 'kpm' or 'maxent'")
    out.update(E=E, rho=rho, rho_err=rho_err, idos=idos, idos_err=idos_err)
    return out


def fermi_and_band_energy(res, n_elec, spin=2.0):
    """E_F and E_BS (with error bars if available) for either method."""
    N, a, b = res["N"], res["a"], res["b"]
    if res["method"] == "kpm":
        val = band_quantities(res["mu"], a, b, N, n_elec, spin)
        samples = ([band_quantities(m, a, b, N, n_elec, spin) for m in res["mu_samples"]]
                   if res["R"] > 1 else None)
        if samples is not None:
            err = np.array(samples).std(axis=0, ddof=1) / np.sqrt(res["R"])
        else:
            err = (np.nan, np.nan)
    else:
        d = res["maxent_default"]
        E, rho, idos = maxent_density(res["lam"], a, b, d)
        val = band_quantities_grid(E, rho, idos, N, n_elec, spin)
        if res["lam_jackknife"] is not None:
            R = res["R"]
            jk = np.array([band_quantities_grid(*maxent_density(l, a, b, d), N, n_elec, spin)
                           for l in res["lam_jackknife"]])
            err = np.sqrt((R - 1) / R * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
        else:
            err = (np.nan, np.nan)
    return val[0], val[1], err[0], err[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("matrix", help="triplet file: i j Re [Im]  (or Matrix Market)")
    p.add_argument("-M", "--moments", type=int, default=256,
                   help="number of Chebyshev moments (resolution ~ pi*halfwidth/M)")
    p.add_argument("-R", "--samples", type=int, default=10,
                   help="number of independent random samples (for averaging and error bars)")
    p.add_argument("--probe", type=int, default=0,
                   help="distance-d coloring for probing (0 = plain random vectors)")
    p.add_argument("--base", choices=["auto", "0", "1"], default="auto")
    p.add_argument("--storage", choices=["auto", "triangle", "full"], default="auto")
    p.add_argument("--emin", type=float, help="lower spectral bound (skip estimation)")
    p.add_argument("--emax", type=float, help="upper spectral bound (skip estimation)")
    p.add_argument("--complex-phases", action="store_true",
                   help="use exp(i phi) vectors even for real H (half the variance, 2x cost)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--npts", type=int, default=None, help="energy grid points")
    p.add_argument("--method", choices=["kpm", "maxent", "si"], default="kpm",
                   help="kpm (Jackson kernel, default), maxent, or si (shift-and-invert "
                        "Lanczos in a window; needs --si-range)")
    p.add_argument("--si-range", type=float, nargs=2, metavar=("E1", "E2"),
                   help="energy window for --method si (e.g. a gap/tail/defect region)")
    p.add_argument("--si-shifts", type=int, default=1,
                   help="number of shifts (LU factorizations) spread over the window")
    p.add_argument("--si-steps", type=int, default=150,
                   help="Lanczos steps per vector at each shift")
    p.add_argument("--si-eta", type=float, default=None,
                   help="Gaussian broadening of Ritz nodes (default: mean node spacing)")
    p.add_argument("--maxent-default", choices=["chebyshev", "flat"], default="chebyshev",
                   help="Maxent default model: 1/(pi sqrt(1-x^2)) (default) or flat")
    p.add_argument("--maxent-noise", type=float, default=1.0,
                   help="scale on moment standard errors in the Maxent dual (0 = fit moments exactly)")
    p.add_argument("--electrons", type=float, default=None,
                   help="number of electrons: report Fermi level and band energy")
    p.add_argument("--spin", type=float, default=2.0, help="spin degeneracy (default 2)")
    p.add_argument("-o", "--output", default="dos.dat")
    p.add_argument("--moments-out", default=None, help="write averaged moments to file")
    p.add_argument("--plot", default=None, help="save a plot (png/pdf)")
    args = p.parse_args(argv)

    H = read_matrix(args.matrix, base=args.base, storage=args.storage)
    bounds = None
    if args.emin is not None and args.emax is not None:
        bounds = (args.emin, args.emax)
    if args.method == "si":
        if args.si_range is None:
            p.error("--method si requires --si-range E1 E2")
        res = shift_invert_dos(H, args.si_range, nshift=args.si_shifts, m=args.si_steps,
                               R=args.samples, eta=args.si_eta, npts=args.npts,
                               seed=args.seed, complex_phases=args.complex_phases or None)
        if args.electrons is not None:
            print("note: --electrons ignored for --method si (window DOS only)", file=sys.stderr)
            args.electrons = None
    else:
        res = estimate_dos(H, M=args.moments, R=args.samples, probe=args.probe,
                           bounds=bounds, complex_phases=args.complex_phases or None,
                           seed=args.seed, npts=args.npts, method=args.method,
                           maxent_default=args.maxent_default, maxent_noise=args.maxent_noise)

    N = res["N"]
    if res["method"] == "kpm":
        recon = f"Jackson resolution ~ {res['resolution']:.4g}"
    elif res["method"] == "si":
        recon = f"shift-invert Lanczos, window {res['erange']}, eta ~ {res['resolution']:.4g}"
    else:
        recon = f"Maxent, {res['maxent_default']} default"
    header = (f"sdos.py  N={N}  M={res['M']}  samples={res['R']}  colors={res['ncolors']}  "
              f"{recon}\n"
              "E   rho(E) [per orbital, integrates to 1]   err   IDOS(E) [fraction]   err")
    np.savetxt(args.output, np.column_stack([res["E"], res["rho"], res["rho_err"],
                                             res["idos"], res["idos_err"]]),
               header=header, fmt="%.10e")
    print(f"wrote {args.output}")
    if args.moments_out:
        np.savetxt(args.moments_out, res["mu"], header="mu_n = Tr T_n(Ht)/N", fmt="%.12e")

    if args.electrons is not None:
        EF, Ebs, dEF, dEbs = fermi_and_band_energy(res, args.electrons, args.spin)
        if res["R"] > 1:
            print(f"E_F = {EF:.6f} +- {dEF:.2g}    E_BS = {Ebs:.6f} +- {dEbs:.2g}  "
                  f"({Ebs / args.electrons:.6f} per electron)")
        else:
            print(f"E_F = {EF:.6f}    E_BS = {Ebs:.6f}  ({Ebs / args.electrons:.6f} per electron)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(res["E"], res["rho"], lw=1)
        if res["R"] > 1:
            ax.fill_between(res["E"], res["rho"] - 2 * res["rho_err"],
                            res["rho"] + 2 * res["rho_err"], alpha=0.3, lw=0)
        ax.set_xlabel("E")
        ax.set_ylabel("DOS per orbital")
        ax.set_title(f"N = {N}, M = {res['M']}, {res['R']} sample(s), {res['method'].upper()}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
