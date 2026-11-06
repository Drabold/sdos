"""Eigenpairs in an energy window, with a per-state inverse participation ratio (IPR).

Two routes, for the standard problem A x = lambda x and the generalized problem A x = lambda B x (B Hermitian positive definite):

* ``window_eigs(..., method="shift-invert")``: ``scipy.sparse.linalg.eigsh(A, k, M=B, sigma=sigma)`` around the window centre, with
  k increased until the farthest returned eigenvalue lies outside the window (so the window is complete).  **NOT O(N)**: it
  factorizes A - sigma B, whose fill grows superlinearly for 3D connectivity.  Explicit sparse matrices only.
* ``window_eigs(..., method="filter")``: Chebyshev-filtered subspace iteration, the scalable fallback.  A Jackson-damped Chebyshev
  expansion of the window indicator is applied to a block of vectors (matrix products only; for B != 1 each product also applies
  B^-1 through the PCG / Chebyshev machinery of ``GeneralizedOperator``), followed by a Rayleigh-Ritz step in the B inner product.
  The subspace size comes from the KPM estimate of the number of states in the window, which is also used to warn if the
  converged set looks incomplete (spare block vectors give unconverged Ritz values; these are dropped, not reported).  Cost: block size x degree x iterations
  products, plus the dense block algebra, O(N k^2).

IPR convention: with orbital weights w_i = Re conj(psi_i) (B psi)_i / (psi^+ B psi) (the Mulliken weights; w_i = |psi_i|^2 for B = 1),
IPR = sum_i w_i^2 (per site if ``block_size`` > 1: weights of the orbitals of a site are summed first).  IPR * N_sites ~ 1 for an extended
state and ~ N_sites / (number of sites it lives on) for a localized one.
"""

import time
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .dos import estimate_dos
from .generalized import GeneralizedOperator, generalized_bounds
from .operators import is_complex
from .reconstruct import jackson


class WindowResult(dict):
    """E, V (columns), ipr, residual, plus info; attribute access."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc


def mulliken_weights(V, B=None):
    """Orbital weights w[:, s] = Re conj(v_is) (B v_s)_i / (v_s^+ B v_s) of the columns of V."""
    V = np.asarray(V)
    BV = V if B is None else B @ V
    w = np.real(np.conj(V) * BV)
    return w / w.sum(axis=0, keepdims=True)


def ipr(V, B=None, block_size=1):
    """IPR of every column of V (Mulliken weights; summed over the ``block_size`` orbitals of a site first)."""
    w = mulliken_weights(V, B)
    if block_size > 1:
        n = w.shape[0]
        w = w.reshape(n // block_size, block_size, -1).sum(axis=1)
    return (w ** 2).sum(axis=0)


def _count_estimate(A, B, window, bounds, seed):
    r = estimate_dos(A, B=B, M=256, R=6, seed=seed, bounds=bounds, verbose=False, exact_moments=0 if B is not None else 4,
                     check_bounds=False, diagonal_shortcut=True)
    idos = np.interp([window[0], window[1]], r.E, r.idos)
    err = np.interp([window[0], window[1]], r.E, r.idos_err)
    return float(r.N * (idos[1] - idos[0])), float(r.N * np.hypot(err[0], err[1]))


def _shift_invert(A, B, window, k0, block_size, seed):
    if not sp.issparse(A):
        raise ValueError("shift-invert needs an explicit sparse A")
    e1, e2 = window
    sigma = 0.5 * (e1 + e2)
    half = 0.5 * (e2 - e1)
    n = A.shape[0]
    k = int(min(max(k0, 6), n - 2))
    while True:
        # complex Hermitian: eigsh handles it (use the Hermitian solver); B as M
        w, V = spla.eigsh(sp.csc_matrix(A), k=k, M=(sp.csc_matrix(B) if B is not None else None), sigma=sigma, which="LM")
        if np.max(np.abs(w - sigma)) > half or k >= n - 2:
            break
        k = int(min(2 * k, n - 2))
    keep = (w >= e1) & (w <= e2)
    order = np.argsort(w[keep])
    return w[keep][order], V[:, keep][:, order], dict(k=k, method="shift-invert", complete=bool(np.max(np.abs(w - sigma)) > half))


def _window_filter_coeffs(x1, x2, degree):
    """Jackson-damped Chebyshev coefficients of the indicator of [x1, x2] (x in [-1, 1])."""
    t1, t2 = np.arccos(np.clip(x2, -1, 1)), np.arccos(np.clip(x1, -1, 1))        # theta of the upper / lower edges
    j = np.arange(1, degree + 1)
    c = np.empty(degree + 1)
    c[0] = (t2 - t1) / np.pi
    c[1:] = 2.0 * (np.sin(j * t2) - np.sin(j * t1)) / (j * np.pi)
    return c * jackson(degree + 1)


def _filter_block(Xop, V, c):
    """p(X) V by the Chebyshev three-term recurrence (Clenshaw-free forward form)."""
    out = c[0] * V
    Tprev, Tcur = V, Xop(V)
    out = out + c[1] * Tcur
    for j in range(2, len(c)):
        Tnext = 2.0 * Xop(Tcur) - Tprev
        out = out + c[j] * Tnext
        Tprev, Tcur = Tcur, Tnext
    return out


def _filtered(A, B, window, bounds, k, degree, maxiter, tol, seed, margin, binv, precond, block_size, verbose):
    n = A.shape[0]
    cplx = bool(is_complex(A) or (B is not None and np.iscomplexobj(B.data)))
    dtype = np.complex128 if cplx else np.float64
    if B is None:
        Yop = lambda V: A @ V
        Bm = None
        inner_B = None
        if bounds is None:
            from .bounds import spectral_bounds
            lo, hi = spectral_bounds(A, verbose=False)
        else:
            lo, hi = bounds
        Amul = lambda V: A @ V
    else:
        op = GeneralizedOperator(A, B, binv=binv, precond=precond, block_size=block_size, check=False)
        if binv != "pcg":
            raise ValueError("the filter route needs the original coordinates: use binv='pcg' for the generalized problem")
        Yop = op.apply
        Bm = op.B
        Amul = lambda V: A @ V
        if bounds is None:
            lo, hi, rs = generalized_bounds(op.apply, op.B, n, m=40, seed=seed, is_cplx=cplx)
            lo, hi = lo - rs, hi + rs
        else:
            lo, hi = bounds
    w = hi - lo
    lo, hi = lo - margin * w, hi + margin * w
    a, b = 0.5 * (hi - lo), 0.5 * (hi + lo)
    Xop = lambda V: (Yop(V) - b * V) / a
    c = _window_filter_coeffs((window[0] - b) / a, (window[1] - b) / a, degree)
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, k)).astype(dtype)
    if cplx:
        V = V + 1j * rng.normal(size=(n, k))
    nprod = 0
    res = np.full(k, np.inf)
    for it in range(maxiter):
        V = _filter_block(Xop, V, c)
        nprod += degree * k
        # B-orthonormalise (Cholesky QR in the B inner product, SVD fallback) and Rayleigh-Ritz
        BV = V if Bm is None else Bm @ V
        G = V.conj().T @ BV
        G = 0.5 * (G + G.conj().T)
        try:
            R = sla.cholesky(G, lower=False)
            V = sla.solve_triangular(R, V.T, trans="T", lower=False).T if not cplx else sla.solve_triangular(R, V.conj().T, trans="C", lower=False).conj().T
        except sla.LinAlgError:
            u, s, _ = np.linalg.svd(V, full_matrices=False)
            V = u
            BV = V if Bm is None else Bm @ V
        AV = Amul(V)
        BV = V if Bm is None else Bm @ V
        Hs = V.conj().T @ AV
        Ss = V.conj().T @ BV
        theta, Q = sla.eigh(0.5 * (Hs + Hs.conj().T), 0.5 * (Ss + Ss.conj().T))
        V = V @ Q
        AV, BV = AV @ Q, BV @ Q
        R = AV - BV * theta[None, :]
        res = np.linalg.norm(R, axis=0) / (a * np.maximum(np.linalg.norm(BV, axis=0), 1e-300))
        inside = (theta >= window[0]) & (theta <= window[1])
        if verbose:
            print(f"  filter iteration {it + 1}: {inside.sum()} Ritz values in the window, max residual {res[inside].max() if inside.any() else np.nan:.2e}")
        if inside.any() and res[inside].max() < tol:
            break
    return theta, V, res, dict(method="filter", degree=degree, iterations=it + 1, block=k, products=nprod, bounds=(lo, hi))


def window_eigs(A, B=None, window=(0.0, 1.0), method="auto", k=None, degree=300, maxiter=30, tol=1e-8, seed=0, bounds=None,
                margin=0.01, binv="pcg", precond="block-jacobi", block_size=1, verbose=False):
    """Eigenpairs (E, V) of A x = lambda B x with E in ``window``, plus IPR and residuals.

    ``method``: 'shift-invert' (exact within the window, not O(N)), 'filter' (Chebyshev-filtered subspace iteration, scalable; only
    converged states are returned, and the window is guaranteed complete only away from its edges, by ``info['edge_width']`` ~ 2 pi a / degree) or
    'auto' (shift-invert for explicit sparse A below ~20000 rows, else filter).  ``k``: number of states to look for (default from a
    KPM estimate, with margin).  Returns a :class:`WindowResult` with E (sorted), V, ipr, residual and ``info``.
    ``block_size``: orbitals per site for the site-resolved IPR (and for the block-Jacobi preconditioner of the generalized filter).
    """
    t0 = time.time()
    if isinstance(B, np.ndarray) and B.ndim == 1:
        B = sp.diags(B).tocsr()
    n = A.shape[0]
    if method == "auto":
        method = "shift-invert" if (sp.issparse(A) and n <= 20000) else "filter"
    n_est, n_err = _count_estimate(A, B, window, bounds, seed)
    if k is None:
        k = int(np.ceil(1.3 * n_est)) + 12
    if method == "shift-invert":
        E, V, info = _shift_invert(A, B, window, k, block_size, seed)
        BV = V if B is None else B @ V
        resid = np.linalg.norm(A @ V - BV * E[None, :], axis=0) / np.maximum(np.linalg.norm(BV, axis=0), 1e-300)
    elif method == "filter":
        theta, V, res, info = _filtered(A, B, window, bounds, k, degree, maxiter, tol, seed, margin, binv, precond, block_size, verbose)
        inside = (theta >= window[0]) & (theta <= window[1])
        lo_b, hi_b = info["bounds"]
        delta = 2.0 * np.pi * 0.5 * (hi_b - lo_b) / degree                 # width of the filter's transition band at the window edges
        conv = res < 10 * tol
        keep = inside & conv
        E, V, resid = theta[keep], V[:, keep], res[keep]
        info.update(edge_width=float(delta), dropped_unconverged=int((inside & ~conv).sum()))
    else:
        raise ValueError("method must be 'auto', 'shift-invert' or 'filter'")
    info.update(n_estimated=n_est, n_estimated_err=n_err, n_found=len(E), seconds=time.time() - t0)
    if method == "filter" and len(E) < n_est - 3.0 * n_err - 3.0:
        warnings.warn(f"filtered subspace iteration found {len(E)} converged states but the KPM estimate is {n_est:.1f} +- {n_err:.1f}: "
                      "the window may be incomplete (states near the edges are returned only if converged); raise `degree`, `maxiter` "
                      "or `k`", stacklevel=2)
    if method == "filter" and len(E) >= k - 1:
        warnings.warn("the subspace may be too small for the window (block full of in-window Ritz values): raise k", stacklevel=2)
    return WindowResult(E=E, V=V, ipr=ipr(V, B, block_size) if len(E) else np.array([]), residual=resid, info=info,
                        window=tuple(window), block_size=block_size)


def count_states_below(A, sigma, B=None):
    """Exact number of eigenvalues of A x = lambda B x below ``sigma`` (Sylvester's law of inertia).

    LDL^+ of (A - sigma B) by SuperLU with a symmetric permutation and no pivoting; the number of negative pivots equals the number of eigenvalues
    below sigma (B positive definite).  Real symmetric or complex Hermitian, explicit sparse matrices.  Raises ArithmeticError for a shift where SuperLU had to swap
    rows or met a zero pivot (callers nudge sigma).  Cost: one sparse factorization (not O(N): fill grows
    superlinearly in 3D, mildly for surface-like connectivity such as fullerene shells).  Raises if sigma coincides with an eigenvalue (zero pivot)."""
    n = A.shape[0]
    Bm = sp.identity(n, format="csc") if B is None else sp.csc_matrix(B)
    M = (sp.csc_matrix(A) - sigma * Bm).tocsc()
    lu = spla.splu(M, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise ArithmeticError(f"sigma = {sigma!r}: the factorization swapped rows (a vanishing diagonal entry), so the inertia count is not valid; "
                              "choose a slightly different shift")
    d = np.real(lu.U.diagonal())
    scale = max(abs(M).max(), 1e-300)
    if np.any(np.abs(d) < 1e-13 * scale):
        raise ArithmeticError(f"sigma = {sigma!r} is (numerically) an eigenvalue: choose a slightly different shift")
    return int((d < 0).sum())


def _kth_eigenvalue(A, k, lo, hi, B=None, tol=1e-9, max_steps=100):
    """The k-th smallest eigenvalue (1-based) by bisection on the inertia count: count(sigma) >= k exactly when sigma > lambda_k."""
    steps = 0
    while hi - lo > tol and steps < max_steps:
        mid = 0.5 * (lo + hi)
        try:
            c = count_states_below(A, mid, B)
        except ArithmeticError:
            mid += 1.234e-7
            c = count_states_below(A, mid, B)
        if c >= k:
            hi = mid
        else:
            lo = mid
        steps += 1
    return 0.5 * (lo + hi), hi, steps


def homo_lumo(A, n_occupied, B=None, bounds=None, tol=1e-9):
    """Exact HOMO, LUMO, gap and midpoint Fermi level when the lowest ``n_occupied`` states are occupied.

    The n-th eigenvalue is found by bisection on the exact inertia count (``count_states_below``): no eigensolver, no assumption about the gap size or the
    degeneracy of the levels.  Accurate to ``tol``.  Returns dict(homo, lumo, gap, fermi, n_steps)."""
    n = A.shape[0]
    if bounds is None:
        if B is not None:
            lo, hi = generalized_bounds(GeneralizedOperator(A, B, check=False).apply, sp.csr_matrix(B), n)[:2]
        else:
            from .bounds import spectral_bounds
            lo, hi = spectral_bounds(A, "gershgorin", verbose=False)
    else:
        lo, hi = bounds
    lo, hi = lo - 1e-3, hi + 1e-3
    homo, hi1, s1 = _kth_eigenvalue(A, n_occupied, lo, hi, B, tol)
    lumo, _, s2 = _kth_eigenvalue(A, n_occupied + 1, hi1 - 1e-6, hi, B, tol)
    return dict(homo=float(homo), lumo=float(lumo), gap=float(lumo - homo), fermi=float(0.5 * (homo + lumo)), n_steps=s1 + s2)
