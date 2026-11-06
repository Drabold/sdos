"""Generalized problem A x = lambda B x (B Hermitian positive definite): the operator Y = B^-1 A.

Y is similar to the Hermitian B^-1/2 A B^-1/2, so it has the same real spectrum and Tr T_m(Y) = Tr T_m(B^-1/2 A B^-1/2):
the stochastic moments x^+ T_m(Y) x are unbiased for any x with E[x x^+] = I (the trace is similarity-invariant).
Y itself is not Hermitian, so the doubling identities are not used: one application of Y (one B^-1 solve) per moment.

Two ways to apply B^-1 (``binv``):
  "pcg":       preconditioned conjugate gradients to a user tolerance (default 1e-12 relative residual);
               preconditioners: block-Jacobi (default; inverts the per-site diagonal blocks), "ic" (incomplete Cholesky IC(0), own
               implementation, with a diagonal shift on breakdown), "amg" (pyamg, optional;
               the best performer in our tests), "none", and "ilu" (EXPERIMENTAL: SciPy's incomplete LU is not symmetric,
               so CG convergence is not guaranteed and it can stall; SciPy has no incomplete Cholesky, which is an open item).
  "chebyshev": a FIXED polynomial p(B~) ~ B~^-1 of degree chosen from the condition number and the tolerance, on the
               block-Jacobi-scaled B~ = L^-1 B L^-+ (B_diag = L L^+).  The whole moment recursion is then an exact
               polynomial map (no inner-solve inconsistency between steps).  The scaling is a similarity transform of Y.
"""

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import check_hermitian, is_complex


def is_diagonal(B):
    B = sp.csr_matrix(B)
    off = B - sp.diags(B.diagonal())
    return off.nnz == 0 or abs(off).max() == 0


def block_diagonal(B, bs):
    """Diagonal blocks of B as an array (n/bs, bs, bs)."""
    B = sp.csr_matrix(B)
    n = B.shape[0]
    if n % bs:
        raise ValueError("block_size must divide the matrix dimension")
    blocks = np.zeros((n // bs, bs, bs), dtype=B.dtype)
    coo = B.tocoo()
    m = (coo.row // bs) == (coo.col // bs)
    blocks[coo.row[m] // bs, coo.row[m] % bs, coo.col[m] % bs] = coo.data[m]
    return blocks


def _bsr_from_blocks(blocks):
    nb, bs, _ = blocks.shape
    return sp.bsr_matrix((blocks, np.arange(nb), np.arange(nb + 1)), shape=(nb * bs, nb * bs)).tocsr()


def block_jacobi_inverse(B, bs):
    """Sparse block-diagonal inverse of the diagonal blocks of B (the block-Jacobi preconditioner)."""
    return _bsr_from_blocks(np.linalg.inv(block_diagonal(B, bs)))


def block_cholesky_scaling(B, bs):
    """Sparse block-diagonal L^-1 with (diagonal blocks of B) = L L^+ ; B~ = L^-1 B L^-+ has unit diagonal blocks."""
    L = np.linalg.cholesky(block_diagonal(B, bs))
    return _bsr_from_blocks(np.linalg.inv(L))


def pcg(Bmat, RHS, precond, tol=1e-12, maxiter=None):
    """Batched preconditioned CG for B U = RHS (columns independent; conjugated inner products).

    ``precond``: callable applying M^-1 to a block of columns (or None).  Returns (U, info) with info = dict(iters, max_res).
    """
    RHS = np.asarray(RHS)
    n, k = RHS.shape
    maxiter = maxiter or max(1000, 10 * int(np.sqrt(n)))
    X = np.zeros_like(RHS)
    R = RHS.copy()
    bn = np.linalg.norm(RHS, axis=0)
    bn[bn == 0] = 1.0
    Z = precond(R) if precond else R
    P = Z.copy()
    rz = np.real(np.einsum("ij,ij->j", R.conj(), Z))
    res = np.linalg.norm(R, axis=0) / bn
    active = res > tol
    it = 0
    while active.any() and it < maxiter:
        BP = Bmat @ P
        pBp = np.real(np.einsum("ij,ij->j", P.conj(), BP))
        alpha = np.where(active, rz / np.where(pBp == 0, 1.0, pBp), 0.0)
        X += P * alpha
        R -= BP * alpha
        res = np.linalg.norm(R, axis=0) / bn
        active = res > tol
        Z = precond(R) if precond else R
        rz_new = np.real(np.einsum("ij,ij->j", R.conj(), Z))
        beta = np.where(active, rz_new / np.where(rz == 0, 1.0, rz), 0.0)
        P = Z + P * beta
        rz = rz_new
        it += 1
    if active.any():
        warnings.warn(f"PCG did not reach tol={tol:g} in {maxiter} iterations (max residual {res.max():.2e})", stacklevel=2)
    return X, dict(iters=it, max_res=float(res.max()))


def _make_precond(B, kind, bs):
    if kind in (None, "none"):
        return None
    if kind == "block-jacobi":
        Mi = block_jacobi_inverse(B, bs)
        return lambda R: Mi @ R
    if kind == "ic":
        from .ic import ic_preconditioner
        apply, info = ic_preconditioner(B, verbose=True)
        return apply
    if kind == "ilu":
        warnings.warn("precond='ilu' is experimental: SciPy's incomplete LU is not symmetric, so PCG may converge slowly or "
                      "stall (an incomplete Cholesky is not available in SciPy); prefer 'block-jacobi' or 'amg'", stacklevel=3)
        ilu = spla.spilu(sp.csc_matrix(B), drop_tol=1e-4, fill_factor=10)
        return lambda R: np.column_stack([ilu.solve(R[:, j]) for j in range(R.shape[1])])
    if kind == "amg":
        try:
            import pyamg
        except ImportError as exc:
            raise ImportError("precond='amg' needs the optional dependency pyamg (pip install pyamg)") from exc
        ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(B))
        M = ml.aspreconditioner(cycle="V")
        return lambda R: np.column_stack([M @ R[:, j] for j in range(R.shape[1])])
    raise ValueError("precond must be 'block-jacobi', 'ic', 'amg', 'ilu' or 'none'")


def chebyshev_inverse_coeffs(lo, hi, tol, max_degree=5000):
    """Chebyshev coefficients c_j of the polynomial p(x) ~ 1/x on [lo, hi] with max |1 - x p(x)| <= tol (checked on a grid)."""
    kappa = hi / lo
    q = (np.sqrt(kappa) - 1.0) / (np.sqrt(kappa) + 1.0)
    d = int(np.ceil(np.log(2.0 / tol) / np.log(1.0 / q))) if q > 0 else 2
    d = max(d, 2)
    xs = np.linspace(lo, hi, 4001)
    al, be = 0.5 * (hi - lo), 0.5 * (hi + lo)
    while True:
        c = np.polynomial.chebyshev.chebinterpolate(lambda t: 1.0 / (al * t + be), d)
        t = (xs - be) / al
        err = np.max(np.abs(1.0 - xs * np.polynomial.chebyshev.chebval(t, c)))
        if err <= tol or d >= max_degree:
            return c, d, float(err)
        d = int(d * 1.15) + 1


class GeneralizedOperator:
    """Y = B^-1 A (applied to blocks of columns) with a chosen B^-1 mode; counts solves and iterations.

    ``A``: explicit sparse or LinearOperator (matrix-free A is fine); ``B``: explicit sparse Hermitian positive definite.
    """

    def __init__(self, A, B, binv="pcg", precond="block-jacobi", block_size=1, tol=1e-12, cheb_tol=None, bmargin=0.02,
                 maxiter=None, hermitian_tol=1e-10, check=True, seed=0, verbose=False):
        if not sp.issparse(B):
            raise TypeError("B must be an explicit sparse matrix (Hermitian positive definite)")
        if check:
            check_hermitian(B, tol=hermitian_tol, name="B")
            if sp.issparse(A) or isinstance(A, np.ndarray):
                check_hermitian(A, tol=hermitian_tol, name="A")
        self.n = B.shape[0]
        self.dtype = np.result_type(A.dtype, B.dtype)
        self.binv, self.tol, self.maxiter = binv, tol, maxiter
        self.nsolve = 0
        self.iters = 0
        self.info = dict(binv=binv)
        bs = block_size
        if binv == "pcg":
            self.A, self.B = A, sp.csr_matrix(B)
            self.pre = _make_precond(self.B, precond, bs)
            self.info.update(precond=precond, block_size=bs, tol=tol)
        elif binv == "chebyshev":
            Linv = block_cholesky_scaling(B, bs)
            self.Linv, self.LinvH = Linv, Linv.conj().T.tocsr()
            Bt = (Linv @ sp.csr_matrix(B) @ self.LinvH).tocsr()
            Bt = 0.5 * (Bt + Bt.conj().T)
            self.A = A
            self.B = Bt                                   # B~ (also the inner-product matrix for the Lanczos bounds)
            if sp.issparse(A):
                self.At = (Linv @ sp.csr_matrix(A) @ self.LinvH).tocsr()
            else:
                self.At = None
            k = 1
            lam_max = float(spla.eigsh(Bt, k=1, which="LA", tol=1e-8, return_eigenvectors=False)[0])
            lam_min = float(spla.eigsh(Bt, k=1, which="SA", tol=1e-8, return_eigenvectors=False)[0])
            lo, hi = lam_min * (1.0 - bmargin), lam_max * (1.0 + bmargin)
            if lo <= 0:
                raise ValueError("B is not positive definite (estimated smallest eigenvalue of the scaled B <= 0)")
            ct = cheb_tol if cheb_tol is not None else tol
            self.coef, self.degree, err = chebyshev_inverse_coeffs(lo, hi, ct)
            self.lo, self.hi = lo, hi
            self.info.update(kappa=hi / lo, degree=self.degree, poly_err=err, lam_min=lam_min, lam_max=lam_max, block_size=bs)
            # verify on a random vector that the polynomial inverts B~ to the tolerance (guards against a too-tight bound)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=(self.n, 1)).astype(self.dtype)
            res = np.linalg.norm(Bt @ self._cheb_apply(v) - v) / np.linalg.norm(v)
            self.info["verify_residual"] = float(res)
            if res > 100 * max(ct, 1e-14):
                warnings.warn(f"Chebyshev B^-1: verification residual {res:.2e} exceeds the tolerance; "
                              "increase bmargin or cheb_degree (the spectral bounds of B may be too tight)", stacklevel=2)
        else:
            raise ValueError("binv must be 'pcg' or 'chebyshev'")

    # ---- B^-1 ---------------------------------------------------------------------------------------------
    def _cheb_apply(self, X):
        al, be = 0.5 * (self.hi - self.lo), 0.5 * (self.hi + self.lo)
        tmul = lambda V: (self.B @ V - be * V) / al
        c = self.coef
        b1 = np.zeros_like(X)
        b2 = np.zeros_like(X)
        for j in range(len(c) - 1, 0, -1):
            b1, b2 = c[j] * X + 2.0 * tmul(b1) - b2, b1
        return c[0] * X + tmul(b1) - b2

    def _A_apply(self, X):
        if self.binv == "pcg":
            return self.A @ X
        if self.At is not None:
            return self.At @ X
        return self.Linv @ (self.A @ (self.LinvH @ X))

    def apply(self, X):
        """Y X = B^-1 (A X) (in the scaled coordinates for binv='chebyshev'; a similarity transform of Y)."""
        X = np.asarray(X)
        one = X.ndim == 1
        if one:
            X = X[:, None]
        AX = self._A_apply(X)
        if self.binv == "pcg":
            U, inf = pcg(self.B, AX, self.pre, self.tol, self.maxiter)
            self.iters += inf["iters"]
        else:
            U = self._cheb_apply(AX)
            self.iters += self.degree
        self.nsolve += X.shape[1]
        return U[:, 0] if one else U

    def apply_adjoint(self, X):
        """Y^+ X = A (B^-1 X): the operator whose diagonal gives the Mulliken-projected DOS (Y^+ = B Y B^-1 = A B^-1)."""
        X = np.asarray(X)
        one = X.ndim == 1
        if one:
            X = X[:, None]
        if self.binv == "pcg":
            U, inf = pcg(self.B, X, self.pre, self.tol, self.maxiter)
            self.iters += inf["iters"]
        else:
            U = self._cheb_apply(X)
            self.iters += self.degree
        self.nsolve += X.shape[1]
        out = self._A_apply(U)
        return out[:, 0] if one else out

    __matmul__ = apply


def generalized_bounds(Y, Bmat, n, m=40, seed=0, is_cplx=False):
    """Extremal eigenvalue estimates of Y = B^-1 A by Lanczos in the B inner product <u,v>_B = u^+ B v (Y is B-self-adjoint).

    Returns (lam_min, lam_max, resid): Ritz extremes (they lie inside the spectrum) and the Ritz residual norm; the caller
    widens the interval by the residual plus its safety margin.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n) + (1j * rng.normal(size=n) if is_cplx else 0.0)
    v = v / np.sqrt(np.real(np.vdot(v, Bmat @ v)))
    V = [v]
    BV = [Bmat @ v]
    alpha, beta = [], []
    m = min(m, n)
    for j in range(m):
        w = Y(V[j][:, None])[:, 0]
        a = np.real(np.vdot(BV[j], w))
        w = w - a * V[j] - (beta[-1] * V[j - 1] if j > 0 else 0.0)
        for _ in range(2):                                  # full B-reorthogonalisation
            for i in range(len(V)):
                w = w - np.vdot(BV[i], w) * V[i]
        Bw = Bmat @ w
        b = float(np.sqrt(max(np.real(np.vdot(w, Bw)), 0.0)))
        alpha.append(a)
        beta.append(b)
        if b < 1e-12 or j + 1 == m:
            break
        V.append(w / b)
        BV.append(Bw / b)
    k = len(alpha)
    T = np.diag(alpha) + np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
    th, S = np.linalg.eigh(T)
    res = np.abs(beta[k - 1] * S[-1, :])
    return float(th[0]), float(th[-1]), float(max(res[0], res[-1]))
