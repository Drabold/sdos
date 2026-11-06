"""Operators and input checks.

A problem matrix may be an explicit SciPy sparse matrix, a dense array, or a matrix-free
``scipy.sparse.linalg.LinearOperator``.  Everything in the package works with *conjugated*
inner products, so real symmetric and complex Hermitian matrices are handled by the same code;
real arithmetic is kept as the fast path whenever the input is real.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class HermitianError(ValueError):
    """Raised when a matrix that must be Hermitian is not, to the requested tolerance."""


def is_complex(A):
    """True if the operator has a complex dtype (explicit data or declared dtype)."""
    if sp.issparse(A):
        return np.iscomplexobj(A.data)
    if isinstance(A, spla.LinearOperator):
        return np.issubdtype(A.dtype, np.complexfloating)
    return np.iscomplexobj(A)


def is_matrix_free(A):
    return isinstance(A, spla.LinearOperator) and not sp.issparse(A)


def check_hermitian(A, tol=1e-10, name="A", nprobe=4, seed=12345, raise_error=True):
    """Return the relative Hermiticity defect of ``A`` and raise/warn if it exceeds ``tol``.

    Explicit matrices: max|A - A^H| / max|A|.  Matrix-free operators: the randomized defect
    |<x, A y> - <A x, y>| / (|x| |A y|) maximised over ``nprobe`` random complex vector pairs
    (a necessary, not sufficient, test; the operator is only applied, never stored).
    """
    if sp.issparse(A) or isinstance(A, np.ndarray):
        As = sp.csr_matrix(A)
        scale = abs(As).max() if As.nnz else 1.0
        defect = (abs(As - As.conj().T).max() / scale) if As.nnz else 0.0
    else:
        rng = np.random.default_rng(seed)
        n = A.shape[0]
        defect = 0.0
        for _ in range(nprobe):
            x = rng.normal(size=n) + 1j * rng.normal(size=n)
            y = rng.normal(size=n) + 1j * rng.normal(size=n)
            if not is_complex(A):
                x, y = x.real.copy(), y.real.copy()
            Ay, Ax = A @ y, A @ x
            num = abs(np.vdot(x, Ay) - np.vdot(Ax, y))
            den = np.linalg.norm(x) * np.linalg.norm(Ay) + 1e-300
            defect = max(defect, num / den)
    if defect > tol:
        msg = f"{name} is not Hermitian: relative defect {defect:.3e} > tolerance {tol:.1e}"
        if raise_error:
            raise HermitianError(msg)
        import warnings
        warnings.warn(msg, stacklevel=2)
    return float(defect)


class ScaledOperator:
    """Ht = (A - b I) / a with ``Ht @ X`` for vectors or blocks of column vectors.

    For an explicit sparse A the scaled CSR matrix is built once (as in the original sdos.py);
    for a matrix-free A the shift and scale are applied on the fly.
    """

    def __init__(self, A, a, b):
        self.a, self.b = float(a), float(b)
        self.shape = A.shape
        self.dtype = A.dtype
        n = A.shape[0]
        if sp.issparse(A):
            self.M = ((A - self.b * sp.identity(n, format="csr")) / self.a).tocsr()
            self.A = None
        else:
            self.M = None
            self.A = A

    def __matmul__(self, X):
        if self.M is not None:
            return self.M @ X
        return (self.A @ X - self.b * X) / self.a

    def traces(self):
        """Exact (Tr Ht / N, Tr Ht^2 / N) in O(nnz); needs an explicit sparse matrix."""
        if self.M is None:
            raise ValueError("exact low moments need an explicit sparse matrix; pass traces=(tr1, tr2)")
        n = self.shape[0]
        tr1 = float(np.real(self.M.diagonal()).sum()) / n
        tr2 = float((np.abs(self.M.data) ** 2).sum()) / n      # Tr Ht^2 = ||Ht||_F^2 for Hermitian Ht
        return tr1, tr2
