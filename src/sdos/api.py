"""The simple front door: ``sdos.dos(A)`` gives the density of states of a big sparse matrix with sensible defaults;
every advanced option of :func:`sdos.estimate_dos` is available as a keyword when needed."""

import os

import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .bounds import spectral_bounds
from .dos import estimate_dos
from .generalized import GeneralizedOperator, generalized_bounds, is_diagonal
from .io import read_matrix
from .operators import is_complex


def _as_matrix(A, name):
    if isinstance(A, (str, os.PathLike)):
        return read_matrix(A, verbose=False)
    if isinstance(A, np.ndarray) and A.ndim == 2:
        return sp.csr_matrix(A)
    if sp.issparse(A):
        return A.tocsr()
    if isinstance(A, spla.LinearOperator):
        return A
    raise TypeError(f"{name}: expected a file path, a 2-D array, a SciPy sparse matrix or a LinearOperator")


def dos(A, B=None, *, M=None, R=None, resolution=None, quick=False, exact_moments="auto", method="kpm", seed=None,
        verbose=False, **options):
    """Density of states of a large sparse Hermitian matrix A (or of A x = lambda B x).

    ``A`` and ``B`` can be a file name (triplets, Matrix Market or .npz), a dense/sparse matrix, or (A only) a matrix-free
    ``LinearOperator``; a 1-D ``B`` is a diagonal B (e.g. masses).  Real or complex Hermitian is detected automatically.

    Defaults: 256 moments and 10 random vectors (``quick=True``: 128 and 4), Jackson kernel, the first 4 moments exact (cheap local
    method; set ``exact_moments=0`` to turn it off; the default is 0 for method="maxent" with R = 1), PCG with block-Jacobi for a generalized B.  Give an energy ``resolution`` instead
    of ``M`` to choose the number of moments from it.  Anything else of :func:`sdos.estimate_dos` can be passed as a keyword
    (``method='maxent'``, ``vectors='impartial'``, ``binv='chebyshev'``, ``probe=2`` ...).

    Returns a :class:`~sdos.DOSResult`: ``res.E``, ``res.rho`` (per orbital, integrates to 1), ``res.rho_err``, ``res.idos``,
    ``res.plot()``, ``res.save(path)``.  Multiply ``rho`` by N for states per unit energy.
    """
    A = _as_matrix(A, "A")
    if B is not None and not (isinstance(B, np.ndarray) and B.ndim == 1):
        B = _as_matrix(B, "B")
    if R is None:
        R = 4 if quick else 10
    explicit = sp.issparse(A)
    diag_B = B is None or (isinstance(B, np.ndarray) and B.ndim == 1) or (sp.issparse(B) and is_diagonal(B))
    if exact_moments == "auto":
        exact_moments = 4 if (explicit and diag_B and method == "kpm" or explicit and diag_B and method == "maxent") else 0
        if method == "maxent" and R < 2:
            exact_moments = 0              # Maxent with one vector and spliced moments does not converge (see estimate_dos); no substitution by default
    if M is None and resolution is not None:
        # window width from the bounds, then M ~ pi a / resolution
        if B is None or diag_B:
            if B is not None:
                d = np.real(B.diagonal()) if sp.issparse(B) else np.asarray(B, float)
                Aw = sp.diags(1 / np.sqrt(d)) @ A @ sp.diags(1 / np.sqrt(d))
            else:
                Aw = A
            lo, hi = spectral_bounds(Aw, verbose=False)
        else:
            op = GeneralizedOperator(A, B, binv=options.get("binv", "pcg"), tol=options.get("binv_tol", 1e-12),
                                     block_size=options.get("block_size", 1), check=False)
            lo, hi, rs = generalized_bounds(op.apply, op.B, A.shape[0], seed=0, is_cplx=bool(is_complex(A)))
            lo, hi = lo - rs, hi + rs
        M = int(np.ceil(np.pi * 0.5 * (hi - lo) * 1.02 / resolution))
    if M is None:
        M = 128 if quick else 256
    if method == "maxent":
        options.setdefault("maxent_default", "flat")          # uniform default model; measured equal in accuracy and conditioning to the alternatives (docs/theory.md)
    if M >= A.shape[0]:
        warnings.warn(f"M={M} >= N={A.shape[0]}: the resolution pi*a/M is finer than the mean level spacing, the density resolves individual levels and sampling speckle; "
                      "for such a small matrix dense diagonalization (numpy.linalg.eigvalsh) is cheaper and exact.", stacklevel=2)
    if exact_moments and exact_moments >= M:
        exact_moments = max(0, M - 1)
    return estimate_dos(A, B=B, M=M, R=R, method=method, seed=seed, verbose=verbose, exact_moments=exact_moments, **options)
