"""Incomplete Cholesky IC(0) for Hermitian positive-definite sparse matrices (real or complex), with a diagonal shift on breakdown.

B ~ L L^+ with L restricted to the sparsity pattern of the lower triangle of B.  Right-looking, NumPy-vectorised per column
(one small group of array operations per column: O(N d) numpy calls for connectivity d; roughly a minute at 10^6 unknowns).
If a pivot is not positive (possible for matrices that are not M-matrices) the factorisation is repeated for
B + alpha diag(B) with alpha increased geometrically (Manteuffel shift); the alpha used is reported.
The preconditioner is applied as M^-1 r = L^-+ (L^-1 r) with SuperLU triangular solves (no fill, O(nnz(L)) per solve).
"""

import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _ic0_factor(Bl, shift):
    """IC(0) of the lower-triangular CSC pattern ``Bl`` (diagonal included); returns the CSC data of L or None on breakdown."""
    n = Bl.shape[0]
    indptr, indices = Bl.indptr, Bl.indices
    data = Bl.data.astype(np.result_type(Bl.dtype, np.float64)).copy()
    # diagonal positions (first entry of each column since the rows are sorted and the diagonal is the smallest row index)
    diag_pos = indptr[:-1]
    if np.any(indices[diag_pos] != np.arange(n)):
        raise ValueError("matrix has a structurally zero diagonal entry")
    if shift:
        data[diag_pos] = data[diag_pos] * (1.0 + shift)
    for k in range(n):
        s, e = indptr[k], indptr[k + 1]
        piv = data[s].real
        if not piv > 1e-300:
            return None
        piv = np.sqrt(piv)
        data[s] = piv
        if e - s > 1:
            data[s + 1:e] /= piv
            rows = indices[s + 1:e]
            vals = data[s + 1:e]
            # right-looking update: A[i, j] -= L[i, k] conj(L[j, k]) for i >= j in the column pattern, only where (i, j) exists
            for t in range(len(rows)):
                j = rows[t]
                cs, ce = indptr[j], indptr[j + 1]
                col = indices[cs:ce]
                tgt = rows[t:]                                       # i >= j (rows sorted)
                pos = np.searchsorted(col, tgt)
                ok = pos < len(col)
                ok[ok] = col[pos[ok]] == tgt[ok]
                data[cs + pos[ok]] -= vals[t:][ok] * np.conj(vals[t])
    return data


def incomplete_cholesky(B, shift0=0.0, max_tries=12):
    """IC(0) factor L (sparse CSC lower triangular, B ~ L L^+) and the shift alpha that was needed."""
    B = sp.csr_matrix(B)
    Bl = sp.tril(B, format="csc")
    Bl.sort_indices()
    shift = shift0
    for _ in range(max_tries):
        data = _ic0_factor(Bl, shift)
        if data is not None:
            L = sp.csc_matrix((data, Bl.indices.copy(), Bl.indptr.copy()), shape=B.shape)
            return L, shift
        shift = 1e-3 if shift == 0 else shift * 2.0
    raise ValueError("incomplete Cholesky failed even with a large diagonal shift (is B positive definite?)")


def ic_preconditioner(B, verbose=False):
    """Callable applying (L L^+)^-1 to a block of columns, plus the info dict (shift, nnz)."""
    L, shift = incomplete_cholesky(B)
    lu = spla.splu(L, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=False))
    cplx = np.iscomplexobj(L.data)

    def apply(R):
        out = np.empty_like(R, dtype=np.result_type(R.dtype, L.dtype))
        for j in range(R.shape[1]):
            y = lu.solve(np.ascontiguousarray(R[:, j]).astype(L.dtype if cplx else np.float64, copy=False))
            out[:, j] = lu.solve(y, trans="H" if cplx else "T")
        return out

    if shift and verbose:
        warnings.warn(f"incomplete Cholesky needed a diagonal shift of {shift:g} (B is not an M-matrix)", stacklevel=2)
    return apply, dict(shift=shift, nnz=int(L.nnz))
