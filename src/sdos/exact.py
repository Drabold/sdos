"""Exact low-order Chebyshev moments by the local (closed-walk) method.

(T_n(Ht))_ii depends only on a neighbourhood of radius ~n/2 of site i, so mu_n = Tr T_n(Ht)/N is
computable *exactly* at a cost linear in N for fixed order n: with P_m = T_m(Ht) and the doubling identities
    mu_2m   = (2 ||P_m||_F^2 - N) / N,
    mu_2m+1 = (2 Re sum(conj(P_m+1) * P_m) - Tr Ht) / N,
moments up to order 2m+1 follow from P_0 ... P_m+1 (each column P_m e_i is built by sparse recursion and
only its neighbourhood is stored).  The cost per site grows polynomially with the order (the neighbourhood
volume, ~ (n/2)^D in D dimensions, times the connectivity), but does NOT grow with N.  Columns are processed
in chunks, so the memory is bounded by chunk x neighbourhood size.

This is exact for the *standard* problem and for diagonal B (after mass weighting).  For a generalized problem
with a non-diagonal B the operator B^-1 A is not local, so no exact local moments exist (see the documentation).
"""

import numpy as np
import scipy.sparse as sp


def exact_chebyshev_moments(Ht, K, chunk=256):
    """Exact mu_0 ... mu_K = Tr T_n(Ht)/N of an explicit sparse Hermitian Ht (spectrum inside [-1, 1]).

    Returns an array of length K+1.  ``chunk`` is the number of unit vectors propagated at once.
    """
    Ht = sp.csr_matrix(Ht)
    n = Ht.shape[0]
    mu = np.zeros(K + 1)
    mu[0] = 1.0
    if K == 0:
        return mu
    tr1 = float(np.real(Ht.diagonal()).sum())
    mu[1] = tr1 / n
    m_max = K // 2                      # need P_0 .. P_m_max (+ P_{m_max+1} for odd K)
    need_next = (K % 2 == 1)
    acc_even = np.zeros(m_max + 1)      # sum_i ||P_m e_i||^2
    acc_odd = np.zeros(m_max + 1)       # sum_i <P_{m+1} e_i, P_m e_i>
    cdtype = Ht.dtype
    HtT = Ht.T.tocsr()                  # columns P_m e_i are held as ROWS Q_m = (P_m e_i)^T of a (chunk x n) CSR matrix:
    for c0 in range(0, n, chunk):       # a CSR product then costs O(nnz), not O(n) (a CSC block of n rows has an n-long index)
        c1 = min(n, c0 + chunk)
        cols = np.arange(c0, c1)
        Qm = sp.csr_matrix((np.ones(len(cols), dtype=cdtype), (np.arange(len(cols)), cols)), shape=(len(cols), n))
        Q1 = HtT[c0:c1]
        for m in range(0, m_max + 1):
            # Qm = (T_m e_i)^T , Q1 = (T_{m+1} e_i)^T
            acc_even[m] += float(np.real((Qm.conj().multiply(Qm)).sum()))
            if 2 * m + 1 <= K:
                acc_odd[m] += float(np.real((Q1.conj().multiply(Qm)).sum()))
            if m < m_max:
                Q2 = (2.0 * (Q1 @ HtT) - Qm).tocsr()
                Qm, Q1 = Q1, Q2
    for m in range(0, m_max + 1):
        if 2 * m <= K and m > 0:
            mu[2 * m] = 2.0 * acc_even[m] / n - 1.0
        if 2 * m + 1 <= K and m > 0:
            mu[2 * m + 1] = (2.0 * acc_odd[m] - tr1) / n
    # m = 0: mu_1 is already exact (diagonal); the m = 0 odd formula would give 2<T1 e,e>-tr1 = tr1/n (consistent)
    return mu
