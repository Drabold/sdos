"""Non-orthogonal tight binding: A x = lambda B x with an overlap matrix B, both B^-1 modes, complex Hermitian (k != 0)."""
import sys
sys.path.insert(0, "../tests")
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import sdos
from conftest import random_hermitian
from test_generalized import spd_B

n = 400
A = random_hermitian(n, True, seed=3)          # complex Hermitian "Hamiltonian"
B = spd_B(n, 50, True, seed=2)                 # Hermitian positive-definite "overlap"
ev = sla.eigh(A.toarray(), B.toarray(), eigvals_only=True)   # exact, for comparison only
for binv, kw in (("pcg", dict(precond="block-jacobi")), ("pcg", dict(precond="ic")), ("chebyshev", {})):
    r = sdos.dos(A, B=B, M=64, R=20, seed=1, bounds=(ev[0] - 1e-6, ev[-1] + 1e-6), binv=binv, **kw)
    mex = np.cos(np.outer(np.arange(64), np.arccos((ev - r.b) / r.a))).mean(axis=1)
    z = np.abs(r.mu - mex)[2:] / sdos.moment_errors(r.mu_samples)[2:]
    print(f"binv={binv:9s} {kw}: max |moment - exact| / standard error = {z.max():.1f};  {r.binv_info}")
