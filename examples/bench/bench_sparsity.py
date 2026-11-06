"""Cost of the Chebyshev recursion against the number of nonzeros per row: random symmetric sparse matrix, N = 50000, M = 100, R = 2 Rademacher vectors
(bare recursion, no bounds, no reconstruction).  Writes bench_sparsity.json.  Run with OMP/OPENBLAS/MKL_NUM_THREADS=4."""
import json, time
import numpy as np, scipy.sparse as sp
rng = np.random.default_rng(0); N, M, R = 50000, 100, 2
out = []
for z in (4, 8, 16, 32, 64, 128, 256):
    r = np.repeat(np.arange(N), z // 2); c = (r + rng.integers(1, N, r.size)) % N
    A = sp.coo_matrix((rng.normal(size=r.size), (r, c)), shape=(N, N)).tocsr(); A = ((A + A.T) / (2 * np.sqrt(z))).tocsr()
    V = rng.choice([-1., 1.], size=(N, R)); t0 = time.time()
    a = V.copy(); b = A @ V
    for m in range(2, M): a, b = b, 2 * (A @ b) - a
    t = time.time() - t0
    out.append(dict(nnz_per_row=A.nnz // N, nnz=A.nnz, sec=t, ns_per_nnz_M_R=t / (A.nnz * M * R) * 1e9))
    print(out[-1], flush=True)
json.dump(dict(N=N, M=M, R=R, threads=4, rows=out), open("bench_sparsity.json", "w"), indent=1)
