import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # cap BLAS threads (shared machine; also keeps timings honest)
    os.environ.setdefault(_v, "4")
import importlib.util
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

REF = pathlib.Path(__file__).parent / "reference" / "sdos_v0.py"


@pytest.fixture(scope="session")
def v0():
    """The original sdos.py (frozen copy) used for regression tests."""
    spec = importlib.util.spec_from_file_location("sdos_v0", REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cubic(L, W=0.0, t=1.0, seed=0, phases=False):
    """Cubic Anderson model (periodic).  phases=True: random Peierls phases on every bond (complex Hermitian)."""
    rng = np.random.default_rng(seed)
    n = L ** 3
    idx = np.arange(n).reshape(L, L, L)
    rows, cols = [], []
    for ax in range(3):
        rows.append(idx.ravel())
        cols.append(np.roll(idx, -1, axis=ax).ravel())
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    val = -t * (np.exp(1j * rng.uniform(0, 2 * np.pi, len(r))) if phases else np.ones(len(r)))
    Hoff = sp.coo_matrix((val, (r, c)), shape=(n, n))
    H = Hoff + Hoff.conj().T + sp.diags(W * (rng.random(n) - 0.5))
    return H.tocsr()


def random_hermitian(N=600, complex_=True, seed=1):
    rng = np.random.default_rng(seed)
    A = sp.random(N, N, density=6 / N, random_state=seed + 1, format="csr")
    if complex_:
        A = A + 1j * sp.random(N, N, density=6 / N, random_state=seed + 2, format="csr")
    return (A + A.conj().T + sp.diags(rng.normal(size=N))).tocsr()


def exact_moments(evals, a, b, M):
    x = np.clip((evals - b) / a, -1, 1)
    return np.cos(np.outer(np.arange(M), np.arccos(x))).mean(axis=1)


@pytest.fixture
def cubic_fn():
    return cubic


@pytest.fixture
def herm_fn():
    return random_hermitian
