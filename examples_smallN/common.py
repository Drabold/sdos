"""Shared helpers for the small-N study: test systems, exact references (dense eigh), moments of random vectors
computed in the eigenbasis (mathematically identical to the Chebyshev recursion, but O(N^2) per batch of vectors),
the Toeplitz (Caratheodory) consistency test, and the KPM / Maxent estimators used in E1-E4."""
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from sdos.moments import impartial_vector
from sdos.operators import ScaledOperator
from sdos.reconstruct import reconstruct, jackson, band_quantities
from sdos.maxent import maxent_fit, maxent_density, band_quantities_grid
from sdos.exact import exact_chebyshev_moments
from sdos.interfaces import kwon_si as KW

# ---------------------------------------------------------------- systems
def chain(N):
    i = np.arange(N); j = (i + 1) % N
    return sp.csr_matrix((-np.ones(2 * N), (np.r_[i, j], np.r_[j, i])), shape=(N, N))

def ssh(N, t1=1.0, t2=0.5):
    i = np.arange(N); j = (i + 1) % N; t = np.where(i % 2 == 0, t1, t2)
    return sp.csr_matrix((-np.r_[t, t], (np.r_[i, j], np.r_[j, i])), shape=(N, N))

def kwon(ncell):
    pos, box = KW.diamond_positions(ncell)
    return sp.csr_matrix(KW.kwon_hamiltonian(pos, box))

def anderson(L, W=3.0, seed=1):
    rng = np.random.default_rng(seed); N = L ** 3
    idx = np.arange(N).reshape(L, L, L); r, c = [], []
    for ax in range(3):
        j = np.roll(idx, -1, axis=ax); r += [idx.ravel(), j.ravel()]; c += [j.ravel(), idx.ravel()]
    r = np.concatenate(r); c = np.concatenate(c)
    return sp.csr_matrix((np.r_[-np.ones(len(r))], (r, c)), shape=(N, N)) + sp.diags(rng.uniform(-W / 2, W / 2, N))

SYSTEMS = {  # name -> (builder, list of sizes (the argument), N(arg))
    "chain": (chain, [32, 64, 128, 256, 512, 1024, 2048, 4096], lambda s: s),
    "ssh": (ssh, [32, 64, 128, 256, 512, 1024, 2048, 4096], lambda s: s),
    "kwon": (kwon, [2, 3, 4, 5], lambda s: 32 * s ** 3),
    "anderson": (anderson, [8, 10, 16], lambda s: s ** 3),
}

class Ref:
    """Dense reference of a system: eigenpairs, scaled spectrum (sdos convention: bounds = exact extremes widened by margin*width)."""
    def __init__(self, name, size, margin=0.01):
        build, _, nf = SYSTEMS[name]
        self.name, self.size = name, size
        self.H = build(size).tocsr(); self.N = self.H.shape[0]
        w, V = np.linalg.eigh(self.H.toarray())
        self.ev, self.V = w, V
        width = w[-1] - w[0]; lo, hi = w[0] - margin * width, w[-1] + margin * width
        self.a, self.b = 0.5 * (hi - lo), 0.5 * (hi + lo)
        self.x = (w - self.b) / self.a
        self.Ht = ScaledOperator(self.H, self.a, self.b)
        self.nocc = self.N // 2
        self.Ebs_exact = float(w[:self.nocc].sum())

    def cheb(self, M):
        """T_n(x_i), shape (M, N)."""
        th = np.arccos(self.x); return np.cos(np.outer(np.arange(M), th))

    def exact_moments(self, M):
        return self.cheb(M).mean(axis=1)

    def vec_moments(self, X, M, T=None):
        """Moments (M, S) of the columns of X (N, S), each normalised to |x|^2 = N: mu_n = N^-1 sum_i c_i^2 T_n(x_i), c = V^T x."""
        T = self.cheb(M) if T is None else T
        c = self.V.T @ X
        return T @ (c * c) / self.N

    def rademacher(self, rng, S):
        return rng.choice(np.array([-1.0, 1.0]), size=(self.N, S))

    def impartial(self, rng, S):
        tr = self.Ht.traces(); X = np.empty((self.N, S))
        for s in range(S):
            x, _ = impartial_vector(self.Ht, self.N, rng, *tr); X[:, s] = np.sqrt(self.N) * x
        return X

# ---------------------------------------------------------------- Toeplitz test
def lam_min_T(mu, m):
    """lambda_min of the (m+1)x(m+1) Toeplitz matrix T_jk = mu_|j-k|."""
    return sla.eigvalsh(sla.toeplitz(mu[:m + 1]), subset_by_index=[0, 0])[0]

def first_violation(mu, tol=-1e-10, mmax=None):
    """Smallest m with lambda_min(T_m) < tol (T_m for the list mu_0..mu_m), or -1 if none up to len(mu)-1.  lambda_min(T_m) is non-increasing in m
    (interlacing), so bisection is exact."""
    mmax = len(mu) - 1 if mmax is None else mmax
    if lam_min_T(mu, mmax) >= tol: return -1
    lo, hi = 0, mmax
    while lo < hi:
        mid = (lo + hi) // 2
        if lam_min_T(mu, mid) < tol: hi = mid
        else: lo = mid + 1
    return lo

def splice(mu, mu_ex, K):
    """Replace mu_0..mu_K by the exact values (column-wise for a (M, S) array or a vector)."""
    out = np.array(mu, copy=True); out[:K + 1] = mu_ex[:K + 1].reshape((K + 1,) + (1,) * (out.ndim - 1)); return out

# ---------------------------------------------------------------- estimators of E_BS, cumulative error
def kpm_eval(mu, ref, npts=4096, kernel="jackson"):
    """E_BS (half filling, spin 1) and the idos on a grid from one moment list (KPM, Jackson)."""
    E, rho, idos = reconstruct(mu[None], ref.a, ref.b, npts, kernel=kernel)
    EF, Ebs = band_quantities(mu, ref.a, ref.b, ref.N, ref.nocc, 1.0, kernel=kernel)
    return Ebs, E, idos[0]

def maxent_eval(mus, ref, sigma_floor=1e-4, sigma_mode="sampled", npts=4096):
    """Maxent from (R, M) per-sample moments.  sigma_mode: 'sampled' (the package: standard error over the R samples, floored),
    'floor' (sigma = floor for all n>=1)."""
    R = mus.shape[0]; mu = mus.mean(0)
    sigma = None
    if sigma_mode == "floor" or R < 2:
        sigma = np.zeros(mus.shape[1])
    lam, _, info = maxent_fit(mus, ref.a, ref.b, "chebyshev", 1.0, sigma=sigma, jackknife=False, verbose=False, sigma_floor=sigma_floor)
    E, rho, idos = maxent_density(lam, ref.a, ref.b, "chebyshev", mu=mu)
    EF, Ebs = band_quantities_grid(E, rho, idos, ref.N, ref.nocc, 1.0)
    return Ebs, E, idos, info

def cum_exact(ref, E):
    """Exact cumulative N(E)/N (right-continuous step) on the grid E."""
    return np.searchsorted(ref.ev, E, side="right") / ref.N
