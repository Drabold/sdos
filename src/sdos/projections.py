"""Local and projected densities of states.

Convention (documented, as the directive requires): for A x = lambda B x with eigenvectors normalised psi^+ B psi = 1, the weight of
orbital i in eigenstate n is the **Mulliken** weight  w_in = Re conj(psi_in) (B psi_n)_i  (sum_i w_in = 1; w_in = |psi_in|^2 for B = 1, and the
weights may be negative for strongly overlapping orbitals, as always with Mulliken populations).  Then

    LDOS_i(E) = sum_n w_in delta(E - E_n) = [ delta(E - A B^-1) ]_ii ,   PDOS_g(E) = sum_{i in g} LDOS_i(E),

because sum_n psi_n psi_n^+ f(E_n) = f(B^-1 A) B^-1 and B (B^-1 A) B^-1 = A B^-1.  So the moments are those of the operator A B^-1
(the adjoint of B^-1 A): Chebyshev moments of (A B^-1 - b)/a, applied through ``GeneralizedOperator.apply_adjoint``.  Sum rules:
sum_i LDOS_i = N x total DOS, every LDOS_i integrates to 1 (completeness: sum_n w_in = [B B^-1]_ii = 1), and PDOS_g integrates to n_g / N.

Estimators:  ``mode="shared"`` (default): one random vector x per sample is propagated once, and every group reads
sum_{i in g} conj(x_i) (T_n x)_i from the same recursion: all groups for the price of one DOS (the group sums add up to the total DOS exactly),
at the cost of extra noise for small groups.  ``mode="separate"``: a vector supported on the group only, one propagation per group (plain
Hermitian case uses the doubling identities): lower noise for small groups.  ``ldos`` is deterministic: it propagates unit vectors.
"""

import numpy as np
import scipy.sparse as sp

from .generalized import GeneralizedOperator, generalized_bounds, is_diagonal
from .bounds import spectral_bounds
from .moments import chebyshev_moments, chebyshev_moments_plain, make_vectors
from .operators import ScaledOperator, is_complex


class ProjectedResult(dict):
    """E, rho (dict group -> DOS contribution, per orbital of the whole system), rho_total, plus bookkeeping; attribute access."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc


def site_groups(labels, orbitals_per_site):
    """Orbital index arrays per site label, e.g. site_groups(["Si", "H", "Si"], 4) -> {"Si": [0..3, 8..11], "H": [4..7]}."""
    out = {}
    for s, lab in enumerate(labels):
        out.setdefault(lab, []).extend(range(s * orbitals_per_site, (s + 1) * orbitals_per_site))
    return {k: np.array(v, dtype=int) for k, v in out.items()}


def _setup(A, B, bounds, margin, seed, binv, precond, block_size, binv_tol, lanczos_steps):
    n = A.shape[0]
    cplx = bool(is_complex(A) or (B is not None and not (isinstance(B, np.ndarray) and B.ndim == 1) and np.iscomplexobj(B.data)))
    if isinstance(B, np.ndarray) and B.ndim == 1:
        d = np.asarray(B, float)
        s = sp.diags(1.0 / np.sqrt(d))
        A = (s @ A @ s).tocsr()                      # diagonal B: Mulliken weights of the mass-weighted problem equal |psi|^2
        B = None
    if B is not None and sp.issparse(B) and is_diagonal(B):
        s = sp.diags(1.0 / np.sqrt(np.real(B.diagonal())))
        A = (s @ A @ s).tocsr()
        B = None
    if B is None:
        emin, emax = bounds if bounds is not None else spectral_bounds(A, verbose=False)
        w = emax - emin
        emin, emax = emin - margin * w, emax + margin * w
        a, b = 0.5 * (emax - emin), 0.5 * (emax + emin)
        return dict(kind="standard", n=n, cplx=cplx, a=a, b=b, Ht=ScaledOperator(A, a, b), op=None)
    op = GeneralizedOperator(A, B, binv=binv, precond=precond, block_size=block_size, tol=binv_tol, check=True)
    if bounds is None:
        lo, hi, rs = generalized_bounds(op.apply, op.B, n, m=lanczos_steps, seed=seed, is_cplx=cplx)
        lo, hi = lo - rs, hi + rs
    else:
        lo, hi = bounds
    w = hi - lo
    lo, hi = lo - margin * w, hi + margin * w
    a, b = 0.5 * (hi - lo), 0.5 * (hi + lo)
    return dict(kind="generalized", n=n, cplx=cplx, a=a, b=b, Ht=None, op=op)


def pdos(A, B=None, groups=None, M=256, R=10, mode="shared", seed=None, bounds=None, margin=0.01, kernel="jackson", lorentz_lambda=4.0,
         vectors="auto", binv="pcg", precond="block-jacobi", block_size=1, binv_tol=1e-12, lanczos_steps=40, npts=None):
    """Projected DOS of the orbital groups ``groups`` (dict name -> orbital indices; default: every orbital in one group).

    Returns a :class:`ProjectedResult`: ``E``, ``rho[name]`` (DOS contribution of the group, per orbital of the whole system, so that the groups
    add up to the total DOS and the integral of rho[name] is the group's Mulliken population over N), ``rho_err[name]`` (standard error over the
    R samples), ``rho_total`` and ``mu`` (group moments (R, M)).  Convention and estimators: see the module docstring.
    """
    from .reconstruct import reconstruct
    if mode not in ("shared", "separate"):
        raise ValueError("mode must be 'shared' or 'separate'")
    st = _setup(A, B, bounds, margin, seed, binv, precond, block_size, binv_tol, lanczos_steps)
    n = st["n"]
    if groups is None:
        groups = {"all": np.arange(n)}
    groups = {k: np.asarray(v, dtype=int) for k, v in groups.items()}
    a, b = st["a"], st["b"]
    rng = np.random.default_rng(seed)
    cplx = st["cplx"]
    if st["kind"] == "standard":
        Zop = lambda V: st["Ht"] @ V
    else:
        op = st["op"]
        Zop = lambda V: (op.apply_adjoint(V) - b * V) / a
    mus = {g: np.zeros((R, M)) for g in groups}
    for r in range(R):
        if mode == "shared":
            x = make_vectors(n, None, 1, rng, cplx, kind=vectors, is_complex=cplx)[:, 0]
            xh = np.conj(x)
            vprev = x
            vcur = Zop(x[:, None])[:, 0]
            for g, idx in groups.items():
                mus[g][r, 0] = np.real(np.vdot(x[idx], vprev[idx]))
                mus[g][r, 1] = np.real(np.vdot(x[idx], vcur[idx]))
            for m in range(2, M):
                vnext = 2.0 * Zop(vcur[:, None])[:, 0] - vprev
                for g, idx in groups.items():
                    mus[g][r, m] = np.real(np.vdot(x[idx], vnext[idx]))
                vprev, vcur = vcur, vnext
        else:
            for g, idx in groups.items():
                x = make_vectors(n, None, 1, rng, cplx, kind=vectors, is_complex=cplx)[:, 0]
                xg = np.zeros_like(x)
                xg[idx] = x[idx]
                if st["kind"] == "standard":
                    mus[g][r] = chebyshev_moments(st["Ht"], xg[:, None], M)
                else:
                    mus[g][r] = chebyshev_moments_plain(Zop, xg[:, None], M)
    npts = npts or max(2 * M, 1000)
    E = None
    rho, rho_err, idos = {}, {}, {}
    for g in groups:
        m_ = mus[g] / n                                  # group trace per orbital of the whole system
        E, rs, ids = reconstruct(m_, a, b, npts, kernel=kernel, lam=lorentz_lambda)
        rho[g] = rs.mean(axis=0)
        rho_err[g] = rs.std(axis=0, ddof=1) / np.sqrt(R) if R > 1 else np.full(npts, np.nan)
        idos[g] = ids.mean(axis=0)
    total = sum(rho.values())
    return ProjectedResult(E=E, rho=rho, rho_err=rho_err, idos=idos, rho_total=total, mu={g: mus[g] / n for g in groups}, a=a, b=b, N=n, M=M, R=R,
                           mode=mode, groups={g: len(v) for g, v in groups.items()}, kernel=kernel, problem=st["kind"])


def ldos(A, B=None, orbitals=None, M=256, bounds=None, margin=0.01, kernel="jackson", lorentz_lambda=4.0, binv="pcg", precond="block-jacobi",
         block_size=1, binv_tol=1e-12, lanczos_steps=40, npts=None, seed=0):
    """Deterministic (exact-trace) LDOS of the given orbitals: unit vectors propagated, no random noise (resolution ~ pi a / M only).

    Returns a :class:`ProjectedResult` with ``rho[i]`` for each requested orbital i (states per energy per orbital i; integrates to the Mulliken
    population of i), ``mu[i]`` the exact moments.  Cost: M products per orbital (block-propagated in groups of 64).
    """
    from .reconstruct import reconstruct
    st = _setup(A, B, bounds, margin, seed, binv, precond, block_size, binv_tol, lanczos_steps)
    n = st["n"]
    orbitals = np.arange(n) if orbitals is None else np.asarray(orbitals, dtype=int)
    a, b = st["a"], st["b"]
    dtype = np.complex128 if st["cplx"] else np.float64
    mu = {}
    for c0 in range(0, len(orbitals), 64):
        idx = orbitals[c0:c0 + 64]
        X = np.zeros((n, len(idx)), dtype=dtype)
        X[idx, np.arange(len(idx))] = 1.0
        if st["kind"] == "standard":
            Ht = st["Ht"]
            Zop = lambda V: Ht @ V
            V0, V1 = X, Zop(X)
            m = np.zeros((M, len(idx)))
            m[0] = 1.0
            m[1] = np.real(V1[idx, np.arange(len(idx))])
            vp, vc = V0, V1
            for k in range(2, M):
                vn = 2.0 * Zop(vc) - vp
                m[k] = np.real(vn[idx, np.arange(len(idx))])
                vp, vc = vc, vn
        else:
            op = st["op"]
            Zop = lambda V: (op.apply_adjoint(V) - b * V) / a
            m = np.zeros((M, len(idx)))
            m[0] = 1.0
            vp, vc = X, Zop(X)
            m[1] = np.real(vc[idx, np.arange(len(idx))])
            for k in range(2, M):
                vn = 2.0 * Zop(vc) - vp
                m[k] = np.real(vn[idx, np.arange(len(idx))])
                vp, vc = vc, vn
        for j, i in enumerate(idx):
            mu[int(i)] = m[:, j]
    npts = npts or max(2 * M, 1000)
    rho, idos = {}, {}
    E = None
    for i, m_ in mu.items():
        E, rs, ids = reconstruct(m_, a, b, npts, kernel=kernel, lam=lorentz_lambda)
        rho[i], idos[i] = rs, ids
    return ProjectedResult(E=E, rho=rho, idos=idos, mu=mu, a=a, b=b, N=n, M=M, R=0, mode="exact", kernel=kernel, problem=st["kind"],
                           rho_total=sum(rho.values()))
