"""Stochastic Chebyshev moments.

Moments mu_n = Tr T_n(Ht) / N of the scaled operator Ht (spectrum inside (-1, 1)) are estimated
from random vectors x with E[x x^+] = I, using the conjugated product x^+ T_n(Ht) x.  The doubling
identities  T_2n = 2 T_n^2 - T_0,  T_2n+1 = 2 T_n+1 T_n - T_1  give 2K moments from K products.
"""

import numpy as np

# Vector ensembles.  "phase": exp(i phi) (complex, |x_i| = 1); "rademacher": +-1 (real, |x_i| = 1);
# "gaussian": real N(0,1) or complex (N + iN)/sqrt(2); "auto" (the original behaviour): phase for
# complex problems or when complex_phases is requested, Rademacher otherwise.  "impartial": the
# Drabold-Sankey penalty-function vectors (see impartial_vector); they need the operator, so they
# are generated in stochastic_moments, not in make_vectors.
VECTOR_KINDS = ("auto", "phase", "rademacher", "gaussian", "impartial")


def make_vectors(n, sample_colors, ncolors, rng, complex_phases=False, kind="auto", is_complex=None):
    """Random vectors for one sample: one vector per color (a single one without probing).

    ``complex_phases`` is the original flag (complex problem, or phases requested for a real one).
    """
    if kind not in VECTOR_KINDS:
        raise ValueError(f"kind must be one of {VECTOR_KINDS}")
    if kind == "impartial":
        raise ValueError("impartial vectors need the operator: use stochastic_moments(..., kind='impartial')")
    if kind == "auto":
        kind = "phase" if complex_phases else "rademacher"
    if kind == "phase":
        z = np.exp(2j * np.pi * rng.random(n))
    elif kind == "rademacher":
        z = rng.choice(np.array([-1.0, 1.0]), size=n)
    else:  # gaussian
        cplx = bool(complex_phases if is_complex is None else is_complex)
        z = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2.0) if cplx else rng.normal(size=n)
    if sample_colors is None:
        return z[:, None]
    X = np.zeros((n, ncolors), dtype=z.dtype)
    X[np.arange(n), sample_colors] = z
    return X


def chebyshev_moments(Ht, X, M):
    """Return mu[0..M-1] = sum over columns of x^+ T_n(Ht) x, via doubling (unnormalised)."""
    K = (M + 1) // 2           # need v_0 .. v_K
    mu = np.zeros(2 * K + 2)
    v0 = X
    v1 = Ht @ X
    mu0 = np.real(np.vdot(v0, v0))
    mu1 = np.real(np.vdot(v0, v1))
    mu[0], mu[1] = mu0, mu1
    vprev, vcur = v0, v1
    for nn in range(1, K + 1):
        vnext = 2.0 * (Ht @ vcur) - vprev
        mu[2 * nn] = 2.0 * np.real(np.vdot(vcur, vcur)) - mu0
        mu[2 * nn + 1] = 2.0 * np.real(np.vdot(vnext, vcur)) - mu1
        vprev, vcur = vcur, vnext
    return mu[:M]


def chebyshev_moments_plain(Xop, X, M):
    """mu_n = Re x^+ T_n(Xop) x, n < M, by the plain three-term recursion (no doubling): one application of Xop per moment.

    Used for the generalized problem, where Xop = (B^-1 A - b)/a is not Hermitian.  Returns the sum over the columns of X.
    """
    mu = np.zeros(M)
    v0 = X
    mu[0] = np.real(np.vdot(X, v0))
    if M == 1:
        return mu
    v1 = Xop(v0)
    mu[1] = np.real(np.vdot(X, v1))
    vprev, vcur = v0, v1
    for m in range(2, M):
        vnext = 2.0 * Xop(vcur) - vprev
        mu[m] = np.real(np.vdot(X, vnext))
        vprev, vcur = vcur, vnext
    return mu


def impartial_vector(Ht, n, rng, tr1, tr2, is_complex=False, maxiter=500, gtol=1e-14):
    """Drabold-Sankey importance-sampled ("impartial") vector, PRL 70, 3631 (1993), Eqs. (4)-(5).

    Start from a normally distributed vector and minimise, by conjugate gradients,
        P(x) = (<x|x> - 1)^2 + (<x|Ht|x> - mu_1)^2 + (<x|Ht^2|x> - mu_2)^2,
    with the exact mu_1 = Tr Ht / N and mu_2 = Tr Ht^2 / N, so that the first three moments of x
    are exact.  Returns (x, info); x has unit norm.  Complex vectors are used for complex problems
    (the conjugated products and the Wirtinger gradient are used).  Cost: two products per CG step.
    """
    from scipy.optimize import minimize

    x0 = rng.normal(size=n)
    if is_complex:
        x0 = x0 + 1j * rng.normal(size=n)
    x0 = x0 / np.linalg.norm(x0)

    def unpack(v):
        return v[:n] + 1j * v[n:] if is_complex else v

    def fun(v):
        x = unpack(v)
        y = Ht @ x
        z = Ht @ y
        f0 = np.real(np.vdot(x, x)) - 1.0
        f1 = np.real(np.vdot(x, y)) - tr1
        f2 = np.real(np.vdot(y, y)) - tr2
        g = 4.0 * (f0 * x + f1 * y + f2 * z)          # gradient wrt (Re x, Im x) = 2 Re/Im of dP/dx*
        grad = np.concatenate([g.real, g.imag]) if is_complex else np.real(g)
        return f0 * f0 + f1 * f1 + f2 * f2, grad

    v0 = np.concatenate([x0.real, x0.imag]) if is_complex else x0
    r = minimize(fun, v0, jac=True, method="CG", options=dict(maxiter=maxiter, gtol=gtol))
    x = unpack(r.x)
    return x, dict(P=float(r.fun), nit=int(r.nit), nfev=int(r.nfev))


def stochastic_moments(Ht, n, M, R, rng, colors=None, ncolors=1, complex_phases=False, kind="auto",
                       is_complex=None, traces=None, info=None):
    """R independent samples of the normalised moments; returns an (R, M) array (mu_0 = 1).

    kind="impartial" uses Drabold-Sankey vectors (not with probing); ``traces`` = (Tr Ht/N, Tr Ht^2/N)
    is computed from an explicit sparse Ht if not given.  ``info`` (a list) collects the CG statistics.
    """
    mus = np.zeros((R, M))
    if kind == "impartial":
        if colors is not None:
            raise ValueError("impartial vectors cannot be combined with probing")
        cplx = bool(complex_phases if is_complex is None else is_complex)
        tr1, tr2 = traces if traces is not None else Ht.traces()
    for r in range(R):
        if kind == "impartial":
            x, inf = impartial_vector(Ht, n, rng, tr1, tr2, cplx)
            if info is not None:
                info.append(inf)
            X = (np.sqrt(n) * x)[:, None]
        else:
            X = make_vectors(n, colors, ncolors, rng, complex_phases, kind=kind, is_complex=is_complex)
        mus[r] = chebyshev_moments(Ht, X, M) / n
    return mus


def moment_errors(mus):
    """Per-moment standard error of the mean from the spread over the R samples."""
    R = mus.shape[0]
    if R < 2:
        return np.full(mus.shape[1], np.nan)
    return mus.std(axis=0, ddof=1) / np.sqrt(R)
