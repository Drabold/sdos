"""Main entry points: stochastic-trace DOS by KPM or Maxent, Fermi level and band energy.

``estimate_dos(A, B=None, ...)`` handles the standard problem A x = lambda x and the generalized problem
A x = lambda B x (B Hermitian positive definite; a 1-D array means a diagonal B, e.g. masses).
"""

import time
import warnings

import numpy as np
import scipy.sparse as sp

from .bounds import distance_coloring, spectral_bounds
from .exact import exact_chebyshev_moments
from .generalized import GeneralizedOperator, generalized_bounds, is_diagonal
from .maxent import band_quantities_grid, maxent_density, maxent_fit
from .moments import chebyshev_moments_plain, make_vectors, stochastic_moments
from .operators import ScaledOperator, check_hermitian, is_complex, is_matrix_free
from .reconstruct import band_quantities, kernel_factors, reconstruct


class DOSResult(dict):
    """A dict with attribute access (``res.E``, ``res.rho`` ...).  Keys are those of the original sdos.py."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __repr__(self):
        return self.summary()

    def summary(self):
        """One-paragraph description of what was computed."""
        r = self
        kind = r.get("problem", "standard")
        bits = [f"DOSResult: {kind} problem, N = {r['N']}, {r['method'].upper()}"
                + (f" ({r['kernel']} kernel, resolution ~ {r['resolution']:.3g})" if r["method"] == "kpm" else ""),
                f"  M = {r['M']} moments, R = {r['R']} random vectors ({r.get('vectors', 'auto')}), "
                f"exact moments: {r.get('exact_moments', 0)}",
                f"  spectral window [{r['b'] - r['a']:.5g}, {r['b'] + r['a']:.5g}]; "
                f"E in r.E, DOS per orbital in r.rho (integrates to 1), error bars r.rho_err, IDOS r.idos"]
        if "binv_info" in r:
            bi = r["binv_info"]
            bits.append(f"  B^-1 by {bi['binv']}: {bi['nsolve']} applications" +
                        (f", degree {bi['degree']}, cond(B~) ~ {bi['kappa']:.3g}" if "degree" in bi else f", {bi['iterations']} PCG iterations"))
        return "\n".join(bits)

    def plot(self, ax=None, show_error=True, xlabel="E", **kw):
        """Plot the DOS (with +-2 standard errors); returns the matplotlib axes (needs matplotlib)."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self["E"], self["rho"], lw=kw.pop("lw", 1.2), **kw)
        if show_error and self["R"] > 1 and np.all(np.isfinite(self["rho_err"])):
            ax.fill_between(self["E"], self["rho"] - 2 * self["rho_err"], self["rho"] + 2 * self["rho_err"], alpha=0.25, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("DOS per orbital")
        return ax

    def save(self, path):
        """Write the result: a ``.npz`` path saves everything (moments, window, settings; read back with :func:`sdos.load`);
        any other path writes E, rho, rho_err, IDOS, IDOS_err as text."""
        if str(path).endswith(".npz"):
            from .results import save_result
            return save_result(path, self)
        from .io import write_dos
        write_dos(path, self, f"sdos: N={self['N']} M={self['M']} R={self['R']} method={self['method']}")


def _check_moment_bounds(mu, N, R, verbose=True):
    """|Tr T_m(Ht)/N| <= 1 when the spectrum lies inside [-1, 1]; a larger value means the bounds were too tight."""
    tol = 1e-8 + 6.0 / np.sqrt(N * max(R, 1))
    bad = np.nonzero(np.abs(mu[1:]) > 1.0 + tol)[0]
    if len(bad):
        msg = (f"moment mu_{bad[0] + 1} = {mu[bad[0] + 1]:.4g} exceeds 1 (+ statistical tolerance {tol:.2g}): the spectral "
               "bounds are probably too tight (part of the spectrum lies outside the window); increase `margin` or pass "
               "guaranteed `bounds`")
        warnings.warn(msg, stacklevel=3)
        return False
    return True


def _known_count(known):
    return int(sum(c for _, c in known)) if known else 0


def _remove_known(mus, a, b, n, known):
    """Deflate known eigenvalues [(lambda, multiplicity), ...] from the moments: each contributes exactly T_m(x_lambda)/N to mu_m, so its
    (kernel-smeared) weight is removed exactly by linearity, and the remainder is renormalised to unit weight (N - sum of multiplicities states).
    Needed e.g. for the rigid-translation zero modes before taking thermodynamic averages that diverge at omega -> 0."""
    if not known:
        return mus
    cnt = _known_count(known)
    m = np.arange(mus.shape[1])
    out = mus.copy()
    for lam, c in known:
        x0 = np.clip((lam - b) / a, -1.0, 1.0)
        out = out - (c / n) * np.cos(m * np.arccos(x0))[None, :]
    return out * (n / (n - cnt))


def _reconstruct_result(out, mus, a, b, R, method, npts, kernel, lorentz_lambda, maxent_default, maxent_noise, M, verbose):
    if method == "kpm":
        if npts is None:
            npts = max(2 * M, 1000)
        E, rho_s, idos_s = reconstruct(mus, a, b, npts, kernel=kernel, lam=lorentz_lambda)
        rho = rho_s.mean(axis=0)
        idos = idos_s.mean(axis=0)
        if R > 1:
            rho_err = rho_s.std(axis=0, ddof=1) / np.sqrt(R)
            idos_err = idos_s.std(axis=0, ddof=1) / np.sqrt(R)
        else:
            rho_err = np.full_like(rho, np.nan)
            idos_err = np.full_like(idos, np.nan)
        out["resolution"] = np.pi * a / M
    elif method == "maxent":
        lam, lams, info = maxent_fit(mus, a, b, maxent_default, maxent_noise, verbose=verbose)
        E, rho, idos = maxent_density(lam, a, b, maxent_default, npts, mu=mus.mean(axis=0))
        if lams is not None:
            jk = [maxent_density(l, a, b, maxent_default, npts, mu=mus.mean(axis=0)) for l in lams]
            rj = np.array([j[1] for j in jk])
            ij = np.array([j[2] for j in jk])
            fac = np.sqrt((R - 1) / R) * np.sqrt(R)   # jackknife: sqrt((R-1)/R * sum)
            rho_err = rj.std(axis=0) * fac
            idos_err = ij.std(axis=0) * fac
        else:
            rho_err = np.full_like(rho, np.nan)
            idos_err = np.full_like(idos, np.nan)
        out.update(lam=lam, lam_jackknife=lams, maxent_info=info, resolution=np.nan)
    else:
        raise ValueError("method must be 'kpm' or 'maxent'")
    out.update(E=E, rho=rho, rho_err=rho_err, idos=idos, idos_err=idos_err)
    return out


def estimate_dos(H, B=None, M=256, R=10, probe=0, bounds=None, margin=0.01, complex_phases=None, seed=None,
                 npts=None, verbose=True, method="kpm", maxent_default="chebyshev", maxent_noise=1.0,
                 kernel="jackson", lorentz_lambda=4.0, vectors="auto", hermitian_tol=1e-10,
                 check=True, traces=None, exact_moments=0, binv="pcg", precond="block-jacobi", block_size=1,
                 binv_tol=1e-12, cheb_tol=None, bmargin=0.02, lanczos_steps=40, diagonal_shortcut=True,
                 check_bounds=True, exact_chunk=256, known_eigenvalues=None):
    """Density of states of a large sparse Hermitian problem (explicit sparse or matrix-free A).

    Standard problem (B=None) or generalized A x = lambda B x.  For the generalized problem the spectral variable is lambda
    and the Chebyshev recursion runs on (B^-1 A - b)/a, applying B^-1 by ``binv`` = 'pcg' (preconditioner ``precond``, tolerance
    ``binv_tol``) or 'chebyshev' (fixed polynomial of degree set by cond(B) and ``cheb_tol``); see the module documentation.
    A 1-D ``B`` (or a diagonal B with ``diagonal_shortcut``) is mass-weighted once and no solves are done.
    ``known_eigenvalues=[(lam, multiplicity), ...]`` removes eigenvalues you know exactly (e.g. [(0.0, 3)] for rigid translations) from the moments
    before reconstruction: the DOS is then that of the remaining N - sum(multiplicity) states, normalised to 1.

    ``exact_moments=K`` replaces the first K+1 Chebyshev moments of every sample by their exact values (local O(N) method;
    unbiased; standard and diagonal-B problems only).  Returns a :class:`DOSResult` (E, rho, rho_err, idos, idos_err, mu,
    mu_samples, a, b, bounds_raw ...).  rho is normalised to 1; ``rho_err`` is the standard error over the R samples.
    """
    t0 = time.time()
    n = H.shape[0]
    B_mode = "none"
    if method == "maxent" and R < 2 and exact_moments:
        # A list with exact low moments and the tail of ONE vector is in general not the moment list of any positive density, and with only the
        # floor as uncertainty the Maxent dual has no solution: Newton does not converge and the result is far worse than with plain vectors
        # (small-N study, report Sec. III D).  With R >= 2 the sampled uncertainties absorb the inconsistency (about 3-8% of the runs did not converge in the small-N tests, none systematically).
        warnings.warn(f"method='maxent' with R={R} and exact_moments={exact_moments}: the spliced moment list is inconsistent and Maxent does not converge; "
                      "using exact_moments=0. Use R >= 2 vectors to keep the exact low moments, or method='kpm'.", stacklevel=2)
        exact_moments = 0
    if B is not None:
        if isinstance(B, np.ndarray) and B.ndim == 1:
            B = sp.diags(B).tocsr()
        if np.any(np.real(B.diagonal()) <= 0):
            raise ValueError("B must be positive definite (non-positive diagonal entries found)")
        if diagonal_shortcut and is_diagonal(B):
            d = np.real(B.diagonal())
            if not sp.issparse(H):
                raise ValueError("a diagonal B with matrix-free A: pass the mass-weighted operator and B=None")
            s = sp.diags(1.0 / np.sqrt(d))
            H = (s @ H @ s).tocsr()
            B = None
            B_mode = "diagonal (mass-weighted once, no solves)"
    if check:
        check_hermitian(H, tol=hermitian_tol, name="A")
    if B is not None:
        return _estimate_generalized(H, B, M, R, probe, bounds, margin, complex_phases, seed, npts, verbose, method,
                                     maxent_default, maxent_noise, kernel, lorentz_lambda, vectors, hermitian_tol, check,
                                     exact_moments, binv, precond, block_size, binv_tol, cheb_tol, bmargin, lanczos_steps,
                                     check_bounds, t0, known_eigenvalues)
    if complex_phases is None:
        complex_phases = is_complex(H)
    if probe > 0 and is_matrix_free(H):
        raise ValueError("probing needs the sparsity pattern; use an explicit sparse matrix or probe=0")
    if bounds is None:
        emin, emax = spectral_bounds(H, verbose=verbose)
    else:
        emin, emax = bounds
    bounds_raw = (float(emin), float(emax))
    w = emax - emin
    emin -= margin * w
    emax += margin * w
    a = 0.5 * (emax - emin)
    b = 0.5 * (emax + emin)
    Ht = ScaledOperator(H, a, b)
    if verbose:
        print(f"spectral window [{emin:.6g}, {emax:.6g}], a = {a:.6g}, b = {b:.6g}  "
              f"({time.time() - t0:.2f} s)")

    colors, ncolors = (None, 1)
    if probe > 0:
        tc = time.time()
        colors, ncolors = distance_coloring(H, probe)
        if verbose:
            print(f"distance-{probe} coloring: {ncolors} colors ({time.time() - tc:.2f} s); "
                  f"moments n <= {probe} are exact")

    rng = np.random.default_rng(seed)
    tm = time.time()
    cg_info = []
    mus = stochastic_moments(Ht, n, M, R, rng, colors, ncolors, complex_phases, kind=vectors,
                             is_complex=is_complex(H), traces=traces, info=cg_info)
    if verbose:
        nmv = R * ncolors * ((M + 1) // 2 + 1)
        print(f"{R} sample(s) x {ncolors} vector(s), M = {M} moments, "
              f"{nmv} mat-vecs ({time.time() - tm:.2f} s)")
    if exact_moments:
        K = int(exact_moments)
        if K >= M:
            raise ValueError("exact_moments must be < M")
        if Ht.M is None:
            raise ValueError("exact_moments needs an explicit sparse matrix")
        te = time.time()
        mu_ex = exact_chebyshev_moments(Ht.M, K, chunk=exact_chunk)
        mus = mus.copy()
        mus[:, :K + 1] = mu_ex[None, :]
        if verbose:
            print(f"exact moments 0..{K} by the local method ({time.time() - te:.2f} s)")

    mus = _remove_known(mus, a, b, n, known_eigenvalues)
    mu = mus.mean(axis=0)
    if check_bounds:
        _check_moment_bounds(mu, n, R, verbose)
    out = DOSResult(mu=mu, mu_samples=mus, a=a, b=b, N=n, M=M, R=R, ncolors=ncolors,
                    method=method, maxent_default=maxent_default, kernel=kernel, vectors=vectors,
                    bounds_raw=bounds_raw, problem="standard" if B_mode == "none" else "generalized, diagonal B",
                    exact_moments=int(exact_moments), B_mode=B_mode, known_removed=_known_count(known_eigenvalues))
    if cg_info:
        out["impartial_info"] = cg_info
    return _reconstruct_result(out, mus, a, b, R, method, npts, kernel, lorentz_lambda, maxent_default, maxent_noise, M, verbose)


def _estimate_generalized(H, B, M, R, probe, bounds, margin, complex_phases, seed, npts, verbose, method, maxent_default,
                          maxent_noise, kernel, lorentz_lambda, vectors, hermitian_tol, check, exact_moments, binv, precond,
                          block_size, binv_tol, cheb_tol, bmargin, lanczos_steps, check_bounds, t0, known_eigenvalues=None):
    n = H.shape[0]
    if probe > 0:
        raise ValueError("probing is not available for the generalized problem")
    if vectors == "impartial":
        raise ValueError("impartial vectors need the exact traces of B^-1 A and its square, which are not available "
                         "for a non-diagonal B")
    if exact_moments:
        raise NotImplementedError("exact local moments do not exist for a generalized problem with a non-diagonal B "
                                  "(B^-1 A is not local); use exact_moments=0, or mass-weight a diagonal B")
    cplx = bool(is_complex(H) or np.iscomplexobj(B.data))
    if complex_phases is None:
        complex_phases = cplx
    op = GeneralizedOperator(H, B, binv=binv, precond=precond, block_size=block_size, tol=binv_tol, cheb_tol=cheb_tol,
                             bmargin=bmargin, hermitian_tol=hermitian_tol, check=check, seed=0 if seed is None else seed)
    if bounds is None:
        lo, hi, resid = generalized_bounds(op.apply, op.B, n, m=lanczos_steps, seed=0 if seed is None else seed, is_cplx=cplx)
        emin, emax = lo - resid, hi + resid
    else:
        emin, emax = bounds
    bounds_raw = (float(emin), float(emax))
    w = emax - emin
    emin -= margin * w
    emax += margin * w
    a = 0.5 * (emax - emin)
    b = 0.5 * (emax + emin)
    if verbose:
        print(f"generalized problem, B^-1 by {binv}: spectral window [{emin:.6g}, {emax:.6g}] ({time.time() - t0:.2f} s)")
    Xop = lambda V: (op.apply(V) - b * V) / a
    rng = np.random.default_rng(seed)
    mus = np.zeros((R, M))
    tm = time.time()
    s0 = op.nsolve
    for r in range(R):
        X = make_vectors(n, None, 1, rng, complex_phases, kind=vectors, is_complex=cplx)
        mus[r] = chebyshev_moments_plain(Xop, X, M) / n
    if verbose:
        print(f"{R} sample(s), M = {M} moments, {op.nsolve - s0} B^-1 applications, {op.iters} inner iterations/matvec-blocks "
              f"({time.time() - tm:.2f} s)")
    mus = _remove_known(mus, a, b, n, known_eigenvalues)
    mu = mus.mean(axis=0)
    if check_bounds:
        _check_moment_bounds(mu, n, R, verbose)
    out = DOSResult(mu=mu, mu_samples=mus, a=a, b=b, N=n, M=M, R=R, ncolors=1, method=method, maxent_default=maxent_default,
                    kernel=kernel, vectors=vectors, bounds_raw=bounds_raw, problem="generalized", exact_moments=0,
                    B_mode=binv, binv_info=dict(op.info, nsolve=op.nsolve, iterations=op.iters),
                    known_removed=_known_count(known_eigenvalues))
    return _reconstruct_result(out, mus, a, b, R, method, npts, kernel, lorentz_lambda, maxent_default, maxent_noise, M, verbose)


def fermi_interval(res, n_elec, spin=2.0, nsigma=2.0):
    """Interval [E_lo, E_hi] on which the KPM integrated DOS equals the electron count within ``nsigma`` standard errors (needs R > 1), and its midpoint.

    A single Fermi level is only well defined when this interval is narrow.  In a gapped system (or when the count noise exceeds the DOS slope) the
    IDOS is flat at the filling over the whole gap, so the interval spans it: that is information, not an error.  Returns (E_lo, E_hi, E_mid)."""
    if res["method"] != "kpm" or res["R"] < 2:
        raise ValueError("fermi_interval needs a KPM result with R > 1 vectors")
    target = n_elec / (spin * res["N"])
    err = np.nan_to_num(res["idos_err"], nan=0.0)
    inside = np.abs(res["idos"] - target) <= nsigma * err + 1e-12
    if not inside.any():                                        # error bars too small to reach the target anywhere: fall back to the crossing
        k = int(np.searchsorted(res["idos"], target))
        k = min(max(k, 1), len(res["E"]) - 1)
        return float(res["E"][k - 1]), float(res["E"][k]), float(0.5 * (res["E"][k - 1] + res["E"][k]))
    idx = np.nonzero(inside)[0]
    lo, hi = float(res["E"][idx[0]]), float(res["E"][idx[-1]])
    return lo, hi, 0.5 * (lo + hi)


def fermi_and_band_energy(res, n_elec, spin=2.0):
    """E_F and E_BS (with error bars if available) for either method."""
    N, a, b = res["N"], res["a"], res["b"]
    if res["method"] == "kpm":
        kw = dict(kernel=res.get("kernel", "jackson"))
        val = band_quantities(res["mu"], a, b, N, n_elec, spin, **kw)
        samples = ([band_quantities(m, a, b, N, n_elec, spin, **kw) for m in res["mu_samples"]]
                   if res["R"] > 1 else None)
        if samples is not None:
            err = np.array(samples).std(axis=0, ddof=1) / np.sqrt(res["R"])
        else:
            err = (np.nan, np.nan)
    else:
        d = res["maxent_default"]
        E, rho, idos = maxent_density(res["lam"], a, b, d, mu=res["mu"])
        val = band_quantities_grid(E, rho, idos, N, n_elec, spin)
        if res["lam_jackknife"] is not None:
            R = res["R"]
            jk = np.array([band_quantities_grid(*maxent_density(l, a, b, d, mu=res["mu"]), N, n_elec, spin)
                           for l in res["lam_jackknife"]])
            err = np.sqrt((R - 1) / R * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
        else:
            err = (np.nan, np.nan)
    if res["method"] == "kpm" and res["R"] > 1:
        lo, hi, _ = fermi_interval(res, n_elec, spin)
        width = hi - lo
        if width > 3.0 * res.get("resolution", 0.0) and width > 0.1:
            warnings.warn(f"the Fermi level is ill-defined here: the integrated DOS equals the electron count within noise over [{lo:.3g}, {hi:.3g}] "
                          f"(width {width:.3g}, resolution {res.get('resolution', float('nan')):.3g}), i.e. a gap or stochastic noise. E_F = {val[0]:.4g} is only "
                          "one point of that interval; it matters as a reference energy for plots. The band energy is (nearly) independent of where E_F lies in a "
                          "true gap, since the DOS there is zero (the residual KPM dependence, from kernel leakage and noise, is about the size of the method's own "
                          "error). See fermi_interval, and sdos.homo_lumo for the exact HOMO/LUMO where a factorization is feasible.", stacklevel=2)
    return val[0], val[1], err[0], err[1]
