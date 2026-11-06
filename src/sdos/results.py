"""Self-describing result files and the ``sdos analyze`` command.

A result file (``.npz``) holds everything the DOS stage produced: rho(E), its error, the averaged and per-sample Chebyshev moments,
the spectral window, the method and its settings (and omega, g(omega) for vibrations).  It is written by ``sdos H.dat -o result.npz``
or ``res.save("result.npz")`` and read by :func:`load` or ``sdos analyze result.npz``.  The DOS stage knows only the matrix; whatever
needs outside information (number of electrons, spin degeneracy, temperature) is applied afterwards, from the file, with
no access to the matrix.  Plain-text output (any other file extension) is unchanged.
"""

import json
import sys

import numpy as np

from . import __version__

_JSON = "_json_"
FORMAT = "sdos-result-1"


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o).__name__)


def save_result(path, res, extra=None):
    """Write a :class:`DOSResult` (or :func:`sdos.vdos` result) to a ``.npz`` file; ``extra`` is a dict of notes (e.g. the command line)."""
    arrays = {}
    for k, v in res.items():
        if isinstance(v, np.ndarray) and v.dtype != object:
            arrays[k] = v
        elif isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_)):
            arrays[k] = np.asarray(v)
        elif isinstance(v, str):
            arrays[_JSON + k] = np.asarray(json.dumps(v))
        else:
            try:
                arrays[_JSON + k] = np.asarray(json.dumps(v, default=_jsonable))
            except TypeError:
                print(f"note: key {k!r} ({type(v).__name__}) is not saved", file=sys.stderr)
    arrays["_format"] = np.asarray(FORMAT)
    arrays["_sdos_version"] = np.asarray(__version__)
    arrays["_extra"] = np.asarray(json.dumps(extra or {}, default=_jsonable))
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez_compressed(path, **arrays)
    return path


def load(path):
    """Read a result file written by :func:`save_result` / ``sdos -o x.npz``; returns a :class:`~sdos.DOSResult`."""
    from .dos import DOSResult
    out = DOSResult()
    with np.load(str(path), allow_pickle=False) as z:
        if "_format" not in z or str(z["_format"]) != FORMAT:
            raise ValueError(f"{path} is not an sdos result file")
        for k in z.files:
            if k.startswith("_") and not k.startswith(_JSON):
                continue
            if k.startswith(_JSON):
                out[k[len(_JSON):]] = json.loads(str(z[k]))
            else:
                v = z[k]
                out[k] = v.item() if v.ndim == 0 else v
        out["_sdos_version"] = str(z["_sdos_version"])
        out["_extra"] = json.loads(str(z["_extra"]))
    # JSON lists that are really arrays of numbers (jackknife moments etc.)
    for k in ("lam", "lam_jackknife"):
        if isinstance(out.get(k), list):
            out[k] = np.asarray(out[k], float)
    return out


def electrons_from_valence(path, charge=0.0):
    """Number of electrons from a text file with the valence (Z) of every atom, one number per line (or whitespace-separated), minus ``charge``."""
    z = np.loadtxt(path, comments="#", ndmin=1)
    return float(z.sum()) - float(charge)


def _rebuild(res, M2, method, maxent_default, kernel, npts):
    """Re-reconstruct from the saved per-sample moments (first M2 of them), optionally with another method; no matrix needed."""
    from .dos import DOSResult, _reconstruct_result
    if "mu_samples" not in res:
        raise ValueError("this result has no saved moments")
    if res.get("problem", "standard") != "standard" and "omega" in res:
        raise ValueError("re-reconstruction of vibrational results is not supported; recompute with sdos --omega")
    mus = np.asarray(res["mu_samples"])
    M2 = min(M2 or mus.shape[1], mus.shape[1])
    mus = mus[:, :M2]
    new = DOSResult({k: v for k, v in res.items()
                     if k not in ("E", "rho", "rho_err", "idos", "idos_err", "lam", "lam_jackknife", "maxent_info", "resolution")})
    new.update(mu=mus.mean(axis=0), mu_samples=mus, M=M2, method=method or res["method"],
               maxent_default=maxent_default or res.get("maxent_default", "chebyshev"), kernel=kernel or res.get("kernel", "jackson"))
    return _reconstruct_result(new, mus, new["a"], new["b"], new["R"], new["method"], npts, new["kernel"], 4.0,
                               new["maxent_default"], 1.0, M2, False)


def analyze(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="sdos analyze", description="Analyse a saved sdos result file (no matrix needed): "
                                "Fermi level and band energy, harmonic thermodynamics, re-reconstruction, plots, text export.")
    p.add_argument("result", help="result file written by `sdos H.dat -o result.npz`")
    p.add_argument("--info", action="store_true", help="print what the file contains (default if nothing else is asked)")
    g = p.add_argument_group("electrons")
    g.add_argument("--electrons", type=float, help="number of electrons")
    g.add_argument("--valence-file", help="text file with the valence electrons (Z) of each atom, one per line; sum = number of electrons")
    g.add_argument("--charge", type=float, default=0.0, help="net charge (electrons removed), with --valence-file")
    g.add_argument("--spin", type=float, default=2.0, help="spin degeneracy (default 2)")
    g = p.add_argument_group("vibrations")
    g.add_argument("--thermo", type=float, nargs=3, metavar=("TMIN", "TMAX", "NT"), help="harmonic U, F, S, Cv per atom (needs a --omega result)")
    g.add_argument("--dof", type=int, default=3, help="degrees of freedom per atom")
    g = p.add_argument_group("re-reconstruction from the saved moments")
    g.add_argument("--moments", type=int, help="use only the first M moments")
    g.add_argument("--method", choices=["kpm", "maxent"])
    g.add_argument("--maxent-default", choices=["chebyshev", "flat", "kpm"])
    g.add_argument("--kernel", choices=["jackson", "lorentz"])
    g.add_argument("--npts", type=int)
    g = p.add_argument_group("output")
    g.add_argument("-o", "--output", help="write the (re-reconstructed) spectrum as text: E rho err idos err")
    g.add_argument("--save", help="write the (re-reconstructed) result to a new .npz")
    g.add_argument("--plot", help="save a plot (png/pdf)")
    a = p.parse_args(argv)

    res = load(a.result)
    if any([a.moments, a.method, a.maxent_default, a.kernel, a.npts]):
        res = _rebuild(res, a.moments, a.method, a.maxent_default, a.kernel, a.npts)
        print(f"re-reconstructed: {res['method']} from the first {res['M']} moments")
    asked = any([a.electrons is not None, a.valence_file, a.thermo, a.output, a.save, a.plot])
    if a.info or not asked:
        print(res.summary())
        ex = res.get("_extra") or {}
        if ex:
            print("  made with: " + ", ".join(f"{k}={v}" for k, v in ex.items()))
        print(f"  written by sdos {res.get('_sdos_version', '?')}")
    nel = a.electrons
    if a.valence_file:
        nel = electrons_from_valence(a.valence_file, a.charge)
        print(f"electrons from {a.valence_file}: {nel:g}")
    if nel is not None:
        from .dos import fermi_and_band_energy
        EF, Ebs, dEF, dEbs = fermi_and_band_energy(res, nel, a.spin)
        err = lambda x: f" +- {x:.2g}" if np.isfinite(x) else ""
        print(f"E_F = {EF:.6f}{err(dEF)}    E_BS = {Ebs:.6f}{err(dEbs)}  ({Ebs / nel:.6f} per electron)")
    if a.thermo:
        from .thermo import harmonic_thermo
        if "omega" not in res:
            p.error("--thermo needs a result made with --omega (vibrational spectrum)")
        T = np.linspace(a.thermo[0], a.thermo[1], int(a.thermo[2]))
        th = harmonic_thermo(res, T, dof_per_atom=a.dof)
        print(f"zero-point energy {th['zpe']:.6g} eV/atom; reliable for T >~ {th['info']['T_reliable']:.0f} K")
        print("T[K]  U[eV]  F[eV]  S[kB]  Cv[kB]")
        for row in zip(T, th["U"], th["F"], th["S"], th["Cv"]):
            print("  ".join(f"{x:.6g}" for x in row))
    if a.output:
        from .io import write_dos
        write_dos(a.output, res, f"sdos analyze {a.result}: N={res['N']} M={res['M']} R={res['R']} method={res['method']}")
        print(f"wrote {a.output}")
    if a.save:
        print(f"wrote {save_result(a.save, res, {'from': a.result})}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        ax = res.plot()
        ax.figure.tight_layout()
        ax.figure.savefig(a.plot, dpi=150)
        print(f"wrote {a.plot}")
