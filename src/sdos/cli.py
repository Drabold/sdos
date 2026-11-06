"""Command-line interface: ``sdos H.dat -M 512 -R 10 --probe 2 -o dos.dat --electrons 4096``."""

import argparse
import sys

import numpy as np

from . import __version__
from .api import dos as dos_api
from .dos import fermi_and_band_energy
from .phonon import vdos
from .io import read_matrix, write_dos
from .windows import shift_invert_dos

__doc__ += """

Run ``sdos -h`` for all options.  Input: triplets ``i j Re(A_ij) [Im(A_ij)]`` (0/1-based, one
triangle or both), Matrix Market, or SciPy ``.npz``.
"""

def main(argv=None):
    if (argv if argv is not None else sys.argv[1:])[:1] == ["analyze"]:
        from .results import analyze
        return analyze((argv if argv is not None else sys.argv[1:])[1:])
    p = argparse.ArgumentParser(prog="sdos", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"sdos {__version__}")
    p.add_argument("matrix", help="triplet file: i j Re [Im]  (or Matrix Market)")
    p.add_argument("-M", "--moments", type=int, default=256,
                   help="number of Chebyshev moments (resolution ~ pi*halfwidth/M)")
    p.add_argument("-R", "--samples", type=int, default=10,
                   help="number of independent random samples (for averaging and error bars)")
    p.add_argument("--probe", type=int, default=0,
                   help="distance-d coloring for probing (0 = plain random vectors)")
    p.add_argument("--B", default=None, help="second matrix B of A x = lambda B x (same formats as the matrix; Hermitian positive definite)")
    p.add_argument("--mass", default=None, help="text file with one mass (diagonal B) per degree of freedom: mass-weights once, no solves")
    p.add_argument("--binv", choices=["pcg", "chebyshev"], default="pcg", help="how B^-1 is applied: PCG, or a fixed Chebyshev polynomial")
    p.add_argument("--precond", choices=["block-jacobi", "ilu", "amg", "none"], default="block-jacobi", help="PCG preconditioner")
    p.add_argument("--block-size", type=int, default=1, help="per-site block size for block-Jacobi / scaling (e.g. orbitals per atom)")
    p.add_argument("--binv-tol", type=float, default=1e-12, help="PCG relative tolerance / polynomial accuracy")
    p.add_argument("--exact-moments", default="auto", help="'auto' (first 4 moments exact when cheap), 0 (off) or K")
    p.add_argument("--thermo", type=float, nargs=3, metavar=("TMIN", "TMAX", "NT"), help="with --omega: harmonic thermodynamics (U, F, S, Cv per atom) on a T grid, "
                   "written to <output>.thermo (lambda in eV/(A^2 amu); use --zero-modes 3 for crystals/molecules)")
    p.add_argument("--zero-modes", type=int, default=0, help="number of known zero modes (rigid translations: 3) to remove from the moments exactly")
    p.add_argument("--omega", action="store_true", help="phonon output: write omega, g(omega)=2 omega rho(omega^2), error (lambda = omega^2)")
    p.add_argument("--window", type=float, nargs=2, metavar=("E1", "E2"), help="eigenpairs inside [E1, E2] with the IPR of each state (see --window-method)")
    p.add_argument("--window-method", choices=["auto", "shift-invert", "filter"], default="auto",
                   help="shift-invert (exact, NOT O(N)) or Chebyshev-filtered subspace iteration (scalable)")
    p.add_argument("--window-degree", type=int, default=300, help="filter polynomial degree")
    p.add_argument("--vectors-out", default=None, help="save the window eigenvectors (npz) as well")
    p.add_argument("--base", choices=["auto", "0", "1"], default="auto")
    p.add_argument("--format", choices=["auto", "triplet", "mm", "npz"], default="auto", help="input format")
    p.add_argument("--kernel", choices=["jackson", "lorentz"], default="jackson", help="KPM damping kernel")
    p.add_argument("--lorentz-lambda", type=float, default=4.0, help="lambda of the Lorentz kernel")
    p.add_argument("--vectors", choices=["auto", "phase", "rademacher", "gaussian", "impartial"], default="auto", help="random-vector ensemble; impartial = Drabold-Sankey penalty-function vectors (Maxent-friendly, first 3 moments exact)")
    p.add_argument("--storage", choices=["auto", "triangle", "full"], default="auto")
    p.add_argument("--emin", type=float, help="lower spectral bound (skip estimation)")
    p.add_argument("--emax", type=float, help="upper spectral bound (skip estimation)")
    p.add_argument("--complex-phases", action="store_true",
                   help="use exp(i phi) vectors even for real H (half the variance, 2x cost)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--npts", type=int, default=None, help="energy grid points")
    p.add_argument("--method", choices=["kpm", "maxent", "si"], default="kpm",
                   help="kpm (Jackson kernel, default), maxent, or si (shift-and-invert "
                        "Lanczos in a window; needs --si-range)")
    p.add_argument("--si-range", type=float, nargs=2, metavar=("E1", "E2"),
                   help="energy window for --method si (e.g. a gap/tail/defect region)")
    p.add_argument("--si-shifts", type=int, default=1,
                   help="number of shifts (LU factorizations) spread over the window")
    p.add_argument("--si-steps", type=int, default=150,
                   help="Lanczos steps per vector at each shift")
    p.add_argument("--si-eta", type=float, default=None,
                   help="Gaussian broadening of Ritz nodes (default: mean node spacing)")
    p.add_argument("--maxent-default", choices=["chebyshev", "flat", "kpm"], default="chebyshev",
                   help="Maxent default model: 1/(pi sqrt(1-x^2)) (default), flat, or kpm (the Jackson density; robust to a too-wide spectral window)")
    p.add_argument("--maxent-noise", type=float, default=1.0,
                   help="scale on moment standard errors in the Maxent dual (0 = fit moments exactly)")
    p.add_argument("--electrons", type=float, default=None,
                   help="number of electrons: report Fermi level and band energy")
    p.add_argument("--spin", type=float, default=2.0, help="spin degeneracy (default 2)")
    p.add_argument("-o", "--output", default="dos.dat",
                   help="output file; a name ending in .npz saves a full result file (moments, settings) for `sdos analyze`")
    p.add_argument("--moments-out", default=None, help="write averaged moments to file")
    p.add_argument("--plot", default=None, help="save a plot (png/pdf)")
    args = p.parse_args(argv)

    H = read_matrix(args.matrix, base=args.base, storage=args.storage, fmt=args.format)
    bounds = None
    if args.emin is not None and args.emax is not None:
        bounds = (args.emin, args.emax)
    if args.window is not None:
        from .eigwin import window_eigs
        Bw = None
        if args.B:
            Bw = read_matrix(args.B, base=args.base, storage=args.storage, fmt=args.format)
        elif args.mass:
            Bw = np.loadtxt(args.mass).ravel()
        w = window_eigs(H, Bw, tuple(args.window), method=args.window_method, degree=args.window_degree,
                        block_size=args.block_size, precond=args.precond)
        np.savetxt(args.output, np.column_stack([w["E"], w["ipr"], w["residual"]]),
                   header=f"sdos window [{args.window[0]}, {args.window[1]}] method={w['info']['method']} n_found={len(w['E'])} "
                          f"(KPM estimate {w['info']['n_estimated']:.1f})\nE   IPR   residual", fmt="%.10e")
        if args.vectors_out:
            np.savez(args.vectors_out, E=w["E"], V=w["V"], ipr=w["ipr"])
        print(f"{len(w['E'])} states in the window ({w['info']['method']}); wrote {args.output}")
        return
    if args.method == "si":
        if args.si_range is None:
            p.error("--method si requires --si-range E1 E2")
        res = shift_invert_dos(H, args.si_range, nshift=args.si_shifts, m=args.si_steps,
                               R=args.samples, eta=args.si_eta, npts=args.npts,
                               seed=args.seed, complex_phases=args.complex_phases or None)
        if args.electrons is not None:
            print("note: --electrons ignored for --method si (window DOS only)", file=sys.stderr)
            args.electrons = None
    else:
        Bobj = None
        if args.B:
            Bobj = read_matrix(args.B, base=args.base, storage=args.storage, fmt=args.format)
        elif args.mass:
            Bobj = np.loadtxt(args.mass).ravel()
        exact = "auto" if args.exact_moments == "auto" else int(args.exact_moments)
        kw = dict(M=args.moments, R=args.samples, probe=args.probe, bounds=bounds, complex_phases=args.complex_phases or None,
                  seed=args.seed, npts=args.npts, method=args.method, maxent_default=args.maxent_default,
                  maxent_noise=args.maxent_noise, kernel=args.kernel, lorentz_lambda=args.lorentz_lambda, vectors=args.vectors,
                  exact_moments=exact, binv=args.binv, precond=args.precond, block_size=args.block_size, binv_tol=args.binv_tol,
                  verbose=True)
        res = vdos(H, B=Bobj, zero_modes=args.zero_modes, **kw) if args.omega else dos_api(H, B=Bobj, **kw)

    N = res["N"]
    if res["method"] == "kpm":
        recon = f"Jackson resolution ~ {res['resolution']:.4g}"
    elif res["method"] == "si":
        recon = f"shift-invert Lanczos, window {res['erange']}, eta ~ {res['resolution']:.4g}"
    else:
        recon = f"Maxent, {res['maxent_default']} default"
    header = (f"sdos {__version__}  N={N}  M={res['M']}  samples={res['R']}  colors={res['ncolors']}  "
              f"{recon}\n"
              "E   rho(E) [per orbital, integrates to 1]   err   IDOS(E) [fraction]   err")
    if args.output.endswith(".npz"):
        from .results import save_result
        save_result(args.output, res, {"command": " ".join(sys.argv[1:]) if argv is None else " ".join(argv), "matrix": args.matrix})
    elif args.omega and "omega" in res:
        np.savetxt(args.output, np.column_stack([res["omega"], res["g"], res["g_err"]]),
                   header=header + "\nomega  g(omega)=2 omega rho(omega^2) [integrates to 1 - neg. weight]  err", fmt="%.10e")
    else:
        np.savetxt(args.output, np.column_stack([res["E"], res["rho"], res["rho_err"],
                                                 res["idos"], res["idos_err"]]),
                   header=header, fmt="%.10e")
    print(f"wrote {args.output}")
    if args.thermo and args.omega:
        from .thermo import harmonic_thermo
        Tg = np.linspace(args.thermo[0], args.thermo[1], int(args.thermo[2]))
        th = harmonic_thermo(res, Tg)
        np.savetxt(args.output + ".thermo", np.column_stack([Tg, th["U"], th["F"], th["S"], th["Cv"]]),
                   header=f"harmonic thermodynamics per atom; zero-point energy {th['zpe']:.6g} eV; reliable for T >~ {th['info']['T_reliable']:.0f} K\nT[K]  U[eV]  F[eV]  S[kB]  Cv[kB]", fmt="%.8e")
        print(f"wrote {args.output}.thermo (zero-point energy {th['zpe']:.5g} eV/atom; reliable for T >~ {th['info']['T_reliable']:.0f} K)")
    if args.moments_out:
        np.savetxt(args.moments_out, res["mu"], header="mu_n = Tr T_n(Ht)/N", fmt="%.12e")

    if args.electrons is not None:
        EF, Ebs, dEF, dEbs = fermi_and_band_energy(res, args.electrons, args.spin)
        if res["R"] > 1:
            print(f"E_F = {EF:.6f} +- {dEF:.2g}    E_BS = {Ebs:.6f} +- {dEbs:.2g}  "
                  f"({Ebs / args.electrons:.6f} per electron)")
        else:
            print(f"E_F = {EF:.6f}    E_BS = {Ebs:.6f}  ({Ebs / args.electrons:.6f} per electron)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(res["E"], res["rho"], lw=1)
        if res["R"] > 1:
            ax.fill_between(res["E"], res["rho"] - 2 * res["rho_err"],
                            res["rho"] + 2 * res["rho_err"], alpha=0.3, lw=0)
        ax.set_xlabel("E")
        ax.set_ylabel("DOS per orbital")
        ax.set_title(f"N = {N}, M = {res['M']}, {res['R']} sample(s), {res['method'].upper()}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")

if __name__ == "__main__":
    main()
