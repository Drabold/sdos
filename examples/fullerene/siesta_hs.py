"""SIESTA Harris-functional, single-zeta (H, S) for a fullerene geometry (npy of Cartesian Angstrom positions).
usage: python siesta_hs.py C960_lcbop.npy OUTDIR [ranks] [diag|ntpoly] [MeshCutoff Ry]   (needs the siesta micromamba env of ~/siesta_runs/README.md and the PseudoDojo C.psml)"""
import os, subprocess, sys, time
import numpy as np

SIESTA_ENV = os.path.expanduser("~/opt/micromamba/root/envs/siesta")
PSML = os.path.expanduser("~/siesta_runs/pseudos/library/LDA/C.psml")


def write_fdf(x, path, label="fuller", vacuum=14.0, mesh=150.0, solver="diag"):
    x = x - x.mean(axis=0)
    L = float(np.ptp(x, axis=0).max() + vacuum)
    x = x + L / 2
    with open(path, "w") as f:
        f.write(f"SystemName {label}\nSystemLabel {label}\nNumberOfAtoms {len(x)}\nNumberOfSpecies 1\n%block ChemicalSpeciesLabel\n  1  6  C\n%endblock ChemicalSpeciesLabel\n"
                f"LatticeConstant 1.0 Ang\n%block LatticeVectors\n  {L:.4f} 0 0\n  0 {L:.4f} 0\n  0 0 {L:.4f}\n%endblock LatticeVectors\nAtomicCoordinatesFormat Ang\n%block AtomicCoordinatesAndAtomicSpecies\n")
        for p in x:
            f.write(f"  {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} 1\n")
        f.write("%endblock AtomicCoordinatesAndAtomicSpecies\n"
                f"PAO.BasisType split\nPAO.BasisSize SZ\nPAO.EnergyShift 0.02 Ry\nXC.functional LDA\nXC.authors CA\nMeshCutoff {mesh} Ry\n"
                + ("SolutionMethod diagon\n" if solver == "diag" else "SolutionMethod ELSI\nELSI.Solver NTPOLY\n") + "MaxSCFIterations 1\nHarris_functional true\nDM.UseSaveDM false\nSaveHS true\nWriteEigenvalues true\n")
    return L


def run(npy, outdir, ranks=4, solver="diag", mesh=150.0):
    os.makedirs(outdir, exist_ok=True)
    x = np.load(npy)
    L = write_fdf(x, f"{outdir}/fuller.fdf", mesh=mesh, solver=solver)
    if not os.path.exists(f"{outdir}/C.psml"):
        subprocess.run(["cp", PSML, f"{outdir}/C.psml"], check=True)
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(f"{outdir}/fuller.out", "w") as out:
        r = subprocess.run([f"{SIESTA_ENV}/bin/mpirun", "-np", str(ranks), f"{SIESTA_ENV}/bin/siesta", "fuller.fdf"], cwd=outdir, env=env, stdout=out, stderr=subprocess.STDOUT)
    return r.returncode, time.time() - t0, L


if __name__ == "__main__":
    rc, t, L = run(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4, sys.argv[4] if len(sys.argv) > 4 else "diag",
                   float(sys.argv[5]) if len(sys.argv) > 5 else 150.0)
    print(f"rc={rc}  {t:.1f} s  cell {L:.1f} A")
