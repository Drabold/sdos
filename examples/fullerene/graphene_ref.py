"""Graphene reference from the SAME SIESTA Harris/single-zeta Hamiltonian as the fullerenes: H(R), S(R) of the 2-atom cell, then the exact DOS from a dense k-mesh."""
import os, subprocess, sys, warnings
import numpy as np

SIESTA_ENV = os.path.expanduser("~/opt/micromamba/root/envs/siesta")
PSML = os.path.expanduser("~/siesta_runs/pseudos/library/LDA/C.psml")


def run_graphene(outdir, a=2.4595, mesh=150.0):
    os.makedirs(outdir, exist_ok=True)
    subprocess.run(["cp", PSML, f"{outdir}/C.psml"], check=True)
    b = a / 3 ** 0.5
    open(f"{outdir}/gra.fdf", "w").write(f"""SystemName graphene
SystemLabel gra
NumberOfAtoms 2
NumberOfSpecies 1
%block ChemicalSpeciesLabel
  1  6  C
%endblock ChemicalSpeciesLabel
LatticeConstant 1.0 Ang
%block LatticeVectors
  {a:.6f} 0.0 0.0
  {a / 2:.6f} {a * 3 ** 0.5 / 2:.6f} 0.0
  0.0 0.0 20.0
%endblock LatticeVectors
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
  0.0 0.0 0.0 1
  {a / 2:.6f} {a / (2 * 3 ** 0.5):.6f} 0.0 1
%endblock AtomicCoordinatesAndAtomicSpecies
PAO.BasisType split
PAO.BasisSize SZ
PAO.EnergyShift 0.02 Ry
XC.functional LDA
XC.authors CA
MeshCutoff {mesh} Ry
%block kgrid_Monkhorst_Pack
  12 0 0 0.0
  0 12 0 0.0
  0 0 1 0.0
%endblock kgrid_Monkhorst_Pack
SolutionMethod diagon
MaxSCFIterations 1
Harris_functional true
DM.UseSaveDM false
SaveHS true
WriteEigenvalues true
""")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(f"{outdir}/gra.out", "w") as out:
        r = subprocess.run([f"{SIESTA_ENV}/bin/mpirun", "-np", "2", f"{SIESTA_ENV}/bin/siesta", "gra.fdf"], cwd=outdir, env=env, stdout=out, stderr=subprocess.STDOUT)
    return r.returncode


def graphene_dos(outdir, E, eta=0.06, nk=240):
    """Exact DOS per orbital: Gaussian-broadened sum over a nk x nk k-mesh (eigenvalues relative to the Fermi level as in sisl)."""
    import sisl
    warnings.filterwarnings("ignore")
    H = sisl.get_sile(f"{outdir}/gra.fdf").read_hamiltonian()
    ks = (np.arange(nk) + 0.5) / nk
    ev = []
    for i in range(nk):
        for chunk in np.array_split(ks, 4):
            kk = np.array([[ks[i], c, 0.0] for c in chunk])
            ev.append(np.array([np.sort(H.eigh(k=k)) for k in kk]))
    ev = np.concatenate(ev).ravel()
    dos = np.zeros_like(E)
    for chunk in np.array_split(ev, 200):
        dos += np.exp(-0.5 * ((E[:, None] - chunk[None]) / eta) ** 2).sum(axis=1)
    return dos / (len(ev) * eta * np.sqrt(2 * np.pi))


if __name__ == "__main__":
    out = os.path.expanduser("~/siesta_runs/fullerene/graphene")
    print("siesta rc =", run_graphene(out))
