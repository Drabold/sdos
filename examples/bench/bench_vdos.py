"""Cost scaling, vibrational case: matrix-free Hessian of the Stillinger-Weber potential (LAMMPS python module) on tiled, SW-relaxed DTW4096; KPM M=100, R=2, 2 force calls per product."""
import sys, json, time, warnings
import numpy as np
import sdos
from sdos.interfaces.lmp import LammpsForces
from common import rss_gb, Timer
warnings.filterwarnings("ignore")
DIR = "/home/drabold/lammps-runs/topoMC/sdos_dtw"
out = []
for n in map(int, sys.argv[1:]):
    # tiled data file from the SW-relaxed 4096-atom cell
    import re
    txt = open(f"{DIR}/dtw4096_sw_relaxed.data").read().splitlines()
    L = [float(re.findall(r"[-0-9.eE+]+", txt[i])[1]) for i in (5, 6, 7)]
    k = next(i for i, l in enumerate(txt) if l.startswith("Atoms")) + 2
    P = np.array([[float(x) for x in l.split()[2:5]] for l in txt[k:k + 4096]])
    sh = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3) * np.array(L)
    X = (P[None] + sh[:, None, :]).reshape(-1, 3)
    with open(f"{DIR}/dtw_sw_tile{n}.data", "w") as f:
        f.write(f"tiled {n}\n\n{len(X)} atoms\n1 atom types\n\n0 {L[0] * n:.8f} xlo xhi\n0 {L[1] * n:.8f} ylo yhi\n0 {L[2] * n:.8f} zlo zhi\n\nMasses\n\n1 28.0855\n\nAtoms # atomic\n\n")
        for i, p in enumerate(X): f.write(f"{i + 1} 1 {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    open(f"{DIR}/in.swtile{n}", "w").write(f"package omp 4\nsuffix omp\nunits metal\nboundary p p p\natom_style atomic\natom_modify map array sort 0 0.0\nread_data {DIR}/dtw_sw_tile{n}.data\nmass 1 28.0855\npair_style sw\npair_coeff * * /home/drabold/src/lammps/potentials/Si.sw Si\nneighbor 1.0 bin\n")
    with Timer() as ts: F = LammpsForces(f"{DIR}/in.swtile{n}")
    op = F.operator(delta=1e-3)
    with Timer() as tm: r = sdos.vdos(op, zero_modes=3, M=100, R=2, method="kpm", seed=1, hermitian_tol=op.hermitian_tol)
    th = sdos.harmonic_thermo(r, [300.0])
    row = dict(n=n, atoms=len(X), dof=3 * len(X), t_setup=ts.s, t_vdos=tm.s, force_calls=F.calls, sec_per_call=tm.s / F.calls, peak_rss_gb=rss_gb(), zpe_meV=float(th.zpe * 1e3), cv300=float(np.atleast_1d(th.Cv)[0]))
    out.append(row); print(row, flush=True)
    json.dump(out, open("bench_vdos.json", "w"), indent=1)
    F.lmp.close()
