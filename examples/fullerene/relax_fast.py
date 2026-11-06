"""Loose LCBOP relaxation (FIRE only, ftol 1e-4 eV/A) of big Goldberg fullerenes: enough for the DOS (bond lengths to ~0.01 A).  usage: relax_fast.py n"""
import os, re, subprocess, sys, tempfile, time
import numpy as np
import goldberg as G
import relax_compare as RC

def relax_fast(pos, style="lcbop", ftol=1e-4, maxit=40000):
    with tempfile.TemporaryDirectory() as d:
        G.write_lammps_data(f"{d}/c.data", pos)
        open(f"{d}/in", "w").write(f"""units metal\\nboundary f f f\\natom_style atomic\\nread_data c.data\\n{RC.STYLES[style][0]}\\n{RC.STYLES[style][1]}
thermo_style custom step pe fmax\\nthermo 500\\nmin_style fire\\nminimize 0 {ftol} {maxit} {10 * maxit}\\nvariable e equal pe/atoms\\nprint "EPA $e"
write_dump all custom out.xyz id x y z modify sort id\\n""".replace("\\n", "\n"))
        log, r = RC.run(f"{d}/in", d)
        epa = float(re.search(r"EPA (-?\d+\.\d+)", log).group(1))
        its = re.findall(r"^\s*(\d+)\s+-?\d+\.\d+\s+(\d+\.\d+)", log, re.M)
        return np.loadtxt(f"{d}/out.xyz", skiprows=9)[:, 1:4], epa, (its[-1] if its else None)

if __name__ == "__main__":
    n = int(sys.argv[1]); t0 = time.time()
    pos, nb, T = G.fullerene(n, n)
    x, epa, last = relax_fast(pos)
    eg = RC.graphene_energy("lcbop"); N = len(x)
    r = np.linalg.norm(x - x.mean(0), axis=1)
    bl = np.array([np.linalg.norm(x[i] - x[j]) for i in range(N) for j in nb[i] if i < j])
    np.save(f"C{N}_lcbop.npy", x)
    print(f"C{N}: E_rel {epa - eg:.4f} eV/atom | mean radius {r.mean():.2f} sd/r {r.std() / r.mean():.3f} | bonds {bl.min():.3f}-{bl.max():.3f} | last thermo {last} | {time.time() - t0:.0f} s", flush=True)
