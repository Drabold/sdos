"""Relax Goldberg fullerenes with LAMMPS carbon potentials (rebo, airebo, lcbop) and compare bond lengths and strain energy to graphene."""
import os, re, subprocess, sys, tempfile
import numpy as np
import goldberg as G

POT = os.path.expanduser("~/src/lammps/potentials")
STYLES = {"rebo": ("pair_style rebo", "pair_coeff * * CH.rebo C"), "airebo": ("pair_style airebo 3.0", "pair_coeff * * CH.airebo C"),
          "lcbop": ("pair_style lcbop", "pair_coeff * * C.lcbop C")}
LMP = os.path.expanduser("~/bin/lmp_kk")


def run(infile, d):
    env = dict(os.environ, LAMMPS_POTENTIALS=POT, OMP_NUM_THREADS="2")
    r = subprocess.run([LMP, "-k", "off", "-sf", "off", "-in", infile, "-screen", "none", "-log", os.path.join(d, "log")], cwd=d, env=env, capture_output=True, text=True)
    return open(os.path.join(d, "log")).read(), r


def energy_from_log(log):
    m = re.findall(r"^\s*\d+\s+(-?\d+\.\d+)\s", log, re.M)
    return None


def graphene_energy(style):
    with tempfile.TemporaryDirectory() as d:
        a = 2.46
        ax, ay = a, a * 3 ** 0.5
        pos = [(0, 0), (a / 2, a * 3 ** 0.5 / 6), (a / 2, a * 3 ** 0.5 / 2), (0, a * 3 ** 0.5 * 2 / 3)]
        pos = [(0, 0), (a / 2, ay / 2), (a / 2, ay / 2 + a / 3 ** 0.5 / 2 * 0 + 0), (0, 0)]  # placeholder replaced below
        b = a / 3 ** 0.5
        pos = [(0, 0), (0, b), (a / 2, 1.5 * b), (a / 2, 2.5 * b)]
        with open(f"{d}/g.data", "w") as f:
            f.write(f"graphene\n\n4 atoms\n1 atom types\n\n0 {ax} xlo xhi\n0 {ay} ylo yhi\n-10 10 zlo zhi\n\nMasses\n\n1 12.011\n\nAtoms # atomic\n\n")
            for i, (x, y) in enumerate(pos, 1):
                f.write(f"{i} 1 {x} {y} 0.0\n")
        open(f"{d}/in", "w").write(f"""units metal\nboundary p p f\natom_style atomic\nread_data g.data\nreplicate 6 4 1\n{STYLES[style][0]}\n{STYLES[style][1]}
thermo_style custom step pe\nfix r all box/relax x 0 y 0 vmax 0.001\nminimize 0 1e-10 20000 100000\nunfix r\nminimize 0 1e-12 20000 100000\nvariable e equal pe/atoms\nprint "EPA $e"\n""")
        log, r = run(f"{d}/in", d)
        return float(re.search(r"EPA (-?\d+\.\d+)", log).group(1))


def relax(pos, style):
    with tempfile.TemporaryDirectory() as d:
        G.write_lammps_data(f"{d}/c.data", pos)
        open(f"{d}/in", "w").write(f"""units metal\nboundary f f f\natom_style atomic\nread_data c.data\n{STYLES[style][0]}\n{STYLES[style][1]}
thermo_style custom step pe\nmin_style fire\nminimize 0 1e-8 20000 200000\nmin_style cg\nminimize 0 1e-12 20000 200000\nvariable e equal pe/atoms\nprint "EPA $e"
write_dump all custom out.xyz id x y z modify sort id\n""")
        log, r = run(f"{d}/in", d)
        epa = float(re.search(r"EPA (-?\d+\.\d+)", log).group(1))
        x = np.loadtxt(f"{d}/out.xyz", skiprows=9)[:, 1:4]
        return x, epa


if __name__ == "__main__":
    ns = [int(v) for v in sys.argv[1:]] or [1, 2, 3, 4]
    for style in STYLES:
        eg = graphene_energy(style)
        print(f"{style}: graphene E/atom = {eg:.4f} eV")
        for n in ns:
            pos, nb, T = G.fullerene(n, n)
            x, epa = relax(pos, style)
            # bonds from the generator's topology
            bl = np.array([np.linalg.norm(x[i] - x[j]) for i in range(len(x)) for j in nb[i] if i < j])
            sz = G.faces_from_geometry(x, nb)
            Erel = epa - eg
            ref = 9.1 * len(x) ** -0.75
            extra = f"  bonds {bl.min():.3f}-{bl.max():.3f} A (mean {bl.mean():.3f})"
            print(f"   C{len(x):5d}: E_rel = {Erel:.4f} eV/atom   (1995 TB fit 9.1 N^-0.75 = {ref:.4f}){extra}")
