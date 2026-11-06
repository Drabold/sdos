"""LCBOP relaxation of the Ih series C60 n^2 and comparison with Itoh, Ordejon, Drabold, Martin, PRB 53, 2132 (1996): Table V (E_rel) and Table IV (radii)."""
import sys, time
import numpy as np
import goldberg as G
import relax_compare as RC

BENCH_E = {60: (0.3674, 0.3279), 240: (0.1253, 0.1204), 540: (0.0693, 0.0726), 960: (0.0430, 0.0511), 2160: (0.0166, 0.0221), 3840: (0.0084, None)}
BENCH_R = {240: (7.06, 0.17), 540: (10.53, 0.36), 960: (14.02, 0.53), 2160: (20.95, 0.82), 3840: (27.95, 1.08)}      # ETB: mean radius, sd (A)

if __name__ == "__main__":
    style = sys.argv[1]
    ns = [int(v) for v in sys.argv[2:]]
    eg = RC.graphene_energy(style)
    print(f"{style}: graphene E/atom = {eg:.4f} eV", flush=True)
    for n in ns:
        t0 = time.time()
        pos, nb, T = G.fullerene(n, n)
        x, epa = RC.relax(pos, style)
        N = len(x)
        r = np.linalg.norm(x - x.mean(axis=0), axis=1)
        bl = np.array([np.linalg.norm(x[i] - x[j]) for i in range(N) for j in nb[i] if i < j])
        Erel = epa - eg
        be = BENCH_E.get(N)
        br = BENCH_R.get(N)
        s = f"C{N:5d}: E_rel {Erel:.4f}"
        if be:
            s += f" (ETB {be[0]:.4f}" + (f", AITB {be[1]:.4f})" if be[1] else ")") + f" ratio to ETB {Erel / be[0]:.2f}"
        s += f" | mean radius {r.mean():.2f} sd {r.std():.2f} sd/r {r.std() / r.mean():.3f}"
        if br:
            s += f" (ETB {br[0]:.2f}, {br[1]:.2f}, {br[1] / br[0]:.3f})"
        s += f" | bonds {bl.min():.3f}-{bl.max():.3f} | {time.time() - t0:.0f}s"
        print(s, flush=True)
        np.save(f"C{N}_{style}.npy", x)
