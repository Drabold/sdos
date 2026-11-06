"""Does the band-energy error of the DFTB problem of DTW4096 (Table dd: -70 +- 35 meV/atom, R = 6) fall as R^-1/2?  KPM, M = 170, PCG block-Jacobi, several seeds
per R; error against the dense generalized spectrum (cached dtw_dftb_gamma_ev.npy).  usage: python dtw_dftb_Rscan.py"""
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from sdos.interfaces import skf
d = np.load("/home/drabold/lammps-runs/topoMC/voids/host4096_ref.npz"); X, L = d["X"], d["L"]
sk = skf.SKSet("/home/drabold/opt/slako/pbc-0-3", ["Si"]); H, S = skf.skf_hamiltonian(X, ["Si"] * len(X), L, sk); n = H.shape[0]
ev = np.load("dtw_dftb_gamma_ev.npy"); ebs = 2.0 * ev[:n // 2].sum(); out = []
for R, seeds in ((6, (11, 12, 13, 14)), (24, (21, 22)), (96, (31,))):
    for sd in seeds:
        t = time.time(); r = sdos.dos(H, B=S, M=170, R=R, method="kpm", seed=sd, binv="pcg", precond="block-jacobi", verbose=False)
        ef, eb, _, eeb = sdos.fermi_and_band_energy(r, n)
        out.append(dict(R=R, seed=sd, err_meV_atom=float((eb - ebs) / 4096 * 1e3), sampling_se_meV_atom=float(eeb / 4096 * 1e3), sec=time.time() - t))
        print(out[-1], flush=True); json.dump(out, open("dtw_dftb_Rscan.json", "w"), indent=1)
