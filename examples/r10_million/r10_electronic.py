"""Kwon tight-binding DOS of the 17^3 replica of the 216-atom R10 model (1 061 208 atoms, 4 243 968 orbitals), Gamma point of the supercell, KPM and Maxent from the same
moments; bounds = exact extremes of the replica spectrum (union of the 216-atom cell over the 17^3 k-mesh, exact_kwon_r10.py).  usage: python r10_electronic.py [M] [R] [K]
Writes r10_kwon_result_M{M}_R{R}_K{K}.npz and a log json."""
import sys, json, time, resource, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from sdos.interfaces import kwon_si as KW
from common_r10 import *
M = int(sys.argv[1]) if len(sys.argv) > 1 else 600; R = int(sys.argv[2]) if len(sys.argv) > 2 else 4; Kx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ev = np.load("kwon_r10_exact_ev_17.npy"); lo, hi = float(ev[0]), float(ev[-1])
t = time.time(); pos, box = replica(17); H = KW.kwon_hamiltonian(pos, box, chunk=16000); tb = time.time() - t
print(f"H: {H.shape[0]} orbitals, {H.nnz / H.shape[0]:.1f} nnz/row, build {tb:.0f} s", flush=True)
t = time.time(); r = sdos.dos(H, M=M, R=R, method="kpm", seed=3, exact_moments=Kx, bounds=(lo, hi), verbose=False); tm = time.time() - t
log = dict(atoms=len(pos), orbitals=int(H.shape[0]), M=M, R=R, K=Kx, t_build=tb, t_dos=tm, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, window=[lo, hi])
r.save(f"r10_kwon_result_M{M}_R{R}_K{Kx}.npz"); json.dump(log, open(f"r10_kwon_log_M{M}_R{R}_K{Kx}.json", "w"), indent=1); print(log, flush=True); print("R10E_DONE", flush=True)
