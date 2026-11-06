"""Kwon tight-binding DOS of the 10^6-atom amorphous-silicon model of Morrow et al. (Angew. Chem. Int. Ed. 63, e202403842 (2024); Zenodo 10.5281/zenodo.10794053, CC-BY-4.0).
The structure file is NOT part of this repository: download aSi_1Matom_maintext.xyz from Zenodo and pass its path.  Gamma point, one structure.
Usage:  python morrow_dos.py aSi_1Matom_maintext.xyz [M] [R]    (run with OMP/BLAS threads capped, e.g. systemd-run -p MemoryMax=14G)
Writes morrow1M_result.npz (sdos result file with all moments) and morrow1M_log.json."""
import sys, json, time, warnings
import numpy as np
import sdos
from sdos.interfaces import kwon_si as K
warnings.filterwarnings("ignore")
path = sys.argv[1]; M = int(sys.argv[2]) if len(sys.argv) > 2 else 300; R = int(sys.argv[3]) if len(sys.argv) > 3 else 2
t0 = time.time()
with open(path) as f:
    n = int(f.readline()); hdr = f.readline()
    lat = np.array(hdr.split('Lattice="')[1].split('"')[0].split(), float).reshape(3, 3)
    data = np.loadtxt(f, usecols=(1, 2, 3), max_rows=n)
assert n == len(data) and np.allclose(lat, np.diag(np.diag(lat))), "orthorhombic cell expected"
L = np.diag(lat); X = np.mod(data, L)
print(f"read {n} atoms, box {L}, {time.time() - t0:.0f} s", flush=True)
t = time.time(); H = K.kwon_hamiltonian(X, L, chunk=16000); tb = time.time() - t
print(f"H: {H.shape[0]} orbitals, {H.nnz / H.shape[0]:.1f} nnz/row, build {tb:.0f} s", flush=True)
t = time.time(); lo, hi = sdos.spectral_bounds(H, verbose=False); tbd = time.time() - t
print(f"bounds [{lo:.3f}, {hi:.3f}] {tbd:.0f} s", flush=True)
t = time.time(); r = sdos.dos(H, M=M, R=R, method="kpm", seed=1, exact_moments=0, bounds=(lo, hi), verbose=False); tm = time.time() - t
import resource
log = dict(atoms=int(n), orbitals=int(H.shape[0]), M=M, R=R, t_build=tb, t_bounds=tbd, t_dos=tm, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, window=[float(lo), float(hi)])
r.save("morrow1M_result.npz"); json.dump(log, open("morrow1M_log.json", "w"), indent=1)
print(log, flush=True); print("MORROW_DONE", flush=True)
