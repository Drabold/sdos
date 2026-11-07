"""B2: is the -0.5..-0.7% ZPE error of Table VI (DTW4096, M=170, R=4) sampling, resolution near lambda=0, or the Lanczos window?
Matrix-free KPM with the Lanczos window (as in Table VI) and with the window from the exact spectrum, several seeds, R=4 and 16, M=170 and 340; ZPE and Cv(100 K) errors against the exact dense spectrum.
usage: python zpe_check.py sw|ace  -> zpe_check_<pot>.json"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, sdos
from sdos.interfaces.lmp import LammpsForces
from sdos.thermo import HBAR_OMEGA_METAL
pot = sys.argv[1]; DIR = "/home/drabold/lammps-runs/topoMC/sdos_dtw"
lam = np.sort(np.load("dtw4096_ace_lambda.npy")) if pot == "ace" else np.sort(np.array(json.load(open("dtw_phonons_sw.json"))["exact_lambda"]))
om = np.sqrt(np.clip(lam[3:], 0, None)) * HBAR_OMEGA_METAL; kb = 8.617333262e-5; zpe_ex = 0.5 * om.sum() / 4096
x = om / (kb * 100.0); cv_ex = (x ** 2 * np.exp(x) / np.expm1(x) ** 2).sum() / 4096
print("exact lambda range", lam[0], lam[-1], "ZPE", zpe_ex * 1e3, "Cv100", cv_ex, flush=True)
F = LammpsForces(f"{DIR}/in.{pot}"); op = F.operator(delta=1e-3); out = []
cases = [("lanczos", 170, 4, (1, 2, 3, 4, 5)), ("exact", 170, 4, (1, 2, 3, 4, 5)), ("exact", 170, 16, (1, 2)), ("exact", 340, 4, (1, 2, 3))] if pot == "sw" else [("exact", 170, 4, (3, 4))]
for win, M, R, seeds in cases:
    for s in seeds:
        kw = dict(bounds=(lam[0] - 1e-3, lam[-1] * 1.0005)) if win == "exact" else {}
        F.calls = 0; t = time.time(); r = sdos.vdos(op, zero_modes=3, M=M, R=R, method="kpm", seed=s, hermitian_tol=op.hermitian_tol, verbose=False, **kw)
        th = sdos.harmonic_thermo(r, [100.0]); z = float(th.zpe); c = float(np.atleast_1d(th.Cv)[0])
        row = dict(pot=pot, window=win, M=M, R=R, seed=s, zpe_err_pct=100 * (z / zpe_ex - 1), cv100_err_pct=100 * (c / cv_ex - 1), a=float(r["a"]), b=float(r["b"]), calls=F.calls, sec=time.time() - t)
        out.append(row); print(row, flush=True); json.dump(out, open(f"zpe_check_{pot}.json", "w"), indent=1)
import os; os._exit(0)
