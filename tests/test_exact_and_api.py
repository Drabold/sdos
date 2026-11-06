"""Exact local moments (test of the directive's extra), and the simple front door sdos.dos."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sdos
from conftest import cubic, exact_moments, random_hermitian


@pytest.mark.parametrize("phases", [False, True])
@pytest.mark.parametrize("K", [1, 2, 5, 8, 13])
def test_local_exact_moments(phases, K):
    H = cubic(6, W=3.0, seed=1, phases=phases)
    ev = sla.eigvalsh(H.toarray())
    w = ev[-1] - ev[0]
    lo, hi = ev[0] - 0.01 * w, ev[-1] + 0.01 * w
    a, b = 0.5 * (hi - lo), 0.5 * (hi + lo)
    Ht = sdos.ScaledOperator(H, a, b)
    np.testing.assert_allclose(sdos.exact_chebyshev_moments(Ht.M, K, chunk=40), exact_moments(ev, a, b, K + 1)[:K + 1], atol=1e-11)


def test_exact_moments_option_is_unbiased_and_cuts_band_energy_noise():
    H = cubic(8, W=3.0, seed=1)
    n = H.shape[0]
    ev = sla.eigvalsh(H.toarray())
    b = sdos.spectral_bounds(H, "gershgorin", verbose=False)
    plain = sdos.estimate_dos(H, M=30, R=60, bounds=b, seed=7, verbose=False, vectors="gaussian")
    ex = sdos.estimate_dos(H, M=30, R=60, bounds=b, seed=7, verbose=False, vectors="gaussian", exact_moments=4)
    spread = lambda r: np.std([sdos.band_quantities(m, r.a, r.b, n, n, 2.0)[1] / n for m in r.mu_samples], ddof=1)
    assert spread(ex) < spread(plain) / 4
    mex = exact_moments(ev, ex.a, ex.b, 30)
    np.testing.assert_allclose(ex.mu_samples[:, :5], np.tile(mex[:5], (60, 1)), atol=1e-11)


def test_dos_front_door_inputs(tmp_path):
    H = cubic(7, W=2.0, seed=2)
    r1 = sdos.dos(H, quick=True, seed=1)
    p = tmp_path / "h.dat"
    sdos.write_triplets(H, p)
    r2 = sdos.dos(str(p), quick=True, seed=1)
    r3 = sdos.dos(H.toarray(), quick=True, seed=1)
    np.testing.assert_allclose(r2.rho, r1.rho, rtol=1e-9)
    np.testing.assert_allclose(r3.rho, r1.rho, rtol=1e-9)
    op = spla.LinearOperator(H.shape, matvec=lambda x: H @ x, matmat=lambda X: H @ X, dtype=H.dtype)
    r4 = sdos.dos(op, quick=True, seed=1, bounds=sdos.spectral_bounds(H, "gershgorin", verbose=False))
    assert r4.exact_moments == 0 and r1.exact_moments == 4            # matrix-free: no exact moments
    assert abs(np.sum(r1.rho * np.gradient(r1.E)) - 1) < 2e-2
    assert "DOSResult" in repr(r1)
    r1.save(tmp_path / "dos.txt")
    assert (tmp_path / "dos.txt").exists()


def test_dos_resolution_and_generalized_and_options():
    H = cubic(7, W=2.0, seed=2)
    r = sdos.dos(H, resolution=0.2, quick=True)
    assert abs(r.resolution - 0.2) < 0.03
    m = np.random.default_rng(0).uniform(1, 3, H.shape[0])
    g = sdos.dos(H, B=m, quick=True, seed=1)                          # masses -> mass-weighted, exact moments allowed
    assert g.exact_moments == 4
    me = sdos.dos(H, method="maxent", M=40, R=4, seed=1)
    assert me.method == "maxent"
    im = sdos.dos(H, vectors="impartial", M=64, R=3, seed=1, exact_moments=0)
    assert im.vectors == "impartial"


def test_cli_generalized_mass_omega_and_exact(tmp_path):
    import subprocess, sys
    from keating import keating_hessian
    Phi = keating_hessian(nc=2)
    m = np.repeat(np.random.default_rng(4).choice([28.0855, 72.63], size=Phi.shape[0] // 3), 3)
    sdos.write_triplets(Phi, tmp_path / "phi.dat")
    np.savetxt(tmp_path / "mass.txt", m)
    out = tmp_path / "vdos.dat"
    r = subprocess.run([sys.executable, "-m", "sdos", str(tmp_path / "phi.dat"), "--mass", str(tmp_path / "mass.txt"), "-M", "96", "-R", "8",
                        "--seed", "1", "--omega", "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    d = np.loadtxt(out)
    assert d.shape[1] == 3 and d[:, 0].min() >= 0
    out2 = tmp_path / "gen.dat"
    sdos.write_triplets(sp.diags(m).tocsr(), tmp_path / "B.dat")
    r2 = subprocess.run([sys.executable, "-m", "sdos", str(tmp_path / "phi.dat"), "--B", str(tmp_path / "B.dat"), "--binv", "chebyshev",
                         "-M", "64", "-R", "3", "--seed", "1", "--emin=-1e-6", "--emax=3", "-o", str(out2)], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr[-400:]


def test_cli_window(tmp_path):
    import subprocess, sys
    H = random_hermitian(150, False, seed=4)
    ev = sla.eigvalsh(H.toarray())
    sdos.write_triplets(H, tmp_path / "h.dat")
    out, vec = tmp_path / "w.dat", tmp_path / "v.npz"
    lo, hi = ev[60] - 1e-6, ev[80] + 1e-6
    r = subprocess.run([sys.executable, "-m", "sdos", str(tmp_path / "h.dat"), "--window", str(lo), str(hi), "--window-method", "shift-invert",
                        "--vectors-out", str(vec), "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    d = np.loadtxt(out)
    assert d.shape == (21, 3) and np.abs(d[:, 0] - ev[60:81]).max() < 1e-8 and d[:, 2].max() < 1e-8
    assert np.load(vec)["V"].shape == (150, 21)


def test_maxent_single_vector_drops_exact_moments():
    """Maxent with R = 1 and substituted moments does not converge (inconsistent moment list); estimate_dos must warn and use exact_moments = 0."""
    import warnings
    import numpy as np
    import scipy.sparse as sp
    import sdos
    n = 200
    H = sp.diags([-np.ones(n - 1), np.ones(n) * 0.1, -np.ones(n - 1)], [-1, 0, 1], format="csr")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = sdos.dos(H, M=60, R=1, method="maxent", exact_moments=4, seed=1, verbose=False)
    assert any("exact_moments=0" in str(x.message) for x in w)
    assert r["exact_moments"] == 0


def test_warns_when_M_reaches_N():
    """M >= N: the resolution is finer than the level spacing; dos() warns and suggests dense diagonalization."""
    import warnings
    import numpy as np
    import scipy.sparse as sp
    import sdos
    n = 40
    H = sp.diags([-np.ones(n - 1), np.zeros(n), -np.ones(n - 1)], [-1, 0, 1], format="csr")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sdos.dos(H, M=60, R=2, exact_moments=0, seed=1, verbose=False)
    assert any("dense diagonalization" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sdos.dos(H, M=20, R=2, exact_moments=0, seed=1, verbose=False)
    assert not any("dense diagonalization" in str(x.message) for x in w)
