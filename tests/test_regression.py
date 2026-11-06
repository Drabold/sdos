"""The package reproduces the original sdos.py bit-for-bit (same seeds) on its existing test cases."""
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp

import sdos
from conftest import cubic, random_hermitian

KEYS = ["mu", "mu_samples", "E", "rho", "rho_err", "idos", "idos_err"]


def gersh(H):
    return sdos.spectral_bounds(H, "gershgorin", verbose=False)


CASES = {
    "anderson_real": lambda: cubic(8, W=2.0, seed=0),
    "complex_hermitian": lambda: random_hermitian(500, True),
    "real_random": lambda: random_hermitian(500, False),
    "peierls": lambda: cubic(7, W=1.0, seed=3, phases=True),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("opts", [dict(), dict(probe=2), dict(complex_phases=True)])
def test_kpm_identical(v0, case, opts):
    H = CASES[case]()
    b = gersh(H)
    old = v0.estimate_dos(H, M=64, R=4, bounds=b, seed=11, verbose=False, **opts)
    new = sdos.estimate_dos(H, M=64, R=4, bounds=b, seed=11, verbose=False, **opts)
    for k in KEYS:
        np.testing.assert_allclose(new[k], old[k], rtol=1e-12, atol=1e-14, err_msg=k)
    assert new.a == old["a"] and new.b == old["b"]


def test_maxent_identical(v0):
    H = cubic(8, W=2.0, seed=0)
    b = gersh(H)
    old = v0.estimate_dos(H, M=40, R=5, bounds=b, seed=2, method="maxent", verbose=False)
    new = sdos.estimate_dos(H, M=40, R=5, bounds=b, seed=2, method="maxent", verbose=False)
    for k in ["lam", "E", "rho", "rho_err", "idos", "idos_err"]:
        np.testing.assert_allclose(new[k], old[k], rtol=1e-10, atol=1e-12, err_msg=k)


@pytest.mark.parametrize("method", ["kpm", "maxent"])
def test_fermi_band_energy_identical(v0, method):
    H = cubic(8, W=1.0, seed=5)
    b = gersh(H)
    old = v0.estimate_dos(H, M=64, R=4, bounds=b, seed=3, method=method, verbose=False)
    new = sdos.estimate_dos(H, M=64, R=4, bounds=b, seed=3, method=method, verbose=False)
    n_el = 0.5 * 2 * H.shape[0]
    np.testing.assert_allclose(sdos.fermi_and_band_energy(new, n_el, 2.0),
                               v0.fermi_and_band_energy(old, n_el, 2.0), rtol=1e-10)


@pytest.mark.parametrize("complex_", [False, True])
def test_shift_invert_identical(v0, complex_):
    H = random_hermitian(300, complex_, seed=4)
    old = v0.shift_invert_dos(H, (-2.0, 2.0), nshift=2, m=60, R=3, seed=5, verbose=False)
    new = sdos.shift_invert_dos(H, (-2.0, 2.0), nshift=2, m=60, R=3, seed=5, verbose=False)
    for k in ["E", "rho", "rho_err", "idos", "idos_err"]:
        np.testing.assert_allclose(new[k], old[k], rtol=1e-10, atol=1e-12, err_msg=k)


@pytest.mark.parametrize("triangle", ["upper", "lower", "full"])
@pytest.mark.parametrize("base", [0, 1])
@pytest.mark.parametrize("complex_", [False, True])
def test_read_matrix_identical(v0, tmp_path, triangle, base, complex_):
    H = random_hermitian(120, complex_, seed=6)
    p = tmp_path / "h.dat"
    sdos.write_triplets(H, p, triangle=triangle, base=base)
    old = v0.read_matrix(str(p), verbose=False)
    new = sdos.read_matrix(p, verbose=False)
    assert abs(old - new).max() == 0
    assert abs(new - H).max() < 1e-12


def test_cli_identical(v0, tmp_path):
    H = cubic(6, W=1.0, seed=2)
    p = tmp_path / "h.dat"
    sdos.write_triplets(H, p)
    ref = tmp_path / "ref.py"
    out_new, out_old = tmp_path / "new.dat", tmp_path / "old.dat"
    args = [str(p), "-M", "48", "-R", "3", "--seed", "4", "--emin", "-6.5", "--emax", "6.5", "--electrons", "216"]
    subprocess.run([sys.executable, "-m", "sdos", *args, "--exact-moments", "0", "-o", str(out_new)], check=True, capture_output=True)
    subprocess.run([sys.executable, str(__import__("conftest").REF), *args, "-o", str(out_old)], check=True, capture_output=True)
    np.testing.assert_allclose(np.loadtxt(out_new), np.loadtxt(out_old), rtol=1e-12, atol=1e-14)
