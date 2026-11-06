import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

import sdos
from conftest import random_hermitian


@pytest.mark.parametrize("complex_", [False, True])
def test_matrix_market_symmetric_and_hermitian(tmp_path, complex_):
    """Regression for a bug in the original sdos.py: symmetric/Hermitian Matrix Market files were rejected."""
    H = random_hermitian(80, complex_, seed=3)
    p = tmp_path / "h.mtx"
    scipy.io.mmwrite(str(p), H, symmetry="hermitian" if complex_ else "symmetric")
    G = sdos.read_matrix(p, verbose=False)
    assert abs(G - H).max() < 1e-12


def test_matrix_market_general_and_npz(tmp_path):
    H = random_hermitian(80, True, seed=4)
    scipy.io.mmwrite(str(tmp_path / "g.mtx"), H)
    assert abs(sdos.read_matrix(tmp_path / "g.mtx", verbose=False) - H).max() < 1e-12
    sp.save_npz(tmp_path / "h.npz", H)
    assert abs(sdos.read_matrix(tmp_path / "h.npz", verbose=False) - H).max() < 1e-12


def test_triangle_flag_and_base(tmp_path):
    H = random_hermitian(60, False, seed=5)
    p = tmp_path / "h.dat"
    sdos.write_triplets(H, p, "lower", base=1)
    assert abs(sdos.read_matrix(p, base="1", storage="triangle", verbose=False) - H).max() < 1e-12
    with pytest.raises(ValueError):
        sdos.write_triplets(H, p, "full", base=0)
        sdos.read_matrix(p, storage="triangle", verbose=False)
