import numpy as np
import pytest
import scipy.sparse as sp

import sdos
from sdos.cli import main


def _chain(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 0.3, n)
    o = -np.ones(n - 1)
    return sp.diags([o, d, o], [-1, 0, 1]).tocsr()


@pytest.mark.parametrize("method", ["kpm", "maxent"])
def test_roundtrip_gives_same_fermi_level(tmp_path, method):
    H = _chain()
    r = sdos.dos(H, M=60, R=4, method=method, seed=3, verbose=False)
    p = r.save(tmp_path / "r.npz")
    r2 = sdos.load(p)
    for k in ("E", "rho", "idos", "mu", "mu_samples"):
        np.testing.assert_array_equal(r[k], r2[k])
    a = sdos.fermi_and_band_energy(r, 200, 2.0)
    b = sdos.fermi_and_band_energy(r2, 200, 2.0)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_cli_npz_then_analyze(tmp_path, capsys):
    H = _chain(300)
    coo = sp.triu(H).tocoo()
    f = tmp_path / "h.dat"
    np.savetxt(f, np.column_stack([coo.row + 1, coo.col + 1, coo.data]), fmt="%d %d %.12g")   # one triangle, 1-based
    out = tmp_path / "res.npz"
    main([str(f), "-M", "50", "-R", "3", "--seed", "1", "-o", str(out)])
    val = tmp_path / "z.txt"
    np.savetxt(val, np.full(300, 1.0))
    capsys.readouterr()
    main(["analyze", str(out), "--valence-file", str(val), "--spin", "2"])
    txt = capsys.readouterr().out
    assert "E_F" in txt and "electrons from" in txt
    # re-reconstruct from fewer moments with Maxent, write text
    main(["analyze", str(out), "--moments", "30", "--method", "maxent", "-o", str(tmp_path / "t.dat")])
    assert np.loadtxt(tmp_path / "t.dat").shape[1] == 5
    main(["analyze", str(out), "--info"])
    assert "made with" in capsys.readouterr().out


def test_vibrational_result_thermo(tmp_path, capsys):
    rng = np.random.default_rng(0)
    n = 300
    K = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]).tocsr()
    K = (K + sp.diags(0.1 * rng.random(n))).tocsr()
    r = sdos.vdos(K, M=80, R=3, seed=1, verbose=False)
    p = r.save(tmp_path / "v.npz")
    main(["analyze", str(p), "--thermo", "100", "300", "2"])
    out = capsys.readouterr().out
    assert "zero-point" in out
    a = sdos.harmonic_thermo(r, [100.0, 300.0])
    b = sdos.harmonic_thermo(sdos.load(p), [100.0, 300.0])
    np.testing.assert_allclose(a["F"], b["F"], rtol=1e-12)


def test_not_a_result_file(tmp_path):
    np.savez(tmp_path / "x.npz", a=np.arange(3))
    with pytest.raises(ValueError):
        sdos.load(tmp_path / "x.npz")
