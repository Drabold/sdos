"""DFTB .skf reader/builder. Parameter files are not shipped: set SDOS_SKF_DIR (default ~/opt/slako/pbc-0-3); DFTB+ comparisons also need the executable (SDOS_DFTBPLUS or the
default path below) and are skipped otherwise."""
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
import scipy.linalg as sla

import sdos
from sdos.interfaces import kwon_si, skf

SKDIR = os.environ.get("SDOS_SKF_DIR", os.path.expanduser("~/opt/slako/pbc-0-3"))
DFTB = os.environ.get("SDOS_DFTBPLUS", os.path.expanduser("~/opt/mamba/envs/dft/bin/dftb+"))
need_skf = pytest.mark.skipif(not os.path.exists(os.path.join(SKDIR, "Si-Si.skf")), reason="no .skf files (set SDOS_SKF_DIR)")
need_dftb = pytest.mark.skipif(not (os.path.exists(DFTB) and os.path.exists(os.path.join(SKDIR, "Si-Si.skf"))), reason="no DFTB+ executable or .skf files")
HERE = pathlib.Path(__file__).parent


@need_skf
def test_parser_and_grid():
    sk = skf.SKSet(SKDIR, ["Si"])
    f = sk.files[("Si", "Si")]
    assert f.dx == 0.02 and f.ngrid == 519          # the header says 520: row 1 sits at r = dx and r = 0 is omitted
    assert f.first == 19                            # 19 placeholder rows (r < 0.4 Bohr)
    assert abs(sk.onsite["Si"][0] - (-0.39572506 * skf.HARTREE)) < 1e-9 and abs(sk.onsite["Si"][1] - (-0.15031380 * skf.HARTREE)) < 1e-9
    assert 6.0 < sk.rcut < 6.1                      # table end (5.49 A) + the 1 Bohr quintic tail


@need_skf
def test_block_rotation_covariance_and_hermiticity():
    sk = skf.SKSet(SKDIR, ["Si", "H"])
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    d = rng.normal(size=(5, 3)) * 1.2 + np.array([2.0, 0, 0])
    for a, b in (("Si", "Si"), ("Si", "H"), ("H", "Si")):
        Hb, Sb = sk.blocks(a, b, d)
        Hr, Sr = sk.blocks(a, b, d @ Q.T)
        na, nb = Hb.shape[1:]
        Pa = np.eye(na); Pb = np.eye(nb)
        if na == 4: Pa[1:, 1:] = Q
        if nb == 4: Pb[1:, 1:] = Q
        for X, Y in ((Hb, Hr), (Sb, Sr)):
            assert np.max(np.abs(np.einsum("ij,njk,lk->nil", Pa, X, Pb) - Y)) < 1e-10
    # (a, b, d) and (b, a, -d) blocks are transposes of each other
    H1, S1 = sk.blocks("Si", "H", d)
    H2, S2 = sk.blocks("H", "Si", -d)
    assert np.max(np.abs(H1 - np.transpose(H2, (0, 2, 1)))) < 1e-10 and np.max(np.abs(S1 - np.transpose(S2, (0, 2, 1)))) < 1e-10


@need_skf
def test_supercell_equals_primitive_k_sum_generalized_complex():
    n = 3
    pos, box = kwon_si.diamond_positions(n)
    k = np.array([0.11, -0.19, 0.27])
    sk = skf.SKSet(SKDIR, ["Si"])
    sym = ["Si"] * len(pos)
    H, S = skf.skf_hamiltonian(pos, sym, box, sk, kpoint=k)
    assert np.iscomplexobj(H.data) and abs(H - H.getH()).max() < 1e-12 and abs(S - S.getH()).max() < 1e-12
    ev = sla.eigh(H.toarray(), S.toarray(), eigvals_only=True)
    assert ev[0] > -20 and np.linalg.eigvalsh(S.toarray())[0] > 0.05          # S positive definite
    cell, _ = kwon_si.diamond_positions(1)
    G = 2 * np.pi / (n * kwon_si.A_LATTICE)
    ref = []
    for i in range(n):
        for j in range(n):
            for l in range(n):
                Hk, Sk = skf.skf_hamiltonian(cell, ["Si"] * 8, np.full(3, kwon_si.A_LATTICE), sk, kpoint=k + G * np.array([i, j, l]))
                ref.append(sla.eigh(Hk.toarray(), Sk.toarray(), eigvals_only=True))
    assert np.max(np.abs(np.sort(np.concatenate(ref)) - ev)) < 1e-8


@need_skf
def test_sdos_generalized_on_skf_matrices():
    pos, box = kwon_si.diamond_positions(3)
    pos = pos + np.random.default_rng(1).uniform(-0.1, 0.1, pos.shape)
    sk = skf.SKSet(SKDIR, ["Si"])
    H, S = skf.skf_hamiltonian(pos, ["Si"] * len(pos), box, sk)
    ev = sla.eigh(H.toarray(), S.toarray(), eigvals_only=True)
    n = H.shape[0]
    g = sdos.homo_lumo(H, n // 2, B=S)
    assert abs(g["gap"] - (ev[n // 2] - ev[n // 2 - 1])) < 1e-6


def _run_dftb(tmp, pos, symbols, species, lattice, ang):
    gen = [f"{len(pos)} S", " ".join(species)]
    for i, (s, p) in enumerate(zip(symbols, pos)):
        gen.append(f"{i + 1} {species.index(s) + 1} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}")
    gen.append("0 0 0")
    for k in range(3):
        v = np.zeros(3); v[k] = lattice[k]; gen.append(f"{v[0]:.12f} {v[1]:.12f} {v[2]:.12f}")
    (tmp / "geo.gen").write_text("\n".join(gen) + "\n")
    (tmp / "dftb_in.hsd").write_text(f"""Geometry = GenFormat {{
  <<< "geo.gen"
}}
Hamiltonian = DFTB {{
  SCC = No
  SlaterKosterFiles = Type2FileNames {{
    Prefix = "{SKDIR}/"
    Separator = "-"
    Suffix = ".skf"
  }}
  MaxAngularMomentum {{
{ang}
  }}
  KPointsAndWeights = {{
    0.0 0.0 0.0 1.0
  }}
  Filling = Fermi {{
    Temperature [K] = 0.0001
  }}
}}
Options {{
  WriteRealHS = Yes
}}
ParserOptions {{
  ParserVersion = 12
}}
""")
    subprocess.run([DFTB], cwd=tmp, capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=300)
    import sys
    sys.path.insert(0, str(HERE))
    from dftb_ref import read_real_hs

    def full(path):
        n, no, blocks = read_real_hs(path)
        off = np.concatenate([[0], np.cumsum(no)])
        D = np.zeros((off[-1], off[-1]))
        for i, j, cell, m in blocks:
            D[off[i]:off[i + 1], off[j]:off[j + 1]] += m
        return D + D.T - np.diag(np.diag(D)), no
    Hd, no = full(tmp / "hamreal1.dat")
    Sd, _ = full(tmp / "overreal.dat")
    return Hd * skf.HARTREE, Sd, no


def _compare(H, S, Hd, Sd, no):
    H, S = H.toarray(), S.toarray()
    off = np.concatenate([[0], np.cumsum(no)])
    idx = []
    for a, n in enumerate(no):
        idx += [off[a]] + ([off[a] + 1 + p for p in (1, 2, 0)] if n == 4 else [])      # DFTB+ orders p as (py, pz, px)
    idx = np.array(idx)
    return np.abs(H[np.ix_(idx, idx)] - Hd).max(), np.abs(S[np.ix_(idx, idx)] - Sd).max()


@need_dftb
def test_matrices_against_dftbplus_periodic_si(tmp_path):
    pos, box = kwon_si.diamond_positions(2)                # 64 atoms in a 10.9 A cell: pairs interact through two images
    pos = pos + np.random.default_rng(42).uniform(-0.15, 0.15, pos.shape)
    Hd, Sd, no = _run_dftb(tmp_path, pos, ["Si"] * 64, ["Si"], box, '    Si = "p"')
    H, S = skf.skf_hamiltonian(pos, ["Si"] * 64, box, skf.SKSet(SKDIR, ["Si"]))
    dh, ds = _compare(H, S, Hd, Sd, no)
    assert dh < 5e-6 and ds < 5e-7, (dh, ds)


@need_dftb
def test_matrices_against_dftbplus_si_h_cluster(tmp_path):
    rng = np.random.default_rng(7)
    Si = np.array([[0, 0, 0], [2.35, 0.1, 0.05]])
    Hh = np.array([[-0.6, 0.9, 0.7], [-0.5, -1.0, 0.8], [-0.4, 0.1, -1.4], [2.9, -0.8, -0.9], [3.0, 1.0, -0.7], [3.1, 0.1, 1.3]])
    pos = np.vstack([Si, Hh]) + rng.uniform(-0.08, 0.08, (8, 3))
    pos = pos - pos.mean(0) + 7.0
    sym = ["Si"] * 2 + ["H"] * 6
    Hd, Sd, no = _run_dftb(tmp_path, pos, sym, ["Si", "H"], np.full(3, 14.0), '    Si = "p"\n    H = "s"')
    H, S = skf.skf_hamiltonian(pos, sym, np.full(3, 14.0), skf.SKSet(SKDIR, ["Si", "H"]))
    dh, ds = _compare(H, S, Hd, Sd, no)
    assert dh < 5e-6 and ds < 5e-7, (dh, ds)


@need_skf
def test_chunked_assembly_matches_skf():
    pos, box = kwon_si.diamond_positions(3)
    pos = pos + np.random.default_rng(5).uniform(-0.1, 0.1, pos.shape)
    sk = skf.SKSet(SKDIR, ["Si"])
    for kp in (None, np.array([0.1, -0.2, 0.3])):
        H, S = skf.skf_hamiltonian(pos, ["Si"] * len(pos), box, sk, kp)
        H2, S2 = skf.skf_hamiltonian(pos, ["Si"] * len(pos), box, sk, kp, chunk=40)
        assert abs(H - H2).max() < 1e-10 and abs(S - S2).max() < 1e-12
