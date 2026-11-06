"""Xu orthogonal carbon tight-binding Hamiltonian: parameters, symmetry (C60 has the Ih degeneracies), sparsity."""
import pathlib
import sys

import numpy as np
import pytest
import scipy.linalg as sla

import sdos
from sdos.interfaces import xu_carbon as X

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "examples" / "fullerene"))
import goldberg as G


def test_tail_polynomial_reproduces_table_2_of_the_paper():
    """Table 2 of Xu et al. (OCR digits): c0 = 6.7392620074314e-3, c1 = -8.1885359517898e-2, c2 = 0.1932365259144, c3 = 0.3542874332380.
    Our coefficients are derived from the matching conditions only (value and slope at r1 = 2.45, zero value and slope at rm = 2.6)."""
    paper = np.array([6.7392620074314e-3, -8.1885359517898e-2, 0.1932365259144, 0.3542874332380])
    np.testing.assert_allclose(X.TAIL, paper, rtol=1e-6)
    h = 1e-7                                                          # smooth (value and slope) at r1 and at the cutoff
    assert abs(X.scaling(X.R1 - h) - X.scaling(X.R1 + h)) < 5e-8        # value continuity (slope 0.08 x 2h = 1.6e-8 expected)
    assert abs((X.scaling(X.R1 + 2 * h) - X.scaling(X.R1 + h)) / h - (X.scaling(X.R1 - h) - X.scaling(X.R1 - 2 * h)) / h) < 1e-4
    assert X.scaling(X.RM) == pytest.approx(0.0, abs=1e-12) and X.scaling(X.RM + 0.1) == 0.0


def test_scaling_function_and_hoppings_at_r0():
    assert X.scaling(X.R0) == pytest.approx(1.0)
    B = X.hopping_blocks(np.array([[0.0, 0.0, X.R0]]))[0]          # bond along z at r0: sigma/pi values appear on the diagonal
    assert B[0, 0] == pytest.approx(-5.00) and B[0, 3] == pytest.approx(4.70) and B[3, 0] == pytest.approx(-4.70)
    assert B[3, 3] == pytest.approx(5.50) and B[1, 1] == pytest.approx(-1.55) and B[2, 2] == pytest.approx(-1.55)
    assert X.scaling(3.0) == 0.0 and X.scaling(2.46) < 0.01 and X.scaling(1.42) > 1.0     # decays quickly, larger below r0


def test_hermitian_and_sparsity():
    pos, nb, T = G.fullerene(2, 2, bond=1.42)
    H1 = X.xu_hamiltonian(pos, cutoff=1.9)                       # first neighbours only
    assert abs(H1 - H1.T).max() == 0
    rows = np.diff(H1.indptr)
    assert rows.min() == rows.max() == 1 + 3 * 4                  # diagonal on-site entry + 3 neighbours x 4 orbitals
    H2 = X.xu_hamiltonian(pos, cutoff=2.6)
    assert np.diff(H2.indptr).max() > rows.max() and abs(H2 - H2.T).max() == 0


def test_c60_has_the_icosahedral_degeneracies():
    """Ideal C60 (exact Ih): levels have multiplicities 1, 3, 4, 5 only; HOMO fivefold (hu), LUMO threefold (t1u); gap open."""
    pos, nb, T = G.fullerene(1, 1, bond=1.42)
    ev = sla.eigvalsh(X.xu_hamiltonian(pos, cutoff=1.9).toarray())
    assert len(ev) == 240
    levels = np.split(ev, np.nonzero(np.diff(ev) > 1e-8)[0] + 1)
    mult = {len(l) for l in levels}
    assert mult <= {1, 3, 4, 5}, mult
    homo, lumo = ev[119], ev[120]
    assert np.ptp(ev[115:120]) < 1e-8 and np.ptp(ev[120:123]) < 1e-8          # 5-fold HOMO, 3-fold LUMO
    assert lumo - homo > 0.5                                                  # a real molecular gap
    assert ev[123] - ev[122] > 1e-3                                           # LUMO and LUMO+1 distinct (t1u vs t1g)


def test_standard_sdos_matches_exact_for_xu_hamiltonian():
    pos, nb, T = G.fullerene(2, 2, bond=1.42)
    H = X.xu_hamiltonian(pos, cutoff=2.6)
    ev = sla.eigvalsh(H.toarray())
    r = sdos.dos(H, M=96, R=30, seed=1, bounds=(ev[0] - 1e-6, ev[-1] + 1e-6), exact_moments=4)
    mex = np.cos(np.outer(np.arange(96), np.arccos((ev - r.b) / r.a))).mean(axis=1)
    se = sdos.moment_errors(r.mu_samples)
    assert np.all(np.abs(r.mu - mex)[5:] < 5 * se[5:] + 1e-9)
    np.testing.assert_allclose(r.mu[:5], mex[:5], atol=1e-10)


def test_fermi_level_is_ill_defined_in_a_gap_and_the_warning_says_so():
    """KPM E_F of a gapped cluster: the IDOS is flat across the gap, so fermi_interval spans it and fermi_and_band_energy warns; exact values from homo_lumo."""
    pos, nb, T = G.fullerene(2, 2, bond=1.42)
    H = X.xu_hamiltonian(pos, cutoff=1.9)
    hl = sdos.homo_lumo(H, n_occupied=2 * 240)
    r = sdos.dos(H, M=300, R=6, seed=1, exact_moments=4)
    lo, hi, mid = sdos.fermi_interval(r, n_elec=4 * 240, spin=2.0)
    assert lo <= hl["homo"] + 0.3 and hi >= hl["lumo"] - 0.3 and hi - lo > 0.5 * hl["gap"]       # the interval covers (most of) the gap
    with pytest.warns(UserWarning, match="Fermi level is ill-defined"):
        sdos.fermi_and_band_energy(r, 4 * 240, 2.0)
    # a gapless metal: narrow interval, no warning
    from conftest import cubic
    Hm = cubic(10, W=1.0, seed=1)
    rm = sdos.dos(Hm, M=300, R=6, seed=1)
    lo, hi, _ = sdos.fermi_interval(rm, n_elec=Hm.shape[0], spin=2.0)
    assert hi - lo < 0.5
