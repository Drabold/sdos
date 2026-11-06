"""Local / projected DOS: Mulliken convention for B != 1, exact orbital LDOS, sum rules, stochastic group estimators."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

import sdos
from conftest import random_hermitian
from test_generalized import spd_B


def exact_group_moments(A, B, a, b, M, groups):
    """Moments of the Mulliken-weighted group DOS from the exact eigenpairs of A x = lambda B x."""
    ev, V = sla.eigh(A.toarray(), None if B is None else B.toarray())
    BV = V if B is None else B.toarray() @ V
    W = np.real(np.conj(V) * BV)                                     # (orbital, state) Mulliken weights
    x = np.clip((ev - b) / a, -1, 1)
    T = np.cos(np.outer(np.arange(M), np.arccos(x)))                 # (M, states)
    return {g: (W[idx] .sum(axis=0) @ T.T) / A.shape[0] for g, idx in groups.items()}, ev, W


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("general", [False, True])
def test_ldos_is_exact_and_obeys_sum_rules(cplx, general):
    n = 120
    A = random_hermitian(n, cplx, seed=5)
    B = spd_B(n, 30, cplx, seed=3) if general else None
    ev = sla.eigh(A.toarray(), None if B is None else B.toarray(), eigvals_only=True)
    bd = (ev[0] - 1e-6, ev[-1] + 1e-6)
    orbs = np.array([0, 7, 33, 80, 119])
    M = 64
    r = sdos.ldos(A, B, orbitals=orbs, M=M, bounds=bd, margin=0.0, binv_tol=1e-13)
    ex, _, W = exact_group_moments(A, B, r.a, r.b, M, {int(i): [int(i)] for i in orbs})
    for i in orbs:
        np.testing.assert_allclose(r.mu[int(i)], ex[int(i)] * n, atol=1e-7)          # exact moments, no sampling
    full = sdos.ldos(A, B, M=M, bounds=bd, margin=0.0, binv_tol=1e-13)               # all orbitals
    tot = sum(full.mu[i] for i in range(n)) / n                                      # sum rule: sum_i LDOS_i = N x total DOS
    ev_all = ev
    x = np.clip((ev_all - full.b) / full.a, -1, 1)
    np.testing.assert_allclose(tot, np.cos(np.outer(np.arange(M), np.arccos(x))).mean(axis=1), atol=1e-7)
    assert all(abs(full.mu[i][0] - 1.0) < 1e-12 for i in range(n))                   # every LDOS_i integrates to 1


@pytest.mark.parametrize("mode", ["shared", "separate"])
@pytest.mark.parametrize("general", [False, True])
def test_pdos_group_moments_vs_exact(mode, general):
    n = 160
    cplx = True
    A = random_hermitian(n, cplx, seed=7)
    B = spd_B(n, 20, cplx, seed=4) if general else None
    ev = sla.eigh(A.toarray(), None if B is None else B.toarray(), eigvals_only=True)
    bd = (ev[0] - 1e-6, ev[-1] + 1e-6)
    groups = {"even": np.arange(0, n, 2), "odd": np.arange(1, n, 2), "head": np.arange(0, 20)}
    M, R = 32, 200
    r = sdos.pdos(A, B, groups=groups, M=M, R=R, mode=mode, seed=3, bounds=bd, margin=0.0)
    ex, _, _ = exact_group_moments(A, B, r.a, r.b, M, groups)
    for g in groups:
        samples = np.array([r.mu[g][k] for k in range(R)])
        se = samples.std(axis=0, ddof=1) / np.sqrt(R)
        z = np.abs(samples.mean(axis=0) - ex[g])[2:] / se[2:]
        assert z.max() < 5.0, (g, z.max())
    np.testing.assert_allclose(sum(r.idos[g] for g in ("even", "odd"))[-1], 1.0, atol=1e-2)            # PDOS_g integrates to n_g / N
    assert abs(r.idos["even"][-1] - 0.5) < 2e-2 and abs(r.idos["head"][-1] - 20 / n) < 2e-2


def test_shared_mode_groups_sum_to_total_exactly():
    """Shared mode: the group moments of one vector add up to its plain Hutchinson moments x^+ T_n x / N (same vector, same recursion)."""
    n = 100
    A = random_hermitian(n, False, seed=2)
    ev = sla.eigvalsh(A.toarray())
    bd = (ev[0] - 1e-6, ev[-1] + 1e-6)
    groups = {"a": np.arange(0, 40), "b": np.arange(40, 100)}
    r = sdos.pdos(A, groups=groups, M=48, R=1, seed=1, bounds=bd, margin=0.01)
    x = sdos.make_vectors(n, None, 1, np.random.default_rng(1), False, kind="auto", is_complex=False)
    plain = sdos.chebyshev_moments(sdos.ScaledOperator(A, r.a, r.b), x, 48) / n
    np.testing.assert_allclose(r.mu["a"][0] + r.mu["b"][0], plain, atol=1e-10)


def test_site_groups_helper_and_diagonal_B():
    g = sdos.site_groups(["Si", "H", "Si"], 4)
    assert list(g["Si"]) == [0, 1, 2, 3, 8, 9, 10, 11] and list(g["H"]) == [4, 5, 6, 7]
    n = 80
    A = random_hermitian(n, False, seed=9)
    m = np.random.default_rng(1).uniform(1, 3, n)
    r = sdos.ldos(A, m, orbitals=[0, 5], M=32)                         # diagonal B: Mulliken weights = |psi|^2 of the mass-weighted problem
    assert r.problem == "standard"
