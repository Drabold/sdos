# Method notes

## Stochastic trace and KPM
For a Hermitian matrix with spectrum inside [b-a, b+a], the scaled operator Ht = (A-b)/a has Chebyshev moments mu_n = Tr T_n(Ht)/N.  For a random vector x
with E[x x^+] = 1, x^+ T_n(Ht) x is an unbiased estimate of N mu_n (Hutchinson); the doubling identities T_2n = 2 T_n^2 - T_0 and T_2n+1 = 2 T_n+1 T_n - T_1 give
2K moments from K matrix-vector products.  The DOS is the kernel-damped Chebyshev series (Jackson kernel: positive, resolution about pi a / M; the Lorentz
kernel is a Green-function kernel and is not recommended for a DOS) or the maximum-entropy density that reproduces the moments within their uncertainties.
Complex Hermitian problems use conjugated inner products throughout; real arithmetic is the fast path.

## Impartial vectors and exact moments
Drabold and Sankey (PRL 70, 3631 (1993)) call a vector impartial if it weighs all eigenvectors equally; they approximate one by minimising
P(x) = (<x|x>-1)^2 + (<x|H|x>-mu_1)^2 + (<x|H^2|x>-mu_2)^2 (exact mu_1, mu_2 are O(N)).  We reproduce their narrowing of the band-energy distribution (about
tenfold in our tests).  Constraining more moments gives a measurably biased estimator (rms z of the moment residuals 1.0-1.5 for K = 4, 8, 16 at N = 500-2000, against 0.8-1.0 for three moments; see examples_smallN/zscore_constrained_impartial.py).  The unbiased way to use exact low moments is substitution: **mu_n for n <= K can be computed exactly in O(N)**, because
(T_n(Ht))_ii depends only on a neighbourhood of radius about n/2 of site i; with P_m = T_m(Ht), mu_2m = (2 ||P_m||_F^2 - N)/N and
mu_2m+1 = (2 Re sum conj(P_m+1) o P_m - Tr Ht)/N, evaluated column-chunk by column-chunk.  The cost per site grows polynomially with n (the neighbourhood
volume), not with N.

## Generalized problem
A x = lambda B x with B Hermitian positive definite: Y = B^-1 A is self-adjoint in the B inner product, similar to B^-1/2 A B^-1/2, so Tr T_m(Y) =
Tr T_m(B^-1/2 A B^-1/2) and x^+ T_m(Y) x is unbiased for any x with E[x x^+] = 1.  Y is not Hermitian, so the doubling identities are not used: one
application of B^-1 per moment.  B^-1 is applied by PCG to a tolerance (1e-12), or by a fixed polynomial p(B~) ~ B~^-1 on the block-Jacobi-scaled
B~ = L^-1 B L^-+ (a similarity transform of Y), degree from the condition number kappa: about sqrt(kappa)/2 ln(2/tol), verified on a grid; then the
whole recursion is an exact polynomial map.  The two modes give different per-vector estimates (different coordinates) but the same traces.  Spectral
bounds of Y come from a short Lanczos in the B inner product (Ritz extremes lie inside the spectrum; the residual and a margin are added).

## Projections
With psi^+ B psi = 1 the Mulliken weight of orbital i in state n is w_in = Re conj(psi_in)(B psi_n)_i.  Since sum_n psi_n psi_n^+ f(E_n) = f(B^-1 A) B^-1,
LDOS_i(E) = [delta(E - A B^-1)]_ii: the moments are those of A B^-1, the adjoint of B^-1 A.  Sum rules: every LDOS_i integrates to 1 and sum_i LDOS_i = N
rho.  Mulliken weights can be negative for strongly overlapping orbitals.

## Harmonic thermodynamics
Per mode with x = hbar omega / k_B T: u = hbar omega (1/2 + 1/(e^x-1)), f = hbar omega/2 + k_B T ln(1-e^-x), s/k_B = x/(e^x-1) - ln(1-e^-x), c/k_B = x^2 e^x/(e^x-1)^2;
totals per atom are three times the average over the normalised g(omega).  Known eigenvalues (the rigid translations at lambda = 0) are removed from the moments
exactly by linearity: each contributes T_m(x_lambda)/N to mu_m.

## Scaling, honestly
* DOS by KPM / Maxent: cost about nnz x M x R, memory O(N R_block); linear in N.  The statistical error per moment falls like 1/sqrt(N R).  The resolution is
  set by M (about pi a / M), not by N.
* Exact local moments: linear in N at fixed K; the prefactor grows polynomially with K.
* Generalized problem: each moment needs one B^-1 application; PCG iterations depend on cond(B) and the preconditioner, not on N for a local B with a good preconditioner
  (AMG), so the cost stays close to linear; the Chebyshev mode costs degree matrix-vector products per application.
* Window eigenpairs: shift-invert factorizes A - sigma B: **not O(N)** (fill grows superlinearly for 3D connectivity); the filtered subspace route costs
  block x degree x iterations products plus O(N k^2) dense algebra, with k the number of states in the window.
* No method here resolves individual levels of a million-atom system from a global expansion: the resolution is pi a / M.

## Fermi level of a gapped system
In a gap the integrated DOS is flat at the filling, so any energy in the gap is "the" Fermi level and the band energy is the same for all of them (the DOS is
zero there); only the reference energy for plots is ambiguous.  A KPM estimate is further blurred by kernel leakage and stochastic noise (measured on C240 with
the Xu Hamiltonian, 600 moments: band energy varies by 0.03 eV/atom, 0.09%, over E_F positions inside the 1.0 eV gap, about the method's own 0.14% error).
`sdos.fermi_interval` reports the interval in which the IDOS equals the electron count within its standard error, `fermi_and_band_energy` warns when it is wide,
and `sdos.homo_lumo` gives the exact HOMO, LUMO and midpoint by bisection on the LDL^T inertia count (exact, but each step is a sparse factorization: at 15,360 orbitals
35 s in total, at 19,440 orbitals 113 s, more than the whole DOS, so it is capped in the examples).

## Maxent: default model, spectral window, and what "noise" means
For any vector x the moments <x|T_k(H)|x> are the exact moments of a positive measure (the weights |<psi_j|x>|^2 on the eigenvalues), and so is their average over
vectors, so the moment problem always has a solution and a fit with exact moments (``maxent_noise=0``) is always well posed.  The vector-to-vector difference is the
sampling speckle of that measure (it averages out as more states fall in a resolution cell); the default ``maxent_noise=1`` (the sigma^2 lambda^2 term of the dual,
sigma = spread over vectors / sqrt(R)) is a *regularization* that lets the fit miss high-order moments by their sampling spread.
Measured (examples/fullerene/maxent_prior_test.py, maxent_vs_kpm.py):
* Same K exact moments of graphene's DOS: Maxent's L1 error is 1.4-1.7x smaller than Jackson-KPM's at every K from 20 to 300 (e.g. K = 100: 0.111 vs 0.178), but
  Maxent ripples near hard edges while Jackson-KPM is smooth and broad (it misses peak heights and fills the Dirac dip); neither is ringing-free.
* The default model (uniform "flat", Chebyshev weight, KPM density, or a smoothed KPM density from K/5 moments) makes no measurable difference in accuracy or in the size of
  the multipliers: use ``flat``.
* A spectral window wider than the true support makes the Lagrange multipliers explode for a hard-edged density with exact moments (|lambda| ~ 1e4-1e5, solver
  stalls at its iteration cap), because the Chebyshev representation then needs large cancelling coefficients, but the density stays accurate (moment residuals
  1e-9 - 1e-5; L1 error 0.015 -> 0.016 at 2% too wide, 0.050 at 4x too wide, always better than KPM).  With stochastic moments (R > 1, sigma-regularized) every solve converged with
  |lambda| <= 17 even for a window 2x too wide.
* Ringing/lumpiness in a KPM DOS of a finite cluster is mostly real structure plus random-vector sampling error (C960 at 0.12 eV resolution: real structure 0.005 rms,
  sampling 0.0047 at R = 2, 0.0021 at R = 10, 0.0008 at R = 40), not Gibbs oscillation: use more vectors (or probing) before changing the reconstruction.
