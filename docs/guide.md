# sdos user guide

Every snippet here is exercised by `examples/` (run by the test suite).

## 1. The simple way
```python
import sdos
res = sdos.dos("H.npz")            # file | SciPy sparse | array | LinearOperator
res.plot(); res.save("dos.txt")
```
`res.E`, `res.rho` (per orbital, integrates to 1), `res.rho_err` (standard error over the random vectors), `res.idos`.  Multiply by N for
states per unit energy.  Pick the energy resolution instead of the number of moments with `resolution=`; `quick=True` gives a fast look.

## 2. Choosing accuracy
* **Moments M** set the resolution (Jackson kernel: about pi a / M, a = half the spectral width).  **Random vectors R** set the noise; for large N
  a few vectors suffice (the error per moment falls like 1/sqrt(N R)).
* `exact_moments=K` (default 4 when cheap): the first K moments are computed exactly by a local O(N) method and substituted into every sample.
  This removes the noise from the global shape (band energy, Fermi level) at negligible cost.
* `method="maxent"`: maximum entropy from few moments (20-60) with the moment uncertainties; best for smooth spectra.
* `vectors="impartial"` (Drabold-Sankey): each vector has its first three moments exact.  Constraining more than three moments biases the
  estimator slightly; prefer `exact_moments` for that.
* `probe=d` (distance-d probing): moments up to d exact and short-range noise removed, at d^3-ish extra vectors.
* Spectral bounds are estimated from Lanczos and widened by `margin=0.01`; a warning says if a moment exceeds 1 (window too tight).  Give guaranteed
  `bounds=(Emin, Emax)` if you know them.

## 3. Generalized problems A x = lambda B x
```python
res = sdos.dos(A, B=S)                         # S: sparse Hermitian positive definite (overlap); a 1-D array means masses
res = sdos.dos(A, B=S, binv="chebyshev")       # fixed Chebyshev polynomial for B^-1 (an exact polynomial map)
res = sdos.dos(A, B=S, precond="ic")           # PCG preconditioner: "block-jacobi" (default), "ic", "amg" (pyamg), "none"
```
`block_size=4` makes block-Jacobi use per-atom blocks.  A diagonal B is mass-weighted once and no solves are done.  Real or complex Hermitian A and B are
both handled; matrix-free A is allowed, B must be explicit.  Exact local moments, impartial vectors and probing are refused for a non-diagonal B (B^-1 A is
not local).

## 4. Phonons and thermodynamics
```python
res = sdos.vdos(Phi, mass=m, zero_modes=3)      # lambda = omega^2; Phi sparse or matrix-free
th = sdos.harmonic_thermo(res, [100, 300, 600]) # zero-point energy, U, F, S, Cv per atom (eV, kB)
```
`res.omega`, `res.g` (g = 2 omega rho(omega^2)); imaginary modes are reported and warned about, never clipped.  `zero_modes=3` removes the rigid translations
exactly (otherwise the kernel smears their weight into small omega and inflates the entropy: +1.6% at 300 K, +12% at 50 K in our test).  Low-temperature values
are limited by the resolution near omega = 0 (`th.info["T_reliable"]`).  For force fields without a stored Hessian use
`sdos.interfaces.lmp.HessianOperator(forces, x0, mass)` (finite-difference Hessian-vector products; LAMMPS via `LammpsForces`).

## 5. Eigenstates and IPR in a window
```python
w = sdos.window_eigs(A, B, (E1, E2))            # w.E, w.V, w.ipr, w.residual
```
`method="shift-invert"` (exact, **not O(N)**: sparse factorization, fill grows superlinearly in 3D) or `method="filter"` (Chebyshev-filtered subspace iteration;
only converged states are returned and the window is complete away from its edges by `info["edge_width"]`).  IPR uses Mulliken weights when B != 1.

## 6. Local and projected DOS
```python
p = sdos.pdos(A, B, groups={"Si": idx_si, "H": idx_h})   # all groups from one propagation
l = sdos.ldos(A, B, orbitals=[0, 5])                       # exact, no sampling noise
```
Mulliken convention for B != 1 (see theory.md); LDOS_i integrates to 1, PDOS_g to n_g/N, the groups sum to the total DOS.

## 7. Command line
`sdos H.npz -o dos.dat --plot dos.png`; `--B S.npz --binv chebyshev`; `--mass m.txt --omega --zero-modes 3 --thermo 10 1000 50`;
`--window E1 E2 --window-method filter --vectors-out states.npz`; `--method maxent`; `--exact-moments 0` to switch the default exact moments off.

### Result files and `sdos analyze`
`sdos H.dat -o result.npz` (or `res.save("result.npz")`) writes a self-describing file: E, rho, errors, IDOS, the averaged and per-sample
moments, the spectral window, the method and its settings (and omega, g for `--omega` runs).  `sdos.load` reads it back.  `sdos analyze`
works from that file alone: `--info`; `--electrons N` or `--valence-file Z.txt` (Z of each atom, one per line; `--charge q` for an ion) with
`--spin`; `--thermo TMIN TMAX NT` for a vibrational result; `--moments M --method maxent|kpm --kernel ... --maxent-default ...` to reconstruct again
from the first M saved moments (cheap, no matrix); `-o` text spectrum, `--save` new result, `--plot`.  Orbitals per atom do not matter: the
valence file counts electrons, and the DOS is per orbital.

### Cost and sparseness
The moments cost about nnz x M x R multiply-adds, so the time is linear in the number of nonzeros.  Measured (random symmetric matrix, N = 50,000,
M = 100, R = 2, 4 threads): 0.8-0.9 ns per nnz per moment per vector from 15 to 255 nonzeros per row, up to 1.8x more at 3 per row where the row overhead dominates.
Sparseness does not change the stochastic error (it goes as 1/sqrt(R N)); M is set by bandwidth / resolution.  Maxent after the moments costs the same for any
sparsity.  Dense exact diagonalisation (N^3) wins only when the matrix is small or nearly full and M R is comparable to N.

## 8. Optional interfaces
`sdos.interfaces.sisl_if.from_sisl(H, k)` gives (H(k), S(k)) from a `sisl.Hamiltonian` (complex Hermitian for k != 0; generalized when non-orthogonal);
`sdos.interfaces.lmp` is the force-based Hessian operator.  (Not yet: tbham, a DFTB `.skf` reader; see the project status notes.)

## Choosing KPM or Maxent
Maxent gives every moment but mu_0 a minimum uncertainty `sigma_floor` (default 1e-4, `sdos.maxent_fit` / `sdos.maxent_bootstrap`); without it, exact or nearly exact moments of a singular density stop converging beyond ~100 moments. Stochastic moments are unaffected (their sampled errors are larger). `sigma_floor=0` gives the exact-moment problem.


* **KPM (default, `method="kpm"`)**: linear in the moments, always converges, positive with the Jackson kernel; resolution is pi*a/K, so for maximum resolution of a large
  system just raise `M` (a few thousand is cheap: cost is linear).  Past the mean level spacing or the stochastic noise floor, more moments only sharpen sampling speckle
  (isolated delta-like peaks and ringing); check by comparing two independent sets of random vectors, or watch `rho_err`.
* **Maxent (`method="maxent"`, `sdos.maxent_bootstrap`)**: use it when high-order moments are expensive or noisy (<~300 moments), or when positivity and sharp edges matter.  Extracts more
  from few moments (measured 1.4-1.7x lower L1 error than KPM from identical exact moments of graphene), but it is a nonlinear solve: with *exact* moments of a density with
  singularities (log van Hove, Dirac cusp) no finite exp-polynomial fits and it stops improving near 300-400 moments.  `maxent_bootstrap` climbs the moment count with warm starts
  and shrinks the support to the band edges found at lower order; on failure it returns the last converged stage (`info["converged"]`, `info["moments_used"]`).
* To report a result, overlay both: where they agree the feature is in the moments.

## Worked example: from fullerenes to graphene

`examples/06_fullerene_to_graphene.py` strings the tools together on icosahedral C_N (Xu-Wang-Chan-Ho tight binding): exact HOMO-LUMO gaps by inertia bisection, hbar v_F from the
graphene bands of the same Hamiltonian, the confined-Dirac-cone scaling gap ~ 1.9 hbar v_F / R (a smooth sphere gives 2), the states nearest E_F by shift-invert with their Ih multiplets
and IPR, and a non-orthogonal check on SIESTA (H, S) of C240 and C540 (gaps agree with dense eigh(H, S) to 1e-10).  The larger studies are in `examples/fullerene/`.

## Random vectors and exact moments (measured on DTW4096, Kwon TB, band energy, M = 100, 12 seeds x 2 vectors)

| vectors | std of E_BS (meV/atom) | time per run | efficiency vs Rademacher at equal cost |
|---|---|---|---|
| Gaussian | 315 | 0.23 s | 0.4x |
| Rademacher (default) | 193 | 0.23 s | 1x |
| impartial (3 exact moments, Drabold-Sankey) | 26 | 0.35 s | ~37x |
| Rademacher + `exact_moments=4` (default) | 11 | 1.8 s | ~39x |
| Rademacher + `exact_moments=8` | 4.4 | 9 s | ~48x |
| Rademacher + `exact_moments=16` | 1.4 | 80 s | ~24x |

Roughly +-30% (12 seeds).  `exact_moments=8` is the accuracy setting for integrated quantities; impartial vectors constrain only the first three moments (more is biased).
Pointwise DOS profits much less.  See `examples/dtw4096/dtw_electrons.py`.
