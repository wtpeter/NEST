# NTTDA analytic nuclear Hessian

This implementation uses matrix-free iterative response solves and a dense
integral backend for conventional all-electron ROKS, with real full-rank spatial
orbitals. The integral backend still limits it to small systems.
HF, LDA, GGA, tau-dependent MGGA, and their full-range global hybrids are
supported in all three NTTDA sectors. The target must be an isolated state.
DFT holds the quadrature coordinates and weights fixed and requires analytic
XC fourth derivatives. Range separation, nonlocal correlation, Laplacian
functionals, density fitting, ECPs and X2C are explicitly rejected.

```python
td = mf.NTTDA().set(deltaS=-1, nstates=3).run()
hess = td.Hessian().kernel(state=1)
# hess[atom_A, atom_B, cartesian_A, cartesian_B], Eh/Bohr**2
```

The roots are one-based, as for NTTDA gradients.  `atmlst` selects both atom
axes, in the supplied order.  The returned derivative is of the total energy
`E_ROKS + omega`, including nuclear repulsion.  There is no finite-difference
step and no displaced geometry in the implementation.

## Energy and reference constraints

Let `q` contain independent antisymmetric spatial-orbital rotations between
closed/open, closed/virtual and open/virtual spaces.  The reference constraints
in the current MO basis are

\[
r_{oc}=F^\beta_{oc},\qquad
r_{vc}=(F^\alpha+F^\beta)_{vc},\qquad
r_{vo}=F^\alpha_{vo}.
\]

Write `J = partial r / partial q`.  We do not assume this residual Jacobian
has a particular symmetric scaling.  Both the forward solve and its transpose
use this same explicitly differentiated residual.

For a normalized eigenvector of the *complete* channel matrix,

\[
A(q,R)X=\omega X,\quad X^T X=1,\qquad
E(q,R)=E_\mathrm{ref}(q,R)+\omega(q,R).
\]

In HF, at fixed amplitudes, `E_ref`, every element of `A`, and `r` are linear in the
orthonormal MO one-electron integrals and chemists' ERIs.  `_mo_operators`
reuses the production `gen_vind_sfu/sc/sfd` with these integrals and identity
MO coefficients.  It therefore applies the *same* linear maps to integral
derivatives. No second set of spin-adaptation formulas is introduced. In DFT,
semilocal values and derivatives are supplied separately as described below.

In the HF limit the reconstructed `Fz` equals
`-K(D_open)/2 = (F_alpha-F_beta)/2`.  The `nobeta` equal-spin reference gives
the same `F0` in this limit, so both settings follow the same derivatives.

## Semilocal XC chain rule

For grid orbitals `phi_i`, define the pair features

\[
P^0_{ij}=\phi_i\phi_j,\quad
P^k_{ij}=(\partial_k\phi_i)\phi_j+\phi_i(\partial_k\phi_j),\quad
P^\tau_{ij}=\tfrac12\sum_k(\partial_k\phi_i)(\partial_k\phi_j).
\]

LDA uses only `P0`, GGA also uses the three gradient features, and MGGA adds
the tau feature. Spin density features are `rho_sx = sum_i occ_si P^x_ii`.
PySCF's `eval_xc_eff` gives derivatives against these features, including the
transformation from LibXC's sigma variables for GGA/MGGA.

The actual spin potentials and reconstructed kernel in the MO basis are

\[
V^s_{ij}=\sum_g w_g\sum_x v_{sx}(g)P^x_{ij}(g),\qquad
K^{\rm Ref}_{ij,kl}=\sum_g w_g\sum_{xy} f^{\rm Ref}_{xy}(g)
P^x_{ij}(g)P^y_{kl}(g).
\]

The reference energy and residual use the actual spin densities. The
reconstructed kernel is evaluated at equal-spin density features
`rho_equal_sx = (rho_alpha_x + rho_beta_x)/2`. With `u=(1,-1)`,

\[
f^{\rm Ref}_{xy}=\tfrac12\sum_{st}u_su_t f_{sx,ty},\quad
\frac{\partial f^{\rm Ref}_{xy}}{\partial\rho_z}
=\tfrac14\sum_{str}u_su_t k_{sx,ty,rz},\quad
\frac{\partial^2 f^{\rm Ref}_{xy}}{\partial\rho_z\partial\rho_w}
=\tfrac18\sum_{strv}u_su_t l_{sx,ty,rz,vw}.
\]

These derivatives are with respect to total density features. `xc.py`
differentiates these quantities and every pair product by the full product
rule. For `nobeta=True`, the common Fock potential is evaluated at the
equal-spin density, while the actual reference residual remains unchanged.
This separation also holds in first and second derivatives.

The production HF action consumes the semilocal response through an
effective tensor `g_eff[p,r,s,q] = hybrid*g[p,r,s,q] - K_ref[p,q,r,s]`.
Its exchange contraction yields the ordinary kernel action and its Coulomb
contraction yields the recoupled kernel action. Actual reference J/K, energy,
and spin potentials are computed separately; `g_eff` is only a response tensor.
Operator-closure tests compare complete channel matrices with the
production DFT actions, including the recoupled GGA/MGGA terms.

AO values and their gradients require nuclear derivatives up to third order
for GGA/MGGA. Quadrature points and weights have zero derivatives here.
Consequently this DFT Hessian differentiates the current fixed-grid gradient
convention. It does not include moving atom-centered grids or partition-weight
response; HF translation sum-rule tests do not apply to fixed-grid DFT.

## Orthonormal orbital chart and forward response

To handle the overlap explicitly, define a local chart about the reference
geometry with constant `C0`:

\[
C(q,R)=C_0 S_m(R)^{-1/2}\exp K(q),\qquad
S_m(R)=C_0^T S_\mathrm{AO}(R)C_0.
\]

At the expansion point `S_m=I`, `K=0`.  Intra-space rotations are a gauge and
are excluded from `q`.  For nuclear direction `a`, first form the skeleton
residual derivative `r_a` in this chart (`K_a=0`), then solve

\[
Jq_a=-r_a.
\]

Let `K_a` be the antisymmetric matrix packed from `q_a`.  The total first
derivative of the orbitals is

\[
C_a=C_0(K_a-\tfrac12 S_a).
\]

Along a two-direction path with `q_ab=0`,

\[
C_{ab}=C_0\left[
-\tfrac12 S_{ab}+\tfrac38(S_aS_b+S_bS_a)
-\tfrac12(S_aK_b+S_bK_a)
+\tfrac12(K_aK_b+K_bK_a)\right].
\]

Here all `S` derivatives are in the constant `C0` basis.  The AO integral
derivatives and these orbital derivatives are combined by the complete
multilinear product rule.  In particular, ERI second derivatives include
the twelve ordered terms where first orbital derivatives act on different
slots, as well as the four second-orbital-derivative terms.

## Adjoint and state response

The energy is stationary in `X` but not in reference orbital rotations.
Solve the reference adjoint

\[
J^T z=E_q.
\]

Use a bar to denote derivatives along the path carrying the computed first
orbital response but setting `q_ab=0`.  The missing second orbital response
is removed exactly by differentiating the reference constraint twice:

\[
Jq_{ab}+\bar r_{ab}=0,\qquad
E_q^Tq_{ab}=-z^T\bar r_{ab}.
\]

This avoids both explicit second-order CPHF and differentiated Z-vectors.

The default `solver='iterative'` uses diagonally preconditioned GMRES for
`J q_a = -r_a`, `J.T z = E_q`, and the state response. The forward orbital
action follows

\[
D'_\sigma=K n_\sigma-n_\sigma K,\qquad
F'_\sigma=K^T F_\sigma+F_\sigma K+C^T v_\sigma[D']C.
\]

The transpose action is the existing gradient's analytic ROKS adjoint action.
The gradient's M matrix supplies `E_q = pack(M) + 2*r`, including the reference
energy derivative. Neither orbital Jacobian is assembled. Define
`Q=I-X X.T`; the state response solves

\[
[Q(A-\omega)Q+XX^T]X_a=-Q\bar A_aX,\qquad X^T X_a=0.
\]

Each application calls the native NTTDA channel algebra with the cached MO
integrals and XC kernel. DFT state-response iterations therefore avoid repeated
grid integration; this cache is part of the dense integral backend, not an
assembled state matrix. The complete
channel space participates, including the lowering zero mode; only the target
state is projected out. No full state matrix, inverse, or eigensystem is built.
The rank-one term only fixes the parallel component, without shifting physical
gaps. This is equivalent to the spectral reduced resolvent

\[
X_a=\sum_{j\ne I}X_j\frac{X_j^T\bar A_aX_I}{\omega_I-\omega_j}.
\]

GMRES retains at most 40 Krylov vectors, uses an absolute residual tolerance
`conv_tol=1e-10`, and allows `max_cycle=100` restart cycles by default. Diagonal
entries below `1e-4` are clipped only in the preconditioner. The true residual
is checked after every solve; failed convergence raises an error. Diagnostics
are `response_residual` and cumulative `response_iterations` by equation type.
`solver='dense'` retains the independent original Jacobian/eigensystem path for
small-system comparisons. It also retains the original AO product-rule
transforms; the iterative path reuses MO-transformed skeleton derivatives and
contracts only differentiated slots in the second derivative.

The stored target eigenvector is checked against the native action using
`max(1e-7, td.conv_tol)`. The reference residual norm is checked against
`max(1e-7, 3*scf_gradient_tolerance)`, where PySCF defaults the gradient tolerance
to `sqrt(mf.conv_tol)` and relaxes its final check by a factor of three. Ordinary
SCF/TD defaults are accepted; the input residuals are exposed as
`reference_residual` and `state_residual`. A tightly converged response solve
does not remove errors already present in the input SCF/eigenvector.

For high-precision finite-difference validation, lowering `td.lindep` can be
necessary as well as tightening `td.conv_tol`; otherwise small search directions
may be discarded early. These are validation settings, not requirements for
ordinary use. A known
root separation below `1e-7 Eh` is rejected. In iterative mode `state_gap` is
only the minimum separation to roots already present in `td.e` (and the known
lowering zero mode); it is `None` if none is available. This does **not** certify
a full-spectrum gap. Compute neighboring roots and inspect convergence near
crossings. Dense mode checks the full spectrum. Neither path modifies `td.xy`
or `td.e`.

The final electronic Hessian is

\[
E_{ab}=\bar E_{\mathrm{ref},ab}
+X^T\bar A_{ab}X-z^T\bar r_{ab}
+X_a^T\bar A_bX+X_b^T\bar A_aX.
\]

Nuclear repulsion is added using PySCF's analytic nuclear Hessian.  This is
the constrained second derivative of the energy, not a numerical derivative
of the gradient.  It requires only first-order reference and state responses.

## Integral derivatives and validation

The one-electron Hessian uses PySCF's `rhf.Hessian.hcore_generator` (including
attraction-center derivatives).  The overlap uses `int1e_ipipovlp` and
`int1e_ipovlpip`.  ERIs use `int2e_ip1`, `int2e_ipip1`, `int2e_ipvip1`, and
`int2e_ip1ip2`, with all AO slots assigned to the displaced centers.
One nuclear derivative on an AO introduces a minus sign; two introduce a
plus sign.  Chemists' ERI symmetries supply the remaining differentiated slots.

Tests check AO second derivatives against differences of AO first derivatives,
and total Hessians against differences of the *existing independent analytic
gradient*.  Displaced orbitals are aligned by occupation-block Procrustes
rotations before state matching.  This removes arbitrary MO gauge changes
without changing the scalar state.  Translation sum rules, atom ordering and
the closed-shell spin-raising limit are checked separately.  Symmetry alone
is insufficient: the implementation evaluates one triangle and reflects it.

The dense ERIs still limit practical system size, even with iterative response
solves. Integral storage scales as approximately `(40 + 3*Nselected)*NAO**4`
doubles, before DFT grid tensors and existing process memory. For 10 selected
atoms, this estimate is 3.5 GB at 50 AOs and 56 GB at 100 AOs. This is not yet a
large-system Hessian. Direct/shell-blocked derivative contractions and blocked
XC fourth-derivative integration remain necessary for that use case.
The driver estimates memory before allocating derivative tensors, including
effective XC fourth-derivative tensors. The backend must supply analytic
fourth derivatives; a missing capability raises an error, without a numerical
fallback or backend switch.

## Running DFT with fourth derivatives

The original `nest-soc` PySCF 2.13.1 / LibXC 7.0.0 binary supplies only three
derivative orders. LibXC 7.0.0 can be built with fourth derivatives using
[`DISABLE_LXC=OFF`](https://gitlab.com/libxc/libxc/-/blob/7.0.0/CMakeLists.txt).
An isolated build was used for validation; no installed library was replaced.
From the SpinOrbitCoupling workspace root:

```bash
mkdir -p Temp/nttda_hessian
curl -fL https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.gz \
  -o Temp/nttda_hessian/libxc-7.0.0.tar.gz
tar -xzf Temp/nttda_hessian/libxc-7.0.0.tar.gz -C Temp/nttda_hessian
cmake -S Temp/nttda_hessian/libxc-7.0.0 \
  -B Temp/nttda_hessian/libxc-build -G Ninja \
  -DBUILD_SHARED_LIBS=ON -DDISABLE_KXC=OFF -DDISABLE_LXC=OFF \
  -DENABLE_FORTRAN=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build Temp/nttda_hessian/libxc-build -j 16
LIBXC4=$(realpath Temp/nttda_hessian/libxc-build/libxc.so)
cd src/NEST
OMP_NUM_THREADS=36 LD_PRELOAD="$LIBXC4" conda run --no-capture-output -n nest-soc \
  python -m pytest src/nest/hessian/tests/test_nttda.py -q
```

Downloaded archive SHA256:
`8d4e343041c9cd869833822f57744872076ae709a613c118d70605539fb13a77`.
The preload is restricted to the test/process invocation and uses the same
LibXC version as the installed PySCF dependency. It is not set by NEST code.
DFT tests skip with an explicit reason when fourth derivatives are unavailable;
HF tests and the missing-capability error test run in the original environment.


## Independent gradient audit

For the lowering OO-CV block, the supplied matrix formula and its transpose
contribute `-2*gamma/S * Tr(X_OO) * X_CV : Fz`, with
`gamma=sqrt((2*S+1)/(2*S-1))`. The previous gradient ledger used
`-gamma*(1+1/S)` instead, which agrees only at `S=1`. Both CPU and GPU
coefficients are corrected. The GPU execution requires its separate environment
and is not validated by the CPU tests.

The new arbitrary-amplitude tests use `S=3/2` and compare the full gradient
ledger to `X.T A X`, then compare `pack(M)` to finite orbital rotations of the
native energy action for HF, PBE and M06-2X, including `nobeta`. Arbitrary
amplitudes matter: eigenstates orthogonal to the lowering zero mode can have
`Tr(X_OO)=0`, hiding this coefficient error. The OO orientation remains the
native `[w,v]`; no amplitude convention was changed.

Independent nuclear tests use five-point differences of `E_ROKS+omega` to
check both the gradient and directional Hessian at steps `1e-3` and `5e-4`
Bohr. These tests never evaluate the analytic gradient to form the numerical
Hessian. Full HF Hessians and selected DFT columns are additionally checked
against displaced analytic gradients. Displaced DFT calculations retain the
reference grid and match the same state after occupation-block MO alignment.

A runnable example with the iterative eigensolver, analytic Hessian, response
diagnostics, and an independent energy stencil is
`examples/hessian/01_nttda_hessian.py`. The separate
`examples/hessian/02_compare_solvers.py` compares the iterative and dense paths
on CH2/STO-3G, CH2/6-31G and CH2/cc-pVDZ, timing one atom block and checking equality.


## Measured checks and timing

`examples/hessian/01_nttda_hessian.py` uses the public iterative eigensolver.
For CH2/HF/STO-3G, the H1-z component differs from an independent five-point
energy stencil by `3.9e-11 Eh/Bohr` for the gradient and `1.3e-8 Eh/Bohr^2`
for the Hessian. The response residual is `9.6e-11`; the translation sum-rule
residual is `5.3e-12 Eh/Bohr^2`.

One local run of `02_compare_solvers.py`, with `OMP_NUM_THREADS=36` and
`OMP_WAIT_POLICY=PASSIVE`, gave the following times. SCF/TD setup is excluded;
the timed calculation is the first lowering root's H1-H1 3x3 Hessian block.
These are illustrative single-run timings, not large-system performance claims.

| CH2 basis | AOs | Dense / s | Iterative + MO reuse / s | Maximum difference / Eh Bohr^-2 |
| --- | ---: | ---: | ---: | ---: |
| STO-3G | 7 | 0.227 | 0.524 | 2.9e-12 |
| 6-31G | 13 | 0.597 | 0.651 | 4.0e-12 |
| cc-pVDZ | 24 | 8.123 | 2.577 | 9.9e-12 |

Iteration overhead loses at the smallest sizes. The 24-AO case is about 3.2x
faster with the default path. This comparison includes both iterative solves
and the reuse of MO skeleton tensors; it does not isolate GMRES alone.

The original LibXC environment passed 36 tests across the full and incremental
tolerance runs, with 28 fourth-derivative DFT Hessian tests skipped. The combined
command is:

```bash
OMP_NUM_THREADS=36 OMP_WAIT_POLICY=PASSIVE conda run --no-capture-output -n nest-soc \
  python -m pytest --import-mode=importlib -q \
  src/nest/hessian/tests/test_nttda.py src/nest/nttda/tests/test_nttda.py \
  src/nest/soc/tests/test_soc_ao.py src/nest/soc/tests/test_sftda_soc.py \
  src/nest/soc/tests/test_sftddft_soc.py src/nest/soc/tests/test_nttda_soc.py
```

`--import-mode=importlib` avoids the two different `test_nttda.py` modules
colliding during test collection.


With the isolated fourth-derivative LibXC build, the combined Hessian and
analytic-gradient suite passed **60 tests and 18 subtests**, with the
missing-fourth-derivative guard test skipped as expected. The cached-state-action
Hessian was also checked by its full suite (**44 passed, 1 skipped**). The final
response-tolerance adjustment and iterative/dense comparisons passed another
**5 tests** (four overlaps and one new tolerance regression). These runs cover
the final changed numerical paths; no finite differences are used in the
production Hessian. Global `ruff check src/nest` and `git diff --check` passed.

```bash
OMP_NUM_THREADS=36 OMP_WAIT_POLICY=PASSIVE LD_PRELOAD="$LIBXC4" \
  conda run --no-capture-output -n nest-soc python -m pytest -q \
  src/nest/hessian/tests/test_nttda.py src/nest/grad/tests/test_nttda_grad.py \
  src/nest/grad/tests/test_nttda_orbital_derivative.py
```


## Asymmetric H2O2 / 6-31G* validation

`examples/hessian/03_h2o2_631gstar.py` uses the supplied four-atom geometry in
Angstrom, charge 0, spin 2, symmetry disabled, HF, and `nobeta=False` (32 AOs).
It computes the complete Hessian for the lowest state in each spin channel.
Ordinary default SCF/TD calculations are compared with a validation reference;
the latter changes the SCF energy tolerance but leaves the orbital-gradient
threshold at PySCF's default. Extreme SCF convergence is not required for use.

This case exposed a real gradient-solver issue: the old unnormalized
`lib.krylov` expansion terminated with an actual Z-vector residual near `6e-6`
despite a requested `1e-12`. The gradient now uses preconditioned restarted GMRES
and verifies the unpreconditioned equation residual before returning. The
existing `cphf_max_cycle` remains an inner-iteration limit. A regression on this
32-AO system checks the residual and verifies that a one-iteration solve raises
an error rather than returning an inaccurate gradient. With GMRES the observed
residuals were around `1e-14` to `1e-13`.

The same geometry also needs care as a finite-difference oracle. A three-point
Hessian column at a `1e-3 Bohr` step differed by about `0.016`, `0.020`, and
`0.064 Eh/Bohr^2` for deltaS=-1,0,+1. Doubling the step increased those errors
approximately fourfold. Five-point differences remove this leading truncation
error, but shrinking the step indefinitely amplifies the SCF noise floor.
The example reports errors and step dependence instead of disguising this
limitation by silently increasing an acceptance threshold.

For the complete Hessian, ordinary default SCF/TD settings differed from the
validation reference by the following maximum absolute values (Eh/Bohr^2):

| deltaS | Input SCF residual | Input TD residual | Maximum Hessian difference |
| ---: | ---: | ---: | ---: |
| -1 | 1.01e-6 | 1.08e-6 | 4.85e-4 |
| 0 | 1.01e-6 | 3.91e-6 | 2.27e-4 |
| +1 | 1.01e-6 | 5.19e-7 | 6.62e-4 |

These are differences between two input accuracies, not rigorous error bounds
relative to an exact Hessian. In particular, tight GMRES convergence does not
make the input SCF/TD solution more accurate.


An independent dense solve on the first oxygen's 3x3 block agreed with the
iterative result to `1.02e-9`, `6.92e-11`, and `6.82e-10 Eh/Bohr^2` in the
three channels. This checks both the distinct orbital-derivative implementations
and the two state-response solvers; it does not by itself establish the nuclear
finite-difference accuracy. The default-environment Hessian and orbital-gradient
regressions passed **35 tests**, with **28 DFT fourth-derivative tests skipped**.


The completed full five-point run (`h=5e-4 Bohr`) reported:

| deltaS | Max gradient vs energy FD / Eh Bohr^-1 | Max Hessian vs gradient FD / Eh Bohr^-2 | FD Hessian asymmetry / Eh Bohr^-2 |
| ---: | ---: | ---: | ---: |
| -1 | 1.30e-4 | 3.61e-3 | 3.59e-3 |
| 0 | 5.03e-5 | 5.05e-3 | 5.00e-3 |
| +1 | 2.07e-4 | 1.45e-2 | 1.44e-2 |

All matched-state overlaps exceeded `0.999994`. These differences do **not**
satisfy a strict `2e-5 Eh/Bohr^2` Hessian acceptance threshold. They are reported
as an unresolved numerical-validation limit, not as a passing finite-difference
test. The substantial asymmetry of the finite-difference matrix and the step
scans demonstrate significant numerical-reference error, but do not prove that
every remaining discrepancy is numerical noise.

An independent five-point energy curvature on H1-z had errors
`8.13e-4`, `1.45e-4`, `2.38e-3 Eh/Bohr^2` at `h=1e-2 Bohr`. Halving that step
increased the errors to `3.14e-3`, `4.14e-4`, `8.82e-3`, respectively, consistent
with amplified energy/SCF noise. No further tightening of the user's ordinary
SCF/TD settings was imposed.

The existing public-gradient regression additionally passed **4 tests and
18 subtests** after the Z-vector fix. Together with the default-environment
checks above, that is **39 tests passed, 28 skipped, 18 subtests passed**. These
regression passes must not be confused with a strict finite-difference pass on
the new H2O2 case. `ruff check src/nest` and `git diff --check` passed.

### Explicit user-supplied A matrix and step convergence

A subsequent HF check used the user's blockwise `get_ab` builder, diagonalized
with `scipy.linalg.eigh`, for the central and every displaced geometry. Its
native ordering was checked against three random `vind` probes per channel at
every geometry (maximum allowed difference `1e-10`); Hermiticity was checked
before diagonalization. At the central geometry the maximum action difference
was `3.54e-13`, and the tightly converged iterative excitation energies agreed
with the dense eigenvalues within `2.2e-15 Eh`. Dense eigenvector residuals were
of order `1e-14` throughout. The lowering zero mode was excluded, and displaced
states were matched by overlap after C/O/V orbital-space alignment.

SCF settings were unchanged (`conv_tol=1e-12`, default gradient threshold).
Analytic gradients and full Hessians were recomputed using the dense states;
the derivative response equations retained their residual-checked GMRES solves.
The following errors are maxima over the O0-x Hessian **column**, not the full
matrix, in Eh/Bohr^2:

| Step / Bohr | Three-point -1 | Three-point 0 | Three-point +1 | Five-point -1 | Five-point 0 | Five-point +1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.002 | 6.328e-2 | 8.142e-2 | 2.582e-1 | 5.853e-3 | 5.008e-3 | 1.356e-2 |
| 0.001 | 1.569e-2 | 2.021e-2 | 6.408e-2 | 1.727e-4 | 1.890e-4 | 6.390e-4 |
| 0.0005 | 4.177e-3 | 5.126e-3 | 1.618e-2 | 3.383e-4 | 1.083e-4 | 2.142e-4 |
| 0.00025 | 1.043e-3 | 8.233e-4 | 2.524e-3 | 4.022e-5 | 6.109e-4 | 2.028e-3 |

Three-point errors show approximately second-order convergence over the larger
steps. Five-point errors do not decrease monotonically at the smallest steps,
even after removing iterative NTTDA eigensolver error. This rules out that
eigensolver as the sole explanation; it does not prove that the analytic
derivatives are free of remaining errors. SCF error remains in the numerical
reference. Five-point energy-gradient errors at `h=0.001 Bohr` were `9.72e-6`,
`1.74e-6`, and `1.94e-6 Eh/Bohr`, and likewise did not improve monotonically.
The standalone script, supplied builder, log, and raw arrays are retained in
the parent workspace's `Temp/nttda_hessian/get_ab_scan/` directory. This check
does not install the supplied builder as a production NEST API or validate
its DFT implementation.

### PBE and B3LYP on the same asymmetric geometry

The explicit-A scan was repeated for PBE and B3LYP with the same geometry,
basis, spin, and `nobeta=False`. Both used the default level-3 quadrature
(47,784 points), held fixed at all displaced geometries, and LibXC 7.0.0 built
with fourth derivatives. SCF used `conv_tol=1e-12` with the default orbital
gradient threshold; `max_memory=32000` MB was needed because the Hessian's
additional-memory estimate was about 14 GB. No production derivative formulas
were changed for this check.

At every geometry all three explicit A matrices were checked against random
`vind` probes and for Hermiticity before diagonalization. Dense eigenvector
residuals stayed near `1e-14`; all SCFs converged and all matched overlaps
exceeded 0.99. The three central Hessian response residuals were below `1e-10`.
Only the first oxygen's 3x3 analytic block was computed. The errors below are
maxima over its three entries differentiated along O0-x, **not** full-Hessian
errors. Units are Eh/Bohr^2.

| XC | Step / Bohr | Three-point -1 | Three-point 0 | Three-point +1 | Five-point -1 | Five-point 0 | Five-point +1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PBE | 0.002 | 1.750e-4 | 6.323e-5 | 1.001e-4 | 1.920e-4 | 5.792e-5 | 1.181e-4 |
| PBE | 0.001 | 9.880e-6 | 1.212e-5 | 1.102e-4 | 4.517e-5 | 1.159e-5 | 1.802e-4 |
| PBE | 0.0005 | 1.045e-3 | 3.598e-4 | 3.218e-4 | 1.390e-3 | 4.767e-4 | 4.249e-4 |
| PBE | 0.00025 | 3.000e-3 | 9.772e-4 | 2.451e-3 | 3.651e-3 | 1.183e-3 | 3.250e-3 |
| B3LYP | 0.002 | 5.100e-5 | 6.156e-5 | 1.777e-4 | 1.821e-4 | 4.715e-5 | 3.132e-4 |
| B3LYP | 0.001 | 9.350e-5 | 1.519e-5 | 2.356e-4 | 1.417e-4 | 3.263e-5 | 2.549e-4 |
| B3LYP | 0.0005 | 4.585e-4 | 1.019e-4 | 9.659e-4 | 5.801e-4 | 1.328e-4 | 1.209e-3 |
| B3LYP | 0.00025 | 4.673e-4 | 1.062e-4 | 9.693e-4 | 4.703e-4 | 1.076e-4 | 9.705e-4 |

At `h=0.001 Bohr`, the independent five-point total-energy differences agreed
with the O0-x analytic gradients to `2.40e-6`, `8.46e-7`, `1.93e-6 Eh/Bohr`
for PBE, and `7.99e-6`, `1.04e-6`, `7.42e-6 Eh/Bohr` for B3LYP.
The central SCF orbital-gradient norms were `1.61e-6` and `4.35e-7`,
respectively, so reference-state error was not eliminated by diagonalizing A.

These results support agreement at roughly `1e-5` to a few `1e-4 Eh/Bohr^2`
over the larger tested steps, but do not show clean second-order convergence
in every channel. B3LYP deltaS=0 has an approximately fourfold reduction from
`h=0.002` to `0.001`; most channels deteriorate at smaller steps. The results
must not be presented as a uniform `2e-5` acceptance pass or proof that all
remaining differences are SCF noise. They also do not validate moving-grid
response. The script `Temp/nttda_hessian/get_ab_scan/scan_dft.py` (in the parent
workspace) accepts `PBE` or `B3LYP`; the corresponding logs and `*_scan.npz`
files retain the observations and raw numerical data.
