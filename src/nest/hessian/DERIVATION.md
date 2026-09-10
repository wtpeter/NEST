# HF NTTDA analytic nuclear Hessian

This implementation is a dense, small-system reference implementation for
conventional all-electron `ROKS(xc='HF')`, with real full-rank spatial orbitals.
All three NTTDA sectors are supported.  The target must be an isolated state.
DFT, density fitting, ECPs and X2C are explicitly rejected.

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

At fixed amplitudes, `E_ref`, every element of `A`, and `r` are linear in the
orthonormal MO one-electron integrals and chemists' ERIs.  `_mo_operators`
reuses the production `gen_vind_sfu/sc/sfd` with these integrals and identity
MO coefficients.  It therefore applies the *same* linear maps to integral
derivatives.  No second set of spin-adaptation formulas is introduced.

In the HF limit the reconstructed `Fz` equals
`-K(D_open)/2 = (F_alpha-F_beta)/2`.  The `nobeta` equal-spin reference gives
the same `F0` in this limit, so both settings follow the same derivatives.

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

The state response is obtained by a complete spectral reduced resolvent:

\[
X_a=\sum_{j\ne I} X_j
\frac{X_j^T\bar A_a X_I}{\omega_I-\omega_j},\qquad X_I^T X_a=0.
\]

All channel eigenvectors participate, including any zero mode filtered from
the public lowering-channel root list.  Only the target eigenvector is
projected out.  A gap below `1e-7 Eh` is rejected rather than regularized.
The public root is matched to the dense eigenstate by overlap and energy;
the stored `td.e` and `td.xy` are not modified.

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

The dense ERIs and complete state eigensystem limit practical system size.
The driver estimates memory before allocating derivative tensors.  There is
no DFT support: differentiating the reconstructed semilocal kernel twice
requires XC fourth derivatives, which the tested local LibXC functionals do
not provide.  Extending this implementation requires an analytic source of
those derivatives and an explicit choice of grid-response convention.
