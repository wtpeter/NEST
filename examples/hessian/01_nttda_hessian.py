"""Analytic total-energy Hessian and an independent energy-difference check.

Run from the NEST checkout:
    OMP_NUM_THREADS=36 conda run --no-capture-output -n nest-soc python examples/hessian/01_nttda_hessian.py

Small-system integral backend: response solves are iterative, but AO integral
derivatives are still dense. See src/nest/hessian/DERIVATION.md for limits.
For DFT, use a LibXC build with fourth derivatives; quadrature is held fixed.
"""

import time

import numpy as np
from pyscf import dft, gto

from nest.nttda import NTTDA


mol = gto.M(atom='C 0 0 0; H 1.4 0.2 1.1; H -1.2 0 1.3',
            basis='sto-3g', spin=2, unit='Bohr', verbose=0)
mf = dft.ROKS(mol).set(xc='HF', conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200).run()
assert mf.converged
# deltaS may be -1, 0, or +1. Tighten the eigenvector residual for Hessians.
# Lower lindep as well so small residual directions are not discarded early.
td = NTTDA(mf).set(deltaS=-1, nstates=3, conv_tol=1e-10, lindep=1e-20, max_cycle=200).run()
assert td.converged[0]
gradient = td.Gradients().kernel(state=1)
driver = td.Hessian().set(conv_tol=1e-10)  # default: solver='iterative'
start = time.perf_counter()
hessian = driver.kernel(state=1)
print('Hessian time / s:', time.perf_counter() - start)
print('GMRES iterations:', driver.response_iterations)
print('Maximum response residual:', driver.response_residual)
print('Hessian shape:', hessian.shape)  # [atom_A, atom_B, xyz_A, xyz_B]
print('Hessian / Eh Bohr^-2:\n', hessian.transpose(0, 2, 1, 3).reshape(9, 9))
print('Translation residual:', np.max(abs(hessian.sum(axis=1))))
# For a sub-block: td.Hessian().kernel(state=1, atmlst=[1]).

# Independent total-energy stencil along H1 z. Match states by overlap after
# aligning the C/O/V orbital spaces. For DFT the same coordinates and weights
# must be used at every displaced geometry.
step = 1e-3
energies = []
for factor in (-2, -1, 1, 2):
    coords = mol.atom_coords()
    coords[1, 2] += factor * step
    moved_mol = mol.copy().set_geom_(coords, unit='Bohr')
    moved_mf = dft.ROKS(moved_mol).set(
        xc=mf.xc, conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200,
    )
    if mf._numint._xc_type(mf.xc) != 'HF':
        moved_mf.grids.coords = mf.grids.coords.copy()
        moved_mf.grids.weights = mf.grids.weights.copy()
    moved_mf.kernel()
    assert moved_mf.converged
    cross = gto.intor_cross('int1e_ovlp', moved_mol, mol)
    for occupation in (2, 1, 0):
        indices = np.flatnonzero(mf.mo_occ == occupation)
        overlap = moved_mf.mo_coeff[:, indices].T @ cross @ mf.mo_coeff[:, indices]
        u, _, vh = np.linalg.svd(overlap)
        moved_mf.mo_coeff[:, indices] = moved_mf.mo_coeff[:, indices] @ (u @ vh)
    moved = NTTDA(moved_mf).set(
        deltaS=td.deltaS, nobeta=td.nobeta, nstates=td.nstates,
        conv_tol=1e-10, lindep=1e-20, max_cycle=200,
    ).run()
    overlaps = np.array([abs(np.vdot(td.xy[0][0], xy[0])) for xy in moved.xy])
    root = int(np.argmax(overlaps))
    assert overlaps[root] > 0.99, 'state matching failed'
    assert moved.converged[root], 'selected root did not converge'
    energies.append(moved_mf.e_tot + moved.e[root])

em2, em1, ep1, ep2 = energies
e0 = mf.e_tot + td.e[0]
finite_g = (em2 - 8*em1 + 8*ep1 - ep2) / (12*step)
finite_h = (-em2 + 16*em1 - 30*e0 + 16*ep1 - ep2) / (12*step**2)
print('Gradient: analytic, energy FD:', gradient[1, 2], finite_g)
print('Hessian: analytic, energy FD:', hessian[1, 1, 2, 2], finite_h)
np.testing.assert_allclose(gradient[1, 2], finite_g, atol=2e-7, rtol=0)
np.testing.assert_allclose(hessian[1, 1, 2, 2], finite_h, atol=1e-5, rtol=0)
