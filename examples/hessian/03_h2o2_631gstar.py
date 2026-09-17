"""Full HF NTTDA derivative validation on asymmetric H2O2 / 6-31G*.

    OMP_NUM_THREADS=20 conda run --no-capture-output -n nest-soc python examples/hessian/03_h2o2_631gstar.py

Coordinates are Angstrom; neutral triplet reference, no symmetry, nobeta=False.
All three channels use their lowest computed scalar state. Each full analytic
Hessian is compared to all 12 columns of five-point gradient differences. The gradients
are independently compared to total-energy differences. A five-point energy
stencil checks the first H atom z direction at two steps. No dense TD matrix is
assembled. This example is more expensive than the small CH2 examples.
Differences are reported rather than hidden by a universal acceptance threshold:
this geometry has rapid derivative variation and an SCF noise floor.
"""

import time

import numpy as np
from pyscf import dft, gto

from nest.nttda import NTTDA


def reference(mol, original=None):
    # These settings are only for the finite-difference oracle below.
    # Ordinary default SCF/TD settings are tested separately at the end.
    mf = dft.ROKS(mol).set(xc='HF', conv_tol=1e-12)
    mf.kernel(dm0=None if original is None else original.make_rdm1())
    assert mf.converged, 'reference SCF did not converge'
    if original is not None:
        np.testing.assert_array_equal(mf.mo_occ, original.mo_occ)
        cross = gto.intor_cross('int1e_ovlp', mol, original.mol)
        # Align the whole C/O/V spaces before comparing excitation amplitudes.
        for occupation in (2, 1, 0):
            indices = np.flatnonzero(mf.mo_occ == occupation)
            overlap = mf.mo_coeff[:, indices].T @ cross @ original.mo_coeff[:, indices]
            u, _, vh = np.linalg.svd(overlap)
            mf.mo_coeff[:, indices] = mf.mo_coeff[:, indices] @ (u @ vh)
    return mf


def state(mf, delta_s, target=None):
    td = NTTDA(mf).set(deltaS=delta_s, nobeta=False, nstates=3,
                       conv_tol=1e-10, lindep=1e-20, max_cycle=200).run()
    root, overlap = 0, 1.
    if target is not None:
        overlaps = np.array([abs(np.vdot(target.xy[0][0], xy[0])) for xy in td.xy])
        root = int(np.argmax(overlaps))
        overlap = overlaps[root]
        assert overlap > 0.99, 'state matching failed'
    assert td.converged[root], 'selected NTTDA root did not converge'
    return td, root, overlap


mol = gto.M(atom='''
O  0.64372820  0.14077399 -0.04477253
O -0.64862595 -0.12779073 -0.05445498
H  1.16027512 -0.65947800  0.36730132
H -1.12109306  0.55561188  0.42651873
''', basis='6-31g*', charge=0, spin=2, symmetry=False, unit='Angstrom', verbose=0)
mf = reference(mol)
channels = (-1, 0, 1)
states, gradients, hessians, energies = {}, {}, {}, {}
print('AO count:', mol.nao_nr(), 'reference energy / Eh:', mf.e_tot, flush=True)
print('SCF orbital-gradient norm:', np.linalg.norm(
    mf.get_grad(mf.mo_coeff, mf.mo_occ, mf.get_fock())), flush=True)
for ds in channels:
    td, _, _ = state(mf, ds)
    states[ds] = td
    energies[ds] = mf.e_tot + td.e[0]
    gradient = td.Gradients()
    gradients[ds] = gradient.kernel(state=1)
    print('deltaS:', ds, 'gradient Z-vector residual:', gradient.nttda_details.residual, flush=True)
    start = time.perf_counter()
    driver = td.Hessian()
    hessians[ds] = driver.kernel(state=1)
    print('deltaS:', ds, 'omega / Eh:', td.e[0],
          'full Hessian seconds:', time.perf_counter() - start,
          'response residual:', driver.response_residual,
          'iterations:', driver.response_iterations, flush=True)
    np.testing.assert_allclose(hessians[ds].sum(axis=1), 0, atol=1e-8, rtol=0)

step = 5e-4  # Bohr; five-point stencil removes the dominant O(step**2) error
finite_gradients = {ds: np.zeros((4, 3)) for ds in channels}
finite_hessians = {ds: np.zeros((4, 4, 3, 3)) for ds in channels}
minimum_overlap = {ds: 1. for ds in channels}
for atom in range(4):
    for xyz in range(3):
        values, forces = {}, {}
        for factor in (-2, -1, 1, 2):
            coords = mol.atom_coords()
            coords[atom, xyz] += factor * step
            moved_mol = mol.copy().set_geom_(coords, unit='Bohr')
            moved_mf = reference(moved_mol, mf)
            for ds in channels:
                moved, root, overlap = state(moved_mf, ds, states[ds])
                minimum_overlap[ds] = min(minimum_overlap[ds], overlap)
                values[factor, ds] = moved_mf.e_tot + moved.e[root]
                forces[factor, ds] = moved.Gradients().kernel(state=root + 1)
        for ds in channels:
            finite_gradients[ds][atom, xyz] = (values[-2, ds] - 8*values[-1, ds] + 8*values[1, ds] - values[2, ds]) / (12*step)
            finite_hessians[ds][:, atom, :, xyz] = (forces[-2, ds] - 8*forces[-1, ds] + 8*forces[1, ds] - forces[2, ds]) / (12*step)
        print('Finite-difference column complete:', atom, xyz, flush=True)

for ds in channels:
    print('deltaS:', ds,
          'max gradient-energy FD error / Eh Bohr^-1:', np.max(abs(gradients[ds] - finite_gradients[ds])),
          'max Hessian-gradient FD error / Eh Bohr^-2:', np.max(abs(hessians[ds] - finite_hessians[ds])),
          'FD Hessian asymmetry:', np.max(abs(finite_hessians[ds] - finite_hessians[ds].transpose(1, 0, 3, 2))),
          'minimum matched overlap:', minimum_overlap[ds], flush=True)

# Independent energy-only curvature check on H1 z. The full gradient stencil
# above already covers all mixed atom/Cartesian blocks. Larger energy-stencil
# steps reduce cancellation; both step sizes are reported without tightening SCF.
direction = np.zeros((4, 3))
direction[2, 2] = 1.
for step in (1e-2, 5e-3):
    values = {}
    for factor in (-2, -1, 1, 2):
        moved_mol = mol.copy().set_geom_(mol.atom_coords() + factor*step*direction, unit='Bohr')
        moved_mf = reference(moved_mol, mf)
        for ds in channels:
            moved, root, _ = state(moved_mf, ds, states[ds])
            values[factor, ds] = moved_mf.e_tot + moved.e[root]
    for ds in channels:
        em2, em1, ep1, ep2 = (values[factor, ds] for factor in (-2, -1, 1, 2))
        finite = (-em2 + 16*em1 - 30*energies[ds] + 16*ep1 - ep2) / (12*step**2)
        analytic = np.einsum('ax,abxy,by', direction, hessians[ds], direction)
        print('deltaS:', ds, 'energy FD step / Bohr:', step,
              'directional Hessian:', analytic, 'energy FD:', finite,
              'absolute error:', abs(analytic - finite), flush=True)


# Actual everyday use: no custom SCF/TD convergence settings.
default_mf = dft.ROKS(mol).set(xc='HF').run()
assert default_mf.converged
for ds in channels:
    default_td = NTTDA(default_mf).set(deltaS=ds).run()
    assert default_td.converged[0]
    driver = default_td.Hessian()
    default_hessian = driver.kernel()
    error = np.max(abs(default_hessian - hessians[ds]))
    print('DEFAULT settings, deltaS:', ds,
          'SCF residual:', driver.reference_residual, 'TD residual:', driver.state_residual,
          'max full Hessian difference from validation reference:', error, flush=True)
