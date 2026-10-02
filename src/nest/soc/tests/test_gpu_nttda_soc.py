"""GPU integration tests; require the NT-TDA GPU branch and a CUDA worker."""

import numpy as np
import pytest
from pyscf import gto


@pytest.fixture(scope='module')
def solution():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CUDA device required')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('Usable CUDA driver required')
    from gpu4pyscf.dft.roks import ROKS
    from gpu4pyscf.sftda import NTTDA_ROKS
    mol = gto.M(atom='C .02 -.03 .01; H -.02 .8 .62; H .03 -.91 .5',
                basis='sto-3g', spin=2, verbose=0)
    mf = ROKS(mol, xc='PBE').set(conv_tol=1e-12)
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.run()
    td = NTTDA_ROKS(mf).set(deltaS=-1, nstates=3, conv_tol=1e-10, max_cycle=200).run()
    assert mf.converged and np.all(td.converged)
    return td


def test_bridge_against_independent_nest_solver(solution):
    import cupy as cp
    from nest.soc.gpu_nttda import from_gpu_nttda, to_nest
    td = solution
    actual = from_gpu_nttda(td, include_reference=True).run()
    cpu = to_nest(td)
    cpu._scf.grids.coords = cp.asnumpy(td._scf.grids.coords)
    cpu._scf.grids.weights = cp.asnumpy(td._scf.grids.weights)
    cpu.set(conv_tol=1e-8, max_cycle=200).run()
    np.testing.assert_allclose(cpu.e, td.e, atol=1e-8, rtol=0)
    # Align independently solved scalar root phases before comparing SOC.
    cpu.xy = [(x*np.sign(np.vdot(x, cp.asnumpy(cp.asarray(gpu_x))).real), y)
              for (x, y), (gpu_x, _) in zip(cpu.xy, td.xy)]
    expected = cpu.SOC(include_reference=True).run()
    np.testing.assert_allclose(actual.h_soc, expected.h_soc, atol=1e-8, rtol=0)
    assert np.linalg.norm(actual.h_soc[:3, 3:]) > 1e-6
    np.testing.assert_allclose(actual.h_soc, actual.h_soc.conj().T, atol=1e-13)


def test_bridge_freezes_geometry_and_electronic_solution(solution):
    import copy
    import cupy as cp
    from nest.soc.gpu_nttda import to_nest
    moving = copy.copy(solution)
    moving._scf = copy.copy(solution._scf)
    moving.mol = solution.mol.copy()
    moving._scf.mol = moving.mol
    moving._scf.mo_coeff = solution._scf.mo_coeff.copy()
    moving.xy = [(x.copy(), y) for x, y in solution.xy]
    carrier = to_nest(moving)
    original_coords = moving.mol.atom_coords()
    original_coeff = cp.asnumpy(moving._scf.mo_coeff)
    original_x = cp.asnumpy(moving.xy[0][0])
    assert carrier.mol is carrier._scf.mol
    assert carrier.mol is not moving.mol
    displaced = original_coords.copy()
    displaced[1, 0] += .1
    moving.mol.set_geom_(displaced, unit='Bohr', inplace=True)
    moving._scf.mo_coeff[:] = 0
    moving.xy[0][0][:] = 0
    np.testing.assert_array_equal(carrier.mol.atom_coords(), original_coords)
    np.testing.assert_array_equal(carrier._scf.mo_coeff, original_coeff)
    np.testing.assert_array_equal(carrier.xy[0][0], original_x)


def test_all_gpu_gradients_against_nest(solution):
    import cupy as cp
    from gpu4pyscf.grad.nttda import compute_frame
    from nest.soc.gpu_nttda import to_nest
    cpu = to_nest(solution)
    cpu._scf.grids.coords = cp.asnumpy(solution._scf.grids.coords)
    cpu._scf.grids.weights = cp.asnumpy(solution._scf.grids.weights)
    actual = compute_frame(solution, 1, gradient_states=[1, 2, 3])
    for state in (1, 2, 3):
        expected = cpu.Gradients().kernel(state=state)
        np.testing.assert_allclose(actual['gradients'][state], expected, atol=1e-7, rtol=0)


def test_nac_against_displaced_wavefunction_overlap(solution):
    nac = solution.NAC()
    actual = nac.kernel(state_I=1, state_J=2, atmlst=[1], ediff=True, use_etfs=False)
    expected = nac.finite_difference(state_I=1, state_J=2, atmlst=[1], step=1e-4)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


def test_soc_dipoles_and_overlap_follow_root_phase(solution):
    import copy
    from nest.soc.gpu_nttda import from_gpu_nttda, to_nest
    from nest.interfaces.sharc_nttda import canonical_order, overlaps, _dipoles
    flipped = copy.copy(solution)
    flipped.xy = [(-x if i == 0 else x, y) for i, (x, y) in enumerate(solution.xy)]
    original_soc = from_gpu_nttda(solution, include_reference=True).run()
    flipped_soc = from_gpu_nttda(flipped, include_reference=True).run()
    order = canonical_order(original_soc)
    gauge = np.diag([-1., 1., 1., 1., 1., 1.])
    original_h = original_soc.h_soc[np.ix_(order, order)]
    flipped_h = flipped_soc.h_soc[np.ix_(order, order)]
    np.testing.assert_allclose(flipped_h, gauge @ original_h @ gauge, atol=1e-12)
    np.testing.assert_allclose(overlaps(to_nest(solution), flipped), gauge, atol=1e-9)
    original_dm, flipped_dm = _dipoles(solution), _dipoles(flipped)
    np.testing.assert_allclose(flipped_dm, gauge @ original_dm @ gauge, atol=1e-10)


def test_ensemble_bridge_keeps_selected_energy_zero(solution):
    import cupy as cp
    from pyscf import dft
    from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS
    from nest.soc.gpu_nttda import to_nest
    from nest.interfaces.sharc_nttda import _snapshot
    mf = EnsembleROKS(solution.mol, xc='PBE').set(conv_tol=1e-12, conv_tol_grad=1e-8)
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.run()
    assert mf.converged
    td = NTTDA_EnsembleROKS(mf).set(deltaS=-1, nstates=3, conv_tol=1e-10).run()
    assert np.all(td.converged)
    # Independently evaluate the selected high-spin reference on the CPU,
    # with identical ensemble orbitals and quadrature; do not optimize it.
    cpu = dft.ROKS(mf.mol, xc=mf.xc)
    cpu.grids.coords = cp.asnumpy(mf.grids.coords)
    cpu.grids.weights = cp.asnumpy(mf.grids.weights)
    cpu.small_rho_cutoff = 0
    dm = cpu.make_rdm1(cp.asnumpy(mf.mo_coeff), cp.asnumpy(mf.mo_occ))
    expected = cpu.energy_tot(dm, cpu.get_hcore(), cpu.get_veff(dm=dm))
    assert abs(expected - mf.e_tot) > 1e-4
    carrier = to_nest(td)
    np.testing.assert_allclose(carrier._scf.e_tot, expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(_snapshot(td)['e_tot'], expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(carrier._scf.e_tot + carrier.e, td.e_tot, atol=1e-12, rtol=0)


@pytest.mark.parametrize('ensemble', [False, True])
def test_triplet_gradient_batch_against_serial_and_cpu(solution, ensemble):
    import cupy as cp
    from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS, NTTDA_ROKS
    from nest.interfaces.sharc_nttda import _triplet_gradients
    from nest.soc.gpu_nttda import to_nest
    if ensemble:
        mf = EnsembleROKS(solution.mol.copy(), xc='PBE').set(
            conv_tol=1e-12, conv_tol_grad=1e-8)
        mf.grids.level = 0
        mf.small_rho_cutoff = 0
        mf.run()
        cls = NTTDA_EnsembleROKS
    else:
        mf, cls = solution._scf, NTTDA_ROKS
    assert mf.converged
    td = cls(mf).set(deltaS=0, nstates=3, conv_tol=1e-10, max_cycle=200).run()
    assert np.all(td.converged)
    serial = np.asarray(_triplet_gradients(td, backend='serial'))
    batch = np.asarray(_triplet_gradients(td, backend='batch'))
    np.testing.assert_allclose(batch, serial, atol=1e-8, rtol=0)
    if not ensemble:
        cpu = to_nest(td)
        cpu._scf.grids.coords = cp.asnumpy(mf.grids.coords)
        cpu._scf.grids.weights = cp.asnumpy(mf.grids.weights)
        expected = np.array([cpu.Gradients().kernel(state=i) for i in (1, 2, 3)])
        np.testing.assert_allclose(batch, expected, atol=1e-7, rtol=0)
