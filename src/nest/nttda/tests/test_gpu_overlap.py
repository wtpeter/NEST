"""GPU/CPU AWF agreement and excited-triplet DF numerical checks."""
import copy
import os
from pathlib import Path

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
        pytest.skip('CUDA device required')
    from gpu4pyscf.dft.roks import ROKS
    from gpu4pyscf.sftda import NTTDA_ROKS
    mol = gto.M(atom='C .02 -.03 .01; H -.02 .8 .62; H .03 -.91 .5',
                basis='sto-3g', spin=2, verbose=0)
    mf = ROKS(mol, xc='PBE').density_fit(auxbasis='def2-universal-jkfit')
    mf.set(conv_tol=1e-12)
    mf.grids.level = 3
    mf.small_rho_cutoff = 0
    mf.run()
    s, t = [NTTDA_ROKS(mf).set(deltaS=sector, nstates=3, conv_tol=1e-10, max_cycle=200).run()
            for sector in (-1, 0)]
    assert mf.converged and np.all(s.converged) and np.all(t.converged)
    return s, t


def test_gpu_singlet_awf_and_dipoles(solution):
    from gpu4pyscf.nac.nttda import build_spin_adapted_awf, interstate_rdm1
    from nest.nttda.overlap import build_awf, state_density
    from nest.soc.gpu_nttda import to_nest
    s, _ = solution
    cpu = to_nest(s)
    for i in range(3):
        expected = build_spin_adapted_awf(s, s.xy[i])
        actual = build_awf(cpu, i)
        for det in actual.coefficients.keys() | expected.coefficients.keys():
            assert abs(actual.coefficients.get(det, 0)-expected.coefficients.get(det, 0)) < 1e-12
        for j in range(3):
            np.testing.assert_allclose(state_density(cpu, i, j),
                                       interstate_rdm1(s, s.xy[i], s.xy[j], method='determinant'), atol=1e-12)


def test_triplet_against_independent_cpu_nttda(solution):
    import cupy as cp
    from nest.soc.gpu_nttda import to_nest
    from nest.nttda.overlap import build_awf
    _, triplet = solution
    cpu = to_nest(triplet)
    expected = [build_awf(cpu, i) for i in range(3)]
    cpu._scf = cpu._scf.density_fit(auxbasis='def2-universal-jkfit')
    cpu._scf.grids.coords = cp.asnumpy(triplet._scf.grids.coords)
    cpu._scf.grids.weights = cp.asnumpy(triplet._scf.grids.weights)
    cpu.set(conv_tol=1e-9, lindep=1e-20, max_cycle=200).run()
    assert np.all(cpu.converged)
    np.testing.assert_allclose(cpu.e, triplet.e, atol=1e-8, rtol=0)
    for i in range(3):
        x = cpu.xy[i][0]
        phase = np.sign(np.vdot(x, cp.asnumpy(cp.asarray(triplet.xy[i][0]))).real)
        cpu.xy[i] = (x*phase, 0)
        actual = build_awf(cpu, i)
        for det in actual.coefficients.keys() | expected[i].coefficients.keys():
            assert abs(actual.coefficients.get(det, 0)-expected[i].coefficients.get(det, 0)) < 1e-7


def test_triplet_df_gradient_against_energy_fd(solution):
    from gpu4pyscf.dft.roks import ROKS
    from gpu4pyscf.sftda import NTTDA_ROKS
    s, t = solution
    h = 1e-4
    energies = []
    for displacement in (-h, h):
        coordinates = t.mol.atom_coords().copy()
        coordinates[1, 0] += displacement
        mol = t.mol.set_geom_(coordinates, unit='Bohr', inplace=False)
        mf = ROKS(mol, xc='PBE').density_fit(auxbasis='def2-universal-jkfit')
        mf.set(conv_tol=1e-12)
        mf.grids.level = 3
        mf.small_rho_cutoff = 0
        mf.kernel(dm0=t._scf.make_rdm1())
        td = NTTDA_ROKS(mf).set(deltaS=0, nstates=3, conv_tol=1e-10, max_cycle=200).run()
        assert mf.converged and np.all(td.converged)
        energies.append(float(mf.e_tot)+td.e)
    fd = (energies[1]-energies[0])/(2*h)
    for i in range(3):
        actual = t.Gradients().set(cphf_conv_tol=1e-10).kernel(state=i+1)
        np.testing.assert_allclose(actual[1, 0], fd[i], atol=1e-5, rtol=0)


@pytest.mark.parametrize('backend', ['serial', 'batch'])
def test_shared_triplet_gradients_against_independent_drivers(solution, backend):
    from nest.interfaces.sharc_nttda import _triplet_gradients
    _, triplet = solution
    actual = _triplet_gradients(triplet, backend=backend)
    expected = [triplet.Gradients().set(cphf_conv_tol=1e-10).kernel(state=i)
                for i in (1, 2, 3)]
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def test_multichannel_phase_covariance_and_wfoverlap(solution):
    from nest.soc.gpu_nttda import to_nest, from_gpu_nttda
    from nest.interfaces.sharc_nttda import overlaps, _dipoles, canonical_order
    s, t = solution
    flipped = copy.copy(t)
    flipped.xy = [(-x if i == 1 else x, y) for i, (x, y) in enumerate(t.xy)]
    gauge = np.diag([1.]*3 + [1., -1., 1.]*3)
    before, after = [from_gpu_nttda(s, td, soctype='Zeff').run() for td in (t, flipped)]
    order = canonical_order(before)
    np.testing.assert_allclose(after.h_soc[np.ix_(order, order)],
                               gauge @ before.h_soc[np.ix_(order, order)] @ gauge, atol=1e-12)
    np.testing.assert_allclose(_dipoles(s, flipped), gauge @ _dipoles(s, t) @ gauge, atol=1e-12)
    expected = overlaps(to_nest(s), s, to_nest(t), flipped)
    np.testing.assert_allclose(expected, gauge, atol=1e-12)
    actual = overlaps(to_nest(s), s, to_nest(t), flipped, backend='gpu')
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    if os.environ.get('NTTDA_WFOVERLAP_TEST_DIR'):
        actual = overlaps(to_nest(s), s, to_nest(t), flipped,
                          directory=Path(os.environ['NTTDA_WFOVERLAP_TEST_DIR'])/'gpu', backend='wfoverlap')
        np.testing.assert_allclose(actual, expected, atol=6e-11)


def test_gpu_contractions_against_displaced_determinants(solution):
    from nest.soc.gpu_nttda import to_nest
    from nest.nttda.overlap_contraction import gpu_overlap_matrix, overlap_from_mo
    from nest.nttda.overlap import build_awf, determinant_overlap
    from scipy.linalg import expm
    import cupy as cp
    rng = np.random.default_rng(417)
    for td in solution:
        cpu = to_nest(td)
        n = len(cpu._scf.mo_occ)
        rotation = rng.normal(size=(n, n))*.04
        mo = expm(rotation-rotation.T)
        actual = cp.asnumpy(gpu_overlap_matrix(cpu, td, mo_overlap=cp.asarray(mo)))
        wf = [build_awf(cpu, i) for i in range(3)]
        expected = np.array([[determinant_overlap(a, b, mo) for b in wf] for a in wf])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual, overlap_from_mo(cpu, cpu, mo), atol=1e-12, rtol=0)


def test_rectangular_cuda_cubic_contractions(solution):
    import cupy as cp
    from nest.nttda.overlap_contraction import overlap_from_mo
    from nest.nttda.overlap import build_awf, determinant_overlap
    from nest.nttda.tests.test_overlap import rectangular_pair
    for sector in (-1, 0):
        left, right, mo = rectangular_pair(sector)
        expected = np.array([[determinant_overlap(build_awf(left, i), build_awf(right, j), mo)
                              for j in range(3)] for i in range(4)])
        actual = cp.asnumpy(overlap_from_mo(left, right, cp.asarray(mo), xp=cp))
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
        reverse = cp.asnumpy(overlap_from_mo(right, left, cp.asarray(mo.conj().T), xp=cp))
        np.testing.assert_allclose(reverse, expected.conj().T, atol=1e-13, rtol=0)
