"""Audit the gradient ledger against the energy action at arbitrary amplitudes."""

import copy

import numpy as np
import pytest
from pyscf import dft, gto
from scipy.linalg import expm

from nest.nttda import NTTDA
from nest.grad.nttda import delta_s_minus_one, delta_s_zero, delta_s_plus_one
from nest.grad.nttda.roks import canonical_pairs, pack_m_matrix


@pytest.fixture(scope='module')
def quartet():
    mol = gto.M(atom='N 0 0 0; H 1.4 0.2 1.1; H -1.2 0 1.3',
                basis='sto-3g', spin=3, unit='Bohr', verbose=0)
    mf = dft.ROKS(mol).set(xc='HF', conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200).run()
    assert mf.converged
    return mf


@pytest.mark.parametrize('delta_s', [-1, 0, 1])
@pytest.mark.parametrize('xc,nobeta', [('HF', False), ('PBE', False), ('PBE', True), ('M06-2X', True)])
def test_arbitrary_amplitude_energy_and_orbital_derivative(quartet, delta_s, xc, nobeta):
    # Keep the same orbitals for this algebra test. SCF stationarity and an
    # eigenvector are deliberately NOT assumed: Tr(X_OO)=0 would hide errors.
    mf = copy.copy(quartet)
    mf.xc = xc
    mf.grids = dft.gen_grid.Grids(mf.mol)
    mf.grids.level = 0
    mf.grids.build()
    td = NTTDA(mf).set(deltaS=delta_s, nobeta=nobeta, verbose=0)
    action, diagonal = {-1: td.gen_vind_sfd, 0: td.gen_vind_sc, 1: td.gen_vind_sfu}[delta_s]()
    rng = np.random.default_rng(83)
    x = rng.normal(size=len(diagonal))
    x /= np.linalg.norm(x)
    nc, no, nv = (np.count_nonzero(mf.mo_occ == n) for n in (2, 1, 0))
    shape = {-1: (nc + no, no + nv), 0: x.shape, 1: (nc, nv)}[delta_s]
    xy = (x.reshape(shape), 0)
    ledger = {-1: delta_s_minus_one.spin_lowering_ledger_scalar,
              0: delta_s_zero.same_spin_ledger_scalar,
              1: delta_s_plus_one.spin_raising_ledger_scalar}[delta_s]
    np.testing.assert_allclose(ledger(td, xy), x @ action(x[None])[0], atol=1e-11, rtol=0)

    pairs = canonical_pairs(td)
    direction = rng.normal(size=len(pairs))
    direction /= np.linalg.norm(direction)
    k = np.zeros((len(mf.mo_occ), len(mf.mo_occ)))
    for value, (p, q, _) in zip(direction, pairs):
        k[p, q], k[q, p] = value, -value
    m = td.Gradients()._analytic_components(xy, (), with_response=False)
    analytic = pack_m_matrix(m, pairs) @ direction
    step = 1e-5
    values = []
    for sign in (-1, 1):
        moved_mf = copy.copy(mf)
        moved_mf.mo_coeff = mf.mo_coeff @ expm(sign * step * k)
        moved = NTTDA(moved_mf).set(deltaS=delta_s, nobeta=nobeta, verbose=0)
        moved_action, _ = {-1: moved.gen_vind_sfd, 0: moved.gen_vind_sc, 1: moved.gen_vind_sfu}[delta_s]()
        values.append(x @ moved_action(x[None])[0])
    np.testing.assert_allclose(analytic, (values[1] - values[0]) / (2*step), atol=2e-7, rtol=0)


def test_large_basis_zvector_residual_and_failure():
    mol = gto.M(atom='''
    O  0.64372820  0.14077399 -0.04477253
    O -0.64862595 -0.12779073 -0.05445498
    H  1.16027512 -0.65947800  0.36730132
    H -1.12109306  0.55561188  0.42651873
    ''', basis='6-31g*', spin=2, symmetry=False, unit='Angstrom', verbose=0)
    mf = dft.ROKS(mol).set(xc='HF').run()
    assert mf.converged
    td = NTTDA(mf).set(deltaS=-1).run()
    assert td.converged[0]
    gradient = td.Gradients()
    gradient.kernel()
    # The old unnormalized Krylov solver silently left a residual near 6e-6.
    assert gradient.nttda_details.residual <= gradient.cphf_conv_tol
    with pytest.raises(RuntimeError, match='Z-vector GMRES failed'):
        td.Gradients().set(cphf_max_cycle=1).kernel()
