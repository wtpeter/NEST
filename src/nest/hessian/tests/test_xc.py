"""Blocked XC actions against the original explicit four-index implementation."""
import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.dft import libxc

from nest.hessian.xc import Semilocal
from nest.hessian.tests._xc_dense_reference import Semilocal as DenseSemilocal


@pytest.mark.parametrize('xc', ['SVWN', 'PBE', 'B3LYP', 'TPSS'])
def test_blocked_xc_mixed_derivatives(xc, monkeypatch):
    if libxc.max_deriv_order(xc) < 4:
        pytest.skip('requires fourth XC derivatives')
    mol = gto.M(atom='Li 0 0 0; H .2 .1 2', basis='sto-3g', unit='Bohr', verbose=0)
    mf = dft.ROKS(mol).set(xc=xc, conv_tol=1e-12).run()
    assert mf.converged
    # A fixed subset suffices for this quadrature algebra check; no displaced
    # SCF or grid convergence is involved in the dense-vs-contracted identity.
    mf.grids.coords = mf.grids.coords[::31].copy()
    mf.grids.weights = mf.grids.weights[::31].copy()
    # Isolate contraction algebra from the ill-conditioned MGGA vacuum tail:
    # roundoff in rho ~ 1e-15 can change TPSS fourth derivatives by 1e47.
    # End-to-end MGGA Hessian tests separately retain their complete grids.
    phi = mf._numint.eval_ao(mol, mf.grids.coords) @ mf.mo_coeff
    keep = np.einsum('gi,i,gi->g', phi, mf.mo_occ, phi) > 1e-6
    mf.grids.coords = mf.grids.coords[keep].copy()
    mf.grids.weights = mf.grids.weights[keep].copy()
    old = DenseSemilocal(mf)
    new = Semilocal(mf)
    new.block_size = 113
    c, n = mf.mo_coeff, mol.nao_nr()
    rng = np.random.default_rng(197)
    ca, cb, cab = rng.normal(scale=.01, size=(3, n, n))
    masks = [(np.arange(n) >= p0) & (np.arange(n) < p1)
             for p0, p1 in mol.aoslice_by_atom()[:, 2:]]
    phia = old.ao_derivative(masks[0], 0) @ c + old.ao0 @ ca
    phib = old.ao_derivative(masks[1], 2) @ c + old.ao0 @ cb
    phiab = (old.ao_derivative(masks[0]*masks[1], 0, 2) @ c
             + old.ao_derivative(masks[0], 0) @ cb
             + old.ao_derivative(masks[1], 2) @ ca + old.ao0 @ cab)
    # Non-symmetric probes and a multidimensional batch catch transposed J/K
    # layouts and the action's probe batching, which real densities can hide.
    dm = rng.normal(size=(3, 3, n, n))
    dense_args = [(), (phia,), (phia, phib, phiab)]
    direct_args = [(), ((0, 0, ca),), ((0, 0, ca), (1, 2, cb), cab)]
    for args0, args1 in zip(dense_args, direct_args):
        expected = old.terms(*args0)
        original_eval = mf._numint.eval_xc_eff
        def blocked_eval(code, rho, *args, **kwargs):
            assert rho.shape[-1] <= new.block_size
            return original_eval(code, rho, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(mf._numint, 'eval_xc_eff', blocked_eval)
            actual = new.terms(*args1)
        for slot in (0, 1, 3):
            np.testing.assert_allclose(actual[slot], expected[slot], atol=2e-10, rtol=2e-10)
        direct = np.einsum('ijkl,...kl->...ij', expected[2], dm)
        exchange = np.einsum('ijkl,...jl->...ik', expected[2], dm)
        np.testing.assert_allclose(actual[2].apply(dm), direct, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(actual[2].apply(dm, exchange=True), exchange, atol=2e-10, rtol=2e-10)
        for block in actual[2].blocks:
            for coefficient, left, right in block:
                assert coefficient.ndim == 3 and coefficient.shape[-1] <= new.block_size
                for factors in (left, right):
                    for lphi, rphi in factors:
                        assert lphi.ndim == rphi.ndim == 3
                        assert lphi.shape[1] <= new.block_size
