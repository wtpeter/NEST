"""Direct derivative contractions against explicit, unsymmetrized AO tensors."""

import numpy as np
import pytest
from pyscf import gto

from nest.hessian.eri import DirectERI
from nest.hessian.nttda import _first_transform, _second_transform, _transform
from nest.hessian.tests._eri_dense_reference import _eri_first, _eri_second


@pytest.mark.parametrize('cart', [False, True])
def test_direct_derivative_actions(cart, monkeypatch):
    mol = gto.M(atom='Li 0 0 0; H .2 .1 2', basis='6-31g*', unit='Bohr', cart=cart, verbose=0)
    n = mol.nao_nr()
    rng = np.random.default_rng(41)
    c, ca, cb, cab = rng.normal(scale=.2, size=(4, n, n))
    dm = rng.normal(size=(2, 3, n, n))  # Directed densities, not symmetric probes.
    eri = mol.intor('int2e', aosym='s1')
    ip1 = mol.intor('int2e_ip1', comp=3, aosym='s1')
    second = [mol.intor(name, comp=9, aosym='s1').reshape(3, 3, n, n, n, n)
              for name in ('int2e_ipip1', 'int2e_ipvip1', 'int2e_ip1ip2')]
    masks = [(np.arange(n) >= p0) & (np.arange(n) < p1) for p0, p1 in mol.aoslice_by_atom()[:, 2:]]
    a = (0, 0, ca)
    ga = _eri_first(ip1, masks[0], 0)
    cases = [((), _transform(eri, [c]*4)),
             ((a,), _first_transform(eri, ga, c, ca)),
             (((None, 0, ca),), _first_transform(eri, np.zeros_like(eri), c, ca))]
    for atom, xyz in ((0, 1), (0, 2), (1, 2), (1, 1)):
        b = (atom, xyz, cb)
        gb = _eri_first(ip1, masks[atom], xyz)
        gab = _eri_second(*second, masks[0], masks[atom], 0, xyz)
        cases.append(((a, b, cab), _second_transform(eri, ga, gb, gab, c, ca, cb, cab)))
    base = DirectERI(mol, c)
    for args, tensor in cases:
        direct = base.derivative(*args) if args else base
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError('direct action allocated a full AO integral tensor')
            patch.setattr(mol, 'intor', forbidden)
            actual_j = direct.apply(dm)
            actual_k = direct.apply(dm, exchange=True)
        np.testing.assert_allclose(actual_j, np.einsum('pqrs,...sr->...pq', tensor, dm), atol=2e-10, rtol=0)
        np.testing.assert_allclose(actual_k, np.einsum('prsq,...rs->...pq', tensor, dm), atol=2e-10, rtol=0)
    assert 0 < base.cache_bytes <= base.cache_limit
    base.clear_potential_cache()
    assert base.cache_bytes == 0 and not base.potential_cache
    base.cache_limit = 0
    np.testing.assert_allclose(direct.apply(dm), actual_j, atol=2e-10, rtol=0)
    assert base.cache_bytes == 0 and not base.potential_cache


@pytest.mark.parametrize('xc,delta_s', [('HF', -1), ('PBE', 0), ('B3LYP', 1)])
def test_hessian_never_builds_full_eri(xc, delta_s, monkeypatch):
    from pyscf.dft import libxc
    from nest.hessian.tests.test_nttda_hessian import reference, dense_td
    if xc != 'HF' and libxc.max_deriv_order(xc) < 4:
        pytest.skip('requires fourth XC derivatives')
    mf = reference(gto.M(atom='C 0 0 0; H 1.4 .2 1.1; H -1.2 0 1.3',
                         basis='sto-3g', spin=2, unit='Bohr', verbose=0), xc)
    td = dense_td(mf, delta_s)
    cached_eri = mf._eri
    original = gto.Mole.intor

    def checked_intor(mol, name, *args, **kwargs):
        assert not name.startswith('int2e'), 'Hessian requested a full ERI tensor'
        return original(mol, name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(gto.Mole, 'intor', checked_intor)
        hessian = td.Hessian().kernel(atmlst=[1])
        if xc == 'HF':
            from nest.nttda import NTTDA
            automatic = NTTDA(mf).set(deltaS=delta_s)
            generated = automatic.Hessian().kernel(atmlst=[1])
            assert automatic.xy is not None and automatic.converged[0]
            np.testing.assert_allclose(generated, hessian, atol=2e-5, rtol=0)
    assert np.isfinite(hessian).all()
    assert mf._eri is cached_eri, 'Hessian modified the user reference cache'
