"""Independent finite-gradient-difference checks of the analytic HF Hessian."""

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.hessian import rhf as rhf_hess

from nest.nttda import NTTDA
from nest.hessian.nttda import _eri_first, _eri_second, _overlap_second


def reference(mol):
    mf = dft.ROKS(mol).set(xc='HF', conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200)
    mf.kernel()
    assert mf.converged
    return mf


def dense_td(mf, delta_s, target=None, nobeta=False):
    td = NTTDA(mf).set(deltaS=delta_s, nobeta=nobeta, verbose=0)
    action, diagonal = {-1: td.gen_vind_sfd, 0: td.gen_vind_sc, 1: td.gen_vind_sfu}[delta_s]()
    a = action(np.eye(len(diagonal))).T
    np.testing.assert_allclose(a, a.T, atol=1e-12, rtol=0)
    energies, vectors = np.linalg.eigh(a)
    if delta_s == -1:
        keep = abs(energies) > 1e-8
        energies, vectors = energies[keep], vectors[:, keep]
    if target is not None:
        overlaps = abs(vectors.T @ target)
        root = np.argmax(overlaps)
        assert overlaps[root] > 0.99
        energies, vectors = energies[root:root + 1], vectors[:, root:root + 1]
    occ = mf.mo_occ
    nc, no, nv = (np.count_nonzero(occ == n) for n in (2, 1, 0))
    shape = {-1: (nc + no, no + nv), 0: (len(diagonal),), 1: (nc, nv)}[delta_s]
    td.e = energies
    td.xy = [(v.reshape(shape), 0) for v in vectors.T]
    td.converged = np.ones(len(energies), dtype=bool)
    td.nstates = len(energies)
    return td


def displaced_gradient(td, atom, xyz, displacement):
    mol = td.mol.copy()
    coords = mol.atom_coords()
    coords[atom, xyz] += displacement
    mol.set_geom_(coords, unit='Bohr')
    mf = reference(mol)
    # Align entire occupied-space blocks, not individual MO signs.  This also
    # handles rotations within degenerate C/O/V subspaces in the displaced SCF.
    cross = gto.intor_cross('int1e_ovlp', mol, td.mol)
    for occupation in (2, 1, 0):
        indices = np.flatnonzero(mf.mo_occ == occupation)
        overlap = mf.mo_coeff[:, indices].T @ cross @ td._scf.mo_coeff[:, indices]
        u, _, vh = np.linalg.svd(overlap)
        mf.mo_coeff[:, indices] = mf.mo_coeff[:, indices] @ (u @ vh)
    displaced = dense_td(mf, td.deltaS, td.xy[0][0].ravel(), td.nobeta)
    return displaced.Gradients().kernel(state=1)


@pytest.fixture(scope='module')
def ch2():
    return reference(gto.M(
        atom='C 0 0 0; H 1.4 0.2 1.1; H -1.2 0 1.3',
        basis='sto-3g', spin=2, unit='Bohr', verbose=0,
    ))


@pytest.mark.parametrize('delta_s', [-1, 0, 1])
def test_hessian_against_gradient_difference(ch2, delta_s):
    td = dense_td(ch2, delta_s)
    driver = td.Hessian()
    analytic = driver.kernel(state=1)
    assert analytic.shape == (3, 3, 3, 3)
    assert np.isfinite(analytic).all()
    assert driver.response_residual < 1e-9
    np.testing.assert_allclose(analytic, analytic.transpose(1, 0, 3, 2), atol=1e-12, rtol=0)
    np.testing.assert_allclose(analytic.sum(axis=1), 0, atol=1e-9, rtol=0)
    step = 2e-4
    finite = np.empty_like(analytic)
    for atom in range(3):
        for xyz in range(3):
            plus = displaced_gradient(td, atom, xyz, step)
            minus = displaced_gradient(td, atom, xyz, -step)
            finite[:, atom, :, xyz] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(analytic, finite, atol=2e-5, rtol=0)


def test_closed_shell_and_atom_selection():
    mf = reference(gto.M(atom='Li 0 0 0; H 0.2 0.1 2', basis='sto-3g', unit='Bohr', verbose=0))
    td = dense_td(mf, 1, nobeta=True)
    hessian = td.Hessian().kernel()
    selected = td.Hessian().kernel(atmlst=[1, 0])
    np.testing.assert_allclose(selected, hessian[[1, 0]][:, [1, 0]], atol=1e-10, rtol=0)
    only = td.Hessian().kernel(atmlst=[1])
    np.testing.assert_allclose(only[0, 0], hessian[1, 1], atol=1e-10, rtol=0)
    step = 2e-4
    finite = (displaced_gradient(td, 1, 2, step) - displaced_gradient(td, 1, 2, -step)) / (2 * step)
    np.testing.assert_allclose(hessian[:, 1, :, 2], finite, atol=2e-5, rtol=0)


def test_unsupported_reference_and_state(ch2):
    td = dense_td(ch2, 0)
    with pytest.raises(ValueError, match='one-based'):
        td.Hessian().kernel(state=0)
    with pytest.raises(ValueError, match='atom indices'):
        td.Hessian().kernel(atmlst=[0, 0])
    df = ch2.density_fit()
    df_td = NTTDA(df)
    with pytest.raises(NotImplementedError, match='conventional'):
        df_td.Hessian().kernel()
    dft_td = NTTDA(dft.ROKS(ch2.mol).set(xc='PBE'))
    with pytest.raises(NotImplementedError, match='HF'):
        dft_td.Hessian().kernel()
    scaled = NTTDA(dft.ROKS(ch2.mol).set(xc='0.5*HF'))
    with pytest.raises(NotImplementedError, match='100% HF'):
        scaled.Hessian().kernel()


@pytest.mark.parametrize('spin,delta_s,basis', [(1, 0, 'sto-3g'), (3, -1, 'sto-3g'), (1, 1, '6-31g')])
def test_other_spins_and_second_root(spin, delta_s, basis):
    mf = reference(gto.M(
        atom='N 0 0 0; H 1.4 0.2 1.1; H -1.2 0 1.3',
        basis=basis, spin=spin, unit='Bohr', verbose=0,
    ))
    td = dense_td(mf, delta_s, nobeta=True)
    hessian = td.Hessian().kernel(state=2, atmlst=[1])
    selected = dense_td(mf, delta_s, td.xy[1][0].ravel(), nobeta=True)
    step = 2e-4
    finite = (displaced_gradient(selected, 1, 2, step)
              - displaced_gradient(selected, 1, 2, -step)) / (2 * step)
    np.testing.assert_allclose(hessian[0, 0, :, 2], finite[1], atol=2e-5, rtol=0)


def test_ao_second_derivative_slots(ch2):
    mol = ch2.mol
    n = mol.nao_nr()
    shape = (3, 3, n, n, n, n)
    aa = mol.intor('int2e_ipip1', comp=9).reshape(shape)
    ab = mol.intor('int2e_ipvip1', comp=9).reshape(shape)
    ac = mol.intor('int2e_ip1ip2', comp=9).reshape(shape)
    saa, sab, _ = rhf_hess.get_ovlp(mol)
    masks = []
    for _, _, p0, p1 in mol.aoslice_by_atom():
        masks.append((np.arange(n) >= p0) & (np.arange(n) < p1))
    step = 1e-4
    for a, b, u, v in [(0, 0, 2, 2), (0, 1, 0, 2), (0, 1, 2, 2), (1, 1, 1, 2)]:
        plus, minus = mol.copy(), mol.copy()
        cp, cm = mol.atom_coords(), mol.atom_coords()
        cp[b, v] += step
        cm[b, v] -= step
        plus.set_geom_(cp, unit='Bohr')
        minus.set_geom_(cm, unit='Bohr')
        fd = (_eri_first(plus.intor('int2e_ip1', comp=3), masks[a], u)
              - _eri_first(minus.intor('int2e_ip1', comp=3), masks[a], u)) / (2 * step)
        exact = _eri_second(aa, ab, ac, masks[a], masks[b], u, v)
        np.testing.assert_allclose(exact, fd, atol=2e-8, rtol=0)
        sp = -plus.intor('int1e_ipovlp', comp=3)[u]
        sm = -minus.intor('int1e_ipovlp', comp=3)[u]
        sf = ((sp - sm) * masks[a][:, None] + (sp - sm).T * masks[a][None, :]) / (2 * step)
        exact_s = _overlap_second(saa, sab, masks[a], masks[b], u, v)
        np.testing.assert_allclose(exact_s, sf, atol=2e-8, rtol=0)
