"""Independent energy/gradient difference checks of the analytic Hessian."""

import copy

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.hessian import rhf as rhf_hess

from nest.nttda import NTTDA
from nest.hessian.nttda import _eri_first, _eri_second, _overlap_second


def reference(mol, xc='HF', grid=None):
    mf = dft.ROKS(mol).set(xc=xc, conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200)
    mf.grids.level = 0
    if grid is not None:
        mf.grids.coords = np.array(grid[0], copy=True)
        mf.grids.weights = np.array(grid[1], copy=True)
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


def displaced_state(td, displacement):
    mol = td.mol.copy()
    coords = mol.atom_coords()
    coords += displacement
    mol.set_geom_(coords, unit='Bohr')
    grid = None
    if td._scf._numint._xc_type(td._scf.xc) != 'HF':
        grid = (td._scf.grids.coords, td._scf.grids.weights)
    mf = reference(mol, td._scf.xc, grid)
    # Align entire occupied-space blocks, not individual MO signs.  This also
    # handles rotations within degenerate C/O/V subspaces in the displaced SCF.
    cross = gto.intor_cross('int1e_ovlp', mol, td.mol)
    for occupation in (2, 1, 0):
        indices = np.flatnonzero(mf.mo_occ == occupation)
        overlap = mf.mo_coeff[:, indices].T @ cross @ td._scf.mo_coeff[:, indices]
        u, _, vh = np.linalg.svd(overlap)
        mf.mo_coeff[:, indices] = mf.mo_coeff[:, indices] @ (u @ vh)
    return dense_td(mf, td.deltaS, td.xy[0][0].ravel(), td.nobeta)


def displaced_gradient(td, atom, xyz, displacement):
    shift = np.zeros((td.mol.natm, 3))
    shift[atom, xyz] = displacement
    return displaced_state(td, shift).Gradients().kernel(state=1)


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
    dft_td = NTTDA(dft.ROKS(ch2.mol).set(xc='CAM-B3LYP'))
    with pytest.raises(NotImplementedError, match='range separation'):
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


@pytest.mark.parametrize('delta_s,nobeta', [(-1, False), (0, False), (1, False),
                                          (-1, True), (0, True), (1, True)])
@pytest.mark.parametrize('xc', ['SVWN', 'PBE'])
def test_semilocal_fixed_grid_hessian(ch2, delta_s, nobeta, xc):
    from pyscf.dft import libxc
    if libxc.max_deriv_order(xc) < 4:
        pytest.skip('requires LibXC compiled with fourth derivatives (DISABLE_LXC=OFF)')
    mf = reference(ch2.mol, xc)
    td = dense_td(mf, delta_s, nobeta=nobeta)
    hessian = td.Hessian().kernel(atmlst=[0, 1])
    # Core-centered fixed grids need a smaller step: the Cxx discrepancy
    # decreases by a factor of four when the central-difference step is halved.
    step = 1e-4
    for atom, xyz in [(0, 0), (1, 2)]:
        finite = (displaced_gradient(td, atom, xyz, step)
                  - displaced_gradient(td, atom, xyz, -step)) / (2 * step)
        np.testing.assert_allclose(hessian[:, atom, :, xyz], finite[:2], atol=2e-5, rtol=0)


@pytest.mark.parametrize('xc', ['0.25*HF + 0.75*LDA_X + LDA_C_VWN', 'PBE', 'B3LYP', 'TPSS', 'M06-2X'])
def test_semilocal_operator_closure(ch2, xc):
    from pyscf.dft import libxc
    from nest.hessian.xc import Semilocal
    from nest.hessian.nttda import _mo_operators, _transform
    from nest.grad.nttda.roks import canonical_pairs
    if libxc.max_deriv_order(xc) < 4:
        pytest.skip('requires LibXC compiled with fourth derivatives (DISABLE_LXC=OFF)')
    mf = reference(ch2.mol, xc)
    c = mf.mo_coeff
    hm = c.T @ mf.get_hcore() @ c
    gm = _transform(mf.mol.intor('int2e'), [c] * 4)
    xc_grid = Semilocal(mf)
    xc_terms = xc_grid.terms()
    for delta_s in (-1, 0, 1):
        td = dense_td(mf, delta_s, nobeta=True)
        energy, residual, action = _mo_operators(td, hm, gm, canonical_pairs(td), xc_terms)
        np.testing.assert_allclose(energy + mf.mol.energy_nuc(), mf.e_tot, atol=1e-10, rtol=0)
        np.testing.assert_allclose(residual, 0, atol=1e-8, rtol=0)
        original, diagonal = {-1: td.gen_vind_sfd, 0: td.gen_vind_sc, 1: td.gen_vind_sfu}[delta_s]()
        unit = np.eye(len(diagonal))
        np.testing.assert_allclose(action(unit), original(unit), atol=1e-10, rtol=0)
    hessian = td.Hessian().kernel(atmlst=[1])
    step = 2e-4
    finite = (displaced_gradient(td, 1, 2, step) - displaced_gradient(td, 1, 2, -step)) / (2 * step)
    np.testing.assert_allclose(hessian[0, 0, :, 2], finite[1], atol=2e-5, rtol=0)


@pytest.mark.parametrize('xc,delta_s,nobeta', [('TPSS', -1, False), ('TPSS', 0, False),
                                             ('M06-2X', -1, True), ('M06-2X', 0, True)])
def test_mgga_hessian(ch2, xc, delta_s, nobeta):
    from pyscf.dft import libxc
    if libxc.max_deriv_order(xc) < 4:
        pytest.skip('requires LibXC compiled with fourth derivatives (DISABLE_LXC=OFF)')
    mf = reference(ch2.mol, xc)
    td = dense_td(mf, delta_s, nobeta=nobeta)
    hessian = td.Hessian().kernel(atmlst=[0, 1])
    step = 1e-4
    finite = (displaced_gradient(td, 1, 2, step) - displaced_gradient(td, 1, 2, -step)) / (2 * step)
    np.testing.assert_allclose(hessian[:, 1, :, 2], finite[:2], atol=2e-5, rtol=0)


def test_missing_lda_fourth_derivative(ch2):
    from pyscf.dft import libxc
    if libxc.max_deriv_order('SVWN') >= 4:
        pytest.skip('this LibXC build supplies fourth derivatives')
    td = NTTDA(dft.ROKS(ch2.mol).set(xc='SVWN'))
    with pytest.raises(NotImplementedError, match='derivative order 4'):
        td.Hessian().kernel()


@pytest.mark.parametrize('xc,delta_s', [('HF', -1), ('HF', 0), ('HF', 1),
                                       ('PBE', -1), ('PBE', 0), ('PBE', 1),
                                       ('M06-2X', -1), ('M06-2X', 0), ('M06-2X', 1)])
def test_energy_gradient_hessian_chain(ch2, xc, delta_s):
    from pyscf.dft import libxc
    if xc != 'HF' and libxc.max_deriv_order(xc) < 4:
        pytest.skip('requires LibXC fourth derivatives')
    mf = ch2 if xc == 'HF' else reference(ch2.mol, xc)
    td = dense_td(mf, delta_s, nobeta=True)
    direction = np.zeros((3, 3))
    direction[1] = [0.3, -0.4, 0.5]
    direction /= np.linalg.norm(direction)
    gradient = td.Gradients().kernel()
    hessian = td.Hessian().kernel(atmlst=[1])
    exact_g = np.einsum('ax,ax', gradient, direction)
    exact_h = direction[1] @ hessian[0, 0] @ direction[1]
    energy0 = mf.e_tot + td.e[0]
    errors = []
    # Independent five-point TOTAL ENERGY stencil; no analytic gradient enters it.
    for step in (1e-3, 5e-4):
        values = []
        for factor in (-2, -1, 1, 2):
            moved = displaced_state(td, factor * step * direction)
            values.append(moved._scf.e_tot + moved.e[0])
        em2, em1, ep1, ep2 = values
        finite_g = (em2 - 8*em1 + 8*ep1 - ep2) / (12*step)
        finite_h = (-em2 + 16*em1 - 30*energy0 + 16*ep1 - ep2) / (12*step**2)
        errors.append((abs(finite_g - exact_g), abs(finite_h - exact_h)))
        np.testing.assert_allclose(finite_g, exact_g, atol=2e-7, rtol=0)
        np.testing.assert_allclose(finite_h, exact_h, atol=1e-5, rtol=0)
    print(xc, delta_s, 'energy FD errors (gradient, Hessian):', errors)


@pytest.mark.parametrize('xc,delta_s', [('HF', -1), ('HF', 0), ('HF', 1), ('PBE', -1)])
def test_iterative_matches_dense(ch2, xc, delta_s, monkeypatch):
    from pyscf.dft import libxc
    if xc != 'HF' and libxc.max_deriv_order(xc) < 4:
        pytest.skip('requires LibXC fourth derivatives')
    td = dense_td(ch2 if xc == 'HF' else reference(ch2.mol, xc), delta_s)
    iterative = td.Hessian()
    name = {-1: 'gen_vind_sfd', 0: 'gen_vind_sc', 1: 'gen_vind_sfu'}[delta_s]
    original = getattr(NTTDA, name)

    def checked_action(obj):
        action, diagonal = original(obj)

        def apply(vectors):
            assert np.asarray(vectors).shape[0] == 1, 'assembled full NTTDA matrix'
            return action(vectors)

        return apply, diagonal

    with monkeypatch.context() as patch:
        patch.setattr(NTTDA, name, checked_action)
        actual = iterative.kernel(atmlst=[1])
    expected = td.Hessian().set(solver='dense').kernel(atmlst=[1])
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
    assert iterative.response_residual <= iterative.conv_tol
    assert iterative.response_iterations['orbital'] > 0
    assert iterative.response_iterations['state'] > 0


def test_response_solver_residual_and_failure():
    from nest.hessian.response import solve
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(60, 60)) + 10*np.eye(60)
    rhs = rng.normal(size=60)
    solution, residual, cycles = solve(lambda v: matrix @ v, rhs, matrix.diagonal(), 1e-10, 100)
    np.testing.assert_allclose(matrix @ solution, rhs, atol=1e-10, rtol=0)
    assert residual <= 1e-10 and cycles > 0
    with pytest.raises(RuntimeError, match='GMRES failed'):
        solve(lambda v: matrix @ v, rhs, matrix.diagonal(), 1e-14, 1)


def test_response_tolerance(ch2):
    td = dense_td(ch2, -1)
    driver = td.Hessian().set(conv_tol=1e-6)
    actual = driver.kernel(atmlst=[1])
    expected = td.Hessian().set(solver='dense').kernel(atmlst=[1])
    assert 1e-8 < driver.response_residual <= driver.conv_tol
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize('delta_s', [-1, 0, 1])
def test_default_convergence_settings(ch2, delta_s):
    mf = dft.ROKS(ch2.mol).set(xc='HF').run()
    assert mf.converged
    td = NTTDA(mf).set(deltaS=delta_s).run()
    assert td.converged[0]
    driver = td.Hessian()
    actual = driver.kernel(atmlst=[1])
    expected = dense_td(ch2, delta_s).Hessian().kernel(atmlst=[1])
    print('Default-settings Hessian error:', delta_s, np.max(abs(actual - expected)),
          'input residuals:', driver.reference_residual, driver.state_residual)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)


def test_stale_converged_inputs_are_rejected(ch2):
    td = dense_td(ch2, 0)
    changed = copy.copy(ch2)
    changed.mo_coeff = ch2.mo_coeff.copy()
    changed.mo_coeff[:, 0] = np.cos(.01)*ch2.mo_coeff[:, 0] + np.sin(.01)*ch2.mo_coeff[:, -1]
    changed.mo_coeff[:, -1] = -np.sin(.01)*ch2.mo_coeff[:, 0] + np.cos(.01)*ch2.mo_coeff[:, -1]
    stale = copy.copy(td)
    stale._scf = changed
    with pytest.raises(RuntimeError, match='reference convergence tolerance'):
        stale.Hessian().kernel(atmlst=[1])
    stale = copy.copy(td)
    stale.xy = list(td.xy)
    stale.xy[0] = (td.xy[0][0] + .01*td.xy[1][0], 0)
    with pytest.raises(RuntimeError, match='TD convergence tolerance'):
        stale.Hessian().kernel(atmlst=[1])
