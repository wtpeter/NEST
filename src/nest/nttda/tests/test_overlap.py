"""Determinant oracles for scalar/rank-one NT-TDA densities and WFoverlap."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.linalg import expm
from nest.nttda.overlap import (build_awf, determinant_rdm, determinant_overlap,
                                state_density, wfoverlap, AuxiliaryWavefunction,
                                _combine, _one_body)
from nest.soc.nttda import SOC
from nest.soc.soc import clebsch_gordan_rank1


def synthetic(delta_s, seed=83):
    # Two closed, two open, two virtual orbitals; traceless OO for singlets.
    occ = np.array([2, 2, 1, 1, 0, 0])
    rng = np.random.default_rng(seed)
    size = 16 if delta_s == -1 else 17
    x = rng.normal(size=(size, 4))
    if delta_s == -1:
        # OO entries (2,0) and (3,1) of the 4x4 excitation matrix.
        trace = x[8]+x[13]
        x[8] -= trace/2
        x[13] -= trace/2
    x = np.linalg.qr(x)[0].T
    mol = SimpleNamespace(nao_nr=lambda: 6)
    mf = SimpleNamespace(mo_coeff=np.eye(6), mo_occ=occ, mol=mol,
                         converged=True, verbose=0, stdout=None)
    return SimpleNamespace(_scf=mf, mol=mol, xy=[(a, 0) for a in x],
                           e=np.arange(4)*.1, deltaS=delta_s, nobeta=False,
                           converged=np.ones(4, dtype=bool))


class OverlapTests(unittest.TestCase):
    def test_spin_purity_and_ms_invariance(self):
        rng = np.random.default_rng(43)
        rotation = rng.normal(size=(6, 6))*.03
        mo = expm(rotation-rotation.T)
        for sector in (-1, 0):
            td = synthetic(sector)
            for i in range(4):
                wf = build_awf(td, i)
                raised = _combine([(1, _one_body(wf.coefficients, p, 6+p)) for p in range(6)])
                self.assertFalse(raised, 'Highest-Ms AWF contains a higher-spin component')
        a, b = build_awf(synthetic(0), 0), build_awf(synthetic(0, 94), 1)
        expected = determinant_overlap(a, b, mo)
        for _ in range(2):
            def lower(wf):
                state = _combine([(1/np.sqrt(2), _one_body(wf.coefficients, 6+p, p))
                                  for p in range(6)])
                return AuxiliaryWavefunction(6, wf.nalpha-1, wf.nbeta+1, state)
            a, b = lower(a), lower(b)
            self.assertAlmostEqual(a.norm, 1, places=13)
            np.testing.assert_allclose(determinant_overlap(a, b, mo), expected, atol=1e-13)

    def test_mo_phase_gauge(self):
        signs = np.array([-1., 1., -1., 1., 1., -1.])
        ref_phase = np.prod(signs[2:4])
        for sector in (-1, 0):
            td, changed = synthetic(sector), synthetic(sector)
            if sector == -1:
                factors = (signs[:4, None]*signs[None, 2:]).ravel()
            else:
                factors = np.concatenate([(signs[:2, None]*signs[None, 2:4]).ravel(),
                                          (signs[:2, None]*signs[None, 4:]).ravel(), [1],
                                          (signs[2:4, None]*signs[None, 4:]).ravel(),
                                          (signs[:2, None]*signs[None, 4:]).ravel()])
            changed.xy = [(x*factors*ref_phase, y) for x, y in changed.xy]
            for i in range(4):
                for j in range(4):
                    actual = determinant_overlap(build_awf(td, i), build_awf(changed, j), np.diag(signs))
                    np.testing.assert_allclose(actual, float(i == j), atol=1e-13)

    def test_scalar_densities_and_identity(self):
        for delta_s in (-1, 0):
            td = synthetic(delta_s)
            wf = [build_awf(td, i) for i in range(4)]
            for i in range(4):
                for j in range(4):
                    np.testing.assert_allclose(determinant_overlap(wf[i], wf[j], np.eye(6)),
                                               float(i == j), atol=1e-13)
                    np.testing.assert_allclose(determinant_rdm(wf[i], wf[j]),
                                               state_density(td, i, j), atol=1e-13)

    def test_cubic_contractions_against_determinants(self):
        from nest.nttda.overlap_contraction import overlap_from_mo
        rng = np.random.default_rng(584)
        mo = np.eye(6)+rng.normal(size=(6, 6))*.035
        for sector in (-1, 0):
            left, right = synthetic(sector), synthetic(sector, 143)
            # Complex phases test conjugation as well as fermionic signs.
            left.xy = [(x*np.exp(.3j*i), y) for i, (x, y) in enumerate(left.xy)]
            right.xy = [(x*np.exp(-.2j*i), y) for i, (x, y) in enumerate(right.xy)]
            a, b = [build_awf(left, i) for i in range(4)], [build_awf(right, i) for i in range(4)]
            expected = np.array([[determinant_overlap(x, y, mo) for y in b] for x in a])
            np.testing.assert_allclose(overlap_from_mo(left, right, mo), expected,
                                       atol=1e-13, rtol=0)
            np.testing.assert_allclose(overlap_from_mo(left, left, np.eye(6)), np.eye(4),
                                       atol=1e-13, rtol=0)

    def test_cubic_contractions_singular_core(self):
        from nest.nttda.overlap_contraction import overlap_from_mo, SingularCoreOverlapError
        td = synthetic(0)
        metric = np.eye(6)
        metric[0, 0] = 0
        with self.assertRaises(SingularCoreOverlapError):
            overlap_from_mo(td, td, metric)

    def test_cubic_contractions_orbital_partitions(self):
        from nest.nttda.overlap_contraction import overlap_from_mo
        rng = np.random.default_rng(337)
        for occupations in [(1, 1, 0), (2, 1, 1), (0, 1, 2, 0, 1)]:
            nc, nv, nmo = occupations.count(2), occupations.count(0), len(occupations)
            for sector in (-1, 0):
                size = (nc+2)*(nv+2) if sector == -1 else 2*nc+2*nc*nv+1+2*nv
                x = rng.normal(size=(size, min(size, 2)))
                if sector == -1:
                    i, j = nc*(nv+2), (nc+1)*(nv+2)+1
                    trace = x[i]+x[j]
                    x[i] -= trace/2
                    x[j] -= trace/2
                x = np.linalg.qr(x)[0].T
                mf = SimpleNamespace(mo_occ=np.array(occupations))
                td = SimpleNamespace(_scf=mf, xy=[(a, 0) for a in x], deltaS=sector)
                mo = np.eye(nmo)+rng.normal(size=(nmo, nmo))*.06
                wf = [build_awf(td, i) for i in range(len(x))]
                expected = np.array([[determinant_overlap(a, b, mo) for b in wf] for a in wf])
                np.testing.assert_allclose(overlap_from_mo(td, td, mo), expected, atol=1e-13, rtol=0)

    def test_reference_projection_continuity(self):
        from pyscf import gto
        from nest.interfaces.sharc_nttda import project_reference, reference_overlap
        mol = gto.M(atom='C 0 0 0; H 0 .8 .6; H 0 -.9 .5', basis='sto-3g', spin=2, verbose=0)
        metric = mol.intor_symmetric('int1e_ovlp')
        eig, vectors = np.linalg.eigh(metric)
        coeff = (vectors*eig**-.5)@vectors.T
        occ = np.array([2, 2, 2, 1, 1, 0, 0])
        previous = SimpleNamespace(mol=mol, _scf=SimpleNamespace(mo_coeff=coeff, mo_occ=occ))
        coords = mol.atom_coords()
        coords[1, 0] += 1e-4
        moved = mol.set_geom_(coords, unit='Bohr', inplace=False)
        projected = project_reference(previous, moved)
        np.testing.assert_allclose(projected.T @ moved.intor_symmetric('int1e_ovlp') @ projected,
                                   np.eye(7), atol=1e-13)
        current = SimpleNamespace(mol=moved, _scf=SimpleNamespace(mo_coeff=projected, mo_occ=occ))
        self.assertGreater(abs(reference_overlap(previous, current)), .999999)

    def test_soc_phase_and_spin(self):
        t, s = synthetic(0), synthetic(-1)
        driver = SOC([t, s])
        driver.states = driver._initialize_states()
        wt, ws = [build_awf(t, i) for i in range(4)], [build_awf(s, i) for i in range(4)]
        for i in range(4):
            for j in range(4):
                # All singlet rank-one matrix elements vanish.
                np.testing.assert_allclose(determinant_rdm(ws[i], ws[j], 0), 0, atol=1e-13)
                for ket, wket, projection in ((driver.states[j], wt[j], 0),
                                              (driver.states[4+j], ws[j], 1)):
                    cg = clebsch_gordan_rank1(ket.spin, ket.spin, projection, 1, 1)
                    expected = driver.reduced_transition_density(driver.states[i], ket).T*cg
                    actual = determinant_rdm(wt[i], wket, projection)
                    # NEST's SOC density expressions omit open-orbital
                    # diagonal terms. Real-orbital AO/MO SOC operators are
                    # antisymmetric and cannot contract with these terms.
                    # All off-diagonal entries (including phases) must agree.
                    np.fill_diagonal(actual, 0)
                    np.fill_diagonal(expected, 0)
                    np.testing.assert_allclose(actual, expected, atol=1e-13)

    @unittest.skipUnless(os.environ.get('NTTDA_WFOVERLAP_TEST_DIR') and
                         os.environ.get('NTTDA_WFOVERLAP_EXECUTABLE'),
                         'Set executable and generated-output directory for external WFoverlap test')
    def test_wfoverlap_displaced_orbital_metric(self):
        rng = np.random.default_rng(66)
        rotation = rng.normal(size=(6, 6))*.03
        mo = expm(rotation-rotation.T)
        for delta_s in (-1, 0):
            left, right = synthetic(delta_s), synthetic(delta_s, seed=93)
            a, b = [build_awf(left, i) for i in range(4)], [build_awf(right, i) for i in range(4)]
            expected = np.array([[determinant_overlap(x, y, mo) for y in b] for x in a])
            actual = wfoverlap(a, b, mo, Path(os.environ['NTTDA_WFOVERLAP_TEST_DIR'])/str(delta_s),
                               executable=os.environ['NTTDA_WFOVERLAP_EXECUTABLE'])
            np.testing.assert_allclose(actual, expected, atol=6e-11, rtol=0)
            # A root phase flip must flip only its overlap column.
            right.xy[1] = (-right.xy[1][0], 0)
            b[1] = build_awf(right, 1)
            flipped = wfoverlap(a, b, mo, Path(os.environ['NTTDA_WFOVERLAP_TEST_DIR'])/f'{delta_s}-flip',
                                executable=os.environ['NTTDA_WFOVERLAP_EXECUTABLE'])
            expected[:, 1] *= -1
            np.testing.assert_allclose(flipped, expected, atol=6e-11, rtol=0)


if __name__ == '__main__':
    unittest.main()
