# Copyright 2026 The NEST Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import null_space
from pyscf import dft, gto
from pyscf.solvent import pcm

from nest.solvent import StateSpecificPCM, state_density
from nest.solvent.pcm import _potential_action
from nest.nttda.nttda import _orbital_indices


class PCMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                        basis='sto-3g', spin=2, verbose=0)
        cls.mf = dft.ROKS(cls.mol, xc='HF').run(conv_tol=1e-11)
        cls.uks = dft.UKS(cls.mol, xc='SVWN')
        cls.uks.grids.level = 1
        cls.uks.run(conv_tol=1e-11)

    def solvent(self, eps=10.0):
        obj = pcm.PCM(self.mol)
        obj.eps = eps
        obj.lebedev_order = 7
        return obj

    def test_vacuum_limit_all_nt_sectors_and_sf_collinear_branches(self):
        methods = [self.mf.NTTDA().set(deltaS=ds, nstates=3) for ds in (-1, 0, 1)]
        methods += [self.uks.TDA_SF().set(collinear=col, collinear_samples=20, nstates=3)
                    for col in ('col', 'mcol')]
        for td in methods:
            with self.subTest(method=type(td).__name__, branch=getattr(td, 'deltaS', getattr(td, 'collinear', None))):
                td.conv_tol = 1e-10
                td.kernel()
                original = td.e.copy()
                sol = StateSpecificPCM(td, self.solvent(1), eps_optical=1)
                sol.kernel(state=1)
                np.testing.assert_allclose(sol.e_tot, td._scf.e_tot + original[0], atol=1e-9)
                for method in ('frozen', 'ptss', 'ptlr'):
                    np.testing.assert_allclose(sol.vertical(method=method), original-original[0], atol=1e-9)
                np.testing.assert_array_equal(td.e, original)
                self.assertFalse(hasattr(td._scf, 'with_solvent'))

    def test_nt_equilibrium_and_ptss_reference(self):
        td = self.mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-10)
        pcm_obj = self.solvent()
        sol = StateSpecificPCM(td, pcm_obj, eps_optical=2)
        energy = sol.kernel()
        self.assertTrue(sol.converged)
        self.assertLess(sol.cycles, 12)
        # Fixed-orbital model, water/STO-3G, coarse cavity: numerical regression,
        # not a spectroscopic benchmark. Densities/operators have independent tests.
        self.assertAlmostEqual(energy, -74.957667030629, places=9)
        gaps = sol.vertical()
        np.testing.assert_allclose(gaps, [0.0, 0.435565616, 0.520631761], atol=2e-8)
        self.assertLess(np.max(sol.ptss_correction), 1e-12)
        self.assertEqual(pcm_obj.eps, 10)
        self.assertEqual(sol.with_solvent.eps, 10)
        np.testing.assert_allclose(sol.history[-1][1], 0, atol=sol.conv_tol_dm)

        # With eps_optical = eps, the quadratic expansion recovers the exact
        # equilibrium PCM energy evaluated on each *fixed-field* wavefunction.
        sol.eps_optical = 10
        sol.vertical()
        vacuum_vind, _ = sol._vacuum_response
        direct = []
        for k, (x, _) in enumerate(sol.td.xy):
            x = x.ravel()
            dm = state_density(sol.td, k + 1)
            epcm, _ = self.solvent().kernel(dm)
            direct.append(self.mf.e_tot + np.vdot(x, vacuum_vind(x[None, :])[0]).real + epcm)
        np.testing.assert_allclose(sol.e_ptss, direct, atol=2e-9)

    def test_projected_nt_down_against_dense_spectrum_and_warm_starts(self):
        vacuum_td = self.mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-10)
        sol = StateSpecificPCM(vacuum_td, self.solvent(), eps_optical=2)
        vacuum_vind, diagonal = sol._vacuum_response
        matrix = vacuum_vind(np.eye(len(diagonal))).T
        cs, os, vs = _orbital_indices(vacuum_td)
        rotation = np.zeros((len(cs)+len(os), len(os)+len(vs)))
        rotation[len(cs)+np.arange(len(os)), np.arange(len(os))] = 1 / np.sqrt(len(os))
        physical = null_space(rotation.reshape(1, -1))
        _, potential = self.solvent().kernel(sol._dm_ref)
        solvent_vind, _ = _potential_action(vacuum_td, potential)
        matrix += solvent_vind(np.eye(len(diagonal))).T
        expected = np.linalg.eigvalsh(physical.T @ matrix @ physical)
        # Put a physical root exactly at zero, next to the redundant direction.
        shift = expected[0]

        def shifted_vind(z):
            return vacuum_vind(z) - shift*np.asarray(z)

        response = shifted_vind, diagonal-shift
        for nstates in (1, 3, len(diagonal)):
            with self.subTest(nstates=nstates):
                vacuum_td.nstates = nstates
                previous = None
                for _ in range(3):
                    field_td = sol._solve(vacuum_td, response, potential, x0=previous)
                    previous = np.array([x.ravel() for x, _ in field_td.xy])
                    n = min(nstates, len(expected))
                    np.testing.assert_allclose(field_td.e, expected[:n]-shift, atol=1e-10)
                    np.testing.assert_allclose(previous @ rotation.ravel(), 0, atol=1e-12)
                    np.testing.assert_allclose(previous @ previous.T, np.eye(n), atol=1e-11)
                    action, _ = field_td.gen_vind_sfd()
                    residual = action(previous) - field_td.e[:, None]*previous
                    self.assertLess(np.linalg.norm(residual, axis=1).max(), 1e-10)

    def test_root_order_and_optional_tracking_at_a_crossing(self):
        vacuum_td = self.mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-10)
        sol = StateSpecificPCM(vacuum_td, self.solvent(1), eps_optical=1)
        sol.kernel()
        initial = sol.td.copy()
        crossed = initial.copy()
        # Prescribe a crossing between two normalized physical roots. This
        # isolates root selection from the convergence of a particular solvent.
        crossed.xy = [initial.xy[1], initial.xy[0], initial.xy[2]]
        for method in ('kernel', 'state_specific'):
            for state in (1, 2):
                for follow in (False, True):
                    with self.subTest(method=method, state=state, follow_state=follow):
                        sol.kernel()
                        kwargs = {'follow_state': True} if follow else {}
                        with patch.object(sol, '_solve', side_effect=[initial, crossed, crossed, crossed]):
                            getattr(sol, method)(state=state, **kwargs)
                        expected_state = 3-state if follow else state
                        if method == 'kernel':
                            actual_state, dm = sol.state, sol.dm
                        else:
                            actual_state, dm = sol.ss_state, sol.ss_dm
                        self.assertEqual(actual_state, expected_state)
                        np.testing.assert_allclose(dm, state_density(crossed, expected_state), atol=1e-12)

    def test_fixed_field_warm_start_can_find_a_new_lower_root(self):
        vacuum_td = self.uks.TDA_SF().set(nstates=1, conv_tol=1e-10)
        sol = StateSpecificPCM(vacuum_td, self.solvent(1), eps_optical=1)
        _, diagonal = sol._vacuum_response
        diagonal = np.arange(len(diagonal), dtype=float)
        # An exact old eigenvector has zero residual after a diagonal level
        # crossing. Extra guesses let the solver discover the new lowest root.
        diagonal[:2] = [1, -1]
        initial = np.eye(1, len(diagonal))
        field_td = sol._solve(vacuum_td, (lambda z: z*diagonal, diagonal),
                              np.zeros_like(sol._dm_ref), x0=initial)
        self.assertAlmostEqual(field_td.e[0], -1, places=12)
        self.assertAlmostEqual(abs(field_td.xy[0][0].ravel()[1]), 1, places=12)

    def test_common_environment_across_spin_sectors(self):
        sol = StateSpecificPCM(self.mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-10),
                               self.solvent(), eps_optical=2)
        sol.kernel()
        dm, v, energy = sol.dm.copy(), sol.v.copy(), sol.e_tot
        target = self.mf.NTTDA().set(deltaS=0, nstates=3, conv_tol=1e-10)
        frozen = sol.vertical(target, method='frozen')
        ptss = sol.vertical(target)
        self.assertGreater(np.max(abs(ptss-frozen)), 1e-5)
        # Scalar charge has zero transition matrix elements between pure
        # different-spin NT sectors, while their state densities can differ.
        ptlr = sol.vertical(target, method='ptlr')
        np.testing.assert_allclose(ptlr, frozen, atol=1e-10)
        np.testing.assert_array_equal(sol.ptlr_correction, np.zeros(len(ptlr)))
        np.testing.assert_array_equal(sol.dm, dm)
        np.testing.assert_array_equal(sol.v, v)
        self.assertEqual(sol.e_tot, energy)
        with self.assertRaisesRegex(ValueError, 'same vacuum'):
            sol.vertical(self.mf.copy().NTTDA())

        # Establish a separate triplet environment for the reverse process.
        triplet = StateSpecificPCM(target, self.solvent(), eps_optical=2)
        triplet.kernel()
        self.assertAlmostEqual(triplet.vertical()[triplet.state - 1], 0, places=9)
        singlets = self.mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-10)
        self.assertLess(triplet.vertical(singlets)[0], 0)
        self.assertGreater(np.linalg.norm(triplet.dm - sol.dm), 1e-3)

    def test_sf_solvated_energy_col_and_mcol(self):
        for collinear in ('col', 'mcol'):
            with self.subTest(collinear=collinear):
                td = self.uks.TDA_SF().set(collinear=collinear, collinear_samples=20,
                                          nstates=3, conv_tol=1e-10)
                sol = StateSpecificPCM(td, self.solvent(), eps_optical=10)
                sol.kernel()
                sol.vertical()
                vacuum_vind, _ = sol._vacuum_response
                for k, (x, _) in enumerate(sol.td.xy):
                    dm = state_density(sol.td, k + 1)
                    epcm, _ = self.solvent().kernel(dm)
                    x = x.ravel()
                    direct = self.uks.e_tot + np.vdot(x, vacuum_vind(x[None, :])[0]).real + epcm
                    self.assertAlmostEqual(sol.e_ptss[k], direct, places=9)
                self.assertGreater(np.linalg.norm(sol.ptss_correction[1:]), 1e-6)
                for method in (sol.td.Gradients, sol.td.NAC, sol.td.as_scanner):
                    with self.assertRaises(NotImplementedError):
                        method()

    def test_ptss_approaches_self_consistent_ss_for_weak_response(self):
        td = self.mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-10)
        errors = []
        for eps in (1.2, 1.1, 1.05):
            ground = StateSpecificPCM(td, self.solvent(eps), eps_optical=eps)
            ground.kernel()
            ptss = ground.vertical()[1] + ground.e_tot
            excited = StateSpecificPCM(td, self.solvent(eps), eps_optical=eps)
            excited.kernel(state=2)
            errors.append(abs(ptss - excited.e_tot))
        # Halving a weak PCM response reduces the omitted amplitude-relaxation
        # energy approximately quadratically. Both calculations use equilibrium
        # solvent here so their energy definitions agree.
        self.assertGreater(errors[0], 1e-7)
        self.assertLess(errors[1], 0.4 * errors[0])
        self.assertLess(errors[2], 0.4 * errors[1])

    def test_physical_root_lr_against_energy_curvature(self):
        td = self.mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-11)
        sol = StateSpecificPCM(td, self.solvent(), eps_optical=2)
        sol.conv_tol_dm = 1e-8
        sol.kernel()
        sol.vertical(method='ptlr')
        self.assertGreater(np.linalg.norm(sol.ptlr_correction[1:]), 1e-6)
        source = sol.td.xy[0][0].ravel()
        target = sol.td.xy[1][0].ravel()
        frozen_gap = sol.e_frozen[1] - sol.e_frozen[0]
        vacuum_vind, vacuum_diag = sol._vacuum_response
        vacuum_matrix = vacuum_vind(np.eye(len(vacuum_diag))).T
        fast = self.solvent(2)
        work = sol.td.copy()
        shape = sol.td.xy[0][0].shape

        def energy(angle):
            x = np.cos(angle) * source + np.sin(angle) * target
            work.xy = [(x.reshape(shape), 0)]
            delta = state_density(work) - sol.dm
            return (x @ vacuum_matrix @ x + np.einsum('ij,ji', sol.v, delta)
                    + 0.5 * np.einsum('ij,ji', delta, fast._B_dot_x(delta)))

        step = 2e-4
        curvature = (energy(step) + energy(-step) - 2*energy(0)) / step**2
        # Real-amplitude curvature measures A+B. Its solvent contribution is
        # twice the first-order pole shift: G'' = 2*gap + 4*ptLR.
        self.assertAlmostEqual(curvature/4-frozen_gap/2, sol.ptlr_correction[1], delta=5e-8)

    def test_pcm_functional_and_potential_gauge(self):
        td = self.mf.NTTDA().set(deltaS=0, nstates=3).run()
        dm = state_density(td)
        solvent = self.solvent()
        e0, v0 = solvent.kernel(dm)
        rng = np.random.default_rng(56)
        delta = rng.normal(size=dm.shape)
        delta = 0.01 * (delta + delta.T)
        e1, _ = solvent.kernel(dm + delta)
        expansion = e0 + np.einsum('ij,ji', v0, delta)
        expansion += 0.5 * np.einsum('ij,ji', delta, solvent._B_dot_x(delta))
        self.assertAlmostEqual(e1, expansion, places=12)
        # A constant electronic potential cancels from normal-ordered gaps.
        for ds in (-1, 0, 1):
            td.deltaS = ds
            action, diagonal = _potential_action(td, self.mf.get_ovlp())
            np.testing.assert_allclose(diagonal, 0, atol=1e-13)
            z = rng.normal(size=(2, len(diagonal)))
            np.testing.assert_allclose(action(z), 0, atol=1e-12)

    def test_failed_iteration_and_unsupported_inputs(self):
        td = self.mf.NTTDA().set(deltaS=-1, nstates=3)
        sol = StateSpecificPCM(td, self.solvent(), eps_optical=2)
        with self.assertRaisesRegex(RuntimeError, 'equilibrate'):
            sol.vertical()
        sol.max_cycle = 1
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            sol.kernel()
        self.assertFalse(sol.converged)
        with self.assertRaises(ValueError):
            sol.kernel(state=0)
        with self.assertRaises(TypeError):
            StateSpecificPCM(self.uks.TDDFT_SF(), self.solvent(), eps_optical=2)
        with self.assertRaises(ValueError):
            StateSpecificPCM(self.mf.PCM().NTTDA(), self.solvent(), eps_optical=2)
        with self.assertRaisesRegex(TypeError, 'ordinary'):
            StateSpecificPCM(self.mf.copy().average_occ().NTTDA(), self.solvent(), eps_optical=2)

        sol.max_cycle = 30
        sol.kernel()
        sol.vertical()
        self.assertIsNotNone(sol.e_ptss)
        sol.kernel(state=2)
        self.assertIsNone(sol.e_ptss)
        self.assertIsNone(sol.e_frozen)
        self.assertIsNone(sol.ptss_correction)
        with self.assertRaises(ValueError):
            StateSpecificPCM(sol.td, self.solvent(), eps_optical=2)
        with self.assertRaises(ValueError):
            sol.vertical(sol.td)
        for method in (sol.td.nuc_grad_method, sol.td.Gradients, sol.td.nac_method,
                       sol.td.NAC, sol.td.get_ab, sol.td.as_scanner):
            with self.assertRaises(NotImplementedError):
                method()
