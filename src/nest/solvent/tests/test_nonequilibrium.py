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

import numpy as np
from scipy.optimize import root
from pyscf import gto, dft
from pyscf.solvent import pcm

from nest.solvent import StateSpecificPCM, state_density
from nest.solvent.pcm import _transition_density


class NonEquilibriumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                        basis='sto-3g', spin=2, verbose=0)
        cls.mf = dft.ROKS(cls.mol, xc='HF').run(conv_tol=1e-11)

    def driver(self, eps=10, optical=2, nstates=3):
        solvent = pcm.PCM(self.mol)
        solvent.eps = eps
        solvent.lebedev_order = 7
        td = self.mf.NTTDA().set(deltaS=-1, nstates=nstates, conv_tol=1e-10)
        return StateSpecificPCM(td, solvent, eps_optical=optical)

    def test_ss_limits_source_preservation_and_failure(self):
        sol = self.driver(optical=1)
        sol.kernel()
        frozen = sol.vertical(method='frozen')
        source = sol.dm.copy(), sol.v.copy(), sol.e_tot, sol.state
        self.assertAlmostEqual(sol.state_specific(2), frozen[1], places=9)
        np.testing.assert_array_equal(sol.dm, source[0])
        np.testing.assert_array_equal(sol.v, source[1])
        self.assertEqual((sol.e_tot, sol.state), source[2:])
        self.assertAlmostEqual(sol.state_specific(sol.state), 0, places=9)
        sol.eps_optical = 10
        ss = sol.state_specific(2) + sol.e_tot
        relaxed = self.driver()
        relaxed.kernel(state=2)
        self.assertAlmostEqual(ss, relaxed.e_tot, places=9)
        # Re-equilibrating the source invalidates target results.
        sol.kernel()
        self.assertIsNone(sol.e_ss)
        self.assertFalse(sol.ss_converged)
        sol.max_cycle = 1
        with self.assertRaisesRegex(RuntimeError, 'Non-equilibrium SS did not converge'):
            sol.state_specific(2)
        self.assertFalse(sol.ss_converged)
        with self.assertRaises(ValueError):
            sol.state_specific(0)

    def test_ss_against_independent_full_space_stationary_equations(self):
        sol = self.driver(nstates=18)
        sol.conv_tol_dm = 1e-9
        sol.kernel()
        sol.vertical()
        frozen = sol.e_frozen.copy()
        work = sol.td.copy()
        n = len(work.xy)
        # All physical NT roots span the finite model space. Matrix elements
        # are evaluated from densities, independently of _potential_action.
        densities = np.array([[_transition_density(work, i, j).real for j in range(n)] for i in range(n)])
        fast = sol.with_solvent.copy().reset()
        fast.eps = sol.eps_optical
        target = 1
        direction = np.eye(n)[:, np.arange(n) != target]
        unit = np.eye(n)[target]

        def evaluate(u):
            c = unit + direction @ u
            c /= np.linalg.norm(c)
            dm = np.einsum('i,j,ijab->ab', c, c, densities)
            delta = dm - sol.dm
            response = fast._B_dot_x(delta)
            h = np.diag(frozen) + np.einsum('ijab,ba->ij', densities, response)
            energy = c @ (frozen*c) + 0.5*np.einsum('ij,ji', delta, response)
            residual = direction.T @ (h @ c - (c @ h @ c)*c)
            return energy, residual, c, dm

        result = root(lambda u: evaluate(u)[1], np.zeros(n-1), tol=1e-9)
        self.assertLess(np.linalg.norm(result.fun), 1e-9)
        expected, _, coeff, dm = evaluate(result.x)
        actual = sol.state_specific(target+1) + sol.e_tot
        self.assertAlmostEqual(actual, expected, places=9)
        np.testing.assert_allclose(sol.ss_dm, dm, atol=2e-8)
        amps = np.array([x.ravel() for x, _ in work.xy])
        self.assertGreater(abs(np.vdot(coeff @ amps, sol.ss_td.xy[sol.ss_state-1][0])), 1-1e-8)

    def test_marcus_pekar_cpcm_energy_equivalence(self):
        sol = self.driver()
        sol.kernel()
        sol.vertical()
        field = sol.with_solvent
        field.kernel(sol.dm)
        surface = field._intermediates['S']
        v0 = field._intermediates['v_grids'].copy()
        q0 = field._intermediates['q_sym'].copy()
        qslow = (field.eps-sol.eps_optical)/(field.eps-1)*q0
        finf = (sol.eps_optical-1)/sol.eps_optical
        qfast0 = q0-qslow
        for state in (1, 2, 3):
            dm = state_density(sol.td, state)
            vk = field.v_grids_n - field._get_v(dm[None])[0]
            qfast = -finf*np.linalg.solve(surface, vk+surface @ qslow)
            cross_work = 0.5*(qfast-qfast0) @ surface @ qslow
            marcus = vk @ (qslow+qfast) - 0.5*v0 @ qslow - 0.5*vk @ qfast + cross_work
            dv = vk-v0
            pekar = 0.5*v0 @ q0 + dv @ q0 - 0.5*finf*dv @ np.linalg.solve(surface, dv)
            self.assertAlmostEqual(marcus, pekar, places=12)

    def test_diis_preserves_stationary_energies(self):
        plain = self.driver()
        plain.kernel()
        plain_gap = plain.state_specific(2)
        accelerated = self.driver()
        accelerated.diis_space = 6
        accelerated.kernel()
        accelerated_gap = accelerated.state_specific(2)
        self.assertAlmostEqual(plain.e_tot, accelerated.e_tot, places=10)
        self.assertAlmostEqual(plain_gap, accelerated_gap, places=9)
        np.testing.assert_allclose(plain.dm, accelerated.dm, atol=1e-6)
        np.testing.assert_allclose(plain.ss_dm, accelerated.ss_dm, atol=1e-6)

    def test_coupled_lr_against_real_and_imaginary_energy_hessians(self):
        sol = self.driver(nstates=18)
        sol.conv_tol_dm = 1e-9
        sol.kernel()
        sol.vertical(method='ptlr')
        work = sol.td
        gaps = work.e[1:]-work.e[0]
        n = len(gaps)
        densities = np.array([[_transition_density(work, i, j).real for j in range(n+1)] for i in range(n+1)])
        fast = sol.with_solvent.copy().reset()
        fast.eps = sol.eps_optical
        transitions = densities[0, 1:]
        response = fast._B_dot_x(transitions)
        coupling = np.einsum('iab,jba->ij', transitions, response)
        np.testing.assert_allclose(coupling, coupling.T, atol=1e-12)
        np.testing.assert_allclose(coupling.diagonal(), sol.ptlr_correction[1:], atol=1e-12)

        def energy(q, p):
            c = np.r_[1, q+1j*p]
            c /= np.linalg.norm(c)
            dm = np.einsum('i,j,ijab->ab', c.conj(), c, densities).real
            delta = dm-sol.dm
            return np.dot(abs(c[1:])**2, gaps) + 0.5*np.einsum('ij,ji', delta, fast._B_dot_x(delta))

        step = 1e-4
        zero = np.zeros(n)
        origin = energy(zero, zero)
        hessians = []
        # These numerical derivatives do not use the transition-density LR
        # formula. Imaginary variations separately determine A-B.
        for imaginary in (False, True):
            def value(z):
                return energy(zero, z) if imaginary else energy(z, zero)
            h = np.zeros((n, n))
            for i in range(n):
                ei = np.eye(n)[i]*step
                h[i, i] = (value(ei)+value(-ei)-2*origin)/step**2
                for j in range(i):
                    ej = np.eye(n)[j]*step
                    h[i, j] = h[j, i] = (value(ei+ej)-value(ei-ej)-value(-ei+ej)+value(-ei-ej))/(4*step**2)
            hessians.append(h)
        a = (hessians[0]+hessians[1])/4
        b = (hessians[0]-hessians[1])/4
        np.testing.assert_allclose(a, np.diag(gaps)+coupling, atol=5e-8)
        np.testing.assert_allclose(b, coupling, atol=5e-8)
        numerical = np.linalg.eigvals(np.block([[a, b], [-b, -a]]))
        analytical = np.linalg.eigvals(np.block([[np.diag(gaps)+coupling, coupling],
                                               [-coupling, -np.diag(gaps)-coupling]]))
        np.testing.assert_allclose(np.sort(numerical.real), np.sort(analytical.real), atol=8e-8)
        self.assertLess(np.max(abs(analytical.imag)), 1e-9)
        errors = []
        for strength in (0.02, 0.01):
            k = strength*coupling
            values = np.linalg.eigvals(np.block([[np.diag(gaps)+k, k], [-k, -np.diag(gaps)-k]]))
            lowest = np.sort(values.real[values.real > 0])[0]
            errors.append(abs(lowest-(gaps[0]+k[0, 0])))
        self.assertGreater(errors[0], 1e-12)
        self.assertLess(errors[1], 0.3*errors[0])

    def test_ss_sf_branches_and_cross_spin_target(self):
        uks = dft.UKS(self.mol, xc='SVWN')
        uks.grids.level = 1
        uks.run(conv_tol=1e-11)
        field = self.driver().with_solvent
        for collinear in ('col', 'mcol'):
            with self.subTest(collinear=collinear):
                td = uks.TDA_SF().set(collinear=collinear, collinear_samples=20, nstates=3)
                sol = StateSpecificPCM(td, field, eps_optical=2)
                sol.kernel()
                sol.vertical()
                old_dm = sol.dm.copy()
                ss = sol.state_specific(2)
                self.assertTrue(np.isfinite(ss))
                self.assertTrue(sol.ss_converged)
                np.testing.assert_array_equal(sol.dm, old_dm)
                np.testing.assert_allclose(sol.ss_dm, state_density(sol.ss_td, sol.ss_state), atol=1e-12)
                delta = sol.ss_dm-sol.dm
                fast = field.copy().reset()
                fast.eps = 2
                x = sol.ss_td.xy[sol.ss_state-1][0].ravel()
                vacuum_vind, _ = sol._vacuum_response
                expected = (uks.e_tot + x @ vacuum_vind(x[None])[0] + field.kernel(sol.dm)[0]
                            + np.einsum('ij,ji', sol.v, delta)
                            + 0.5*np.einsum('ij,ji', delta, fast._B_dot_x(delta)))
                self.assertAlmostEqual(sol.e_ss, expected, places=9)
        sol = self.driver(optical=1)
        sol.kernel()
        triplets = self.mf.NTTDA().set(deltaS=0, nstates=3)
        frozen = sol.vertical(triplets, method='frozen')[0]
        self.assertAlmostEqual(sol.state_specific(1, td=triplets), frozen, places=9)
