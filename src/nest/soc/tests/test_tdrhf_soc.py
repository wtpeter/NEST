#!/usr/bin/env python
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
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pyscf import gto
from pyscf.lib import logger

from nest.soc.tdrhf import SOC
from nest.soc.soc import SpinFreeState


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='H 0 0 0; H 0 0 0.74',
            basis='sto-3g',
            verbose=0,
            output='/dev/null',
        )
        cls.mf = cls.mol.RKS(xc='lda').run()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def make_tds(self):
        singlet = self.mf.TDA().set(singlet=True)
        triplet = self.mf.TDDFT().set(singlet=False)
        singlet.e = np.array([0.4, 0.5])
        singlet.xy = [
            (np.array([[1.0]]), 0),
            (np.array([[0.8]]), 0),
        ]
        triplet.e = np.array([0.3])
        triplet.xy = [(np.array([[0.7]]), np.array([[0.1]]))]
        return singlet, triplet

    def test_collect_states_with_reference_by_default(self):
        singlet, triplet = self.make_tds()
        driver = SOC(singlet, triplet)
        states = driver._initialize_states()

        self.assertIs(driver._scf, self.mf)
        self.assertEqual([state.spin for state in states], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual([state.energy for state in states], [0.0, 0.4, 0.5, 0.3])
        self.assertEqual(states[0].label, 'reference')
        self.assertIsNone(states[0].source)
        np.testing.assert_allclose(states[1].amplitude, [[1.0]])
        np.testing.assert_allclose(states[2].amplitude, [[1.0]])
        np.testing.assert_allclose(states[3].amplitude, [[1.0]])

    def test_tda_and_tddft_pseudo_amplitudes(self):
        tda, tddft = self.make_tds()
        tda.xy[0] = (np.array([[3.0, 4.0]]), 0)
        tddft.xy[0] = (np.array([[1.0, 2.0]]), np.array([[2.0, 2.0]]))
        np.testing.assert_allclose(SOC._state_amplitude(tda, 0), [[0.6, 0.8]])
        np.testing.assert_allclose(SOC._state_amplitude(tddft, 0), [[0.6, 0.8]])

        tda.xy[0] = (np.array([[1.0]]), 1)
        with self.assertRaisesRegex(ValueError, r'\(X, Y\).*\(X, 0\)'):
            SOC._state_amplitude(tda, 0)

    def test_collect_states_without_reference(self):
        singlet, triplet = self.make_tds()
        states = SOC(singlet, triplet, include_reference=False)._initialize_states()
        self.assertEqual([state.spin for state in states], [0.0, 0.0, 1.0])
        self.assertTrue(all(state.source is not None for state in states))

    def test_accept_more_than_two_objects_with_same_multiplicity(self):
        singlet0, _ = self.make_tds()
        singlet1 = self.mf.TDA().set(singlet=True)
        singlet1.e = np.array([0.6])
        singlet1.xy = [(np.array([[0.6]]), 0)]
        _, triplet = self.make_tds()
        states = SOC(singlet0, singlet1, triplet, include_reference=False)._initialize_states()
        self.assertEqual([state.spin for state in states], [0.0, 0.0, 0.0, 1.0])

    def test_accept_one_object_and_warn(self):
        _, triplet = self.make_tds()
        output = StringIO()
        driver = SOC(triplet)
        driver.verbose = logger.WARN
        driver.stdout = output
        states = driver._initialize_states()
        self.assertEqual([state.spin for state in states], [0.0, 1.0])
        self.assertIn('Only one RHF/RKS TD object', output.getvalue())

    def test_accept_list(self):
        singlet, triplet = self.make_tds()
        states = SOC([singlet, triplet], include_reference=False)._initialize_states()
        self.assertEqual([state.spin for state in states], [0.0, 0.0, 1.0])

    def test_reject_no_objects(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            SOC()._initialize_states()

    def test_reject_uhf_td_even_though_singlet_attribute_is_inherited(self):
        uhf = self.mol.UHF().run()
        uhf_td = uhf.TDA()
        self.assertIsNone(uhf_td.singlet)
        _, triplet = self.make_tds()
        with self.assertRaisesRegex(ValueError, 'singlet to True or False'):
            SOC(uhf_td, triplet)._initialize_states()

    def test_reject_uhf_td_with_manually_set_singlet(self):
        uhf_td = self.mol.UHF().run().TDA()
        uhf_td.singlet = False
        _, triplet = self.make_tds()
        with self.assertRaisesRegex(TypeError, 'only RHF/RKS'):
            SOC(uhf_td, triplet)._initialize_states()

    def test_reduced_transition_densities(self):
        driver = SOC()
        driver._scf = SimpleNamespace(
            mo_occ=np.array([2, 2, 0, 0]),
            mo_coeff=np.eye(4),
        )
        reference = SpinFreeState(None, None, 0.0, 0.0, None, 'reference')
        singlet = SpinFreeState(object(), 0, 0.4, 0.0, np.array([[5.0, 6.0], [7.0, 8.0]]), 'singlet')
        triplet_k = SpinFreeState(None, 0, 0.3, 1.0, np.array([[1.0, 2.0], [3.0, 4.0]]), 'triplet k')
        triplet_n = SpinFreeState(None, 1, 0.5, 1.0, np.array([[2.0, 1.0], [4.0, 3.0]]), 'triplet n')

        gamma_t0 = np.zeros((4, 4))
        gamma_t0[2:, :2] = triplet_k.amplitude.T
        np.testing.assert_allclose(
            driver.reduced_transition_density(triplet_k, reference),
            gamma_t0.T,
        )

        gamma_ts = np.zeros((4, 4))
        gamma_ts[2:, 2:] = triplet_k.amplitude.T @ singlet.amplitude / np.sqrt(2.0)
        gamma_ts[:2, :2] = -singlet.amplitude @ triplet_k.amplitude.T / np.sqrt(2.0)
        np.testing.assert_allclose(
            driver.reduced_transition_density(triplet_k, singlet),
            gamma_ts.T,
        )

        gamma_tt = np.zeros((4, 4))
        gamma_tt[2:, 2:] = triplet_n.amplitude.T @ triplet_k.amplitude
        gamma_tt[:2, :2] = triplet_k.amplitude @ triplet_n.amplitude.T
        np.testing.assert_allclose(
            driver.reduced_transition_density(triplet_n, triplet_k),
            gamma_tt.T,
        )
        np.testing.assert_allclose(
            driver.reduced_transition_density(singlet, reference),
            np.zeros((4, 4)),
        )

    def test_build_hamiltonian_is_hermitian(self):
        singlet, triplet = self.make_tds()
        driver = SOC(singlet, triplet)
        nao = self.mol.nao_nr()
        with patch('nest.soc.soc.get_ao_soc', return_value=np.zeros((3, nao, nao), dtype=complex)):
            h_soc = driver.build_hamiltonian()
        self.assertEqual(h_soc.shape, (6, 6))
        np.testing.assert_allclose(h_soc, h_soc.conj().T, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
