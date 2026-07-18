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
import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2WAVENUMBER
from nest import nttda


def assert_allclose_up_to_sign(testcase, actual, desired, atol):
    try:
        np.testing.assert_allclose(actual, desired, atol=atol, rtol=0)
    except AssertionError as error:
        np.testing.assert_allclose(actual, -desired, atol=atol, rtol=0, err_msg=str(error))


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = '/dev/null'
        mol.atom = '''
        O                  0.64372820    0.14077399   -0.04477253
        O                 -0.64862595   -0.12779073   -0.05445498
        H                  1.16027512   -0.65947800    0.36730132
        H                 -1.12109306    0.55561188    0.42651873
        '''
        mol.charge = 0
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_deltam1_soc_with_reference(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        tdm1 = nttda.NTTDA(mf).set(deltaS=-1, nstates=2, verbose=0).run()
        driver = tdm1.SOC(soctype='SOMF')
        driver.include_reference = True
        driver.run()

        self.assertTrue(mf.converged)
        self.assertTrue(np.all(tdm1.converged))
        self.assertAlmostEqual(mf.e_tot, -150.18173594947874, delta=1e-5)
        np.testing.assert_allclose(tdm1.e, [
            -0.2117097904730021, 0.02304623644417658,
        ], atol=1e-5, rtol=0)

        self.assertEqual(len(driver.states), 3)
        self.assertIsNone(driver.states[0].source)
        self.assertIsNone(driver.states[0].root)
        self.assertEqual(driver.h_soc.shape, (5, 5))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose(
            (driver.e - driver.e.min()).real * HARTREE2WAVENUMBER,
            [
                0.0, 46464.93224169827, 46464.934340590495,
                46464.93926312047, 51523.00193211141,
            ],
            atol=1e-4, rtol=0,
        )
        assert_allclose_up_to_sign(
            self,
            driver.get_block(0, 1) * HARTREE2WAVENUMBER,
            np.array([
                [-0.6200496897328874 + 6.674778072441238j],
                [0.0 + 14.033973737433145j],
                [-0.6200496897328874 - 6.674778072441238j],
            ]),
            1e-5,
        )

    def test_delta0_deltam1_soc_without_reference(self):
        mf = self.mol.ROKS(xc='SVWN').run()
        td0 = nttda.NTTDA(mf).set(deltaS=0, nstates=1, verbose=0).run()
        tdm1 = nttda.NTTDA(mf).set(deltaS=-1, nstates=2, verbose=0).run()
        driver = td0.SOC(tdm1, soctype='SOMF', include_reference=False).run()

        self.assertTrue(mf.converged)
        self.assertTrue(np.all(td0.converged))
        self.assertTrue(np.all(tdm1.converged))
        self.assertAlmostEqual(mf.e_tot, -150.18173594947874, delta=1e-5)
        np.testing.assert_allclose(td0.e, [
            -0.00141029197365759,
        ], atol=1e-5, rtol=0)
        np.testing.assert_allclose(tdm1.e, [
            -0.2117097904730021, 0.02304623644417658,
        ], atol=1e-5, rtol=0)

        self.assertEqual(len(driver.states), 3)
        self.assertTrue(all(state.source is not None for state in driver.states))
        self.assertEqual(driver.h_soc.shape, (5, 5))
        np.testing.assert_allclose(driver.h_soc, driver.h_soc.conj().T, atol=1e-12)
        np.testing.assert_allclose(
            (driver.e - driver.e.min()).real * HARTREE2WAVENUMBER,
            [
                0.0, 46155.408911919854, 46155.41103102725,
                46155.415973766896, 51523.00193435802,
            ],
            atol=1e-4, rtol=0,
        )
        assert_allclose_up_to_sign(
            self,
            driver.get_block(0, 1) * HARTREE2WAVENUMBER,
            np.array([
                [-0.6013835340881171 + 6.509518824543219j],
                [0.0 + 14.124330522021628j],
                [-0.6013835340881171 - 6.509518824543219j],
            ]),
            1e-5,
        )


if __name__ == '__main__':
    print('Full SOC tests for noncollinear tensor TDA')
    unittest.main()
