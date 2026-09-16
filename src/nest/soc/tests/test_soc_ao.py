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

from types import SimpleNamespace
import unittest

import numpy as np
from pyscf import gto, lib
from pyscf.data.nist import LIGHT_SPEED
from nest.soc import soc_ao


try:
    import socutils  # noqa: F401
except ImportError:
    HAS_SOCUTILS = False
else:
    HAS_SOCUTILS = True


def fp(mat):
    return lib.fp(mat)


class NuclearModels(unittest.TestCase):
    def test_bp1e_gaussian_analytic_and_point_limit(self):
        # Normalized px, py, pz primitives with exponent alpha=1.
        mol = gto.M(atom='He 0 0 0', basis={'He': [[1, [1.0, 1.0]]]},
                    verbose=0)
        point = 4 * 2**1.5 / (3 * np.sqrt(np.pi))
        # The returned q=0 component is the Cartesian z component / sqrt(2).
        scale = -1j * 2 / (2 * LIGHT_SPEED**2 * np.sqrt(2))
        for zeta in (0, 1, 100, 1e8, 1e16):
            with self.subTest(zeta=zeta):
                mol.set_nuc_mod(0, zeta)
                ratio = (zeta / (zeta + 2))**1.5 if zeta else 1
                actual = soc_ao.get_ao_soc_1e(mol)[1, 0, 1]
                np.testing.assert_allclose(actual, scale * point * ratio,
                                           rtol=1e-12, atol=1e-16)

    def test_bp1e_nuclear_models_and_context(self):
        for model in (None, gto.dyall_nuc_mod, gto.filatov_nuc_mod,
                      {'O': gto.filatov_nuc_mod}):
            with self.subTest(model=model):
                mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                            basis='cc-pvdz', nucmod=model, verbose=0)
                expected = soc_ao._cartesian_to_spherical(
                    1j * mol.intor('int1e_pnucxp') / (2 * LIGHT_SPEED**2))
                mol.set_rinv_orig((1, 2, 3))
                mol.set_rinv_zeta(0.75)
                before = mol._env.copy()
                actual = soc_ao.get_ao_soc_1e(mol)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
                np.testing.assert_array_equal(
                    mol._env[gto.PTR_RINV_ORIG:gto.PTR_RINV_ORIG+3],
                    before[gto.PTR_RINV_ORIG:gto.PTR_RINV_ORIG+3])
                self.assertEqual(mol._env[gto.PTR_RINV_ZETA],
                                 before[gto.PTR_RINV_ZETA])

    def test_x2c1e_nuclear_models_and_point_limit(self):
        mol = gto.M(atom='Ne 0 0 0', basis='cc-pvdz', verbose=0)
        point = soc_ao.get_ao_soc_x2c1e(mol)
        for model in (gto.dyall_nuc_mod, gto.filatov_nuc_mod):
            with self.subTest(model=model.__name__):
                finite = gto.M(atom='Ne 0 0 0', basis='cc-pvdz',
                               nucmod=model, verbose=0)
                actual = soc_ao.get_ao_soc_x2c1e(finite)
                self.assertGreater(np.linalg.norm(actual-point) / np.linalg.norm(point),
                                   1e-8)
                finite.set_nuc_mod(0, 1e16)
                np.testing.assert_allclose(soc_ao.get_ao_soc_x2c1e(finite),
                                           point, rtol=1e-7, atol=1e-10)


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

    def test_1e_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.0035217717150048196 - 0.002071753516350664j
        ao_soc = soc_ao.get_ao_soc(mf, '1e')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-11)

    def test_zeff_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.002466043752900166 - 0.001450202630169735j
        ao_soc = soc_ao.get_ao_soc(mf, 'Zeff')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-11)

    def test_somf_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.002331075720748517 - 0.0013735097397886604j
        ao_soc = soc_ao.get_ao_soc(mf, 'SOMF')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    def test_somf_soc_ao_uks(self):
        mf = self.mol.UKS(xc='HF').newton().run()
        self.assertTrue(mf.converged)

        ref = -0.0023423479533036867 - 0.0013759118729275677j
        ao_soc = soc_ao.get_ao_soc(mf, 'SOMF')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    def test_somf_amfi_soc_ao(self):
        mf = self.mol.ROKS(xc='HF').run()
        self.assertTrue(mf.converged)

        ref = -0.002258000497354782 - 0.001309642786448594j
        ao_soc = soc_ao.get_ao_soc(mf, 'SOMF_AMFI')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    def test_x2c1e_soc_ao(self):
        ref = -0.0035149790489170775 - 0.002067624769849547j
        ao_soc = soc_ao.get_ao_soc(SimpleNamespace(mol=self.mol), 'X2C1E')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    @unittest.skipUnless(HAS_SOCUTILS, 'socutils is not installed')
    def test_x2camf_soc_ao(self):
        ref = -0.002270958919990373 - 0.00131052918540656j
        ao_soc = soc_ao.get_ao_soc(SimpleNamespace(mol=self.mol), 'X2CAMF')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)

    @unittest.skipUnless(HAS_SOCUTILS, 'socutils is not installed')
    def test_x2cmp_soc_ao(self):
        ref = -0.0022231653593819713 - 0.0012960492601477023j
        ao_soc = soc_ao.get_ao_soc(SimpleNamespace(mol=self.mol), 'X2CMP')
        self.assertEqual(ao_soc.shape, (3, self.mol.nao_nr(), self.mol.nao_nr()))
        self.assertAlmostEqual(abs(fp(ao_soc) - ref), 0, delta=1e-9)


if __name__ == '__main__':
    print('Full tests for AO spin-orbit integrals')
    unittest.main()
