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
import cupy as cp
from pyscf import gto

from nest import nttda
from nest.gpu import nttda as gpu_nttda


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
        mol.basis = '631g'
        mol.spin = 2
        cls.mol = mol.build()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def _check_gradient(self, xc, delta_s):
        mf = self.mol.ROKS(xc=xc)
        mf.run(conv_tol=1e-12)
        mf_gpu = mf.to_gpu()
        self.assertTrue(mf.converged)
        self.assertTrue(mf_gpu.converged)
        cpu = nttda.NTTDA(mf).set(deltaS=delta_s, nstates=2).run()
        gpu = gpu_nttda.NTTDA(mf_gpu).set(deltaS=delta_s, nstates=2).run()
        self.assertTrue(np.all(cpu.converged))
        self.assertTrue(bool(cp.all(gpu.converged)))
        actual = gpu.Gradients().kernel()
        reference = cpu.Gradients().kernel()
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=0)

    def test_hf_gradient(self):
        self._check_gradient('HF', -1)

    def test_svwn_gradient(self):
        self._check_gradient('SVWN', 0)

    def test_pbe_gradient(self):
        self._check_gradient('PBE', -1)

    def test_tpss_gradient(self):
        self._check_gradient('TPSS', 0)

    def test_cam_b3lyp_gradient(self):
        self._check_gradient('CAM-B3LYP', -1)

    def test_df_cam_b3lyp_gradient(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').density_fit().to_gpu()
        mf.run(conv_tol=1e-12)
        self.assertTrue(mf.converged)
        references = {
            -1: [
                [-0.112061300781, -0.009418241257, -0.009777856681],
                [0.095248944289, 0.041503274494, 0.014115611593],
                [0.017868394151, -0.018988198531, 0.007250656022],
                [-0.001056453719, -0.013096867227, -0.011583815881],
            ],
            0: [
                [-0.247108985900, -0.058958044993, -0.012932963385],
                [0.255296225885, 0.068799005690, 0.004328381699],
                [-0.054100969122, 0.028627307776, 0.007742425840],
                [0.045914232221, -0.038468526377, 0.000865408831],
            ],
        }
        for delta_s, reference in references.items():
            with self.subTest(deltaS=delta_s):
                td = gpu_nttda.NTTDA(mf).set(deltaS=delta_s, nstates=2).run()
                self.assertTrue(bool(cp.all(td.converged)))
                np.testing.assert_allclose(td.Gradients().kernel(), reference, atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
