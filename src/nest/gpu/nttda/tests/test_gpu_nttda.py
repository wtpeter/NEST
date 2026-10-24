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

    def _check_energy(self, xc, delta_s, density_fit=False, nobeta=False):
        mf = self.mol.ROKS(xc=xc)
        if density_fit:
            mf = mf.density_fit()
        mf.run(conv_tol=1e-12)
        mf_gpu = mf.to_gpu()
        cpu = nttda.NTTDA(mf).set(deltaS=delta_s, nstates=2, nobeta=nobeta).run()
        gpu = gpu_nttda.NTTDA(mf_gpu).set(deltaS=delta_s, nstates=2, nobeta=nobeta).run()
        self.assertTrue(np.all(cpu.converged))
        self.assertTrue(bool(cp.all(gpu.converged)))
        np.testing.assert_allclose(cp.asnumpy(gpu.e), cpu.e, atol=1e-6, rtol=0)
        np.testing.assert_allclose(cp.asnumpy(gpu.e_tot), cpu.e_tot, atol=1e-6, rtol=0)

    def test_hf(self):
        self._check_energy('HF', -1)

    def test_svwn(self):
        self._check_energy('SVWN', 0)

    def test_df_pbe(self):
        self._check_energy('PBE', 1, density_fit=True)

    def test_df_tpss(self):
        self._check_energy('TPSS', -1, density_fit=True)

    def test_df_cam_b3lyp(self):
        self._check_energy('CAM-B3LYP', 0, density_fit=True)

    def test_cam_b3lyp_nobeta(self):
        self._check_energy('CAM-B3LYP', 0, nobeta=True)


if __name__ == '__main__':
    unittest.main()
