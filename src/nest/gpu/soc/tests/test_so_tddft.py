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

import cupy as cp
import numpy as np
from pyscf import gto

from nest.gpu.soc import SOTDDFT as GPU_SOTDDFT
from nest.soc import SOTDDFT as CPU_SOTDDFT


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='''
            O   0.64372820   0.14077399  -0.04477253
            O  -0.64862595  -0.12779073  -0.05445498
            H   1.16027512  -0.65947800   0.36730132
            H  -1.12109306   0.55561188   0.42651873
            ''',
            basis='631g',
            spin=0,
            verbose=4,
        )
        cls.mf_cpu = cls.mol.RKS(xc='PBE').run()

    def test_cpu_gpu_somf(self):
        cpu = CPU_SOTDDFT(
            self.mf_cpu, soctype='SOMF', include_reference=True,
        ).set(nstates=5, max_cycle=300).run()

        mf_gpu = self.mf_cpu.to_gpu()
        gpu = mf_gpu.SOTDDFT(
            soctype='SOMF', include_reference=True,
        ).set(nstates=5, max_cycle=300).run()

        self.assertIsInstance(gpu, GPU_SOTDDFT)
        self.assertTrue(bool(cp.all(gpu.converged)))
        self.assertIsInstance(gpu.soc_ao, cp.ndarray)
        self.assertIsInstance(gpu.soc_mo, cp.ndarray)
        self.assertTrue(all(isinstance(x, cp.ndarray) for x, _ in gpu.xy))
        np.testing.assert_allclose(gpu.e, cpu.e, atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
