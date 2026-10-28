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
from pyscf import dft, gto
from pyscf.solvent import pcm

from nest.solvent import StateSpecificPCM


class PCMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                        basis='def2-svp', spin=2, verbose=0)
        mf = dft.ROKS(cls.mol, xc='HF').run(conv_tol=1e-11)
        uks = dft.UKS(cls.mol, xc='SVWN')
        uks.grids.level = 1
        uks.run(conv_tol=1e-11)
        assert mf.converged and uks.converged
        cls.nt = mf.NTTDA().set(deltaS=-1, nstates=3, conv_tol=1e-10)
        cls.sf = uks.TDA_SF().set(extype=1, nstates=3, conv_tol=1e-10)

    def driver(self, td, eps=10, eps_optical=2):
        solvent = pcm.PCM(self.mol)
        solvent.eps = eps
        solvent.lebedev_order = 7
        return StateSpecificPCM(td, solvent, eps_optical=eps_optical)

    def test_vacuum_limit(self):
        for template in (self.nt, self.sf):
            with self.subTest(method=type(template).__name__):
                td = template.copy().run()
                self.assertTrue(np.all(td.converged))
                sol = self.driver(td, eps=1, eps_optical=1)
                energy = sol.kernel(state=1)
                self.assertTrue(sol.converged)
                np.testing.assert_allclose(energy, td._scf.e_tot + td.e[0], atol=2e-8, rtol=0)
                for method in ('frozen', 'ptss'):
                    np.testing.assert_allclose(sol.vertical(method=method), td.e - td.e[0],
                                               atol=2e-8, rtol=0)
                np.testing.assert_allclose(sol.state_specific(state=2), td.e[1] - td.e[0],
                                           atol=2e-8, rtol=0)
                self.assertTrue(sol.ss_converged)

    def test_energy_regression(self):
        # H2O/def2-SVP; NT: HF, SF: SVWN (default mcol).
        # Each gap table contains frozen and ptSS rows, in Hartree.
        # The fixed cavity and grid settings define this numerical reference.
        references = (
            (self.nt, -75.958959924281,
             [[0.0, 0.273871834613, 0.361661508418],
              [0.0, 0.266210798218, 0.351772869828]],
             0.266116456885),
            (self.sf, -75.850037724627,
             [[0.0, 0.356376019168, 0.376024591218],
              [0.0, 0.347207108791, 0.366794552631]],
             0.347117730352),
        )
        for td, expected_energy, expected_gaps, expected_ss in references:
            with self.subTest(method=type(td).__name__):
                sol = self.driver(td)
                energy = sol.kernel(state=1)
                self.assertTrue(sol.converged)
                np.testing.assert_allclose(energy, expected_energy, atol=2e-8, rtol=0)
                for method, gaps in zip(('frozen', 'ptss'), expected_gaps):
                    np.testing.assert_allclose(sol.vertical(method=method), gaps, atol=2e-8, rtol=0)
                gap_ss = sol.state_specific(state=2)
                self.assertTrue(sol.ss_converged)
                np.testing.assert_allclose(gap_ss, expected_ss, atol=2e-8, rtol=0)
