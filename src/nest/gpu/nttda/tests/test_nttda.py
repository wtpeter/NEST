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

import subprocess
import sys
import unittest

import cupy as cp
import numpy as np
from gpu4pyscf.dft import roks
from pyscf import gto
from pyscf.dft import gen_grid

from nest.gpu import nttda
from nest.gpu.nttda import nttda as nttda_module
from nest.nttda.nttda import nr_rks_fxc1_gga as nr_rks_fxc1_gga_cpu
from nest.nttda.nttda import nr_rks_fxc1_mgga as nr_rks_fxc1_mgga_cpu


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

    def assert_nttda(self, mf, ref, **kwargs):
        td = mf.NTTDA().set(nstates=2, **kwargs).run()
        self.assertTrue(bool(cp.all(td.converged)))
        self.assertEqual(len(td.e), len(ref))
        self.assertIsInstance(td.e, np.ndarray)
        self.assertIsInstance(td.converged, cp.ndarray)
        self.assertTrue(all(isinstance(x, cp.ndarray) for x, _ in td.xy))
        np.testing.assert_allclose(td.e, ref, atol=1e-6, rtol=0)

    def test_hf_nttda(self):
        mf = self.mol.ROKS(xc='HF').to_gpu().run()
        self.assertIsInstance(mf, roks.ROKS)

        self.assert_nttda(
            mf,
            np.array([0.26373033968267973, 0.32114587049263738]),
            deltaS=1,
            nobeta=True,
        )
        self.assert_nttda(
            mf,
            np.array([-0.25588162251385815, 0.03179164805915535]),
            deltaS=-1,
            nobeta=False,
        )
        self.assert_nttda(
            mf,
            np.array([-0.021227306082554027, 0.03681224565830669]),
            deltaS=0,
            nobeta=False,
        )

    def test_svwn_nttda(self):
        mf = self.mol.ROKS(xc='SVWN').to_gpu().run()

        self.assert_nttda(
            mf,
            np.array([-0.21136285952298853, 0.022829192982022128]),
            deltaS=-1,
            nobeta=True,
        )
        self.assert_nttda(
            mf,
            np.array([-0.0014224229333087768, 0.029907227771976085]),
            deltaS=0,
            nobeta=True,
        )
        self.assert_nttda(
            mf,
            np.array([0.2621305574444208, 0.3146577468311684]),
            deltaS=1,
            nobeta=False,
        )

    def test_m062x_nttda(self):
        mf = self.mol.ROKS(xc='M062X').to_gpu().run()

        self.assert_nttda(
            mf,
            np.array([-0.24666086824597583, 0.015820053409613927]),
            deltaS=-1,
            nobeta=True,
        )
        self.assert_nttda(
            mf,
            np.array([-0.008184446338165025, 0.025150738879015422]),
            deltaS=0,
            nobeta=False,
        )
        self.assert_nttda(
            mf,
            np.array([0.26880002289621757, 0.3280851476633962]),
            deltaS=1,
            nobeta=False,
        )

    def test_cam_b3lyp_nttda(self):
        mf = self.mol.ROKS(xc='CAM-B3LYP').to_gpu().run()

        self.assert_nttda(
            mf,
            np.array([-0.0044893465927124268, 0.035037117269294718]),
            deltaS=0,
            nobeta=True,
        )
        self.assert_nttda(
            mf,
            np.array([0.27155932081326395, 0.32184531828332463]),
            deltaS=1,
            nobeta=True,
        )
        self.assert_nttda(
            mf,
            np.array([-0.22362676199942616, 0.02217598445976246]),
            deltaS=-1,
            nobeta=False,
        )

    def test_gpu_path(self):
        mf = self.mol.ROKS(xc='HF').to_gpu().run()
        self.assertIsInstance(mf, roks.ROKS)
        self.assertEqual(nttda_module.lr_eigh.__module__, 'gpu4pyscf.tdscf._lr_eig')

        for delta_s, method in ((1, 'gen_vind_sfu'), (0, 'gen_vind_sc'), (-1, 'gen_vind_sfd')):
            td = mf.NTTDA().set(nstates=2, deltaS=delta_s, nobeta=False)
            vind, hdiag = getattr(td, method)()
            self.assertIsInstance(hdiag, cp.ndarray)
            out = vind(cp.eye(1, hdiag.size))
            self.assertIsInstance(out, cp.ndarray)
            self.assertEqual(out.shape, (1, hdiag.size))

        with self.assertRaisesRegex(TypeError, 'gpu4pyscf ROKS'):
            nttda_module.NTTDA(self.mol.ROKS(xc='HF'))

    def test_cpu_only_import(self):
        code = "import sys; import nest; assert 'cupy' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_vref1(self):
        rng = np.random.default_rng(12)
        for xc, gpu_fn, cpu_fn in (
            ('CAM-B3LYP', nttda_module.nr_rks_fxc1_gga, nr_rks_fxc1_gga_cpu),
            ('M062X', nttda_module.nr_rks_fxc1_mgga, nr_rks_fxc1_mgga_cpu),
        ):
            mf = self.mol.ROKS(xc=xc).to_gpu().run()
            ni = mf._numint
            fxc = ni.cache_xc_kernel(self.mol, mf.grids, xc, mf.mo_coeff, mf.mo_occ, 1)[2]
            fxc = 0.5 * (fxc[0, :, 0] - fxc[0, :, 1] - fxc[1, :, 0] + fxc[1, :, 1])
            mo_coeff = cp.asnumpy(mf.mo_coeff)
            mo_occ = cp.asnumpy(mf.mo_occ)
            orbcs = mo_coeff[:, mo_occ == 2]
            orbvs = mo_coeff[:, mo_occ == 0]
            zs = rng.standard_normal((2, orbcs.shape[1], orbvs.shape[1]))
            dms = np.einsum('xia,pa,qi->xpq', zs, orbvs, orbcs)
            left = cp.einsum(
                'xia,pa->xpi', cp.asarray(zs), cp.asarray(orbvs),
            )
            factorized_dms = nttda_module._factorized_density(
                left, cp.asarray(orbcs),
            )

            ni_cpu = ni.to_cpu()
            grids_cpu = gen_grid.Grids(self.mol)
            grids_cpu.coords = cp.asnumpy(mf.grids.coords)
            grids_cpu.weights = cp.asnumpy(mf.grids.weights)
            grids_cpu.cutoff = mf.grids.cutoff
            fxc_cpu = ni_cpu.cache_xc_kernel(
                self.mol,
                grids_cpu,
                xc,
                mo_coeff,
                mo_occ,
                1,
            )[2]
            fxc_cpu = 0.5 * (
                fxc_cpu[0, :, 0] - fxc_cpu[0, :, 1] - fxc_cpu[1, :, 0] + fxc_cpu[1, :, 1]
            )
            actual = gpu_fn(ni, self.mol, mf.grids, xc, cp.asarray(dms), cp.asarray(fxc_cpu))
            actual_gpu_fxc = gpu_fn(ni, self.mol, mf.grids, xc, cp.asarray(dms), fxc)
            expected = cpu_fn(
                ni_cpu,
                self.mol,
                grids_cpu,
                xc,
                dms,
                fxc_cpu,
            )
            self.assertIsInstance(actual, cp.ndarray)
            actual_factorized = gpu_fn(
                ni, self.mol, mf.grids, xc, factorized_dms,
                cp.asarray(fxc_cpu),
            )
            np.testing.assert_allclose(cp.asnumpy(actual), expected, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(
                cp.asnumpy(actual_factorized), expected,
                atol=1e-10, rtol=1e-10,
            )
            np.testing.assert_allclose(cp.asnumpy(actual_gpu_fxc), expected, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
