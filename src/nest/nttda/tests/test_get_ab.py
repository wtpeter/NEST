# Copyright 2026 The NEST Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest

import numpy as np
from pyscf import gto

from nest import nttda


class GetABKnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0 1; H 0 1 0',
            basis='sto-3g',
            spin=2,
            verbose=0,
        )

    def assert_vind_matches_get_ab(self, mf, nobeta=False):
        rng = np.random.default_rng(12)
        generators = {
            -1: 'gen_vind_sfd',
            0: 'gen_vind_sc',
            1: 'gen_vind_sfu',
        }
        for delta_s, generator in generators.items():
            td = nttda.NTTDA(mf).set(
                deltaS=delta_s, nobeta=nobeta, verbose=0,
            )
            vind, _ = getattr(td, generator)()
            a = td.get_ab()
            xs = rng.standard_normal((a.shape[1], 10))
            np.testing.assert_allclose(a @ xs, vind(xs.T).T, atol=1e-12, rtol=0)
            np.testing.assert_allclose(a, a.T, atol=1e-12, rtol=0)

    def test_hf(self):
        mf = self.mol.ROKS(xc='HF').run(conv_tol=1e-11)
        self.assert_vind_matches_get_ab(mf)
        self.assert_vind_matches_get_ab(mf, nobeta=True)

    def test_lda(self):
        self.assert_vind_matches_get_ab(self.mol.ROKS(xc='SVWN').run(conv_tol=1e-11))

    def test_gga(self):
        self.assert_vind_matches_get_ab(self.mol.ROKS(xc='PBE').run(conv_tol=1e-11))

    def test_mgga(self):
        self.assert_vind_matches_get_ab(self.mol.ROKS(xc='M06-L').run(conv_tol=1e-11))

    def test_range_separated_hybrid(self):
        self.assert_vind_matches_get_ab(
            self.mol.ROKS(xc='CAM-B3LYP').run(conv_tol=1e-11),
        )


if __name__ == '__main__':
    unittest.main()
