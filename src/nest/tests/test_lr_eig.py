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
from pyscf.lib import logger

from nest._lr_eig import eigh


class HermitianSolverTests(unittest.TestCase):
    def test_nearly_converged_guess_and_preconditioner_scale(self):
        rng = np.random.default_rng(123)
        matrix = rng.normal(size=(12, 12))
        matrix += matrix.T
        expected, vectors = np.linalg.eigh(matrix)
        guesses = vectors[:, :2].T + 1e-7*rng.normal(size=(2, 12))

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(len(w))

        for scale in (1.0, 1e-8):
            with self.subTest(scale=scale):
                def precond(residual, energy):
                    return scale*residual/(matrix.diagonal()-energy)

                converged, energies, roots = eigh(
                    lambda z: z @ matrix, guesses, precond, nroots=2,
                    tol_residual=1e-10, pick=all_eigs, verbose=logger.new_logger(None, 0))
                self.assertTrue(np.all(converged))
                np.testing.assert_allclose(energies, expected[:2], atol=1e-12)
                residuals = roots @ matrix - energies[:, None]*roots
                self.assertLess(np.linalg.norm(residuals, axis=1).max(), 1e-10)
