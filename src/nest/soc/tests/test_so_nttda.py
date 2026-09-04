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

import io
import unittest

import numpy as np
from pyscf import gto
from pyscf.lib import logger

from nest import nttda
from nest.soc import NTTDASOC, SONTTDA


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atom = '''
        O   0.64372820   0.14077399  -0.04477253
        O  -0.64862595  -0.12779073  -0.05445498
        H   1.16027512  -0.65947800   0.36730132
        H  -1.12109306   0.55561188   0.42651873
        '''
        cls.neutral = gto.M(
            atom=atom, basis='sto-3g', charge=0, spin=2, symmetry=False, verbose=0,
        ).ROKS(xc='SVWN').run(conv_tol=1e-12)
        cls.cation_doublet = gto.M(
            atom=atom, basis='sto-3g', charge=1, spin=1, symmetry=False, verbose=0,
        ).ROKS(xc='SVWN').run(conv_tol=1e-12)
        cls.cation_quartet = gto.M(
            atom=atom, basis='sto-3g', charge=1, spin=3, symmetry=False, verbose=0,
        ).ROKS(xc='SVWN').run(conv_tol=1e-12)

    @staticmethod
    def _complete_branch(mf, direct, delta_s, total_dimension):
        td = nttda.NTTDA(mf).set(deltaS=delta_s, verbose=0)
        if delta_s == -1:
            vind, hdiag = td.gen_vind_sfd()
        elif delta_s == 0:
            vind, hdiag = td.gen_vind_sc()
        else:
            vind, hdiag = td.gen_vind_sfu()
        matrix = vind(np.eye(hdiag.size)).T
        if delta_s == -1:
            block_slice = direct.block_slices[-1][0]
            local_identity = np.zeros((hdiag.size, total_dimension))
            local_identity[:, block_slice] = np.eye(hdiag.size)
            projector = direct._project_deltam1(local_identity)[:, block_slice].T
            values, vectors = np.linalg.eigh(projector)
            physical = vectors[:, values > 0.5]
            energy, vectors = np.linalg.eigh(physical.T @ matrix @ physical)
            vectors = physical @ vectors
        else:
            energy, vectors = np.linalg.eigh(matrix)
        td.e = energy
        td.xy = [(vectors[:, root], 0) for root in range(vectors.shape[1])]
        td.converged = np.ones(len(energy), dtype=bool)
        return td, vectors

    def _full_space_oracle(self, mf, expected_delta_s):
        direct = SONTTDA(mf, soctype='1e').set(verbose=0)
        spin_free_calls = dict.fromkeys(expected_delta_s, 0)
        for delta_s, name in ((-1, 'gen_vind_sfd'), (0, 'gen_vind_sc'), (1, 'gen_vind_sfu')):
            if delta_s not in expected_delta_s:
                continue
            generator = getattr(direct, name)

            def counted_generator(generator=generator, delta_s=delta_s):
                spin_vind, hdiag = generator()

                def counted_vind(xs):
                    spin_free_calls[delta_s] += 1
                    return spin_vind(xs)

                return counted_vind, hdiag

            setattr(direct, name, counted_generator)
        if -1 in expected_delta_s:
            vind, hdiag = direct.gen_vind()
        else:
            with self.assertWarnsRegex(UserWarning, 'Skipping deltaS=-1'):
                vind, hdiag = direct.gen_vind()
        identity = np.eye(hdiag.size, dtype=np.complex128)
        h_direct = vind(identity).T
        self.assertEqual(spin_free_calls, dict.fromkeys(expected_delta_s, 1))
        np.testing.assert_allclose(h_direct, h_direct.conj().T, atol=1e-12, rtol=0)
        self.assertEqual(direct._active_delta_s, expected_delta_s)

        td_objects = []
        branch_vectors = {}
        for delta_s in expected_delta_s:
            td, vectors = self._complete_branch(
                mf, direct, delta_s, hdiag.size,
            )
            td_objects.append(td)
            branch_vectors[delta_s] = vectors
        state_interaction = NTTDASOC(
            td_objects, soctype='1e', include_reference=False,
        )
        state_interaction.build_hamiltonian()

        transform = np.zeros(
            (hdiag.size, direct._physical_dimension), dtype=np.complex128,
        )
        column = 0
        for delta_s in expected_delta_s:
            vectors = branch_vectors[delta_s]
            for root in range(vectors.shape[1]):
                for block_slice in direct.block_slices[delta_s]:
                    transform[block_slice, column] = vectors[:, root]
                    column += 1
        self.assertEqual(column, direct._physical_dimension)
        np.testing.assert_allclose(
            transform.conj().T @ transform,
            np.eye(direct._physical_dimension),
            atol=1e-12,
            rtol=0,
        )
        projected = transform.conj().T @ h_direct @ transform
        np.testing.assert_allclose(
            projected, state_interaction.h_soc, atol=1e-11, rtol=0,
        )
        return direct, projected

    def test_complex_spin_free_batch_uses_one_call(self):
        calls = []

        def real_vind(xs):
            calls.append(xs.copy())
            return 2 * xs

        xs = np.array([[1 + 2j, 0], [0, 3j], [0, 0]], dtype=np.complex128)
        result = SONTTDA._apply_spin_free(real_vind, xs)
        np.testing.assert_allclose(result, 2 * xs, atol=0, rtol=0)
        self.assertEqual(len(calls), 1)
        self.assertFalse(np.iscomplexobj(calls[0]))
        self.assertEqual(calls[0].shape, (3, 2))

    def test_get_ab_matches_vectorized_vind(self):
        rng = np.random.default_rng(13)
        for channels, soctype in (((-1, 0, 1), 'SOMF'), ((-1, 0), '1e')):
            direct = SONTTDA(
                self.neutral, deltaS=channels, soctype=soctype,
            ).set(verbose=0)
            vind, _ = direct.gen_vind()
            a = direct.get_ab()
            xs = (
                rng.standard_normal((a.shape[1], 10))
                + 1j * rng.standard_normal((a.shape[1], 10))
            )
            np.testing.assert_allclose(
                a @ xs, vind(xs.T).T, atol=1e-12, rtol=0,
            )
            np.testing.assert_allclose(a, a.conj().T, atol=1e-12, rtol=0)
            self.assertEqual(direct._active_delta_s, channels)

    def test_neutral_full_space_and_kernel(self):
        subset = self.neutral.SONTTDA(deltaS=[-1, 0], soctype='1e')
        self.assertIsInstance(subset, SONTTDA)
        self.assertEqual(subset._selected_delta_s(1.0), (-1, 0))

        direct, projected = self._full_space_oracle(self.neutral, (-1, 0, 1))
        self.assertEqual(
            [direct.m_values[delta_s].tolist() for delta_s in (-1, 0, 1)],
            [[0.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0, 1.0, 2.0]],
        )
        exact = np.linalg.eigvalsh(projected)[:5]
        np.testing.assert_allclose(
            exact,
            [
                -0.2643416540343013,
                -0.0002251369702839718,
                -0.0002250499879963996,
                -0.0002245162832439073,
                0.03759457343170138,
            ],
            atol=1e-8,
            rtol=0,
        )
        energy, xy = direct.kernel(nstates=5)
        self.assertTrue(np.all(direct.converged))
        np.testing.assert_allclose(energy, exact, atol=1e-8, rtol=0)
        self.assertEqual(len(xy), 5)
        for x, y in xy:
            self.assertEqual(x.shape, (279,))
            self.assertTrue(np.iscomplexobj(x))
            self.assertEqual(y, 0)
            projected_x = direct._project_deltam1(x)
            np.testing.assert_allclose(x, projected_x, atol=1e-12, rtol=0)

        synthetic = np.zeros(279, dtype=np.complex128)
        for delta_s in (-1, 0, 1):
            synthetic[direct.block_slices[delta_s][0].start] = 1 + 1j
        synthetic /= np.linalg.norm(synthetic)
        direct.e = np.zeros(1)
        direct.xy = [(synthetic, 0)]
        np.testing.assert_allclose(direct.spin_square(), [8 / 3], atol=1e-12, rtol=0)

        direct.stdout = io.StringIO()
        self.assertIs(direct.analyze(verbose=logger.INFO), direct)
        output = direct.stdout.getvalue()
        for text in (
            'deltaS=-1', 'deltaS=+0', 'deltaS=+1', 'S= 0.0',
            'M_S=', 'CO(1)', 'CV(1)', 'X=', '|X|^2=', 'sum|X|^2=',
        ):
            self.assertIn(text, output)

    def test_cation_doublet_skips_unphysical_deltam1(self):
        direct, projected = self._full_space_oracle(self.cation_doublet, (0, 1))
        self.assertNotIn(-1, direct.block_slices)
        self.assertEqual(
            [direct.m_values[delta_s].tolist() for delta_s in (0, 1)],
            [[-0.5, 0.5], [-1.5, -0.5, 0.5, 1.5]],
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(projected)[:6],
            [
                -0.000285714498588,
                -0.000285714498578,
                0.042764674190227,
                0.042764674190238,
                0.181611366703026,
                0.181611366703045,
            ],
            atol=1e-8,
            rtol=0,
        )

    def test_cation_quartet_full_space(self):
        direct, projected = self._full_space_oracle(
            self.cation_quartet, (-1, 0, 1),
        )
        self.assertEqual(
            [direct.m_values[delta_s].tolist() for delta_s in (-1, 0, 1)],
            [
                [-0.5, 0.5],
                [-1.5, -0.5, 0.5, 1.5],
                [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
            ],
        )
        self.assertEqual(projected.shape, (406, 406))


if __name__ == '__main__':
    unittest.main()
