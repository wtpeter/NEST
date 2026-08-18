import unittest

import cupy as cp
import numpy as np
from gpu4pyscf.df.df_jk import _DFHF
from pyscf import gto

from nest.gpu.grad.nttda.common import JKDerivativeLedger
from nest.gpu.nttda import NTTDA as GPUNTTDA


class NTTDAGradientKnownValues(unittest.TestCase):
    @staticmethod
    def molecule():
        return gto.M(
            atom="N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )

    def gradient(self, xc, delta_s, nobeta=False, density_fit=False):
        mol = self.molecule()
        mf_gpu = mol.ROKS(xc=xc).to_gpu()
        mf_gpu.grids.level = 0
        if density_fit:
            mf_gpu = mf_gpu.density_fit()
        mf_gpu.run()
        self.assertTrue(mf_gpu.converged)
        td_gpu = GPUNTTDA(mf_gpu).set(
            deltaS=delta_s,
            nobeta=nobeta,
            nstates=2,
            conv_tol=1e-9,
            max_cycle=200,
        ).run()
        gradient = td_gpu.Gradients().set(
            verbose=0,
            cphf_conv_tol=1e-10,
        )
        actual = gradient.kernel(state=1)
        self.assertIsInstance(actual, np.ndarray)
        self.assertIsInstance(gradient.nttda_details.m_matrix, cp.ndarray)
        self.assertIsInstance(gradient.nttda_details.zvector, cp.ndarray)
        self.assertLess(gradient.nttda_details.residual, 1e-5)
        return actual

    def test_jk_ledger_selects_stable_backend_contractions(self):
        mol = self.molecule()
        density = cp.eye(mol.nao)
        calls = []

        class GradientDriver:
            def __init__(self, scf):
                self.base = type("Base", (), {"_scf": scf})()

            def jk_energies_per_atom(
                    self, dm_pairs, j_factor, k_factor, omega,
                    sum_results):
                assert dm_pairs.ndim == 4
                calls.append((
                    omega, tuple(j_factor), tuple(k_factor), sum_results,
                ))
                if sum_results:
                    scale = omega + sum(j_factor) + sum(k_factor)
                    return cp.full((mol.natm, 3), scale)
                return cp.stack([
                    cp.full((mol.natm, 3), omega + j + k)
                    for j, k in zip(j_factor, k_factor)
                ])

        ledger = JKDerivativeLedger()
        ledger.add("j", "a", (
            (density, density, 0.5, None),
            (density, density, 0.25, 0.3),
        ))
        ledger.add("k", "a", (
            (density, density, 0.125, None),
        ))
        ledger.add("j", "b", (
            (density, density, 0.75, None),
        ))

        gradients = ledger.contract(
            GradientDriver(object()),
            mol,
            range(mol.natm),
            slots=("a", "b"),
        )

        self.assertEqual(calls, [
            (0.0, (1.0, 0.0), (0.0, -0.5), True),
            (0.3, (0.5,), (0.0,), True),
            (0.0, (1.5,), (0.0,), True),
        ])
        np.testing.assert_allclose(cp.asnumpy(gradients["a"]), 1.3)
        np.testing.assert_allclose(cp.asnumpy(gradients["b"]), 1.5)

        calls.clear()
        gradients = ledger.contract(
            GradientDriver(object.__new__(_DFHF)),
            mol,
            range(mol.natm),
            slots=("a", "b"),
        )
        self.assertEqual(calls, [
            (0.0, (1.0, 1.5, 0.0), (0.0, 0.0, -0.5), False),
            (0.3, (0.5,), (0.0,), False),
        ])
        np.testing.assert_allclose(cp.asnumpy(gradients["a"]), 1.3)
        np.testing.assert_allclose(cp.asnumpy(gradients["b"]), 1.5)

    def test_analytic_against_cpu(self):
        cases = (
            (
                "HF", 0, False, False,
                [[0.0, 3.63013040, 10.8709903],
                 [0.0, 0.481970055, -12.4274171],
                 [0.0, -4.11210045, 1.55642675]],
            ),
            (
                "SVWN", 0, False, False,
                [[0.0, 3.61792319, 10.7811296],
                 [0.0, 0.475478241, -12.3352829],
                 [0.0, -4.11377287, 1.54960914]],
            ),
            (
                "PBE", 0, False, True,
                [[0.0, 3.61112261, 10.7909544],
                 [0.0, 0.477094482, -12.3818329],
                 [0.0, -4.11603478, 1.55188573]],
            ),
            (
                "TPSS", -1, False, False,
                [[0.0, 3.72623548, 10.6880437],
                 [0.0, 0.479325174, -12.3122842],
                 [0.0, -4.20597591, 1.54945046]],
            ),
            (
                "M06-2X", 0, True, False,
                [[0.0, 3.61814540, 10.7175124],
                 [0.0, 0.463068792, -12.3860681],
                 [0.0, -4.10800230, 1.55111283]],
            ),
            (
                "CAM-B3LYP", -1, False, False,
                [[0.0, 3.72006477, 10.6706030],
                 [0.0, 0.459749204, -12.2321197],
                 [0.0, -4.20039098, 1.54740602]],
            ),
            (
                "HF", 1, False, False,
                [[0.0, 5.76198980, 11.3870699],
                 [0.0, -0.156420838, -12.4241906],
                 [0.0, -5.60556896, 1.03712075]],
            ),
            (
                "PBE", 1, False, False,
                [[0.0, 5.33573208, 10.9641125],
                 [0.0, 0.0714051352, -12.2871295],
                 [0.0, -5.43971674, 1.31070967]],
            ),
        )
        for xc, delta_s, nobeta, density_fit, expected in cases:
            with self.subTest(
                    xc=xc, delta_s=delta_s, nobeta=nobeta,
                    density_fit=density_fit):
                actual = self.gradient(
                    xc, delta_s, nobeta=nobeta,
                    density_fit=density_fit,
                )
                np.testing.assert_allclose(
                    actual, np.asarray(expected), atol=1e-5, rtol=0,
                )

    def finite_difference(
            self, mol, xc, delta_s, state, grid_level,
            density_fit=False, scf_conv_tol=1e-9):
        def make_reference(reference_mol):
            mf = reference_mol.ROKS(xc=xc).to_gpu()
            if density_fit:
                mf = mf.density_fit()
            mf.grids.level = grid_level
            mf.conv_tol = scf_conv_tol
            mf.max_cycle = 200
            return mf

        mf = make_reference(mol).run()
        self.assertTrue(mf.converged)
        td = GPUNTTDA(mf).set(
            deltaS=delta_s,
            nstates=state,
            conv_tol=1e-10,
            max_cycle=200,
        ).run()
        analytic = td.Gradients().set(
            verbose=0,
            cphf_conv_tol=1e-10,
        ).kernel(state=state)

        step = 2e-4
        coords = mol.atom_coords()
        dm0 = mf.make_rdm1()
        finite_difference = np.zeros_like(analytic)
        for atom in range(mol.natm):
            for xyz in range(3):
                energies = []
                for displacement in (step, -step):
                    displaced = mol.copy()
                    displaced_coords = coords.copy()
                    displaced_coords[atom, xyz] += displacement
                    displaced.set_geom_(displaced_coords, unit="Bohr")
                    displaced_mf = make_reference(displaced)
                    displaced_mf.kernel(dm0=dm0)
                    self.assertTrue(displaced_mf.converged)
                    displaced_td = GPUNTTDA(displaced_mf).set(
                        deltaS=delta_s,
                        nstates=state,
                        conv_tol=1e-10,
                        max_cycle=200,
                    ).run()
                    energies.append(
                        displaced_mf.e_tot + displaced_td.e[state - 1]
                    )
                finite_difference[atom, xyz] = (
                    energies[0] - energies[1]
                ) / (2 * step)
        return analytic, finite_difference

    def test_pbe_against_finite_difference(self):
        mol = gto.M(
            atom="O 0 0 0; O 0 0 2.30",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )
        analytic, finite_difference = self.finite_difference(
            mol, "PBE", delta_s=0, state=2, grid_level=6,
        )
        np.testing.assert_allclose(
            analytic, finite_difference, atol=1e-5, rtol=0,
        )

    def test_df_delta_s_minus_one_against_finite_difference(self):
        analytic, finite_difference = self.finite_difference(
            self.molecule(),
            "HF",
            delta_s=-1,
            state=1,
            grid_level=0,
            density_fit=True,
            scf_conv_tol=1e-12,
        )
        np.testing.assert_allclose(
            analytic, finite_difference, atol=1e-5, rtol=0,
        )

    def test_df_delta_s_plus_one_against_finite_difference(self):
        analytic, finite_difference = self.finite_difference(
            self.molecule(),
            "HF",
            delta_s=1,
            state=1,
            grid_level=0,
            density_fit=True,
            scf_conv_tol=1e-12,
        )
        np.testing.assert_allclose(
            analytic, finite_difference, atol=1e-5, rtol=0,
        )

    def test_delta_s_plus_one_closed_shell_finite_difference(self):
        mol = gto.M(
            atom="Li 0 0 0; H 0 0 1.60",
            basis="sto-3g",
            spin=0,
            unit="Bohr",
            verbose=0,
        )
        analytic, finite_difference = self.finite_difference(
            mol,
            "HF",
            delta_s=1,
            state=1,
            grid_level=0,
            scf_conv_tol=1e-12,
        )
        np.testing.assert_allclose(
            analytic, finite_difference, atol=1e-5, rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
