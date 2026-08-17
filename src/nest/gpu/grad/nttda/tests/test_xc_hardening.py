import unittest
from unittest import mock

import cupy as cp
import numpy as np
from pyscf import gto

from nest.gpu.grad.nttda import common
from nest.gpu.grad.nttda import delta_s_zero
from nest.gpu.grad.nttda import xc as gpu_xc
from nest.gpu.nttda import NTTDA


class XCGradientHardeningTests(unittest.TestCase):
    @staticmethod
    def molecule():
        return gto.M(
            atom="N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )

    def test_directed_pair_ao_rows_match_scalar_oracle(self):
        rng = np.random.default_rng(122)
        nao = 7
        ngrids = 19
        npair = 2
        ao = cp.asarray(rng.standard_normal((10, ngrids, nao)))
        densities = cp.asarray(rng.standard_normal((npair, nao, nao)))
        tensor_weights = cp.asarray(
            rng.standard_normal((npair, 4, 4, ngrids)),
        )
        grid_weights = cp.asarray(rng.standard_normal(ngrids))

        _, contracted = gpu_xc.pair_feature_batches(ao, densities)
        expected = cp.zeros((nao, 3))
        for row in range(nao):
            for xyz in range(3):
                delta = gpu_xc._compact_ao_center_derivative(
                    ao, row, row + 1, xyz, "GGA",
                )
                expected[row, xyz] = (
                    gpu_xc.contract_pair_feature_derivatives(
                        ao, densities, delta, contracted,
                        row, row + 1, tensor_weights, grid_weights,
                    )
                )

        actual = cp.zeros_like(expected)
        transpose_buf = cp.empty((npair, ngrids, nao))
        coefficient_buf = cp.empty((4, ngrids, nao))
        gpu_xc._contract_gga_pair_ao_rows(
            actual, ao, densities, contracted, tensor_weights,
            grid_weights, transpose_buf, coefficient_buf,
        )
        cp.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)

        symmetric = 0.5 * (
            densities + densities.swapaxes(-1, -2)
        )
        _, symmetric_contracted = gpu_xc.pair_feature_batches(
            ao, symmetric,
        )
        symmetrized = cp.zeros_like(expected)
        gpu_xc._contract_gga_pair_ao_rows(
            symmetrized, ao, symmetric, symmetric_contracted,
            tensor_weights, grid_weights, transpose_buf, coefficient_buf,
        )
        self.assertGreater(
            float(cp.max(cp.abs(actual - symmetrized)).get()), 1e-6,
        )

    def test_sorted_ao_reduction_handles_noncontiguous_atoms(self):
        class SortedMol:
            natm = 3
            nao = 7
            _bas = np.asarray([[2], [0], [2], [1]], dtype=np.int32)

            @staticmethod
            def ao_loc_nr():
                return np.asarray([0, 1, 3, 5, 7], dtype=np.int32)

        values = cp.arange(2 * 7 * 3, dtype=cp.float64).reshape(2, 7, 3)
        atom_of_ao = (2, 0, 0, 2, 2, 1, 1)
        expected = cp.zeros((2, 2, 3))
        for batch in range(2):
            for output_atom, atom in enumerate((2, 0)):
                rows = [
                    row for row, owner in enumerate(atom_of_ao)
                    if owner == atom
                ]
                expected[batch, output_atom] = values[batch, rows].sum(0)

        actual = gpu_xc._reduce_sorted_ao_to_atoms(
            SortedMol(), values, atmlst=(2, 0),
        )
        cp.testing.assert_array_equal(actual, expected)
        cp.testing.assert_array_equal(
            gpu_xc._reduce_sorted_ao_to_atoms(
                SortedMol(), values[0], atmlst=(2, 0),
            ),
            expected[0],
        )

    def test_ordinary_vxc_sparse_matches_scalar_formula(self):
        for xc_name, xctype, nprobe in (
                ("PBE", "GGA", 1),
                ("TPSS", "MGGA", 2)):
            with self.subTest(xctype=xctype, nprobe=nprobe):
                mf = self.molecule().ROKS(xc=xc_name).to_gpu()
                mf.grids.level = 0
                mf.run()
                self.assertTrue(mf.converged)

                ni = mf._numint
                opt = ni.gdftopt
                sorted_mol = opt._sorted_mol
                nao = sorted_mol.nao
                ao, mask, weights, coords = next(iter(ni.block_loop(
                    sorted_mol, mf.grids, nao, 2, max_memory=None,
                )))
                keep = cp.arange(len(mask))
                keep = keep[keep % 3 != 0]
                sparse_ao = ao[:, keep]
                sparse_mask = mask[keep]
                self.assertLess(len(sparse_mask), nao)

                dense_sorted_ao = cp.zeros(
                    (len(ao), nao, weights.size), dtype=ao.dtype,
                )
                dense_sorted_ao[:, sparse_mask] = sparse_ao
                dense_ao = opt.unsort_orbitals(
                    dense_sorted_ao, axis=[1],
                ).transpose(0, 2, 1)

                mo = cp.asarray(mf.mo_coeff)
                density_alpha = (
                    mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T
                )
                density_beta = (
                    mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T
                )
                rng = np.random.default_rng(20 + nprobe)
                probe_alpha = cp.asarray(
                    rng.standard_normal((nprobe, nao, nao)) * 1e-2,
                )
                probe_beta = cp.asarray(
                    rng.standard_normal((nprobe, nao, nao)) * 1e-2,
                )
                if nprobe == 1:
                    probe_alpha = probe_alpha[0]
                    probe_beta = probe_beta[0]

                probes_alpha, probes_beta, single_probe = (
                    gpu_xc._spin_probe_stacks(probe_alpha, probe_beta)
                )
                density_alpha = 0.5 * (
                    density_alpha + density_alpha.T
                )
                density_beta = 0.5 * (
                    density_beta + density_beta.T
                )
                probe_densities = cp.stack(
                    (probes_alpha, probes_beta), axis=1,
                ).reshape(-1, nao, nao)
                density_stack = cp.concatenate((
                    cp.stack((density_alpha, density_beta)),
                    probe_densities,
                ))
                full_mask = cp.arange(nao)
                rho = cp.asarray([
                    gpu_xc._xc_density(
                        ni, mf.mol, dense_ao, density,
                        full_mask, xctype,
                    )
                    for density in density_stack
                ])
                reference_rho = rho[:2]
                probe_rho = rho[2:].reshape(
                    nprobe, 2, *rho.shape[1:],
                )
                vxc, fxc = ni.eval_xc_eff(
                    mf.xc, reference_rho, deriv=2,
                    xctype=xctype, spin=1,
                )[1:3]

                expected = cp.zeros((nprobe, mf.mol.natm, 3))
                offsets = mf.mol.offset_nr_by_atom()
                for atom in range(mf.mol.natm):
                    p0, p1 = offsets[atom][2:]
                    for xyz in range(3):
                        delta = gpu_xc._xc_ao_center_derivative(
                            dense_ao, p0, p1, xyz, xctype,
                        )
                        density_derivative = (
                            gpu_xc._xc_density_derivatives(
                                dense_ao, density_stack, p0, p1,
                                xctype, delta,
                            )
                        )
                        reference_derivative = density_derivative[:2]
                        probe_derivative = density_derivative[2:].reshape(
                            nprobe, 2, *density_derivative.shape[1:],
                        )
                        expected[:, atom, xyz] += cp.einsum(
                            "nsxg,sxg,g->n",
                            probe_derivative, vxc, weights,
                        )
                        response_weights = cp.einsum(
                            "axg,axbyg,g->byg",
                            reference_derivative, fxc, weights,
                        )
                        expected[:, atom, xyz] += cp.einsum(
                            "nbyg,byg->n", probe_rho, response_weights,
                        )

                contractor = getattr(
                    gpu_xc,
                    f"contract_{xctype.lower()}_vxc_derivative",
                )
                with mock.patch.object(
                        ni, "block_loop",
                        return_value=[
                            (sparse_ao, sparse_mask, weights, coords),
                        ]):
                    actual = contractor(
                        mf, density_alpha, density_beta,
                        probe_alpha, probe_beta,
                    )
                if single_probe:
                    expected = expected[0]
                cp.testing.assert_allclose(
                    actual, expected, atol=1e-11, rtol=0,
                )

    def test_gga_fockz_without_direct_matches_q_only(self):
        mf = self.molecule().ROKS(xc="PBE").to_gpu()
        mf.grids.level = 0
        mf.run()
        self.assertTrue(mf.converged)
        td = NTTDA(mf).set(
            deltaS=0,
            nstates=2,
            conv_tol=1e-9,
            max_cycle=200,
        ).run()
        spaces = common.orbital_spaces(td)
        _, pz = delta_s_zero.same_spin_fock_probes(td, td.xy[0])
        driver = td.Gradients()
        with_direct = gpu_xc.gga_fockz_terms(
            driver, td, spaces, pz,
            atmlst=(2, 0), with_direct=True,
        )

        with mock.patch.object(
                gpu_xc.tdrks_grad, "_gga_eval_mat_",
                side_effect=AssertionError("direct matrix path used")):
            with mock.patch.object(
                    gpu_xc.rks_grad, "_gga_grad_sum_",
                    side_effect=AssertionError("direct gradient path used")):
                q_only = gpu_xc.gga_fockz_terms(
                    driver, td, spaces, pz,
                    atmlst=(2, 0), with_direct=False,
                )

        cp.testing.assert_allclose(
            q_only.q_alpha, with_direct.q_alpha, atol=1e-13, rtol=0,
        )
        cp.testing.assert_allclose(
            q_only.q_beta, with_direct.q_beta, atol=1e-13, rtol=0,
        )
        cp.testing.assert_array_equal(
            q_only.direct, cp.zeros((2, 3)),
        )


if __name__ == "__main__":
    unittest.main()
