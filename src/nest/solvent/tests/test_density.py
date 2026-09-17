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

"""Check NT densities against explicit fermionic creation/annihilation operators."""

import unittest

import numpy as np
from pyscf import gto, scf

from nest.nttda import NTTDA
from nest.nttda.nttda import transition_dm
from nest.solvent import state_density
from nest.solvent.pcm import _potential_action


def excite(vector, p, q):
    """Apply a_p^+ a_q to a full occupation-bitstring vector."""
    out = np.zeros_like(vector)
    for ket in np.flatnonzero(vector):
        if not ket & (1 << q):
            continue
        sign = (-1) ** int((int(ket) & ((1 << q) - 1)).bit_count())
        reduced = int(ket) ^ (1 << q)
        if reduced & (1 << p):
            continue
        sign *= (-1) ** int((reduced & ((1 << p) - 1)).bit_count())
        out[reduced | (1 << p)] += sign * vector[ket]
    return out


def nt_basis(no, ds):
    """Expand the NT CSFs in determinants, using the spin-ladder definitions."""
    n = no + 2
    s = no / 2
    i, a = 0, n - 1
    opened = range(1, no + 1)
    ref = np.zeros(1 << (2 * n))
    ref[(1 << (no + 1)) - 1 | (1 << n)] = 1

    def up(v, p, q):
        return excite(v, p, q)

    def down(v, p, q):
        return excite(v, p + n, q + n)

    def flip(v, p, q):
        return excite(v, p + n, q)

    def raise_spin(v, p, q):
        return excite(v, p, q + n)

    lower = sum(flip(ref, u, u) for u in opened)
    if ds == 1:
        return np.array([-raise_spin(ref, a, i)])
    if ds == 0:
        co = [down(ref, u, i) for u in opened]
        cv = -np.sqrt(s / (2 * (s + 1))) * (up(ref, a, i) - down(ref, a, i))
        cv -= raise_spin(lower, a, i) / np.sqrt(2 * s * (s + 1))
        ov = [-up(ref, a, u) for u in opened]
        cv0 = (up(ref, a, i) + down(ref, a, i)) / np.sqrt(2)
        return np.array(co + [cv, -ref] + ov + [cv0])

    co, ov = [], []
    for u in opened:
        other = sum(flip(ref, w, w) for w in opened if w != u)
        co.append(np.sqrt((2 * s - 1) / (2 * s)) * flip(ref, u, i)
                  + down(other, u, i) / np.sqrt(2 * s * (2 * s - 1)))
        ov.append(np.sqrt((2 * s - 1) / (2 * s)) * flip(ref, a, u)
                  - up(other, a, u) / np.sqrt(2 * s * (2 * s - 1)))
    double_lower = sum(flip(flip(ref, w, w), v, v) for v in opened for w in opened if v < w)
    cv = np.sqrt((2 * s - 1) / (2 * s + 1)) * flip(ref, a, i)
    cv -= np.sqrt((2 * s - 1) / (2 * s + 1)) / (2 * s) * (up(lower, a, i) - down(lower, a, i))
    cv -= raise_spin(double_lower, a, i) / (s * np.sqrt((2 * s - 1) * (2 * s + 1)))
    rows = [co + [cv]]
    for u in opened:
        rows.append([flip(ref, t, u) for t in opened] + [ov[u - 1]])
    return np.array(rows).reshape((no + 1)**2, -1)


class DensityTests(unittest.TestCase):
    def test_nt_density_and_potential_against_determinants(self):
        rng = np.random.default_rng(309)
        for no in (2, 3):
            n = no + 2
            mol = gto.M(atom=[('H', (0, 0, k)) for k in range(n)], basis='sto-3g', spin=no, verbose=0)
            mf = scf.ROHF(mol)
            mf.mo_coeff = np.eye(n)
            mf.mo_occ = np.array([2] + [1] * no + [0])
            mf.converged = True
            potential = rng.normal(size=(n, n))
            potential += potential.T
            for ds in (-1, 0, 1):
                with self.subTest(no=no, ds=ds):
                    td = NTTDA(mf).set(deltaS=ds)
                    csfs = nt_basis(no, ds)
                    np.testing.assert_allclose(csfs @ csfs.T, np.eye(len(csfs)), atol=1e-14)
                    x = rng.normal(size=len(csfs)) + 1j * rng.normal(size=len(csfs))
                    x /= np.linalg.norm(x)
                    y = rng.normal(size=len(csfs)) + 1j * rng.normal(size=len(csfs))
                    y -= np.vdot(x, y) * x
                    if ds == 1:
                        y = x.copy()  # one-dimensional space: only the diagonal check
                    else:
                        y /= np.linalg.norm(y)
                    shape = {-1: (no + 1, no + 1), 0: (len(x),), 1: (1, 1)}[ds]
                    td.xy = [(x.reshape(shape), 0), (y.reshape(shape), 0)]
                    psi, phi = x @ csfs, y @ csfs
                    expected = np.zeros((n, n), dtype=complex)
                    transition = np.zeros_like(expected)
                    for p in range(n):
                        for q in range(n):
                            expected[p, q] = np.vdot(psi, excite(psi, p, q) + excite(psi, p+n, q+n))
                            transition[p, q] = np.vdot(psi, excite(phi, p, q) + excite(phi, p+n, q+n))
                    dm = state_density(td)
                    np.testing.assert_allclose(dm, expected.T, atol=1e-13)
                    np.testing.assert_allclose(dm, dm.conj().T, atol=1e-13)
                    self.assertAlmostEqual(np.trace(dm).real, n, places=12)
                    ref_term = np.vdot(x, y) * np.diag(mf.mo_occ)
                    np.testing.assert_allclose(transition_dm(td, 0, 1) + ref_term, transition, atol=1e-13)
                    action, diag = _potential_action(td, potential)
                    matrix = action(np.eye(len(x))).T
                    np.testing.assert_allclose(matrix, matrix.T, atol=1e-13)
                    np.testing.assert_allclose(diag, matrix.diagonal(), atol=1e-13)
                    expected_value = np.einsum('pq,pq->', transition - ref_term, potential)
                    np.testing.assert_allclose(np.vdot(x, matrix @ y), expected_value, atol=1e-13)

    def test_sf_density_and_one_body_operator(self):
        rng = np.random.default_rng(50)
        mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='sto-3g', spin=2, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-11)
        for extype in (0, 1):
            td = mf.TDA_SF().set(extype=extype)
            potential = rng.normal(size=(mol.nao, mol.nao))
            potential += potential.T
            action, diag = _potential_action(td, potential)
            so, sv = (1, 0) if extype == 0 else (0, 1)
            shape = (np.count_nonzero(mf.mo_occ[so]), np.count_nonzero(mf.mo_occ[sv] == 0))
            x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            x /= np.linalg.norm(x)
            td.xy = [(x, 0)]
            dm = state_density(td)
            self.assertAlmostEqual(np.einsum('ij,ji', dm, mf.get_ovlp()).real, mol.nelectron, places=10)
            np.testing.assert_allclose(dm, dm.conj().T, atol=1e-13)
            dref = mf.make_rdm1().sum(axis=0)
            np.testing.assert_allclose(np.vdot(x, action(x.reshape(1, -1))),
                                       np.einsum('ij,ji', potential, dm-dref), atol=1e-12)
            co = mf.mo_coeff[so][:, mf.mo_occ[so] > 0]
            cv = mf.mo_coeff[sv][:, mf.mo_occ[sv] == 0]
            v_oo, v_vv = co.T @ potential @ co, cv.T @ potential @ cv
            explicit = np.kron(np.eye(shape[0]), v_vv) - np.kron(v_oo, np.eye(shape[1]))
            np.testing.assert_allclose(action(np.eye(x.size)).T, explicit, atol=1e-13)
            np.testing.assert_allclose(diag, explicit.diagonal(), atol=1e-13)
