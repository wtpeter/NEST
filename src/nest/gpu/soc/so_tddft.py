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

"""GPU direct spin-orbit-coupled TDA for closed-shell RHF/RKS.

The spin-free response, SOC transformation, matrix-vector products, and
Davidson diagonalization are evaluated on the GPU.  AO SOC integrals are
constructed by the CPU implementation and transferred to the GPU once.
"""

import cupy as cp
from pyscf import lib

from gpu4pyscf import scf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.tdscf import rhf
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh

from nest.soc.soc_ao import get_ao_soc


class SOTDDFT(rhf.TDA):
    """GPU spin-orbit-coupled, TDA-only TDHF/TDDFT.

    ``xy[root]`` is stored as ``(x, 0)`` on the GPU.  The one-dimensional
    vector ``x`` uses block order ``[S, T-1, T0, T+1]``, optionally preceded
    by the closed-shell reference coefficient.
    """

    device = 'gpu'
    singlet = None
    soctype = 'SOMF'
    include_reference = True

    _keys = rhf.TDA._keys.union({'soctype', 'include_reference', 'soc_ao', 'soc_mo'})

    def __init__(self, mf, soctype='SOMF', include_reference=True, frozen=None):
        if not isinstance(mf, scf.hf.RHF):
            raise TypeError('GPU SOTDDFT requires a gpu4pyscf RHF/RKS reference')
        super().__init__(mf, frozen=frozen)
        self.soctype = soctype
        self.include_reference = include_reference
        self.soc_ao = None
        self.soc_mo = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('soctype = %s', self.soctype)
        log.info('include_reference = %s', self.include_reference)
        return self

    def check_sanity(self):
        super().check_sanity()
        mo_occ = cp.asarray(self._scf.mo_occ)
        mo_coeff = cp.asarray(self._scf.mo_coeff)
        if mo_occ.ndim != 1 or not bool(cp.all((mo_occ == 0) | (mo_occ == 2))):
            raise ValueError('SOTDDFT requires a closed-shell RHF/RKS reference')
        if mo_coeff.ndim != 2 or not cp.isrealobj(mo_coeff):
            raise ValueError('SOTDDFT requires real, spin-restricted MO coefficients')
        if self.frozen is not None:
            raise NotImplementedError('GPU SOTDDFT does not support frozen orbitals')
        if self.wfnsym is not None:
            raise NotImplementedError('SOTDDFT does not support wfnsym because SOC mixes spatial irreps')
        return self

    @staticmethod
    def _apply_one_body(z_vv, z_oo, xs, hole_sign):
        r"""Apply ``delta_ij z_ab + hole_sign delta_ab z_ji``."""
        particle = contract('ab,xib->xia', z_vv, xs)
        hole = contract('ji,xja->xia', z_oo, xs)
        return particle + hole_sign * hole

    def gen_vind(self, mf=None):
        """Generate the GPU matrix-vector product for the SOC Casida A matrix."""
        if mf is None:
            mf = self._scf
        if mf is not self._scf:
            raise ValueError('gen_vind must use the SCF object associated with SOTDDFT')
        self.check_sanity()

        vind_s, hdiag_s = rhf.gen_tda_operation(
            self, mf, singlet=True, wfnsym=None,
        )
        vind_t, hdiag_t = rhf.gen_tda_operation(
            self, mf, singlet=False, wfnsym=None,
        )

        mo_coeff = cp.asarray(mf.mo_coeff)
        mo_occ = cp.asarray(mf.mo_occ)
        occidx = cp.where(mo_occ == 2)[0]
        viridx = cp.where(mo_occ == 0)[0]
        nocc = occidx.size
        nvir = viridx.size
        nov = nocc * nvir

        # AO SOC construction is retained on the CPU.  All subsequent work is
        # performed with this single device copy.
        self.soc_ao = cp.asarray(get_ao_soc(mf.to_cpu(), self.soctype))
        soc_mo = contract('up,xuv->xpv', mo_coeff.conj(), self.soc_ao)
        self.soc_mo = contract('xpv,vq->xpq', soc_mo, mo_coeff)

        z_minus = 2 * self.soc_mo[0]
        z_z = cp.sqrt(2.0) * self.soc_mo[1]
        z_plus = -2 * self.soc_mo[2]

        z_minus_oo = z_minus[cp.ix_(occidx, occidx)]
        z_minus_vv = z_minus[cp.ix_(viridx, viridx)]
        z_plus_oo = z_plus[cp.ix_(occidx, occidx)]
        z_plus_vv = z_plus[cp.ix_(viridx, viridx)]
        z_z_oo = z_z[cp.ix_(occidx, occidx)]
        z_z_vv = z_z[cp.ix_(viridx, viridx)]

        offset = int(self.include_reference)
        dimension = offset + 4 * nov

        if self.include_reference:
            z_minus_vo = z_minus[cp.ix_(viridx, occidx)].T.reshape(-1)
            z_z_vo = z_z[cp.ix_(viridx, occidx)].T.reshape(-1)
            z_plus_vo = z_plus[cp.ix_(viridx, occidx)].T.reshape(-1)
            reference_coupling = cp.concatenate((
                cp.zeros(nov, dtype=cp.complex128),
                0.5 * z_plus_vo,
                z_z_vo / cp.sqrt(2.0),
                -0.5 * z_minus_vo,
            ))

        def apply_spin_free(vind_spin, xs):
            # gpu4pyscf response kernels operate on real trial densities.
            # The real-orbital spin-free operator is real-linear, so apply it
            # independently to the real and imaginary components.
            if cp.iscomplexobj(xs):
                return vind_spin(xs.real) + 1j * vind_spin(xs.imag)
            return vind_spin(xs)

        def vind(zs):
            zs = cp.asarray(zs)
            if zs.ndim == 1:
                zs = zs[None, :]
            if zs.shape[1] != dimension:
                raise ValueError(f'Trial vectors have length {zs.shape[1]}, expected {dimension}')

            nvec = len(zs)
            amplitudes = zs[:, offset:].reshape(nvec, 4, nocc, nvir)
            result = cp.zeros((nvec, dimension), dtype=cp.result_type(zs, self.soc_mo))
            out = result[:, offset:].reshape(nvec, 4, nocc, nvir)

            singlet_input = amplitudes[:, 0].reshape(nvec, nov)
            out[:, 0] = apply_spin_free(vind_s, singlet_input).reshape(nvec, nocc, nvir)
            triplets = amplitudes[:, 1:].reshape(3 * nvec, nov)
            out[:, 1:] = apply_spin_free(vind_t, triplets).reshape(nvec, 3, nocc, nvir)

            singlet, triplet_m1, triplet_0, triplet_p1 = amplitudes.transpose(1, 0, 2, 3)
            d_minus_s = self._apply_one_body(z_minus_vv, z_minus_oo, singlet, -1)
            d_z_s = self._apply_one_body(z_z_vv, z_z_oo, singlet, -1)
            d_plus_s = self._apply_one_body(z_plus_vv, z_plus_oo, singlet, -1)
            q_minus_m1 = self._apply_one_body(z_minus_vv, z_minus_oo, triplet_m1, 1)
            q_minus_0 = self._apply_one_body(z_minus_vv, z_minus_oo, triplet_0, 1)
            q_plus_0 = self._apply_one_body(z_plus_vv, z_plus_oo, triplet_0, 1)
            q_plus_p1 = self._apply_one_body(z_plus_vv, z_plus_oo, triplet_p1, 1)
            q_z_m1 = self._apply_one_body(z_z_vv, z_z_oo, triplet_m1, 1)
            q_z_p1 = self._apply_one_body(z_z_vv, z_z_oo, triplet_p1, 1)

            out[:, 0] += (
                self._apply_one_body(z_minus_vv, z_minus_oo, triplet_m1, -1) / cp.sqrt(8.0)
                + self._apply_one_body(z_z_vv, z_z_oo, triplet_0, -1) / 2
                - self._apply_one_body(z_plus_vv, z_plus_oo, triplet_p1, -1) / cp.sqrt(8.0)
            )
            out[:, 1] += d_plus_s / cp.sqrt(8.0) - q_z_m1 / 2 + q_plus_0 / cp.sqrt(8.0)
            out[:, 2] += d_z_s / 2 + q_minus_m1 / cp.sqrt(8.0) + q_plus_p1 / cp.sqrt(8.0)
            out[:, 3] += -d_minus_s / cp.sqrt(8.0) + q_minus_0 / cp.sqrt(8.0) + q_z_p1 / 2

            if self.include_reference:
                result[:, 0] = contract('p,xp->x', reference_coupling.conj(), zs[:, 1:])
                result[:, 1:] += zs[:, :1] * reference_coupling
            return result

        hdiag = cp.concatenate((hdiag_s, hdiag_t, hdiag_t, hdiag_t)).astype(cp.complex128)
        q_z_diag = (
            z_z_vv.diagonal()[None, :] + z_z_oo.diagonal()[:, None]
        ).reshape(-1)
        hdiag[nov:2 * nov] -= q_z_diag / 2
        hdiag[3 * nov:4 * nov] += q_z_diag / 2
        if self.include_reference:
            hdiag = cp.concatenate((cp.zeros(1, dtype=cp.complex128), hdiag))
        return vind, hdiag

    def get_init_guess(self, hdiag, nstates=None):
        """Return GPU Koopmans guesses, including cutoff degeneracies."""
        if nstates is None:
            nstates = self.nstates
        nstates = min(nstates, hdiag.size)
        if nstates <= 0:
            raise ValueError('nstates must be positive')
        nguess = min(nstates + 3, hdiag.size)
        order = cp.argsort(hdiag.real)
        threshold = hdiag[order[nguess - 1]].real + self.deg_eia_thresh
        idx = order[hdiag[order].real <= threshold]
        x0 = cp.zeros((idx.size, hdiag.size), dtype=cp.complex128)
        x0[cp.arange(idx.size), idx] = 1
        return x0

    def kernel(self, x0=None, nstates=None):
        """Diagonalize the complex Hermitian SOC-TDA matrix on the GPU."""
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        if self._scf.mo_energy is None:
            self._scf.run()
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        vind, hdiag = self.gen_vind()
        nstates = min(nstates, hdiag.size)
        precond = self.get_precond(hdiag)

        def all_eigs(w, v, nroots, envs):
            return w, v, cp.arange(w.size)

        if x0 is None:
            x0 = self.get_init_guess(hdiag, nstates)
        elif isinstance(x0, list):
            x0 = cp.stack([cp.asarray(x).ravel() for x, _ in x0])

        self.converged, self.e, x1 = lr_eigh(
            vind,
            x0,
            precond,
            tol_residual=self.conv_tol,
            lindep=self.lindep,
            nroots=nstates,
            x0sym=None,
            pick=all_eigs,
            max_cycle=self.max_cycle,
            max_memory=self.max_memory,
            verbose=log,
        )
        self.xy = [(xi, 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(
                self.chkfile,
                'tddft/xy',
                [(cp.asnumpy(x), y) for x, y in self.xy],
            )

        log.timer('SOTDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

    def to_cpu(self):
        raise NotImplementedError('GPU SOTDDFT to_cpu is not implemented')


scf.hf.RHF.SOTDDFT = lib.class_as_method(SOTDDFT)


__all__ = ['SOTDDFT']
