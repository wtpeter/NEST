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

"""Direct spin-orbit-coupled noncollinear-tensor TDA.

The trial vector contains one copy of each selected NTTDA configuration space
for every spin projection. Blocks are ordered by ``deltaS=-1, 0, +1`` and,
within each branch, by ascending ``M_S``. The spin-free part is exactly the
corresponding NTTDA matrix-vector product. SOC is applied directly between
configuration amplitudes instead of between converged spin-free roots.
"""

import warnings

import numpy as np
from pyscf import dft, lib
from pyscf.data import nist
from pyscf.lib import logger

from nest._lr_eig import eigh as lr_eigh
from nest.nttda.nttda import (
    MO_BASE,
    NTTDA,
    _get_ab_matrices,
    _orbital_indices,
    _sc_vector_slices,
)
from nest.soc.soc import clebsch_gordan_rank1
from nest.soc.soc_ao import get_ao_soc


class SONTTDA(NTTDA):
    """TDA with SOC included directly in the NTTDA configuration space.

    ``xy[root]`` is ``(x, 0)``, where ``x`` is a one-dimensional complex
    vector containing the selected ``(deltaS, M_S)`` blocks.
    """

    deltaS = (-1, 0, 1)
    soctype = 'SOMF'

    _keys = NTTDA._keys | {
        'soctype', 'soc_ao', 'soc_mo', 'block_slices', 'm_values', 's2',
    }

    def __init__(self, mf, deltaS=(-1, 0, 1), soctype='SOMF', frozen=None):
        super().__init__(mf, frozen=frozen)
        self.deltaS = deltaS
        self.soctype = soctype
        self.soc_ao = None
        self.soc_mo = None
        self.block_slices = None
        self.m_values = None
        self._active_delta_s = None
        self._physical_dimension = None
        self._deltam1_shape = None
        self.s2 = None

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('deltaS = %s', self.deltaS)
        log.info('soctype = %s', self.soctype)
        return self

    def check_sanity(self):
        super().check_sanity()
        mo_occ = np.asarray(self._scf.mo_occ)
        mo_coeff = np.asarray(self._scf.mo_coeff)
        if mo_occ.ndim != 1 or mo_coeff.ndim != 2:
            raise ValueError('SONTTDA requires a spin-restricted ROKS reference')
        if not np.isrealobj(mo_coeff):
            raise ValueError('SONTTDA requires real MO coefficients')
        if self.wfnsym is not None:
            raise NotImplementedError('SONTTDA does not support wfnsym because SOC mixes spatial irreps')
        return self

    def _selected_delta_s(self, reference_spin):
        values = (self.deltaS,) if np.isscalar(self.deltaS) else tuple(self.deltaS)
        invalid = [value for value in values if value not in (-1, 0, 1)]
        if invalid:
            raise ValueError(f'deltaS entries should be -1, 0, or 1; got {invalid}')
        active = []
        for delta_s in (-1, 0, 1):
            if delta_s not in values:
                continue
            if reference_spin + delta_s < 0:
                warnings.warn(
                    f'Skipping deltaS={delta_s} because the final spin would be negative',
                    stacklevel=3,
                )
                continue
            active.append(delta_s)
        if not active:
            raise ValueError('No physical deltaS branch was selected')
        return tuple(active)

    @staticmethod
    def _apply_spin_free(vind, xs):
        """Apply a real NTTDA response to a batched real or complex input."""
        components = (xs.real, xs.imag) if np.iscomplexobj(xs) else (xs,)
        active = [np.any(component != 0, axis=1) for component in components]
        packed = [component[mask] for component, mask in zip(components, active) if np.any(mask)]
        result = np.zeros_like(xs)
        if not packed:
            return result

        values = vind(np.concatenate(packed, axis=0))
        start = 0
        for component_id, mask in enumerate(active):
            count = np.count_nonzero(mask)
            if count:
                if component_id == 0:
                    result.real[mask] = values[start:start + count]
                else:
                    result.imag[mask] = values[start:start + count]
                start += count
        return result

    @staticmethod
    def _split_delta0(xs, nc, no, nv):
        """Unpack ``CO(1), CV(1), OO(1), OV(1), CV(0)`` amplitudes."""
        slices = _sc_vector_slices(nc, no, nv)
        return (
            xs[:, slices['OO(1)']].reshape(-1),
            xs[:, slices['CO(1)']].reshape(-1, nc, no),
            xs[:, slices['CV(1)']].reshape(-1, nc, nv),
            xs[:, slices['OV(1)']].reshape(-1, no, nv),
            xs[:, slices['CV(0)']].reshape(-1, nc, nv),
        )

    @staticmethod
    def _join_delta0(blocks):
        oo, co, cv, ov, cv0 = blocks
        nvec = len(oo)
        return np.concatenate((
            co.reshape(nvec, -1), cv.reshape(nvec, -1), oo[:, None],
            ov.reshape(nvec, -1), cv0.reshape(nvec, -1),
        ), axis=1)

    @staticmethod
    def _split_deltam1(xs, nc, no, nv):
        """Unpack ``CO, CV, OO, OV``; stored OO orientation is ``[w, v]``."""
        full = xs.reshape(-1, nc + no, no + nv)
        return full[:, :nc, :no], full[:, :nc, no:], full[:, nc:, :no], full[:, nc:, no:]

    @staticmethod
    def _join_deltam1(blocks):
        co, cv, oo, ov = blocks
        return np.concatenate((
            np.concatenate((co, cv), axis=2),
            np.concatenate((oo, ov), axis=2),
        ), axis=1).reshape(len(co), -1)

    @staticmethod
    def _zeros_delta0(nvec, nc, no, nv, dtype):
        return (
            np.zeros(nvec, dtype=dtype),
            np.zeros((nvec, nc, no), dtype=dtype),
            np.zeros((nvec, nc, nv), dtype=dtype),
            np.zeros((nvec, no, nv), dtype=dtype),
            np.zeros((nvec, nc, nv), dtype=dtype),
        )

    @staticmethod
    def _zeros_deltam1(nvec, nc, no, nv, dtype):
        return (
            np.zeros((nvec, nc, no), dtype=dtype),
            np.zeros((nvec, nc, nv), dtype=dtype),
            np.zeros((nvec, no, no), dtype=dtype),
            np.zeros((nvec, no, nv), dtype=dtype),
        )

    @staticmethod
    def _apply_p1_p1(z, ket):
        factor = 1 / np.sqrt(2.0)
        return factor * (
            lib.einsum('ji,xja->xia', z['cc'], ket)
            + lib.einsum('ab,xib->xia', z['vv'], ket)
        )

    @staticmethod
    def _apply_p1_0(z, ket, nc, no, nv):
        k_oo, k_co, k_cv, k_ov, k_cv0 = SONTTDA._split_delta0(ket, nc, no, nv)
        s = no * 0.5
        f_cvcv = np.sqrt(s / (2.0 * (s + 1.0)))
        factor = 1 / np.sqrt(2.0)
        out = -k_oo[:, None, None] * z['vc'].T[None]
        out += lib.einsum('av,xiv->xia', z['vo'], k_co)
        out += lib.einsum('vi,xva->xia', z['oc'], k_ov)
        out += f_cvcv * (
            lib.einsum('ji,xja->xia', z['cc'], k_cv)
            + lib.einsum('ab,xib->xia', z['vv'], k_cv)
        )
        out += factor * (
            -lib.einsum('ji,xja->xia', z['cc'], k_cv0)
            + lib.einsum('ab,xib->xia', z['vv'], k_cv0)
        )
        return out

    @staticmethod
    def _apply_p1_0_adjoint(z, higher, nc, no, nv):
        s = no * 0.5
        f_cvcv = np.sqrt(s / (2.0 * (s + 1.0)))
        factor = 1 / np.sqrt(2.0)
        oo, co, cv, ov, cv0 = SONTTDA._zeros_delta0(
            len(higher), nc, no, nv, np.result_type(z['cc'], higher),
        )
        oo -= lib.einsum('ai,xia->x', z['vc'], higher)
        co += lib.einsum('av,xia->xiv', z['vo'], higher)
        ov += lib.einsum('vi,xia->xva', z['oc'], higher)
        cv += f_cvcv * (
            lib.einsum('ji,xia->xja', z['cc'], higher)
            + lib.einsum('ab,xia->xib', z['vv'], higher)
        )
        cv0 += factor * (
            -lib.einsum('ji,xia->xja', z['cc'], higher)
            + lib.einsum('ab,xia->xib', z['vv'], higher)
        )
        return SONTTDA._join_delta0((oo, co, cv, ov, cv0))

    @staticmethod
    def _apply_0_0(z, ket, nc, no, nv):
        k_oo, k_co, k_cv, k_ov, k_cv0 = SONTTDA._split_delta0(ket, nc, no, nv)
        s = no * 0.5
        rt2 = np.sqrt(2.0)
        f_oo_cv = np.sqrt(s / (s + 1.0))
        f_mix = -(s - 1.0) / (2.0 * np.sqrt(s * (s + 1.0)))
        f_cvcv = 1.0 / (rt2 * (s + 1.0))
        f_cvcv0 = np.sqrt(s / (2.0 * (s + 1.0)))
        oo, co, cv, ov, cv0 = SONTTDA._zeros_delta0(
            len(ket), nc, no, nv, np.result_type(z['cc'], ket),
        )
        oo += lib.einsum('iu,xiu->x', z['co'], k_co) / rt2
        oo += lib.einsum('ub,xub->x', z['ov'], k_ov) / rt2
        oo += f_oo_cv * lib.einsum('ib,xib->x', z['cv'], k_cv)
        co += k_oo[:, None, None] * z['oc'].T[None] / rt2
        co += lib.einsum('ji,xju->xiu', z['cc'], k_co) / rt2
        co -= lib.einsum('uv,xiv->xiu', z['oo'], k_co) / rt2
        co += f_mix * lib.einsum('ub,xib->xiu', z['ov'], k_cv)
        co -= 0.5 * lib.einsum('ub,xib->xiu', z['ov'], k_cv0)
        ov += k_oo[:, None, None] * z['vo'].T[None] / rt2
        ov += lib.einsum('ab,xub->xua', z['vv'], k_ov) / rt2
        ov -= lib.einsum('vu,xva->xua', z['oo'], k_ov) / rt2
        ov += f_mix * lib.einsum('ju,xja->xua', z['co'], k_cv)
        ov += 0.5 * lib.einsum('ju,xja->xua', z['co'], k_cv0)
        cv += f_oo_cv * k_oo[:, None, None] * z['vc'].T[None]
        cv += f_mix * lib.einsum('av,xiv->xia', z['vo'], k_co)
        cv += f_mix * lib.einsum('vi,xva->xia', z['oc'], k_ov)
        cv += f_cvcv * (
            lib.einsum('ji,xja->xia', z['cc'], k_cv)
            + lib.einsum('ab,xib->xia', z['vv'], k_cv)
        )
        cv += f_cvcv0 * (
            lib.einsum('ji,xja->xia', z['cc'], k_cv0)
            - lib.einsum('ab,xib->xia', z['vv'], k_cv0)
        )
        cv0 -= 0.5 * lib.einsum('av,xiv->xia', z['vo'], k_co)
        cv0 += 0.5 * lib.einsum('vi,xva->xia', z['oc'], k_ov)
        cv0 += f_cvcv0 * (
            lib.einsum('ji,xja->xia', z['cc'], k_cv)
            - lib.einsum('ab,xib->xia', z['vv'], k_cv)
        )
        return SONTTDA._join_delta0((oo, co, cv, ov, cv0))

    @staticmethod
    def _apply_0_m1(z, ket, nc, no, nv):
        """Apply the reduced ``<S|H_SO|S-1>`` configuration block."""
        k_co, k_cv, k_oo, k_ov = SONTTDA._split_deltam1(ket, nc, no, nv)
        s = no * 0.5
        f1 = np.sqrt((2.0 * s - 1.0) / (2.0 * s))
        f2 = np.sqrt((2.0 * s - 1.0) / (2.0 * s + 1.0))
        f_co_cv = f2 / (2.0 * s)
        f_cv_oo = 1.0 / np.sqrt(2.0 * s * (s + 1.0))
        f_cv_co = np.sqrt((s + 1.0) * (2.0 * s - 1.0)) / (2.0 * s)
        f_cv_cv = np.sqrt(
            (s + 1.0) * (2.0 * s - 1.0) / (2.0 * s * (2.0 * s + 1.0)),
        )
        f_cv0 = np.sqrt((2.0 * s - 1.0) / (4.0 * s))
        f_cv0_cv = np.sqrt((2.0 * s - 1.0) / (2.0 * (2.0 * s + 1.0)))
        oo, co, cv, ov, cv0 = SONTTDA._zeros_delta0(
            len(ket), nc, no, nv, np.result_type(z['cc'], ket),
        )
        oo += lib.einsum('uv,xuv->x', z['oo'], k_oo)
        oo += f1 * lib.einsum('iv,xiv->x', z['co'], k_co)
        oo += f1 * lib.einsum('vb,xvb->x', z['ov'], k_ov)
        oo += f2 * lib.einsum('ib,xib->x', z['cv'], k_cv)
        co += lib.einsum('wi,xwu->xiu', z['oc'], k_oo)
        co += f1 * lib.einsum('ji,xju->xiu', z['cc'], k_co)
        co += lib.einsum('uv,xiv->xiu', z['oo'], k_co) / np.sqrt(
            2.0 * s * (2.0 * s - 1.0),
        )
        co += f_co_cv * lib.einsum('ub,xib->xiu', z['ov'], k_cv)
        ov += lib.einsum('av,xuv->xua', z['vo'], k_oo)
        ov += f1 * lib.einsum('ab,xub->xua', z['vv'], k_ov)
        ov += lib.einsum('vu,xva->xua', z['oo'], k_ov) / np.sqrt(
            2.0 * s * (2.0 * s - 1.0),
        )
        ov += f_co_cv * lib.einsum('ju,xja->xua', z['co'], k_cv)
        trace_oo = np.trace(k_oo, axis1=1, axis2=2)
        cv += f_cv_oo * trace_oo[:, None, None] * z['vc'].T[None]
        cv += f_cv_co * lib.einsum('av,xiv->xia', z['vo'], k_co)
        cv += f_cv_co * lib.einsum('vi,xva->xia', z['oc'], k_ov)
        cv += f_cv_cv * (
            lib.einsum('ji,xja->xia', z['cc'], k_cv)
            + lib.einsum('ab,xib->xia', z['vv'], k_cv)
        )
        cv0 -= f_cv0 * lib.einsum('av,xiv->xia', z['vo'], k_co)
        cv0 += f_cv0 * lib.einsum('vi,xva->xia', z['oc'], k_ov)
        cv0 += f_cv0_cv * (
            lib.einsum('ji,xja->xia', z['cc'], k_cv)
            - lib.einsum('ab,xib->xia', z['vv'], k_cv)
        )
        return SONTTDA._join_delta0((oo, co, cv, ov, cv0))

    @staticmethod
    def _apply_0_m1_adjoint(z, higher, nc, no, nv):
        """Apply the Hermitian adjoint of the reduced ``S,S-1`` block."""
        b_oo, b_co, b_cv, b_ov, b_cv0 = SONTTDA._split_delta0(higher, nc, no, nv)
        s = no * 0.5
        f1 = np.sqrt((2.0 * s - 1.0) / (2.0 * s))
        f2 = np.sqrt((2.0 * s - 1.0) / (2.0 * s + 1.0))
        f_co_cv = f2 / (2.0 * s)
        f_cv_oo = 1.0 / np.sqrt(2.0 * s * (s + 1.0))
        f_cv_co = np.sqrt((s + 1.0) * (2.0 * s - 1.0)) / (2.0 * s)
        f_cv_cv = np.sqrt(
            (s + 1.0) * (2.0 * s - 1.0) / (2.0 * s * (2.0 * s + 1.0)),
        )
        f_cv0 = np.sqrt((2.0 * s - 1.0) / (4.0 * s))
        f_cv0_cv = np.sqrt((2.0 * s - 1.0) / (2.0 * (2.0 * s + 1.0)))
        co, cv, oo, ov = SONTTDA._zeros_deltam1(
            len(higher), nc, no, nv, np.result_type(z['cc'], higher),
        )
        oo += b_oo[:, None, None] * z['oo'][None]
        oo += lib.einsum('wi,xiu->xwu', z['oc'], b_co)
        oo += lib.einsum('av,xua->xuv', z['vo'], b_ov)
        trace_term = f_cv_oo * lib.einsum('ai,xia->x', z['vc'], b_cv)
        diagonal = np.arange(no)
        oo[:, diagonal, diagonal] += trace_term[:, None]
        co += f1 * b_oo[:, None, None] * z['co'][None]
        co += f1 * lib.einsum('ji,xiu->xju', z['cc'], b_co)
        co += lib.einsum('uv,xiu->xiv', z['oo'], b_co) / np.sqrt(
            2.0 * s * (2.0 * s - 1.0),
        )
        co += f_cv_co * lib.einsum('av,xia->xiv', z['vo'], b_cv)
        co -= f_cv0 * lib.einsum('av,xia->xiv', z['vo'], b_cv0)
        ov += f1 * b_oo[:, None, None] * z['ov'][None]
        ov += f1 * lib.einsum('ab,xua->xub', z['vv'], b_ov)
        ov += lib.einsum('vu,xua->xva', z['oo'], b_ov) / np.sqrt(
            2.0 * s * (2.0 * s - 1.0),
        )
        ov += f_cv_co * lib.einsum('vi,xia->xva', z['oc'], b_cv)
        ov += f_cv0 * lib.einsum('vi,xia->xva', z['oc'], b_cv0)
        cv += f2 * b_oo[:, None, None] * z['cv'][None]
        cv += f_co_cv * lib.einsum('ub,xiu->xib', z['ov'], b_co)
        cv += f_co_cv * lib.einsum('ju,xua->xja', z['co'], b_ov)
        cv += f_cv_cv * (
            lib.einsum('ji,xia->xja', z['cc'], b_cv)
            + lib.einsum('ab,xia->xib', z['vv'], b_cv)
        )
        cv += f_cv0_cv * (
            lib.einsum('ji,xia->xja', z['cc'], b_cv0)
            - lib.einsum('ab,xia->xib', z['vv'], b_cv0)
        )
        return SONTTDA._join_deltam1((co, cv, oo, ov))

    @staticmethod
    def _apply_m1_m1(z, ket, nc, no, nv):
        k_co, k_cv, k_oo, k_ov = SONTTDA._split_deltam1(ket, nc, no, nv)
        s = no * 0.5
        rt2 = np.sqrt(2.0)
        c_oo_mix = -(s - 1.0) / np.sqrt(s * (2.0 * s - 1.0))
        c_oo_tr = -1.0 / (2.0 * np.sqrt(s * (2.0 * s - 1.0)))
        c_oo_cv = -np.sqrt((2.0 * s - 1.0) / (2.0 * s + 1.0)) / (rt2 * s)
        c_coco_1 = -((s - 1.0) * (2.0 * s + 1.0)) / (rt2 * s * (2.0 * s - 1.0))
        c_coco_2 = -(s - 1.0) / (rt2 * s)
        c_cv_mix = -((s - 1.0) / (2.0 * s)) * np.sqrt((2.0 * s + 1.0) / s)
        c_cvcv = -(s - 1.0) / (rt2 * s)
        co, cv, oo, ov = SONTTDA._zeros_deltam1(
            len(ket), nc, no, nv, np.result_type(z['cc'], ket),
        )
        trace_oo = np.trace(k_oo, axis1=1, axis2=2)
        oo -= lib.einsum('wu,xwt->xut', z['oo'], k_oo) / rt2
        oo -= lib.einsum('tv,xuv->xut', z['oo'], k_oo) / rt2
        oo += c_oo_mix * lib.einsum('ju,xjt->xut', z['co'], k_co)
        oo += c_oo_mix * lib.einsum('tb,xub->xut', z['ov'], k_ov)
        trace_term = c_oo_tr * (
            lib.einsum('ju,xju->x', z['co'], k_co)
            + lib.einsum('ub,xub->x', z['ov'], k_ov)
        )
        trace_term += c_oo_cv * lib.einsum('ib,xib->x', z['cv'], k_cv)
        diagonal = np.arange(no)
        oo[:, diagonal, diagonal] += trace_term[:, None]
        co += c_oo_mix * lib.einsum('wi,xwu->xiu', z['oc'], k_oo)
        co += c_oo_tr * trace_oo[:, None, None] * z['oc'].T[None]
        co += c_coco_1 * lib.einsum('uv,xiv->xiu', z['oo'], k_co)
        co += c_coco_2 * lib.einsum('ji,xju->xiu', z['cc'], k_co)
        co += c_cv_mix * lib.einsum('ub,xib->xiu', z['ov'], k_cv)
        ov += c_oo_mix * lib.einsum('av,xuv->xua', z['vo'], k_oo)
        ov += c_oo_tr * trace_oo[:, None, None] * z['vo'].T[None]
        ov += c_coco_1 * lib.einsum('vu,xva->xua', z['oo'], k_ov)
        ov += c_coco_2 * lib.einsum('ab,xub->xua', z['vv'], k_ov)
        ov += c_cv_mix * lib.einsum('ju,xja->xua', z['co'], k_cv)
        cv += c_oo_cv * trace_oo[:, None, None] * z['vc'].T[None]
        cv += c_cv_mix * lib.einsum('av,xiv->xia', z['vo'], k_co)
        cv += c_cv_mix * lib.einsum('vi,xva->xia', z['oc'], k_ov)
        cv += c_cvcv * (
            lib.einsum('ji,xja->xia', z['cc'], k_cv)
            + lib.einsum('ab,xib->xia', z['vv'], k_cv)
        )
        return SONTTDA._join_deltam1((co, cv, oo, ov))

    def _project_deltam1(self, vectors):
        """Remove the redundant ``S_-|reference>`` trace from every -1 block."""
        if -1 not in self._active_delta_s:
            return vectors
        nc, no, nv = self._deltam1_shape
        projected = np.array(vectors, copy=True)
        was_vector = projected.ndim == 1
        if was_vector:
            projected = projected[None]
        for block_slice in self.block_slices[-1]:
            block = projected[:, block_slice].reshape(-1, nc + no, no + nv)
            oo = block[:, nc:, :no]
            trace = np.trace(oo, axis1=1, axis2=2) / no
            diagonal = np.arange(no)
            oo[:, diagonal, diagonal] -= trace[:, None]
        return projected[0] if was_vector else projected

    @staticmethod
    def _soc_component(soc_blocks, q):
        if q == 1:
            return {name: -value[0] for name, value in soc_blocks.items()}
        if q == 0:
            return {name: value[1] for name, value in soc_blocks.items()}
        return {name: -value[2] for name, value in soc_blocks.items()}

    def _apply_same_spin_soc(self, delta_s, inputs, outputs, soc_blocks, nc, no, nv):
        spin = no * 0.5 + delta_s
        if spin == 0:
            return
        reference_cg = clebsch_gordan_rank1(spin, spin, 0, spin, spin)
        for bra_id, m_bra in enumerate(self.m_values[delta_s]):
            for ket_id, m_ket in enumerate(self.m_values[delta_s]):
                q = int(round(m_bra - m_ket))
                if q not in (-1, 0, 1) or abs(m_bra - m_ket - q) > 1e-12:
                    continue
                coefficient = clebsch_gordan_rank1(spin, m_ket, q, spin, m_bra)
                if coefficient == 0:
                    continue
                z = self._soc_component(soc_blocks, q)
                z = {name: value * (coefficient / reference_cg) for name, value in z.items()}
                ket = inputs[:, self.block_slices[delta_s][ket_id]]
                if delta_s == -1:
                    contribution = self._apply_m1_m1(z, ket, nc, no, nv)
                elif delta_s == 0:
                    contribution = self._apply_0_0(z, ket, nc, no, nv)
                else:
                    contribution = self._apply_p1_p1(
                        z, ket.reshape(-1, nc, nv),
                    ).reshape(len(ket), -1)
                outputs[:, self.block_slices[delta_s][bra_id]] += contribution

    def _apply_cross_spin_soc(self, higher_delta, inputs, outputs, soc_blocks, nc, no, nv):
        lower_delta = higher_delta - 1
        higher_spin = no * 0.5 + higher_delta
        lower_spin = higher_spin - 1
        reference_cg = clebsch_gordan_rank1(
            lower_spin, lower_spin, 1, higher_spin, higher_spin,
        )
        for high_id, m_high in enumerate(self.m_values[higher_delta]):
            for low_id, m_low in enumerate(self.m_values[lower_delta]):
                q = int(round(m_high - m_low))
                if q not in (-1, 0, 1) or abs(m_high - m_low - q) > 1e-12:
                    continue
                coefficient = clebsch_gordan_rank1(
                    lower_spin, m_low, q, higher_spin, m_high,
                )
                if coefficient == 0:
                    continue
                z = self._soc_component(soc_blocks, q)
                z = {name: value * (coefficient / reference_cg) for name, value in z.items()}
                high_slice = self.block_slices[higher_delta][high_id]
                low_slice = self.block_slices[lower_delta][low_id]
                lower = inputs[:, low_slice]
                higher = inputs[:, high_slice]
                if higher_delta == 1:
                    high_out = self._apply_p1_0(
                        z, lower, nc, no, nv,
                    ).reshape(len(lower), -1)
                    low_out = self._apply_p1_0_adjoint(
                        {name: value.conj() for name, value in z.items()},
                        higher.reshape(-1, nc, nv), nc, no, nv,
                    )
                else:
                    high_out = self._apply_0_m1(z, lower, nc, no, nv)
                    low_out = self._apply_0_m1_adjoint(
                        {name: value.conj() for name, value in z.items()},
                        higher, nc, no, nv,
                    )
                outputs[:, high_slice] += high_out
                outputs[:, low_slice] += low_out

    def _prepare_space(self, mf, caller):
        if mf is not self._scf:
            raise ValueError(f'{caller} must use the SCF object associated with SONTTDA')
        self.check_sanity()

        csidx, osidx, vsidx = _orbital_indices(self)
        nc, no, nv = len(csidx), len(osidx), len(vsidx)
        reference_spin = no * 0.5
        self._active_delta_s = self._selected_delta_s(reference_spin)
        self._deltam1_shape = (nc, no, nv)
        dimensions = {
            -1: (nc + no) * (no + nv),
            0: nc * no + 2 * nc * nv + 1 + no * nv,
            1: nc * nv,
        }

        self.block_slices = {}
        self.m_values = {}
        start = 0
        for delta_s in self._active_delta_s:
            spin = reference_spin + delta_s
            m_values = np.arange(-spin, spin + 0.5, 1.0)
            m_values[abs(m_values) < 1e-12] = 0.0
            self.m_values[delta_s] = m_values
            self.block_slices[delta_s] = []
            for _ in m_values:
                stop = start + dimensions[delta_s]
                self.block_slices[delta_s].append(slice(start, stop))
                start = stop
        self._physical_dimension = start - len(self.m_values.get(-1, ()))

        mo_coeff = np.asarray(mf.mo_coeff)
        self.soc_ao = get_ao_soc(mf, self.soctype)
        self.soc_mo = lib.einsum(
            'up,xuv,vq->xpq', mo_coeff.conj(), self.soc_ao, mo_coeff, optimize=True,
        )
        soc_blocks = {}
        indices = {'c': csidx, 'o': osidx, 'v': vsidx}
        for row_name, rows in indices.items():
            for col_name, cols in indices.items():
                soc_blocks[row_name + col_name] = self.soc_mo[:, rows[:, None], cols]
        return nc, no, nv, dimensions, soc_blocks, start

    def gen_vind(self, mf=None):
        """Generate the matrix-vector product for the direct SO-NTTDA matrix."""
        if mf is None:
            mf = self._scf
        nc, no, nv, dimensions, soc_blocks, dimension = self._prepare_space(
            mf, 'gen_vind',
        )
        spin_free = {}
        for delta_s in self._active_delta_s:
            if delta_s == -1:
                spin_free[delta_s] = self.gen_vind_sfd()
            elif delta_s == 0:
                spin_free[delta_s] = self.gen_vind_sc()
            else:
                spin_free[delta_s] = self.gen_vind_sfu()

        hdiag_parts = []
        for delta_s in self._active_delta_s:
            for _ in self.m_values[delta_s]:
                hdiag_parts.append(np.asarray(spin_free[delta_s][1]))

        def vind(zs):
            zs = np.asarray(zs)
            if zs.ndim == 1:
                zs = zs[None]
            if zs.shape[1] != dimension:
                raise ValueError(f'Trial vectors have length {zs.shape[1]}, expected {dimension}')
            inputs = self._project_deltam1(zs)
            outputs = np.zeros(inputs.shape, dtype=np.result_type(inputs, self.soc_mo))
            for delta_s in self._active_delta_s:
                spin_vind = spin_free[delta_s][0]
                slices = self.block_slices[delta_s]
                branch_inputs = np.stack(
                    [inputs[:, block_slice] for block_slice in slices], axis=1,
                )
                branch_outputs = self._apply_spin_free(
                    spin_vind, branch_inputs.reshape(-1, dimensions[delta_s]),
                ).reshape(len(inputs), len(slices), dimensions[delta_s])
                for block_id, block_slice in enumerate(slices):
                    outputs[:, block_slice] += branch_outputs[:, block_id]
                self._apply_same_spin_soc(
                    delta_s, inputs, outputs, soc_blocks, nc, no, nv,
                )
            for higher_delta in (0, 1):
                if (
                    higher_delta in self._active_delta_s
                    and higher_delta - 1 in self._active_delta_s
                ):
                    self._apply_cross_spin_soc(
                        higher_delta, inputs, outputs, soc_blocks, nc, no, nv,
                    )
            return self._project_deltam1(outputs)

        hdiag = np.concatenate(hdiag_parts).astype(np.complex128)
        return vind, hdiag

    def get_ab(self, mf=None):
        """Return the projected SO-NTTDA A matrix in the ``gen_vind`` basis."""
        if mf is None:
            mf = self._scf
        nc, no, nv, _, soc_blocks, dimension = self._prepare_space(
            mf, 'get_ab',
        )

        matrix = np.zeros(
            (dimension, dimension), dtype=np.complex128, order='F',
        )
        spin_free_matrices = _get_ab_matrices(self, self._active_delta_s, mf)
        for delta_s in self._active_delta_s:
            for block_slice in self.block_slices[delta_s]:
                matrix[block_slice, block_slice] = spin_free_matrices[delta_s]

        # The SOC routines consume trial vectors by row.  Apply columns of the
        # identity in small batches; a full identity and output would each be
        # as large as the final dense Hamiltonian.
        batch_size = 128
        for start in range(0, dimension, batch_size):
            stop = min(start + batch_size, dimension)
            inputs = np.zeros((stop - start, dimension), dtype=np.complex128)
            inputs[np.arange(stop - start), np.arange(start, stop)] = 1.0
            outputs = np.zeros_like(inputs)
            for delta_s in self._active_delta_s:
                self._apply_same_spin_soc(
                    delta_s, inputs, outputs, soc_blocks, nc, no, nv,
                )
            for higher_delta in (0, 1):
                if (
                    higher_delta in self._active_delta_s
                    and higher_delta - 1 in self._active_delta_s
                ):
                    self._apply_cross_spin_soc(
                        higher_delta, inputs, outputs, soc_blocks, nc, no, nv,
                    )
            matrix[:, start:stop] += outputs.T

        # deltaS=-1 contains one redundant trace direction per M_S block.
        # gen_vind applies this orthogonal projector to both input and output,
        # so the explicit operator is P A P and works for arbitrary raw input.
        if -1 in self._active_delta_s:
            width = no + nv
            local_trace = (nc + np.arange(no)) * width + np.arange(no)
            for block_slice in self.block_slices[-1]:
                trace = block_slice.start + local_trace
                matrix[:, trace] -= matrix[:, trace].sum(axis=1)[:, None] / no
                matrix[trace] -= matrix[trace].sum(axis=0)[None] / no
        return matrix

    def get_init_guess(self, hdiag, nstates=None):
        """Build guesses spanning only the physical, trace-free -1 space."""
        if nstates is None:
            nstates = self.nstates
        n_init = min(nstates + 3, self._physical_dimension)
        guesses = []
        for index in np.argsort(hdiag.real):
            guess = np.zeros(hdiag.size, dtype=np.complex128)
            guess[index] = 1
            guess = self._project_deltam1(guess)
            for previous in guesses:
                guess -= np.vdot(previous, guess) * previous
            norm = np.linalg.norm(guess)
            if norm > self.lindep:
                guesses.append(guess / norm)
            if len(guesses) == n_init:
                break
        if len(guesses) < n_init:
            raise RuntimeError('Failed to span the physical SONTTDA trial space')
        return np.asarray(guesses)

    def spin_square(self):
        """Return ``<S^2>`` for every spin-orbit-coupled root."""
        if self.xy is None or self.block_slices is None:
            raise RuntimeError('Run kernel() before spin_square()')
        reference_spin = 0.5 * self._deltam1_shape[1]
        values = []
        for amplitudes, _ in self.xy:
            amplitudes = np.asarray(amplitudes).reshape(-1)
            spin_square = 0.0
            for delta_s, slices in self.block_slices.items():
                spin = reference_spin + delta_s
                weight = sum(
                    np.vdot(amplitudes[block_slice], amplitudes[block_slice]).real
                    for block_slice in slices
                )
                spin_square += spin * (spin + 1.0) * weight
            values.append(spin_square)
        self.s2 = np.asarray(values)
        return self.s2

    @staticmethod
    def _significant_amplitudes(delta_s, amplitudes, csidx, osidx, vsidx):
        """Yield ``(block, transition, amplitude)`` entries above 0.1."""
        nc, no, nv = len(csidx), len(osidx), len(vsidx)
        if delta_s == -1:
            co, cv, oo, ov = SONTTDA._split_deltam1(
                amplitudes[None], nc, no, nv,
            )
            blocks = (
                ('CO(1)', co[0], csidx, osidx),
                ('CV(1)', cv[0], csidx, vsidx),
                ('OO(1)', oo[0], osidx, osidx),
                ('OV(1)', ov[0], osidx, vsidx),
            )
        elif delta_s == 0:
            oo, co, cv, ov, cv0 = SONTTDA._split_delta0(
                amplitudes[None], nc, no, nv,
            )
            blocks = (
                ('CO(1)', co[0], csidx, osidx),
                ('CV(1)', cv[0], csidx, vsidx),
                ('OO(1)', oo[0], None, None),
                ('OV(1)', ov[0], osidx, vsidx),
                ('CV(0)', cv0[0], csidx, vsidx),
            )
        else:
            cv = amplitudes.reshape(nc, nv)
            blocks = (('CV(1)', cv, csidx, vsidx),)

        for label, block, source, target in blocks:
            if source is None:
                if abs(block) > 0.1:
                    yield label, 'reference', block
                continue
            for row, col in zip(*np.where(abs(block) > 0.1)):
                transition = (
                    f'{source[row] + MO_BASE:4d} -> '
                    f'{target[col] + MO_BASE:4d}'
                )
                yield label, transition, block[row, col]

    def analyze(self, verbose=None):
        """Print spin character and significant complex excitation amplitudes."""
        if self.e is None or self.xy is None:
            self.kernel()
        log = logger.new_logger(self, verbose)
        csidx, osidx, vsidx = _orbital_indices(self)
        reference_spin = 0.5 * len(osidx)
        spin_square = self.spin_square()
        origin = np.min(np.asarray(self.e).real)

        log.note('Spin-orbit-coupled NTTDA states')
        for root, (energy, (amplitudes, _)) in enumerate(zip(self.e, self.xy), 1):
            log.note(
                'State %3d:  Delta E=%12.3f cm^-1 (%10.6f eV)  <S^2>=%8.5f',
                root,
                (energy.real - origin) * nist.HARTREE2WAVENUMBER,
                (energy.real - origin) * nist.HARTREE2EV,
                spin_square[root - 1],
            )
            amplitudes = np.asarray(amplitudes).reshape(-1)
            printed = False
            for delta_s, slices in self.block_slices.items():
                spin = reference_spin + delta_s
                for m_s, block_slice in zip(self.m_values[delta_s], slices):
                    block = amplitudes[block_slice]
                    entries = list(self._significant_amplitudes(
                        delta_s, block, csidx, osidx, vsidx,
                    ))
                    if not entries:
                        continue
                    printed = True
                    weight = np.vdot(block, block).real
                    log.info(
                        '  Partition: deltaS=%+d  S=%4.1f  M_S=%5.1f  '
                        'sum|X|^2=%10.6f',
                        delta_s, spin, m_s, weight,
                    )
                    for label, transition, amplitude in entries:
                        log.info(
                            '    %-6s %-13s X=% .6f%+.6fj  |X|^2=%10.6f',
                            label, transition, amplitude.real, amplitude.imag,
                            abs(amplitude) ** 2,
                        )
            if not printed:
                log.info('  No amplitudes with |X| > 0.1')
        return self

    def kernel(self, x0=None, nstates=None):
        """Diagonalize the complex Hermitian direct SO-NTTDA matrix."""
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.s2 = None
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)
        vind, hdiag = self.gen_vind()
        nstates = min(nstates, self._physical_dimension)
        base_precond = self.get_precond(hdiag)

        def precond(residual, energy):
            return self._project_deltam1(base_precond(residual, energy))

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        if x0 is None:
            x0 = self.get_init_guess(hdiag, nstates)
        else:
            x0 = self._project_deltam1(np.asarray(x0))
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
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)
        log.timer('SONTTDA', *cpu0)
        self._finalize()
        return self.e, self.xy


dft.roks.ROKS.SONTTDA = lib.class_as_method(SONTTDA)
dft.rks_symm.SymAdaptedROKS.SONTTDA = lib.class_as_method(SONTTDA)


__all__ = ['SONTTDA']
