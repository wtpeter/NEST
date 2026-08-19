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
#
# Noncollinear-Tensor TDA
#
# Author: Tai Wang <wtpeter@pku.edu.cn>
#

import cupy as cp
import numpy as np
from pyscf import ao2mo, gto, lib
from gpu4pyscf import dft
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import add_sparse, contract, tag_array
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.tdscf.rhf import TDA


def _nr_rks_fxc1(ni, mol, grids, dms, fxc, mgga=False):
    opt = ni.gdftopt
    sorted_mol = opt._sorted_mol
    nao = sorted_mol.nao
    dms = cp.asarray(dms)
    dm_shape = dms.shape
    dms = opt.sort_orbitals(dms.reshape(-1, nao, nao), axis=[1, 2])
    vmat = cp.zeros_like(dms)

    p1 = 0
    for ao, mask, weight, _ in ni.block_loop(
        sorted_mol,
        grids,
        nao,
        1,
        max_memory=None,
        strict_grid_order=True,
    ):
        p0, p1 = p1, p1 + weight.size
        if len(mask) == 0:
            continue

        ao = ao[:4]
        dm = dms[:, mask[:, None], mask]
        c = contract('xqp,iqg->xipg', dm, ao)
        rho = contract('jpg,xipg->xijg', ao, c)
        rho0 = rho[:, 0, 0]
        l_grad = rho[:, 1:4, 0]
        r_grad = rho[:, 0, 1:4]
        tau = rho[:, 1:4, 1:4]
        fxc_w = fxc[:, :, p0:p1] * weight

        u = cp.zeros((len(dms), 4, 4, weight.size))
        u[:, 0, 0] = fxc_w[0, 0] * rho0
        u[:, 0, 0] += contract('ig,xig->xg', fxc_w[1:4, 0], l_grad)
        u[:, 0, 0] += contract('jg,xjg->xg', fxc_w[0, 1:4], r_grad)
        u[:, 0, 0] += contract('ijg,xijg->xg', fxc_w[1:4, 1:4], tau)
        u[:, 1:4, 0] = contract('ig,xg->xig', fxc_w[1:4, 0], rho0)
        u[:, 1:4, 0] += contract('ijg,xjg->xig', fxc_w[1:4, 1:4], r_grad)
        u[:, 0, 1:4] = contract('jg,xg->xjg', fxc_w[0, 1:4], rho0)
        u[:, 0, 1:4] += contract('ijg,xig->xjg', fxc_w[1:4, 1:4], l_grad)
        u[:, 1:4, 1:4] = contract('ijg,xg->xijg', fxc_w[1:4, 1:4], rho0)

        if mgga:
            u[:, 1:4, 0] += 0.5 * contract('g,xig->xig', fxc_w[4, 0], l_grad)
            u[:, 1:4, 0] += 0.5 * contract('jg,xijg->xig', fxc_w[4, 1:4], tau)
            u[:, 0, 1:4] += 0.5 * contract('g,xjg->xjg', fxc_w[0, 4], r_grad)
            u[:, 0, 1:4] += 0.5 * contract('ig,xijg->xjg', fxc_w[1:4, 4], tau)
            u[:, 1:4, 1:4] += 0.5 * contract('ig,xjg->xijg', fxc_w[1:4, 4], r_grad)
            u[:, 1:4, 1:4] += 0.5 * contract('jg,xig->xijg', fxc_w[4, 1:4], l_grad)
            u[:, 1:4, 1:4] += 0.25 * contract('g,xijg->xijg', fxc_w[4, 4], tau)

        aow = contract('xijg,jqg->xiqg', u, ao)
        v_chunk = contract('ipg,xiqg->xpq', ao, aow)
        add_sparse(vmat, v_chunk, mask)

    vmat = opt.unsort_orbitals(vmat, axis=[1, 2])
    return vmat.reshape(dm_shape)


def nr_rks_fxc1_gga(ni, mol, grids, xc_code, dms, fxc, max_memory=2000):
    return _nr_rks_fxc1(ni, mol, grids, dms, fxc)


def nr_rks_fxc1_mgga(ni, mol, grids, xc_code, dms, fxc, max_memory=2000):
    return _nr_rks_fxc1(ni, mol, grids, dms, fxc, mgga=True)


def _factorized_density(left, right, columns=None):
    """Build directed AO densities with reusable low-rank factors."""
    if columns is not None:
        padded = cp.zeros(
            left.shape[:-1] + (right.shape[-1],), dtype=left.dtype,
        )
        padded[..., columns] = left
        left = padded
    density = contract('xpi,qi->xpq', left, right)
    return tag_array(
        density,
        mo1=left,
        # nr_rks_fxc doubles tagged occ_coeff before eval_rho4.
        occ_coeff=right * 0.5,
        factor_l=left,
        factor_r=right,
        symmetrize=0,
    )


def _concatenate_factorized_densities(densities):
    density = cp.concatenate(densities, axis=0)
    if not all(hasattr(item, 'mo1') for item in densities):
        return density
    left = cp.concatenate([item.mo1 for item in densities], axis=0)
    right = densities[0].factor_r
    return tag_array(
        density,
        mo1=left,
        occ_coeff=densities[0].occ_coeff,
        factor_l=left,
        factor_r=right,
        symmetrize=0,
    )


def gen_rohf_response_sfu(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None, log=None):
    '''
    response function for Sf=Si+1 with K^SF_0 = kernel
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    if log is None:
        log = logger.new_logger(mf)
    if not isinstance(mf, dft.roks.ROKS):
        raise TypeError('NTTDA response requires ROKS reference')

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)
    if xctype != 'HF':
        fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
        fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    if mf.do_nlc():
        logger.warn(mf, "NLC contribution in gen_response is NOT included")

    def vind(dms_cv):
        if xctype != 'HF':
            time_xc = (logger.process_clock(), logger.perf_counter())
            v1ao_cv = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms_cv, 0, hermi,
                                    None, None, fxc_ref, max_memory=max_memory)
            time_xc = log.timer('NTTDA response_sfu kernel v1ao_cv', *time_xc)
        else:
            v1ao_cv = cp.zeros_like(dms_cv)

        if hybrid:
            time_jk = (logger.process_clock(), logger.perf_counter())
            vk = mf.get_k(mol, dms_cv, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms_cv, hermi, omega=omega) * (alpha - hyb)
            v1ao_cv -= vk
            log.timer('NTTDA response_sfu kernel get_k total', *time_jk)
        return v1ao_cv

    orbos = mo_coeff[:, cp.where(mo_occ == 1)[0]]
    dmoo = orbos @ orbos.T
    if xctype != 'HF':
        delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, max_memory=max_memory)
    else:
        delta = cp.zeros_like(dmoo)
    if hybrid:
        delta -= mf.get_k(mol, dmoo, 1) * hyb
        if omega != 0:
            delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
    return vind, 0.5 * delta

def gen_rohf_response_sc(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None, log=None):
    '''
    response function for Sf=Si
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    if log is None:
        log = logger.new_logger(mf)
    if not isinstance(mf, dft.roks.ROKS):
        raise TypeError('NTTDA response requires ROKS reference')

    s = (mol.nelec[0] - mol.nelec[1]) * 0.5

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)
    if xctype != 'HF':
        fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
        fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    if mf.do_nlc():
        logger.warn(mf, "NLC contribution in gen_response is NOT included")

    def vind(dms_co, dms_cv, dms_ov, dms_cv0):
        n_co = len(dms_co)
        n_cv = len(dms_cv)
        n_ov = len(dms_ov)
        idx1 = n_co
        idx2 = idx1 + n_cv
        idx3 = idx2 + n_ov

        v1ao_co = cp.zeros_like(dms_co)
        v1ao_cv = cp.zeros_like(dms_cv)
        v1ao_ov = cp.zeros_like(dms_ov)
        v1ao_cv0 = cp.zeros_like(dms_cv0)

        dms0 = _concatenate_factorized_densities(
            (dms_co, dms_cv, dms_ov, dms_cv0),
        )
        dms1 = _concatenate_factorized_densities((dms_co, dms_ov, dms_cv0))

        # kernel part
        if xctype != 'HF':
            time_xc = (logger.process_clock(), logger.perf_counter())
            vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi,
                                  None, None, fxc_ref, max_memory=max_memory)
            time_xc = log.timer('NTTDA response_sc kernel vref0', *time_xc)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi,
                                      None, None, fxc_ref, max_memory=max_memory)
            elif xctype =='GGA':
                vref1 = nr_rks_fxc1_gga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            elif xctype == 'MGGA':
                vref1 = nr_rks_fxc1_mgga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            log.timer('NTTDA response_sc kernel vref1', *time_xc)
        else:
            vref0 = cp.zeros_like(dms0)
            vref1 = cp.zeros_like(dms1)

        if hybrid:
            time_jk = (logger.process_clock(), logger.perf_counter())
            vk = mf.get_k(mol, dms0, hermi) * hyb
            vj = mf.get_j(mol, dms1, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
                with mol.with_range_coulomb(omega):
                    vj += mf.get_j(
                        mol, dms1, hermi, omega=omega
                    ) * (alpha - hyb)
            vref0 -= vk
            vref1 -= vj
            log.timer('NTTDA response_sc kernel get_j/get_k total', *time_jk)

        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_ov = vref0[idx2:idx3]
        vref0_cv0 = vref0[idx3:]
        vref1_co = vref1[:idx1]
        vref1_ov = vref1[idx1:idx1+n_ov]
        vref1_cv0 = vref1[idx1+n_ov:]

        v1ao_co += vref0_co - vref1_co + cp.sqrt((s + 1) / 2 / s) * vref0_cv + vref1_ov
        v1ao_cv += cp.sqrt((s + 1) / 2 / s) * vref0_co + vref0_cv + cp.sqrt((s + 1) / 2 / s) * vref0_ov
        v1ao_ov += vref1_co + cp.sqrt((s + 1) / 2 / s) * vref0_cv - vref1_ov + vref0_ov
        v1ao_co += cp.sqrt(0.5) * vref0_cv0 - cp.sqrt(2.0) * vref1_cv0
        v1ao_ov += -cp.sqrt(0.5) * vref0_cv0 + cp.sqrt(2.0) * vref1_cv0
        v1ao_cv0 += cp.sqrt(0.5) * vref0_co - cp.sqrt(2.0) * vref1_co
        v1ao_cv0 += -cp.sqrt(0.5) * vref0_ov + cp.sqrt(2.0) * vref1_ov
        v1ao_cv0 += vref0_cv0 - 2.0 * vref1_cv0
        return v1ao_co, v1ao_cv, v1ao_ov, v1ao_cv0

    orbos = mo_coeff[:, cp.where(mo_occ == 1)[0]]
    dmoo = orbos @ orbos.T
    if xctype != 'HF':
        delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, max_memory=max_memory)
    else:
        delta = cp.zeros_like(dmoo)
    if hybrid:
        delta -= mf.get_k(mol, dmoo, 1) * hyb
        if omega != 0:
            delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
    return vind, 0.5 * delta

def gen_rohf_response_sfd(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None, log=None):
    '''
    response function for Sf=Si-1
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mol = mf.mol
    if log is None:
        log = logger.new_logger(mf)
    if not isinstance(mf, dft.roks.ROKS):
        raise TypeError('NTTDA response requires ROKS reference')

    s = (mol.nelec[0] - mol.nelec[1]) * 0.5

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)

    if xctype != 'HF':
        fxc_d0 = ni.cache_xc_kernel(mol, mf.grids, mf.xc, mo_coeff, mo_occ, 1)[2]
        fxc_ref = 0.5 * (fxc_d0[0, :, 0] - fxc_d0[0, :, 1] - fxc_d0[1, :, 0] + fxc_d0[1, :, 1])

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    if mf.do_nlc():
        logger.warn(mf, "NLC contribution in gen_response is NOT included")

    def vind(dms_co, dms_cv, dms_oo, dms_ov):
        n_co = len(dms_co)
        n_cv = len(dms_cv)
        n_oo = len(dms_oo)
        idx1 = n_co
        idx2 = n_co + n_cv
        idx3 = n_co + n_cv + n_oo

        v1ao_co = cp.zeros_like(dms_co)
        v1ao_cv = cp.zeros_like(dms_cv)
        v1ao_oo = cp.zeros_like(dms_oo)
        v1ao_ov = cp.zeros_like(dms_ov)

        dms0 = _concatenate_factorized_densities(
            (dms_co, dms_cv, dms_oo, dms_ov),
        )
        dms1 = _concatenate_factorized_densities((dms_co, dms_ov))

        if xctype != 'HF':
            time_xc = (logger.process_clock(), logger.perf_counter())
            vref0 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms0, 0, hermi,
                                  None, None, fxc_ref, max_memory=max_memory)
            time_xc = log.timer('NTTDA response_sf vref0', *time_xc)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dms1, 0, hermi,
                                      None, None, fxc_ref, max_memory=max_memory)
            elif xctype =='GGA':
                vref1 = nr_rks_fxc1_gga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            elif xctype == 'MGGA':
                vref1 = nr_rks_fxc1_mgga(ni, mol, mf.grids, mf.xc, dms1, fxc_ref, max_memory=max_memory)
            log.timer('NTTDA response_sf vref1', *time_xc)
        else:
            vref0 = cp.zeros_like(dms0)
            vref1 = cp.zeros_like(dms1)

        # HF part
        if hybrid:
            time_jk = (logger.process_clock(), logger.perf_counter())
            vk = mf.get_k(mol, dms0, hermi) * hyb
            vj = mf.get_j(mol, dms1, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
                with mol.with_range_coulomb(omega):
                    vj += mf.get_j(
                        mol, dms1, hermi, omega=omega
                    ) * (alpha - hyb)
            vref0 -= vk
            vref1 -= vj
            log.timer('NTTDA response_sf get_j/get_k total', *time_jk)
        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_oo = vref0[idx2:idx3]
        vref0_ov = vref0[idx3:]
        vref1_co = vref1[:idx1]
        vref1_ov = vref1[idx1:]

        v1ao_co += vref0_co + vref1_co / (2 * s - 1) + cp.sqrt((2 * s + 1) / 2 / s) * vref0_cv
        v1ao_co += cp.sqrt(2 * s / (2 * s - 1)) * vref0_oo + 2 * s / (2 * s - 1) * vref0_ov - vref1_ov / (2 * s - 1)
        v1ao_cv += vref0_co * cp.sqrt((2 * s + 1) / 2 / s) + vref0_cv + cp.sqrt((2 * s + 1) / (2 * s - 1)) * vref0_oo
        v1ao_cv += cp.sqrt((2 * s + 1) / 2 / s) * vref0_ov
        v1ao_oo += cp.sqrt(2 * s / (2 * s - 1)) * vref0_co + cp.sqrt((2 * s + 1) / (2 * s - 1)) * vref0_cv
        v1ao_oo += vref0_oo + cp.sqrt(2 * s / (2 * s - 1)) * vref0_ov
        v1ao_ov += 2 * s / (2 * s - 1) * vref0_co - vref1_co / (2 * s - 1) + cp.sqrt((2 * s + 1) / 2 / s) * vref0_cv
        v1ao_ov += cp.sqrt(2 * s / (2 * s - 1)) * vref0_oo + vref0_ov + vref1_ov / (2 * s - 1)

        return v1ao_co, v1ao_cv, v1ao_oo, v1ao_ov

    orbos = mo_coeff[:, cp.where(mo_occ == 1)[0]]
    dmoo = orbos @ orbos.T
    if xctype != 'HF':
        delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, max_memory=max_memory)
    else:
        delta = cp.zeros_like(dmoo)
    if hybrid:
        delta -= mf.get_k(mol, dmoo, 1) * hyb
        if omega != 0:
            delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
    return vind, 0.5 * delta


def _sc_vector_slices(nc, no, nv):
    co = nc * no
    cv = nc * nv
    oo = 1
    ov = no * nv
    p_cv = co
    p_oo = p_cv + cv
    p_ov = p_oo + oo
    p_cv0 = p_ov + ov
    return {
        'CO(1)': slice(0, p_cv),
        'CV(1)': slice(p_cv, p_oo),
        'OO(1)': slice(p_oo, p_ov),
        'OV(1)': slice(p_ov, p_cv0),
        'CV(0)': slice(p_cv0, p_cv0 + cv),
    }


def _orbital_indices(tdobj):
    mo_occ = tdobj._scf.mo_occ
    return cp.where(mo_occ == 2)[0], cp.where(mo_occ == 1)[0], cp.where(mo_occ == 0)[0]


def _get_fock0_fockz(td, response_function):
    mf = td._scf
    _, fockz = response_function(
        mf,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        hermi=0,
        max_memory=td.max_memory,
        log=logger.new_logger(td),
    )
    if td.nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = 0.5 * (dma + dmb)
        fock = mf.get_fock(dm=cp.stack((dm0, dm0)))
    else:
        fock = mf.get_fock()
    return 0.5 * (fock.focka + fock.fockb), fockz


def _transform_eri_to_mo(mf, omega=None):
    mol = mf.mol
    nao = mol.nao_nr()
    if omega is None:
        eri = mol.intor('int2e_sph', aosym='s8')
    else:
        with mol.with_range_coulomb(omega):
            eri = mol.intor('int2e_sph', aosym='s8')
    eri = cp.asarray(ao2mo.restore(1, eri, nao))
    mo = cp.asarray(mf.mo_coeff)
    eri = contract('pjkl,pi->ijkl', eri, mo)
    eri = contract('ipkl,pj->ijkl', eri, mo)
    eri = contract('ijpl,pk->ijkl', eri, mo)
    return contract('ijkp,pl->ijkl', eri, mo)


def _transform_df_pairs(mf, omega=None):
    mo = cp.asarray(mf.mo_coeff)
    cderi_mo = []
    with mf.with_df.range_coulomb(omega) as dfobj:
        if dfobj._cderi is None:
            dfobj.build(omega=omega)
        for cderi, _ in dfobj.loop(unpack=True):
            block = contract('Lpq,pm->Lmq', cderi, mo)
            cderi_mo.append(contract('Lmq,qn->Lmn', block, mo))
    cderi_mo = cp.concatenate(cderi_mo, axis=0)
    return cderi_mo


def _transform_reference_integrals(mf, omega=None):
    if getattr(mf, 'with_df', None):
        return _transform_df_pairs(mf, omega=omega)
    return _transform_eri_to_mo(mf, omega=omega)


def _pair_indices(holes, particles):
    return (
        cp.repeat(holes, len(particles)),
        cp.tile(particles, len(holes)),
    )


def _kernel_layout(ndim, blocks):
    holes = cp.zeros(ndim, dtype=cp.int32)
    particles = cp.zeros(ndim, dtype=cp.int32)
    for block_slice, block_holes, block_particles in blocks:
        if block_holes is None:
            continue
        pair_holes, pair_particles = _pair_indices(
            block_holes, block_particles,
        )
        holes[block_slice] = pair_holes
        particles[block_slice] = pair_particles
    return holes, particles


def _reference_kernels_from_eri(eri_mo, holes, particles):
    row_holes = holes[:, None]
    row_particles = particles[:, None]
    column_holes = holes[None, :]
    column_particles = particles[None, :]
    k0 = -eri_mo[
        row_particles, column_particles, column_holes, row_holes,
    ]
    k1 = -eri_mo[
        row_particles, row_holes, column_particles, column_holes,
    ]
    return k0, k1


def _reference_kernels_from_df(cderi_mo, holes, particles):
    particle_pairs = cderi_mo[
        :, particles[:, None], particles[None, :],
    ]
    hole_pairs = cderi_mo[
        :, holes[None, :], holes[:, None],
    ]
    k0 = -contract('Lxy,Lxy->xy', particle_pairs, hole_pairs)
    particle_hole = cderi_mo[:, particles, holes]
    k1 = -contract('Lx,Ly->xy', particle_hole, particle_hole)
    return k0, k1


def _reference_kernels_from_integrals(integrals, holes, particles):
    if integrals.ndim == 3:
        return _reference_kernels_from_df(
            integrals, holes, particles,
        )
    return _reference_kernels_from_eri(
        integrals, holes, particles,
    )


def _project_reference_potential(potential, mo, holes, particles):
    transformed = contract('xpq,py->xyq', potential, mo[:, particles])
    return contract('xyq,qy->xy', transformed, mo[:, holes]).T


def _reference_kernels_from_xc(td, holes, particles):
    mf = td._scf
    ni = mf._numint
    mol = mf.mol
    mo = cp.asarray(mf.mo_coeff)
    dms = contract(
        'px,qx->xpq', mo[:, particles], mo[:, holes],
    )
    fxc = ni.cache_xc_kernel(
        mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 1,
    )[2]
    fxc_ref = 0.5 * (
        fxc[0, :, 0] - fxc[0, :, 1]
        - fxc[1, :, 0] + fxc[1, :, 1]
    )
    vref0 = ni.nr_rks_fxc(
        mol, mf.grids, mf.xc, None, dms, 0, 0,
        None, None, fxc_ref, max_memory=td.max_memory,
    )
    xctype = ni._xc_type(mf.xc)
    if xctype == 'LDA':
        vref1 = vref0
    elif xctype == 'GGA':
        vref1 = nr_rks_fxc1_gga(
            ni, mol, mf.grids, mf.xc, dms, fxc_ref,
            max_memory=td.max_memory,
        )
    elif xctype == 'MGGA':
        vref1 = nr_rks_fxc1_mgga(
            ni, mol, mf.grids, mf.xc, dms, fxc_ref,
            max_memory=td.max_memory,
        )
    else:
        raise NotImplementedError(
            'GPU NTTDA get_ab XC kernel for %s is not implemented' %
            xctype,
        )
    return (
        _project_reference_potential(
            vref0, mo, holes, particles,
        ),
        _project_reference_potential(
            vref1, mo, holes, particles,
        ),
    )


def _reference_kernels(td, eri_mo, holes, particles):
    mf = td._scf
    xctype = mf._numint._xc_type(mf.xc)
    if xctype == 'HF':
        return _reference_kernels_from_integrals(
            eri_mo, holes, particles,
        )
    if xctype in ('LDA', 'GGA', 'MGGA'):
        k0, k1 = _reference_kernels_from_xc(td, holes, particles)
        omega, alpha, hybrid = mf._numint.rsh_and_hybrid_coeff(
            mf.xc, mf.mol.spin,
        )
        if hybrid:
            exchange, coulomb = _reference_kernels_from_integrals(
                eri_mo, holes, particles,
            )
            k0 += hybrid * exchange
            k1 += hybrid * coulomb
        if omega != 0:
            exchange, coulomb = _reference_kernels_from_integrals(
                _transform_reference_integrals(mf, omega=omega),
                holes,
                particles,
            )
            k0 += (alpha - hybrid) * exchange
            k1 += (alpha - hybrid) * coulomb
        return k0, k1
    raise NotImplementedError(
        'GPU NTTDA get_ab reference kernel for %s is not implemented' %
        xctype,
    )


def _set_symmetric_block(matrix, slices, row, column, block):
    row_slice = slices[row]
    column_slice = slices[column]
    matrix[row_slice, column_slice] = block
    if row != column:
        matrix[column_slice, row_slice] = block.T


def _set_symmetric_coefficient(matrix, slices, row, column, value):
    row_slice = slices[row]
    column_slice = slices[column]
    matrix[row_slice, column_slice] = value
    if row != column:
        matrix[column_slice, row_slice] = value


def _one_body_block(particle_fock, hole_fock):
    nhole = hole_fock.shape[0]
    nparticle = particle_fock.shape[0]
    dtype = particle_fock.dtype
    return (
        cp.kron(cp.eye(nhole, dtype=dtype), particle_fock)
        - cp.kron(hole_fock.T, cp.eye(nparticle, dtype=dtype))
    )


def _add_reference_kernel(
        matrix, td, eri_mo, holes, particles, slices, coefficients):
    k0, k1 = _reference_kernels(td, eri_mo, holes, particles)
    c0 = cp.zeros_like(matrix)
    c1 = cp.zeros_like(matrix)
    for row, column, weight0, weight1 in coefficients:
        if weight0:
            _set_symmetric_coefficient(
                c0, slices, row, column, weight0,
            )
        if weight1:
            _set_symmetric_coefficient(
                c1, slices, row, column, weight1,
            )
    matrix += c0 * k0 + c1 * k1


def _get_ab_sfu(td, eri_mo):
    mf = td._scf
    csidx, _, vsidx = _orbital_indices(td)
    c = mf.mo_coeff[:, csidx]
    v = mf.mo_coeff[:, vsidx]
    nc = c.shape[1]
    nv = v.shape[1]
    ndim = nc * nv
    slices = {'CV': slice(0, ndim)}

    fock0, fockz = _get_fock0_fockz(td, gen_rohf_response_sfu)
    fock_v = v.T @ (fock0 + fockz) @ v
    fock_c = c.T @ (fock0 - fockz) @ c
    matrix = _one_body_block(fock_v, fock_c)

    holes, particles = _kernel_layout(
        ndim, ((slices['CV'], csidx, vsidx),),
    )
    _add_reference_kernel(
        matrix,
        td,
        eri_mo,
        holes,
        particles,
        slices,
        (('CV', 'CV', 1.0, 0.0),),
    )
    return matrix


def _get_ab_sfd(td, eri_mo):
    mf = td._scf
    csidx, osidx, vsidx = _orbital_indices(td)
    c = mf.mo_coeff[:, csidx]
    o = mf.mo_coeff[:, osidx]
    v = mf.mo_coeff[:, vsidx]
    nc = c.shape[1]
    no = o.shape[1]
    nv = v.shape[1]
    spin = 0.5 * no
    denominator = 2.0 * spin - 1.0

    nco = nc * no
    ncv = nc * nv
    noo = no * no
    nov = no * nv
    slices = {
        'CO': slice(0, nco),
        'CV': slice(nco, nco + ncv),
        'OO': slice(nco + ncv, nco + ncv + noo),
        'OV': slice(nco + ncv + noo, nco + ncv + noo + nov),
    }
    ndim = slices['OV'].stop
    matrix = cp.zeros((ndim, ndim))

    fock0, fockz = _get_fock0_fockz(td, gen_rohf_response_sfd)
    fminus = fock0 - fockz
    fplus = fock0 + fockz
    fminus_oo = o.T @ fminus @ o
    fplus_cc = c.T @ fplus @ c
    fz_cc = c.T @ fockz @ c
    fminus_vv = v.T @ fminus @ v
    fz_vv = v.T @ fockz @ v
    fplus_oo = o.T @ fplus @ o

    _set_symmetric_block(
        matrix, slices, 'CO', 'CO',
        _one_body_block(
            fminus_oo,
            fplus_cc + 2.0 / denominator * fz_cc,
        ),
    )
    _set_symmetric_block(
        matrix, slices, 'CV', 'CV',
        _one_body_block(
            fminus_vv - fz_vv / spin,
            fplus_cc + fz_cc / spin,
        ),
    )
    _set_symmetric_block(
        matrix, slices, 'OO', 'OO',
        _one_body_block(fminus_oo, fplus_oo),
    )
    _set_symmetric_block(
        matrix, slices, 'OV', 'OV',
        _one_body_block(
            fminus_vv - 2.0 / denominator * fz_vv,
            fplus_oo,
        ),
    )

    a = cp.sqrt((2.0 * spin + 1.0) / (2.0 * spin))
    b = cp.sqrt(2.0 * spin / denominator)
    ccoef = cp.sqrt((2.0 * spin + 1.0) / denominator)
    d = 1.0 / cp.sqrt(2.0 * spin * denominator)
    _set_symmetric_block(
        matrix, slices, 'CV', 'CO',
        a * cp.kron(cp.eye(nc), v.T @ fminus @ o),
    )
    _set_symmetric_block(
        matrix, slices, 'CV', 'OV',
        -a * cp.kron((o.T @ fplus @ c).T, cp.eye(nv)),
    )

    block = cp.zeros((noo, nco))
    block4 = block.reshape(no, no, nc, no)
    diagonal = cp.arange(no)
    fplus_co = c.T @ fplus @ o
    fminus_co = c.T @ fminus @ o
    for u in range(no):
        block4[u, :, :, :] -= (
            b * cp.eye(no)[:, None, :] * fplus_co[:, u][None, :, None]
        )
    block4[diagonal, diagonal] += d * fminus_co[None]
    _set_symmetric_block(matrix, slices, 'OO', 'CO', block)

    block = cp.zeros((noo, ncv))
    block4 = block.reshape(no, no, nc, nv)
    block4[diagonal, diagonal] = (
        -ccoef / spin * (c.T @ fockz @ v)[None]
    )
    _set_symmetric_block(matrix, slices, 'OO', 'CV', block)

    block = cp.zeros((noo, nov))
    block4 = block.reshape(no, no, no, nv)
    fminus_ov = o.T @ fminus @ v
    fplus_ov = o.T @ fplus @ v
    for u in range(no):
        block4[u, :, u, :] += b * fminus_ov
    block4[diagonal, diagonal] -= d * fplus_ov[None]
    _set_symmetric_block(matrix, slices, 'OO', 'OV', block)

    holes, particles = _kernel_layout(
        ndim,
        (
            (slices['CO'], csidx, osidx),
            (slices['CV'], csidx, vsidx),
            (slices['OO'], osidx, osidx),
            (slices['OV'], osidx, vsidx),
        ),
    )
    _add_reference_kernel(
        matrix,
        td,
        eri_mo,
        holes,
        particles,
        slices,
        (
            ('CO', 'CO', 1.0, 1.0 / denominator),
            ('CV', 'CV', 1.0, 0.0),
            ('OO', 'OO', 1.0, 0.0),
            ('OV', 'OV', 1.0, 1.0 / denominator),
            ('CV', 'CO', a, 0.0),
            ('CV', 'OV', a, 0.0),
            ('OO', 'CO', b, 0.0),
            ('OO', 'CV', ccoef, 0.0),
            ('OO', 'OV', b, 0.0),
            (
                'CO', 'OV',
                2.0 * spin / denominator,
                -1.0 / denominator,
            ),
        ),
    )
    native_to_block = cp.concatenate((
        cp.concatenate((
            cp.arange(nco).reshape(nc, no),
            cp.arange(ncv).reshape(nc, nv) + nco,
        ), axis=1).ravel(),
        cp.concatenate((
            cp.arange(noo).reshape(no, no) + nco + ncv,
            cp.arange(nov).reshape(no, nv) + nco + ncv + noo,
        ), axis=1).ravel(),
    ))
    return matrix[native_to_block[:, None], native_to_block[None, :]]


def _get_ab_sc(td, eri_mo):
    mf = td._scf
    csidx, osidx, vsidx = _orbital_indices(td)
    c = mf.mo_coeff[:, csidx]
    o = mf.mo_coeff[:, osidx]
    v = mf.mo_coeff[:, vsidx]
    nc = c.shape[1]
    no = o.shape[1]
    nv = v.shape[1]
    spin = 0.5 * no
    slices = _sc_vector_slices(nc, no, nv)
    slices = {
        'CO': slices['CO(1)'],
        'CV': slices['CV(1)'],
        'OO': slices['OO(1)'],
        'OV': slices['OV(1)'],
        'CV0': slices['CV(0)'],
    }
    ndim = slices['CV0'].stop
    matrix = cp.zeros((ndim, ndim))

    fock0, fockz = _get_fock0_fockz(td, gen_rohf_response_sc)
    fminus = fock0 - fockz
    fplus = fock0 + fockz
    f0_vv = v.T @ fock0 @ v
    f0_cc = c.T @ fock0 @ c
    fz_vv = v.T @ fockz @ v
    fz_cc = c.T @ fockz @ c

    _set_symmetric_block(
        matrix, slices, 'CV0', 'CV0',
        _one_body_block(f0_vv, f0_cc),
    )
    _set_symmetric_block(
        matrix, slices, 'CV', 'CV',
        _one_body_block(
            f0_vv - fz_vv / spin,
            f0_cc + fz_cc / spin,
        ),
    )
    _set_symmetric_block(
        matrix, slices, 'CO', 'CO',
        _one_body_block(o.T @ fminus @ o, c.T @ fminus @ c),
    )
    _set_symmetric_block(
        matrix, slices, 'OV', 'OV',
        _one_body_block(v.T @ fplus @ v, o.T @ fplus @ o),
    )

    a = cp.sqrt((spin + 1.0) / (2.0 * spin))
    d = cp.sqrt((spin + 1.0) / spin)
    h = cp.sqrt(0.5)
    r2 = cp.sqrt(2.0)
    _set_symmetric_block(
        matrix, slices, 'CV0', 'CV',
        -d * cp.kron(cp.eye(nc), fz_vv)
        + d * cp.kron(fz_cc.T, cp.eye(nv)),
    )
    _set_symmetric_block(
        matrix, slices, 'CV0', 'CO',
        h * cp.kron(cp.eye(nc), v.T @ fminus @ o),
    )
    _set_symmetric_block(
        matrix, slices, 'CV0', 'OV',
        h * cp.kron((o.T @ fplus @ c).T, cp.eye(nv)),
    )
    _set_symmetric_block(
        matrix, slices, 'CV', 'CO',
        a * cp.kron(cp.eye(nc), v.T @ fminus @ o),
    )
    _set_symmetric_block(
        matrix, slices, 'CV', 'OV',
        -a * cp.kron((o.T @ fplus @ c).T, cp.eye(nv)),
    )

    _set_symmetric_block(
        matrix, slices, 'OO', 'CV0',
        (-r2 * (c.T @ fock0 @ v)).reshape(1, -1),
    )
    _set_symmetric_block(
        matrix, slices, 'OO', 'CV',
        (2.0 * a * (c.T @ fockz @ v)).reshape(1, -1),
    )
    _set_symmetric_block(
        matrix, slices, 'OO', 'CO',
        (-(c.T @ fminus @ o)).reshape(1, -1),
    )
    _set_symmetric_block(
        matrix, slices, 'OO', 'OV',
        (o.T @ fplus @ v).reshape(1, -1),
    )

    holes, particles = _kernel_layout(
        ndim,
        (
            (slices['CO'], csidx, osidx),
            (slices['CV'], csidx, vsidx),
            (slices['OO'], None, None),
            (slices['OV'], osidx, vsidx),
            (slices['CV0'], csidx, vsidx),
        ),
    )
    _add_reference_kernel(
        matrix,
        td,
        eri_mo,
        holes,
        particles,
        slices,
        (
            ('CO', 'CO', 1.0, -1.0),
            ('CV', 'CV', 1.0, 0.0),
            ('OV', 'OV', 1.0, -1.0),
            ('CV0', 'CV0', 1.0, -2.0),
            ('CV', 'CO', a, 0.0),
            ('CV', 'OV', a, 0.0),
            ('CV0', 'CO', h, -r2),
            ('CV0', 'OV', -h, r2),
            ('CO', 'OV', 0.0, 1.0),
        ),
    )
    return matrix


def get_ab(td, mf=None):
    """Build the explicit NTTDA A matrix in the native ``vind`` ordering."""
    if mf is None:
        mf = td._scf
    if mf is not td._scf:
        raise ValueError('GPU NTTDA get_ab requires td._scf')
    xctype = mf._numint._xc_type(mf.xc)
    if xctype not in ('HF', 'LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(
            'GPU NTTDA get_ab does not support XC type %s' % xctype,
        )

    hybrid = mf._numint.libxc.is_hybrid_xc(mf.xc)
    eri_mo = (
        _transform_reference_integrals(mf)
        if xctype == 'HF' or hybrid else None
    )
    if td.deltaS == 1:
        matrix = _get_ab_sfu(td, eri_mo)
    elif td.deltaS == 0:
        matrix = _get_ab_sc(td, eri_mo)
    elif td.deltaS == -1:
        matrix = _get_ab_sfd(td, eri_mo)
    else:
        raise ValueError('deltaS should be -1, 0, or 1')
    return matrix.get()


def gen_vind_sfu(td):
    mf = td._scf
    mo_coeff = mf.mo_coeff
    assert mo_coeff[0].dtype == cp.double
    mo_occ = mf.mo_occ

    csidx, _, vsidx = _orbital_indices(td)
    orbcs = mo_coeff[:, csidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nvs = orbvs.shape[1]

    log = logger.new_logger(td)
    vresp, fockz = gen_rohf_response_sfu(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0,
                                         max_memory=td.max_memory, log=log)

    if td.nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = 0.5 * (dma + dmb)
        fock = mf.get_fock(dm=cp.stack((dm0, dm0)))
        fock0 = 0.5 * (fock.focka + fock.fockb)
        focka = fock0 + fockz
        fockb = fock0 - fockz
    else:
        fock = mf.get_fock()
        fock0 = 0.5 * (fock.focka + fock.fockb)
        focka = fock0 + fockz
        fockb = fock0 - fockz

    fock_v = orbvs.T @ focka @ orbvs
    fock_c = orbcs.T @ fockb @ orbcs
    hdiag = (fock_v.diagonal()[None, :] - fock_c.diagonal()[:, None]).ravel()

    def vind(zs):
        time0 = time1 = (logger.process_clock(), logger.perf_counter())
        zs = cp.asarray(zs).reshape(-1, ncs, nvs)
        mo1_cv = contract('xia,pa->xpi', zs, orbvs)
        dms_cv = _factorized_density(mo1_cv, orbcs.conj())
        time1 = log.timer('NTTDA gen_vind_sfu make density matrices', *time1)

        v1ao_cv = vresp(dms_cv)
        time1 = log.timer('NTTDA gen_vind_sfu response vind total', *time1)
        v1mo_cv = contract('xpq,qi->xip', v1ao_cv, orbcs)
        v1mo_cv = contract('xip,pa->xia', v1mo_cv, orbvs.conj())
        time1 = log.timer('NTTDA gen_vind_sfu AO->MO transform', *time1)

        v1mo_cv += contract('ab,xib->xia', fock_v, zs)
        v1mo_cv -= contract('ji,xja->xia', fock_c, zs)
        time1 = log.timer('NTTDA gen_vind_sfu Fock part', *time1)

        v1mo = v1mo_cv.reshape(len(zs), -1)
        time1 = log.timer('NTTDA gen_vind_sfu pack result', *time1)
        log.timer('NTTDA gen_vind_sfu total', *time0)
        return v1mo
    return vind, hdiag

def gen_vind_sc(td):
    mf = td._scf
    mo_coeff = mf.mo_coeff
    assert mo_coeff[0].dtype == cp.double
    mo_occ = mf.mo_occ

    csidx, osidx, vsidx = _orbital_indices(td)
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]
    slices = _sc_vector_slices(ncs, nos, nvs)

    s = nos * 0.5
    assert s >= 0.5, 'NTTDA only supports case that Sf=Si>=1/2.'
    assert s == (mf.mol.nelec[0] - mf.mol.nelec[1]) * 0.5

    log = logger.new_logger(td)
    vresp, fockz = gen_rohf_response_sc(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0,
                                        max_memory=td.max_memory, log=log)

    if td.nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = 0.5 * (dma + dmb)
        fock = mf.get_fock(dm=cp.stack((dm0, dm0)))
        fock0 = 0.5 * (fock.focka + fock.fockb)
        focka = fock0 + fockz
        fockb = fock0 - fockz
    else:
        fock = mf.get_fock()
        fock0 = 0.5 * (fock.focka + fock.fockb)
        focka = fock0 + fockz
        fockb = fock0 - fockz

    fock_coco1 = orbos.T @ (fock0 - fockz) @ orbos
    fock_coco2 = orbcs.T @ (fock0 - fockz) @ orbcs
    fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
    fock_cvcv1 = orbvs.T @ (fock0 - fockz / s) @ orbvs
    fock_cvcv2 = orbcs.T @ (fock0 + fockz / s) @ orbcs
    fock_cocv0 = orbos.T @ fockb @ orbvs
    fock_cvov = orbos.T @ (fock0 + fockz) @ orbcs
    fock_cvcv01 = 0.5 * orbvs.T @ (focka - fockb) @ orbvs
    fock_cvcv02 = 0.5 * orbcs.T @ (focka - fockb) @ orbcs
    fock_ovov1 = orbvs.T @ (fock0 + fockz) @ orbvs
    fock_ovov2 = orbos.T @ (fock0 + fockz) @ orbos
    fock_ovcv0 = orbcs.T @ focka @ orbos
    fock_cv0cv01 = 0.5 * orbvs.T @ (focka + fockb) @ orbvs
    fock_cv0cv02 = 0.5 * orbcs.T @ (focka + fockb) @ orbcs
    fock_cooo = orbos.T @ (fock0 - fockz) @ orbcs
    fock_cvoo = orbvs.T @ fockz @ orbcs
    fock_ovoo = orbvs.T @ (fock0 + fockz) @ orbos
    fock_cv0oo = 0.5 * orbvs.T @ (focka + fockb) @ orbcs

    # diagonal part for preconditioning
    hdiag_co = (fock_coco1.diagonal()[None, :] - fock_coco2.diagonal()[:, None]).ravel()
    hdiag_cv = (fock_cvcv1.diagonal()[None, :] - fock_cvcv2.diagonal()[:, None]).ravel()
    hdiag_oo = cp.array([0.0])
    hdiag_ov = (fock_ovov1.diagonal()[None, :] - fock_ovov2.diagonal()[:, None]).ravel()
    hdiag_cv0 = (fock_cv0cv01.diagonal()[None, :] - fock_cv0cv02.diagonal()[:, None]).ravel()
    hdiag = cp.concatenate((hdiag_co, hdiag_cv, hdiag_oo, hdiag_ov, hdiag_cv0))

    def vind(zs):
        time0 = time1 = (logger.process_clock(), logger.perf_counter())
        zs = cp.asarray(zs)  # (nstates, ndim)
        zs_co = zs[:, slices['CO(1)']].reshape(-1, ncs, nos)
        zs_cv = zs[:, slices['CV(1)']].reshape(-1, ncs, nvs)
        zs_oo = zs[:, slices['OO(1)']].reshape(-1, 1)
        zs_ov = zs[:, slices['OV(1)']].reshape(-1, nos, nvs)
        zs_cv0 = zs[:, slices['CV(0)']].reshape(-1, ncs, nvs)
        right = cp.concatenate((orbcs.conj(), orbos.conj()), axis=1)
        core = slice(None, ncs)
        open_ = slice(ncs, None)
        dms_co = _factorized_density(
            contract('xov,pv->xpo', zs_co, orbos), right, core,
        )
        dms_cv = _factorized_density(
            contract('xov,pv->xpo', zs_cv, orbvs), right, core,
        )
        dms_ov = _factorized_density(
            contract('xov,pv->xpo', zs_ov, orbvs), right, open_,
        )
        dms_cv0 = _factorized_density(
            contract('xov,pv->xpo', zs_cv0, orbvs), right, core,
        )
        time1 = log.timer('NTTDA gen_vind_sc make density matrices', *time1)
        v1ao_co, v1ao_cv, v1ao_ov, v1ao_cv0 = vresp(dms_co, dms_cv, dms_ov, dms_cv0)
        time1 = log.timer('NTTDA gen_vind_sc response vind total', *time1)
        v1mo_co = contract('xpq,qo->xpo', v1ao_co, orbcs)
        v1mo_co = contract('xpo,pv->xov', v1mo_co, orbos.conj())
        v1mo_cv = contract('xpq,qo->xpo', v1ao_cv, orbcs)
        v1mo_cv = contract('xpo,pv->xov', v1mo_cv, orbvs.conj())
        v1mo_ov = contract('xpq,qo->xpo', v1ao_ov, orbos)
        v1mo_ov = contract('xpo,pv->xov', v1mo_ov, orbvs.conj())
        v1mo_cv0 = contract('xpq,qo->xpo', v1ao_cv0, orbcs)
        v1mo_cv0 = contract('xpo,pv->xov', v1mo_cv0, orbvs.conj())
        time1 = log.timer('NTTDA gen_vind_sc AO->MO transform', *time1)

        v1mo_co += contract('uv,xiv->xiu', fock_coco1, zs_co)
        v1mo_co -= contract('ji,xju->xiu', fock_coco2, zs_co)
        v1mo_co += contract('ub,xib->xiu', fock_cocv, zs_cv) * cp.sqrt((s + 1) / 2 / s)
        v1mo_co -= contract('ui,xv->xiu', fock_cooo, zs_oo)
        v1mo_co += contract('ub,xib->xiu', fock_cocv0, zs_cv0) * cp.sqrt(0.5)

        v1mo_cv += contract('av,xiv->xia', fock_cocv.T, zs_co) * cp.sqrt((s + 1) / 2 / s)
        v1mo_cv += contract('ab,xib->xia', fock_cvcv1, zs_cv)
        v1mo_cv -= contract('ji,xja->xia', fock_cvcv2, zs_cv)
        v1mo_cv += contract('ai,xv->xia', fock_cvoo, zs_oo) * cp.sqrt(2 * (s + 1) / s)
        v1mo_cv -= contract('vi,xva->xia', fock_cvov, zs_ov) * cp.sqrt((s + 1) / 2 / s)
        v1mo_cv -= contract('ab,xib->xia', fock_cvcv01, zs_cv0) * cp.sqrt((s + 1) / s)
        v1mo_cv += contract('ji,xja->xia', fock_cvcv02, zs_cv0) * cp.sqrt((s + 1) / s)

        v1mo_ov -= contract('ju,xja->xua', fock_cvov.T, zs_cv) * cp.sqrt((s + 1) / 2 / s)
        v1mo_ov += contract('au,xv->xua', fock_ovoo, zs_oo)
        v1mo_ov += contract('ab,xub->xua', fock_ovov1, zs_ov)
        v1mo_ov -= contract('vu,xva->xua', fock_ovov2, zs_ov)
        v1mo_ov += contract('ju,xja->xua', fock_ovcv0, zs_cv0) * cp.sqrt(0.5)

        v1mo_cv0 += contract('av,xiv->xia', fock_cocv0.T, zs_co) * cp.sqrt(0.5)
        v1mo_cv0 -= contract('ab,xib->xia', fock_cvcv01, zs_cv) * cp.sqrt((s + 1) / s)
        v1mo_cv0 += contract('ji,xja->xia', fock_cvcv02, zs_cv) * cp.sqrt((s + 1) / s)
        v1mo_cv0 -= contract('ai,xv->xia', fock_cv0oo, zs_oo) * cp.sqrt(2)
        v1mo_cv0 += contract('vi,xva->xia', fock_ovcv0.T, zs_ov) * cp.sqrt(0.5)
        v1mo_cv0 += contract('ab,xib->xia', fock_cv0cv01, zs_cv0)
        v1mo_cv0 -= contract('ji,xja->xia', fock_cv0cv02, zs_cv0)

        v1mo_oo = cp.zeros((len(zs), ))
        v1mo_oo -= contract('jv,xjv->x', fock_cooo.T, zs_co)
        v1mo_oo += contract('jb,xjb->x', fock_cvoo.T, zs_cv) * cp.sqrt(2 * (s + 1) / s)
        v1mo_oo += contract('vb,xvb->x', fock_ovoo.T, zs_ov)
        v1mo_oo -= contract('jb,xjb->x', fock_cv0oo.T, zs_cv0) * cp.sqrt(2)
        time1 = log.timer('NTTDA gen_vind_sc Fock part', *time1)

        v1mo = cp.concatenate((v1mo_co.reshape(len(zs), -1),
                                v1mo_cv.reshape(len(zs), -1),
                                v1mo_oo.reshape(len(zs), -1),
                                v1mo_ov.reshape(len(zs), -1),
                                v1mo_cv0.reshape(len(zs), -1)), axis=1)
        assert v1mo.shape == zs.shape
        time1 = log.timer('NTTDA gen_vind_sc pack result', *time1)
        log.timer('NTTDA gen_vind_sc total', *time0)
        return v1mo
    return vind, hdiag

def gen_vind_sfd(td):
    mf = td._scf
    mo_coeff = mf.mo_coeff
    assert mo_coeff[0].dtype == cp.double
    mo_occ = mf.mo_occ

    csidx, osidx, vsidx = _orbital_indices(td)
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    ncs = orbcs.shape[1]
    nos = orbos.shape[1]
    nvs = orbvs.shape[1]
    nocc = ncs + nos
    nvir = nos + nvs
    core_rows = slice(None, ncs)
    open_rows = slice(ncs, None)
    open_cols = slice(None, nos)
    virt_cols = slice(nos, None)

    s = nos * 0.5
    assert s >= 1.0, 'NTTDA for Sf=Si-1 only supports case that Si>=1.'
    assert s == (mf.mol.nelec[0] - mf.mol.nelec[1]) * 0.5

    log = logger.new_logger(td)
    vresp, fockz = gen_rohf_response_sfd(mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0,
                                         max_memory=td.max_memory, log=log)

    if td.nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = 0.5 * (dma + dmb)
        fock = mf.get_fock(dm=cp.stack((dm0, dm0)))
        fock0 = 0.5 * (fock.focka + fock.fockb)
    else:
        fock = mf.get_fock()
        fock0 = 0.5 * (fock.focka + fock.fockb)

    fock_coco0 = orbos.T @ (fock0 - fockz) @ orbos
    fock_coco1 = orbcs.T @ (fock0 + fockz) @ orbcs
    fock_coco2 = orbcs.T @ fockz @ orbcs
    fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
    fock_cooo0 = orbos.T @ (fock0 + fockz) @ orbcs
    fock_cooo1 = orbos.T @ (fock0 - fockz) @ orbcs
    fock_cvcv0 = orbvs.T @ (fock0 - fockz) @ orbvs
    fock_cvcv1 = fock_coco1
    fock_cvcv2 = orbvs.T @ fockz @ orbvs
    fock_cvcv3 = fock_coco2
    fock_cvoo = orbvs.T @ fockz @ orbcs
    fock_cvov = fock_cooo0
    fock_oooo0 = fock_coco0
    fock_oooo1 = orbos.T @ (fock0 + fockz) @ orbos
    fock_ooov0 = fock_cocv
    fock_ooov1 = orbos.T @ (fock0 + fockz) @ orbvs
    fock_ovov0 = fock_cvcv0
    fock_ovov1 = fock_oooo1
    fock_ovov2 = fock_cvcv2

    # diagonal part for preconditioning
    hdiag_co = fock_coco0.diagonal()[None, :] - fock_coco1.diagonal()[:, None]
    hdiag_co -= fock_coco2.diagonal()[:, None] * 2 / (2 * s - 1)
    hdiag_cv = fock_cvcv0.diagonal()[None, :] - fock_cvcv1.diagonal()[:, None]
    hdiag_cv -= fock_cvcv2.diagonal()[None, :] / s + fock_cvcv3.diagonal()[:, None] / s
    hdiag_oo = fock_oooo0.diagonal()[None, :] - fock_oooo1.diagonal()[:, None]
    hdiag_ov = fock_ovov0.diagonal()[None, :] - fock_ovov1.diagonal()[:, None]
    hdiag_ov -= fock_ovov2.diagonal()[None, :] * 2 / (2 * s - 1)
    hdiag = cp.concatenate(
        (cp.concatenate((hdiag_co, hdiag_cv), axis=1),
         cp.concatenate((hdiag_oo, hdiag_ov), axis=1)),
        axis=0,
    ).ravel()
    open_diag = cp.diag_indices(nos)

    def vind(zs):
        time0 = time1 = (logger.process_clock(), logger.perf_counter())
        zs = cp.asarray(zs).reshape(-1, nocc, nvir)
        zs_co = zs[:, core_rows, open_cols]
        zs_cv = zs[:, core_rows, virt_cols]
        zs_oo = zs[:, open_rows, open_cols]
        zs_ov = zs[:, open_rows, virt_cols]
        right = cp.concatenate((orbcs.conj(), orbos.conj()), axis=1)
        core = slice(None, ncs)
        open_ = slice(ncs, None)
        dms_co = _factorized_density(
            contract('xov,pv->xpo', zs_co, orbos), right, core,
        )
        dms_cv = _factorized_density(
            contract('xov,pv->xpo', zs_cv, orbvs), right, core,
        )
        dms_oo = _factorized_density(
            contract('xov,pv->xpo', zs_oo, orbos), right, open_,
        )
        dms_ov = _factorized_density(
            contract('xov,pv->xpo', zs_ov, orbvs), right, open_,
        )
        time1 = log.timer('NTTDA gen_vind_sfd make density matrices', *time1)
        v1ao_co, v1ao_cv, v1ao_oo, v1ao_ov = vresp(dms_co, dms_cv, dms_oo, dms_ov)
        time1 = log.timer('NTTDA gen_vind_sfd response vind total', *time1)
        v1mo_co = contract('xpq,qo->xpo', v1ao_co, orbcs)
        v1mo_co = contract('xpo,pv->xov', v1mo_co, orbos.conj())
        v1mo_cv = contract('xpq,qo->xpo', v1ao_cv, orbcs)
        v1mo_cv = contract('xpo,pv->xov', v1mo_cv, orbvs.conj())
        v1mo_oo = contract('xpq,qo->xpo', v1ao_oo, orbos)
        v1mo_oo = contract('xpo,pv->xov', v1mo_oo, orbos.conj())
        v1mo_ov = contract('xpq,qo->xpo', v1ao_ov, orbos)
        v1mo_ov = contract('xpo,pv->xov', v1mo_ov, orbvs.conj())
        time1 = log.timer('NTTDA gen_vind_sfd AO->MO transform', *time1)

        v1mo_co += contract('uv,xiv->xiu', fock_coco0, zs_co)
        v1mo_co -= contract('ji,xju->xiu', fock_coco1, zs_co)
        v1mo_co -= contract('ji,xju->xiu', fock_coco2, zs_co) * 2 / (2 * s - 1)
        v1mo_co += contract('ub,xib->xiu', fock_cocv, zs_cv) * cp.sqrt((2 * s + 1) / 2 / s)
        v1mo_co -= contract('wi,xwu->xiu', fock_cooo0, zs_oo) * cp.sqrt(2 * s / (2 * s - 1))
        v1mo_co += contract('ui,xvv->xiu', fock_cooo1, zs_oo) / cp.sqrt(2 * s * (2 * s - 1))

        v1mo_cv += contract('av,xiv->xia', fock_cocv.T, zs_co) * cp.sqrt((2 * s + 1) / 2 / s)
        v1mo_cv += contract('ab,xib->xia', fock_cvcv0, zs_cv)
        v1mo_cv -= contract('ji,xja->xia', fock_cvcv1, zs_cv)
        v1mo_cv -= contract('ab,xib->xia', fock_cvcv2, zs_cv) / s
        v1mo_cv -= contract('ji,xja->xia', fock_cvcv3, zs_cv) / s
        v1mo_cv -= contract('ai,xvv->xia', fock_cvoo, zs_oo) / s * cp.sqrt((2 * s + 1) / (2 * s - 1))
        v1mo_cv -= contract('vi,xva->xia', fock_cvov, zs_ov) * cp.sqrt((2 * s + 1) / 2 / s)

        v1mo_oo -= contract('ju,xjt->xut', fock_cooo0.T, zs_co) * cp.sqrt(2 * s / (2 * s - 1))
        v1mo_oo[:, open_diag[0], open_diag[1]] += (
            contract('jv,xjv->x', fock_cooo1.T, zs_co) /
            cp.sqrt(2 * s * (2 * s - 1))
        )[:, None]
        v1mo_oo[:, open_diag[0], open_diag[1]] -= (
            contract('jb,xjb->x', fock_cvoo.T, zs_cv) /
            s * cp.sqrt((2 * s + 1) / (2 * s - 1))
        )[:, None]
        v1mo_oo += contract('tv,xuv->xut', fock_oooo0, zs_oo)
        v1mo_oo -= contract('wu,xwt->xut', fock_oooo1, zs_oo)
        v1mo_oo += contract('tb,xub->xut', fock_ooov0, zs_ov) * cp.sqrt(2 * s / (2 * s - 1))
        v1mo_oo[:, open_diag[0], open_diag[1]] -= (
            contract('vb,xvb->x', fock_ooov1, zs_ov) /
            cp.sqrt(2 * s * (2 * s - 1))
        )[:, None]

        v1mo_ov -= contract('ju,xja->xua', fock_cvov.T, zs_cv) * cp.sqrt((2 * s + 1) / 2 / s)
        v1mo_ov += contract('av,xuv->xua', fock_ooov0.T, zs_oo) * cp.sqrt(2 * s / (2 * s - 1))
        v1mo_ov -= contract('au,xvv->xua', fock_ooov1.T, zs_oo) / cp.sqrt(2 * s * (2 * s - 1))
        v1mo_ov += contract('ab,xub->xua', fock_ovov0, zs_ov)
        v1mo_ov -= contract('vu,xva->xua', fock_ovov1, zs_ov)
        v1mo_ov -= contract('ab,xub->xua', fock_ovov2, zs_ov) * 2 / (2 * s - 1)
        time1 = log.timer('NTTDA gen_vind_sfd Fock part', *time1)

        v1mo = cp.zeros_like(zs)
        v1mo[:, core_rows, open_cols] = v1mo_co
        v1mo[:, core_rows, virt_cols] = v1mo_cv
        v1mo[:, open_rows, open_cols] = v1mo_oo
        v1mo[:, open_rows, virt_cols] = v1mo_ov
        assert v1mo.shape == zs.shape
        time1 = log.timer('NTTDA gen_vind_sfd pack result', *time1)
        log.timer('NTTDA gen_vind_sfd total', *time0)
        return v1mo.reshape(len(v1mo), -1)
    return vind, hdiag


def as_scanner(td):
    if isinstance(td, lib.SinglePointScanner):
        return td

    logger.info(td, 'Set %s as a scanner', td.__class__)
    name = td.__class__.__name__ + NTTDA_Scanner.__name_mixin__
    return lib.set_class(
        NTTDA_Scanner(td),
        (NTTDA_Scanner, td.__class__),
        name,
    )


class NTTDA_Scanner(lib.SinglePointScanner):
    device = 'gpu'

    def __init__(self, td):
        self.__dict__.update(td.__dict__)
        self._scf = td._scf.as_scanner()
        self._basis_fp = np.hstack(td.mol.bas_exps())

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        previous_mol = self.mol
        self.reset(mol)
        mf_scanner = self._scf
        previous_coeff = mf_scanner.mo_coeff
        previous_occ = mf_scanner.mo_occ
        mf_energy = mf_scanner(mol)

        x0 = self.xy
        if x0 is not None:
            if np.array_equal(self._basis_fp, np.hstack(mol.bas_exps())):
                x0 = self._transfer_initial_guess(
                    x0,
                    previous_mol,
                    previous_coeff,
                    previous_occ,
                )
            else:
                x0 = self.xy = None

        self.kernel(x0=x0, **kwargs)
        return mf_energy + self.e


class NTTDA(TDA):
    '''
    Noncollinear-Tensor TDA
    deltaS: -1 for Sf=Si-1, 0 for Sf=Si, 1 for Sf=Si+1
    nobeta: True for problemstic cases where there is no local beta electrons
    '''

    deltaS = -1
    nobeta = False

    _keys = TDA._keys.union({'deltaS', 'nobeta'})

    def __init__(self, mf):
        if not isinstance(mf, dft.roks.ROKS):
            raise TypeError('GPU NTTDA requires a gpu4pyscf ROKS reference')
        super().__init__(mf)

    def nuc_grad_method(self):
        from nest.gpu.grad.nttda import Gradients

        return Gradients(self)

    Gradients = nuc_grad_method

    def init_guess(self, hdiag, nstates=None):
        if nstates is None:
            nstates = self.nstates
        n_init = min(nstates + 3, hdiag.size)
        idx = cp.argsort(hdiag)[:n_init]
        x0 = cp.zeros((n_init, hdiag.size))
        x0[cp.arange(n_init), idx] = 1.0
        return x0

    def _transfer_initial_guess(self, xy, mol, mo_coeff, mo_occ):
        mf = self._scf
        overlap = cp.asarray(gto.intor_cross(
            'int1e_ovlp', mf.mol, mol,
        ))
        old_coeff = cp.asarray(mo_coeff)
        old_occ = cp.asarray(mo_occ)
        new_coeff = cp.asarray(mf.mo_coeff)
        new_occ = cp.asarray(mf.mo_occ)

        def space_projection(occupation):
            old = old_coeff[:, old_occ == occupation]
            new = new_coeff[:, new_occ == occupation]
            return new.T @ overlap @ old

        closed = space_projection(2)
        open_ = space_projection(1)
        virtual = space_projection(0)
        vectors = cp.stack([cp.asarray(x) for x, _ in xy])

        def project(values, left, right):
            projected = contract('ui,nij->nuj', left, values)
            return contract('nuj,vj->nuv', projected, right)

        if self.deltaS == 1:
            return project(vectors, closed, virtual).reshape(
                len(vectors), -1,
            )
        if self.deltaS == -1:
            nclosed_new, nclosed_old = closed.shape
            nopen_new, nopen_old = open_.shape
            nvirtual_new, nvirtual_old = virtual.shape
            holes = cp.zeros((
                nclosed_new + nopen_new,
                nclosed_old + nopen_old,
            ))
            holes[:nclosed_new, :nclosed_old] = closed
            holes[nclosed_new:, nclosed_old:] = open_
            particles = cp.zeros((
                nopen_new + nvirtual_new,
                nopen_old + nvirtual_old,
            ))
            particles[:nopen_new, :nopen_old] = open_
            particles[nopen_new:, nopen_old:] = virtual
            return project(vectors, holes, particles).reshape(
                len(vectors), -1,
            )
        if self.deltaS == 0:
            nclosed_old = closed.shape[1]
            nopen_old = open_.shape[1]
            nvirtual_old = virtual.shape[1]
            old_slices = _sc_vector_slices(
                nclosed_old,
                nopen_old,
                nvirtual_old,
            )

            def block(label, shape, left, right):
                values = vectors[:, old_slices[label]].reshape(
                    (len(vectors),) + shape,
                )
                return project(values, left, right).reshape(
                    len(vectors), -1,
                )

            return cp.concatenate((
                block(
                    'CO(1)',
                    (nclosed_old, nopen_old),
                    closed,
                    open_,
                ),
                block(
                    'CV(1)',
                    (nclosed_old, nvirtual_old),
                    closed,
                    virtual,
                ),
                vectors[:, old_slices['OO(1)']],
                block(
                    'OV(1)',
                    (nopen_old, nvirtual_old),
                    open_,
                    virtual,
                ),
                block(
                    'CV(0)',
                    (nclosed_old, nvirtual_old),
                    closed,
                    virtual,
                ),
            ), axis=1)
        raise ValueError('deltaS should be -1, 0, or 1')

    def kernel(self, x0=None, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())

        self.check_sanity()
        self.dump_flags()

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        requested_nstates = nstates
        if self.deltaS == -1:
            nstates += 1
        log = logger.Logger(self.stdout, self.verbose)

        def all_eigs(w, v, nroots, envs):
            return w, v, cp.arange(w.size)

        if self.deltaS == 0:
            vind, hdiag = self.gen_vind_sc()
            precond = self.get_precond(hdiag)
        elif self.deltaS == -1:
            vind, hdiag = self.gen_vind_sfd()
            precond = self.get_precond(hdiag)
            csidx, osidx, vsidx = _orbital_indices(self)
            nocc = len(csidx) + len(osidx)
            nvir = len(osidx) + len(vsidx)
        elif self.deltaS == 1:
            vind, hdiag = self.gen_vind_sfu()
            precond = self.get_precond(hdiag)
            csidx, _, vsidx = _orbital_indices(self)
            ncs = len(csidx)
            nvs = len(vsidx)
        else:
            raise ValueError('deltaS should be -1, 0, or 1')

        x0sym = None
        if x0 is None:
            x0 = self.init_guess(hdiag)
        elif len(x0) < nstates:
            x0 = cp.vstack((x0, self.init_guess(hdiag, nstates)))

        self.converged, self.e, x1 = lr_eigh(
            vind,
            x0,
            precond,
            tol_residual=self.conv_tol,
            lindep=self.lindep,
            nroots=nstates,
            x0sym=x0sym,
            pick=all_eigs,
            max_cycle=self.max_cycle,
            max_memory=self.max_memory,
            verbose=log,
        )

        if self.deltaS == 0:
            self.xy = [(xi, 0) for xi in x1]
        elif self.deltaS == -1:
            self.xy = [(xi.reshape(nocc, nvir), 0) for xi in x1]
            keep = np.flatnonzero(abs(self.e) > 1e-8)[:requested_nstates]
            self.e = self.e[keep]
            self.converged = self.converged[keep]
            self.xy = [self.xy[i] for i in keep]
            self.nstates = len(self.e)
        elif self.deltaS == 1:
            self.xy = [(xi.reshape(ncs, nvs), 0) for xi in x1]

        log.timer('NTTDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    gen_vind_sfu = gen_vind_sfu
    gen_vind_sc = gen_vind_sc
    gen_vind_sfd = gen_vind_sfd

    def get_ab(self, mf=None):
        return get_ab(self, mf)

    def analyze(self, *args, **kwargs):
        raise NotImplementedError('GPU NTTDA analysis is not implemented')

    as_scanner = as_scanner

    def to_cpu(self):
        raise NotImplementedError('GPU NTTDA to_cpu is not implemented')

    def NAC(self):
        raise NotImplementedError('GPU NTTDA NAC is not implemented')

    def NACGradients(self):
        raise NotImplementedError('GPU NTTDA NAC gradients are not implemented')

    def SOC(self, *args, **kwargs):
        raise NotImplementedError('GPU NTTDA SOC is not implemented')


dft.roks.ROKS.NTTDA = lib.class_as_method(NTTDA)
