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

import cupy as cp
from pyscf import gto, lib
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import add_sparse, contract, tag_array
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.tdscf.rhf import TDBase, TDA
from nest.nttda.nttda import NTTDA as CPU_NTTDA, _sc_vector_slices


def _get_j_lr(mf, dm, hermi, omega):
    """Contract the long-range Coulomb operator needed by NTTDA."""
    if getattr(mf, 'with_df', None) is not None:
        # DF.get_jk rejects long-range J; contract the same fitted integrals
        # used for long-range K through the underlying J/K routine.
        with mf.with_df.range_coulomb(omega, None, None) as dfobj:
            return df_jk.get_jk(dfobj, dm, hermi, with_k=False, omega=omega)[0]
    with mf.mol.with_range_coulomb(omega):
        return mf.get_j(mf.mol, dm, hermi, omega=omega)


def _nr_rks_fxc1(ni, mol, grids, dms, fxc, mgga=False):
    opt = ni.gdftopt
    sorted_mol = opt._sorted_mol
    nao = sorted_mol.nao
    factor_l = getattr(dms, 'factor_l', None)
    factor_r = getattr(dms, 'factor_r', None)
    if factor_l is not None and factor_r is not None:
        factor_l = opt.sort_orbitals(factor_l, axis=[1])
        factor_r = opt.sort_orbitals(factor_r, axis=[0])
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
    ):
        p0, p1 = p1, p1 + weight.size
        if len(mask) == 0:
            continue

        ao = ao[:4]
        if factor_l is not None and factor_r is not None:
            # Contract each directed low-rank factor before forming the small
            # derivative-component density, instead of a batch of G*NAO fields.
            left = contract('xqr,iqg->xirg', factor_l[:, mask], ao)
            right = contract('pr,jpg->jrg', factor_r[mask], ao)
            rho = contract('xirg,jrg->xijg', left, right)
        else:
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
            left.shape[:-1] + (right.shape[-1],),
            dtype=left.dtype,
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
    """
    response function for Sf=Si+1 with K^SF_0 = kernel
    """
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

    if mf.do_nlc():
        logger.warn(mf, 'NLC contribution in gen_response is NOT included')

    def vind(dms_cv):
        if xctype != 'HF':
            time_xc = (logger.process_clock(), logger.perf_counter())
            v1ao_cv = ni.nr_rks_fxc(
                mol, mf.grids, mf.xc, None, dms_cv, 0, hermi, None, None, fxc_ref, max_memory=max_memory
            )
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
    dmoo = tag_array(orbos @ orbos.T, factor_l=orbos[None], factor_r=orbos)
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
    """
    response function for Sf=Si
    """
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

    if mf.do_nlc():
        logger.warn(mf, 'NLC contribution in gen_response is NOT included')

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
            vref0 = ni.nr_rks_fxc(
                mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, max_memory=max_memory
            )
            time_xc = log.timer('NTTDA response_sc kernel vref0', *time_xc)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(
                    mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, max_memory=max_memory
                )
            elif xctype == 'GGA':
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
                vj += _get_j_lr(mf, dms1, hermi, omega) * (alpha - hyb)
            vref0 -= vk
            vref1 -= vj
            log.timer('NTTDA response_sc kernel get_j/get_k total', *time_jk)

        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_ov = vref0[idx2:idx3]
        vref0_cv0 = vref0[idx3:]
        vref1_co = vref1[:idx1]
        vref1_ov = vref1[idx1 : idx1 + n_ov]
        vref1_cv0 = vref1[idx1 + n_ov :]

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
    dmoo = tag_array(orbos @ orbos.T, factor_l=orbos[None], factor_r=orbos)
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
    """
    response function for Sf=Si-1
    """
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

    if mf.do_nlc():
        logger.warn(mf, 'NLC contribution in gen_response is NOT included')

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
            vref0 = ni.nr_rks_fxc(
                mol, mf.grids, mf.xc, None, dms0, 0, hermi, None, None, fxc_ref, max_memory=max_memory
            )
            time_xc = log.timer('NTTDA response_sf vref0', *time_xc)
            if xctype == 'LDA':
                vref1 = ni.nr_rks_fxc(
                    mol, mf.grids, mf.xc, None, dms1, 0, hermi, None, None, fxc_ref, max_memory=max_memory
                )
            elif xctype == 'GGA':
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
                vj += _get_j_lr(mf, dms1, hermi, omega) * (alpha - hyb)
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
    dmoo = tag_array(orbos @ orbos.T, factor_l=orbos[None], factor_r=orbos)
    if xctype != 'HF':
        delta = ni.nr_rks_fxc(mol, mf.grids, mf.xc, None, dmoo, 0, 1, None, None, fxc_ref, max_memory=max_memory)
    else:
        delta = cp.zeros_like(dmoo)
    if hybrid:
        delta -= mf.get_k(mol, dmoo, 1) * hyb
        if omega != 0:
            delta -= mf.get_k(mol, dmoo, 1, omega=omega) * (alpha - hyb)
    return vind, 0.5 * delta


def _orbital_indices(tdobj):
    occ = tdobj._scf.mo_occ
    return cp.flatnonzero(occ == 2), cp.flatnonzero(occ == 1), cp.flatnonzero(occ == 0)


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
    vresp, fockz = gen_rohf_response_sfu(
        mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, max_memory=td.max_memory, log=log
    )

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
    vresp, fockz = gen_rohf_response_sc(
        mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, max_memory=td.max_memory, log=log
    )

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
            contract('xov,pv->xpo', zs_co, orbos),
            right,
            core,
        )
        dms_cv = _factorized_density(
            contract('xov,pv->xpo', zs_cv, orbvs),
            right,
            core,
        )
        dms_ov = _factorized_density(
            contract('xov,pv->xpo', zs_ov, orbvs),
            right,
            open_,
        )
        dms_cv0 = _factorized_density(
            contract('xov,pv->xpo', zs_cv0, orbvs),
            right,
            core,
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

        v1mo_oo = cp.zeros((len(zs),))
        v1mo_oo -= contract('jv,xjv->x', fock_cooo.T, zs_co)
        v1mo_oo += contract('jb,xjb->x', fock_cvoo.T, zs_cv) * cp.sqrt(2 * (s + 1) / s)
        v1mo_oo += contract('vb,xvb->x', fock_ovoo.T, zs_ov)
        v1mo_oo -= contract('jb,xjb->x', fock_cv0oo.T, zs_cv0) * cp.sqrt(2)
        time1 = log.timer('NTTDA gen_vind_sc Fock part', *time1)

        v1mo = cp.concatenate(
            (
                v1mo_co.reshape(len(zs), -1),
                v1mo_cv.reshape(len(zs), -1),
                v1mo_oo.reshape(len(zs), -1),
                v1mo_ov.reshape(len(zs), -1),
                v1mo_cv0.reshape(len(zs), -1),
            ),
            axis=1,
        )
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
    vresp, fockz = gen_rohf_response_sfd(
        mf, mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=0, max_memory=td.max_memory, log=log
    )

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
        (cp.concatenate((hdiag_co, hdiag_cv), axis=1), cp.concatenate((hdiag_oo, hdiag_ov), axis=1)),
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
            contract('xov,pv->xpo', zs_co, orbos),
            right,
            core,
        )
        dms_cv = _factorized_density(
            contract('xov,pv->xpo', zs_cv, orbvs),
            right,
            core,
        )
        dms_oo = _factorized_density(
            contract('xov,pv->xpo', zs_oo, orbos),
            right,
            open_,
        )
        dms_ov = _factorized_density(
            contract('xov,pv->xpo', zs_ov, orbvs),
            right,
            open_,
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
            contract('jv,xjv->x', fock_cooo1.T, zs_co) / cp.sqrt(2 * s * (2 * s - 1))
        )[:, None]
        v1mo_oo[:, open_diag[0], open_diag[1]] -= (
            contract('jb,xjb->x', fock_cvoo.T, zs_cv) / s * cp.sqrt((2 * s + 1) / (2 * s - 1))
        )[:, None]
        v1mo_oo += contract('tv,xuv->xut', fock_oooo0, zs_oo)
        v1mo_oo -= contract('wu,xwt->xut', fock_oooo1, zs_oo)
        v1mo_oo += contract('tb,xub->xut', fock_ooov0, zs_ov) * cp.sqrt(2 * s / (2 * s - 1))
        v1mo_oo[:, open_diag[0], open_diag[1]] -= (
            contract('vb,xvb->x', fock_ooov1, zs_ov) / cp.sqrt(2 * s * (2 * s - 1))
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
    name = td.__class__.__name__ + NTTDA_Scanner.__name_mixin__
    return lib.set_class(NTTDA_Scanner(td), (NTTDA_Scanner, td.__class__), name)


class NTTDA_Scanner(lib.SinglePointScanner):
    device = 'gpu'

    def __init__(self, td):
        self.__dict__.update(td.__dict__)
        self._scf = td._scf.as_scanner()

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)
        self._scf(mol)
        # Rebuild guesses in the new orbital frame, as in the CPU solver.
        self.kernel(**kwargs)
        return self.e_tot


class NTTDA(TDBase):
    """GPU noncollinear tensor TDA for a ROKS reference.

    Amplitude blocks and spin-channel conventions match nest.nttda.NTTDA.
    Conventional and density-fitted references use the same response operator.
    """

    deltaS = CPU_NTTDA.deltaS
    nobeta = CPU_NTTDA.nobeta
    singlet = None
    device = 'gpu'
    _keys = TDBase._keys | {'deltaS', 'nobeta'}

    def __init__(self, mf):
        if not isinstance(mf, dft.roks.ROKS):
            raise TypeError('GPU NTTDA requires a gpu4pyscf ROKS reference')
        super().__init__(mf)

    dump_flags = CPU_NTTDA.dump_flags
    check_sanity = CPU_NTTDA.check_sanity
    get_precond = TDA.get_precond

    def init_guess(self, hdiag, nstates=None):
        if nstates is None:
            nstates = self.nstates
        n_init = min(nstates + 3, hdiag.size)
        idx = cp.argsort(hdiag)[:n_init]
        x0 = cp.zeros((n_init, hdiag.size))
        x0[cp.arange(n_init), idx] = 1.0
        return x0

    def kernel(self, x0=None, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())

        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        log = logger.new_logger(self)

        def all_eigs(w, v, nroots, envs):
            return w, v, cp.arange(w.size)

        if self.deltaS == 0:
            vind, hdiag = self.gen_vind_sc()
            precond = self.get_precond(hdiag)
        elif self.deltaS == -1:
            base_vind, hdiag = self.gen_vind_sfd()
            base_precond = self.get_precond(hdiag)
            csidx, osidx, vsidx = _orbital_indices(self)
            ncs, nos = len(csidx), len(osidx)
            nocc = ncs + nos
            nvir = nos + len(vsidx)
            nstates = min(nstates, hdiag.size - 1)
            self.nstates = nstates
            open_diag = cp.arange(nos)

            def project(zs):
                # Remove only the OO identity direction; keep the stored layout.
                projected = cp.array(zs, copy=True)
                oo = projected.reshape(-1, nocc, nvir)[:, ncs:, :nos]
                trace = cp.trace(oo, axis1=1, axis2=2) / nos
                oo[:, open_diag, open_diag] -= trace[:, None]
                return projected

            def vind(zs):
                return project(base_vind(project(zs)))

            def precond(residual, energy):
                # Unequal diagonal denominators can reintroduce the OO trace.
                return project(base_precond(residual, energy))
        elif self.deltaS == 1:
            vind, hdiag = self.gen_vind_sfu()
            precond = self.get_precond(hdiag)
            csidx, _, vsidx = _orbital_indices(self)
            ncs = len(csidx)
            nvs = len(vsidx)

        x0sym = None
        if x0 is None:
            x0 = self.init_guess(hdiag)
        if self.deltaS == -1:
            # Davidson's initial QR removes dependencies introduced by projection.
            x0 = project(x0)

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
            self.nstates = len(self.e)
        elif self.deltaS == 1:
            self.xy = [(xi.reshape(ncs, nvs), 0) for xi in x1]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', cp.asnumpy(self.e))
            lib.chkfile.save(self.chkfile, 'tddft/xy', [(cp.asnumpy(x), y) for x, y in self.xy])

        log.timer('NTTDA', *cpu0)
        self._finalize()
        return self.e, self.xy

    gen_vind_sfu = gen_vind_sfu
    gen_vind_sc = gen_vind_sc
    gen_vind_sfd = gen_vind_sfd
    as_scanner = as_scanner

    def gen_vind(self, mf=None):
        td = self if mf is None else self.copy().set(_scf=mf, mol=mf.mol)
        td.check_sanity()
        if td.deltaS == -1:
            return td.gen_vind_sfd()
        if td.deltaS == 0:
            return td.gen_vind_sc()
        return td.gen_vind_sfu()

    def get_ab(self, mf=None):
        """Return the independently constructed A matrix as a NumPy array."""
        from nest.gpu.nttda.get_ab import get_ab
        if mf is None:
            mf = self._scf
        return get_ab(mf, deltaS=self.deltaS, nobeta=self.nobeta).get()

    def nuc_grad_method(self):
        from nest.gpu.grad.nttda import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method

    def to_cpu(self):
        out = utils.to_cpu(self, out=CPU_NTTDA(self._scf.to_cpu()))
        if out.xy is not None:
            out.xy = [(cp.asnumpy(x), y) for x, y in out.xy]
        return out

    def analyze(self, verbose=None):
        return self.to_cpu().analyze(verbose)

    def transition_dipole(self, ref=1, state=None):
        return self.to_cpu().transition_dipole(ref, state)

    def oscillator_strength(self, ref=1, state=None):
        return self.to_cpu().oscillator_strength(ref, state)


dft.roks.ROKS.NTTDA = lib.class_as_method(NTTDA)
