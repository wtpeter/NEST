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

"""Explicit NT-TDA matrix elements after reference-kernel reconstruction."""

import numpy as np
import cupy as cp
from pyscf import ao2mo
from gpu4pyscf import dft
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract


def _get_eri(mf, mo, omega=None):
    """Transform conventional or fitted two-electron integrals to the MO basis."""
    nmo = mo.shape[1]
    if getattr(mf, 'with_df', None):
        with mf.with_df.range_coulomb(omega, None, None) as dfobj:
            if dfobj._cderi is None:
                dfobj.build(omega=omega)
            return dfobj.ao2mo(mo, compact=False).reshape((nmo,) * 4)

    mol = mf.mol
    with mol.with_range_coulomb(omega):
        eri = mol.intor('int2e', aosym='s8')
    eri = cp.asarray(ao2mo.restore(1, eri, mol.nao_nr()))
    eri = contract('pjkl,pi->ijkl', eri, mo)
    eri = contract('ipkl,pj->ijkl', eri, mo)
    eri = contract('ijpl,pk->ijkl', eri, mo)
    return contract('ijkp,pl->ijkl', eri, mo)


def get_ab(mf, deltaS=-1, nobeta=False):
    r"""Return the dense CuPy A matrix for a GPU ROKS reference (there is no B matrix).

    i,j denote closed-shell orbitals; a,b virtual orbitals; u,v,t,w open-shell
    orbitals. Amplitudes store the hole before the particle: X[i,a] for CV,
    X[i,u] for CO, X[u,a] for OV, and X[u,t] for the OO amplitude in A_{tu,vw}.

    deltaS=-1 uses [[CO, CV], [OO, OV]].ravel(), including its redundant
    OO identity zero mode. deltaS=0 concatenates CO, CV, scalar OO, OV, CV0.
    deltaS=+1 uses CV.ravel(). For a batch x of flattened row vectors,
    vind(x) = (A @ x.T).T.

    K[p,q,r,s] is K^Ref_{pq,rs}, and Fz[p,q] = 1/2 sum_u K[p,q,u,u].
    XC kernels are contracted directly with MO values and derivatives;
    neither response functions nor gen_vind are used in this construction.
    This dense implementation stores full four-index MO tensors and is
    intended for small systems. With density fitting, both J and K must use
    the same fitted integrals (only_dfj=True is not supported).
    """
    if not isinstance(mf, dft.roks.ROKS):
        raise TypeError('GPU NTTDA get_ab requires a gpu4pyscf ROKS reference')
    if deltaS not in (-1, 0, 1):
        raise ValueError('deltaS must be -1, 0, or 1')
    with_df = getattr(mf, 'with_df', None)
    if with_df and getattr(mf, 'only_dfj', False):
        raise NotImplementedError('NTTDA get_ab does not support only_dfj=True')

    mol = mf.mol
    csidx = cp.where(mf.mo_occ == 2)[0]
    osidx = cp.where(mf.mo_occ == 1)[0]
    vsidx = cp.where(mf.mo_occ == 0)[0]
    nc, no, nv = len(csidx), len(osidx), len(vsidx)
    s = no * 0.5
    if deltaS == -1:
        assert s >= 1, 'NTTDA for Sf=Si-1 only supports case that Si>=1.'
    elif deltaS == 0:
        assert s >= 0.5, 'NTTDA only supports case that Sf=Si>=1/2.'
    assert s == mol.spin * 0.5
    order = cp.concatenate((csidx, osidx, vsidx))
    mo = mf.mo_coeff[:, order]
    assert mo.dtype == np.double
    nmo = nc + no + nv
    c, o, v = slice(0, nc), slice(nc, nc + no), slice(nc + no, nmo)
    ic, io, iv = cp.eye(nc), cp.eye(no), cp.eye(nv)

    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('HF', 'LDA', 'GGA', 'MGGA'):
        raise NotImplementedError('NTTDA get_ab does not support XC type %s' % xctype)
    if mf.do_nlc():
        logger.warn(mf, 'NLC contribution in get_ab is NOT included')
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

    # Reference kernel: exact exchange K_{pq,rs} = -(pr|qs).
    kref = cp.zeros((nmo, nmo, nmo, nmo))
    if hyb != 0:
        eri = _get_eri(mf, mo)
        kref -= hyb * eri.transpose(0, 2, 1, 3)
    if omega != 0:
        eri = _get_eri(mf, mo, omega=omega)
        kref -= (alpha - hyb) * eri.transpose(0, 2, 1, 3)

    # The reference XC Hessian is evaluated at rho_alpha = rho_beta = rho/2.
    # Pair features are phi_p phi_q, grad(phi_p phi_q), and (for MGGA)
    # 1/2 grad(phi_p).grad(phi_q). Contract both pairs with fxc and grid weights.
    if xctype != 'HF':
        if mf.grids.coords is None:
            mf.grids.build()
        opt = getattr(ni, 'gdftopt', None)
        if opt is None or mol not in (opt.mol, opt._sorted_mol):
            ni.build(mol, mf.grids.coords)
            opt = ni.gdftopt
        sorted_mol = opt._sorted_mol
        mo_grid = opt.sort_orbitals(mo, axis=[0])
        ao_deriv = 0 if xctype == 'LDA' else 1
        for ao, mask, weight, coords in ni.block_loop(sorted_mol, mf.grids, sorted_mol.nao, ao_deriv):
            mo_mask = mo_grid[mask]
            rho = ni.eval_rho2(sorted_mol, ao, mo_mask, mf.mo_occ[order], None, xctype, with_lapl=False) * .5
            fxc = ni.eval_xc_eff(mf.xc, cp.stack((rho, rho)), deriv=2, xctype=xctype, spin=1)[2]
            fxc_ref = .5 * (fxc[0, :, 0] - fxc[0, :, 1] - fxc[1, :, 0] + fxc[1, :, 1])
            if xctype == 'LDA':
                phi = contract('pg,pi->gi', ao, mo_mask)
                pair = cp.einsum('gp,gq->gpq', phi, phi)
                weighted_pair = pair * (fxc_ref[0, 0] * weight)[:, None, None]
                kref += contract('gpq,grs->pqrs', pair, weighted_pair)
            else:
                phi = contract('xpg,pi->xgi', ao, mo_mask)
                pair = cp.einsum('xgp,gq->xgpq', phi, phi[0])
                pair[1:4] += cp.einsum('gp,xgq->xgpq', phi[0], phi[1:4])
                if xctype == 'MGGA':
                    tau_pair = .5 * cp.einsum('xgp,xgq->gpq', phi[1:4], phi[1:4])
                    pair = cp.concatenate((pair, tau_pair[None]), axis=0)
                weighted_pair = contract('xyg,ygpq->xgpq', fxc_ref * weight, pair)
                kref += contract('xgpq,xgrs->pqrs', pair, weighted_pair)

    # Reconstructed Fz and spin-averaged F0, both in the (C, O, V) MO basis.
    fz = .5 * cp.einsum('pquu->pq', kref[:, :, o, o])
    if nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = .5 * (dma + dmb)
        fock = mf.get_fock(dm=cp.stack((dm0, dm0)))
    else:
        fock = mf.get_fock()
    f0 = mo.T @ (.5 * (fock.focka + fock.fockb)) @ mo
    fp, fm = f0 + fz, f0 - fz

    if deltaS == 1:
        # CV-CV: delta_ij (F0+Fz)_ab - delta_ab (F0-Fz)_ji + K_ai,bj.
        a = cp.einsum('ij,ab->iajb', ic, fp[v, v])
        a -= cp.einsum('ab,ji->iajb', iv, fm[c, c])
        a += cp.einsum('aibj->iajb', kref[v, c, v, c])
        return a.reshape(nc * nv, nc * nv)

    if deltaS == -1:
        # CV-CV
        a_cvcv = cp.einsum('ij,ab->iajb', ic, fm[v, v])
        a_cvcv -= cp.einsum('ab,ji->iajb', iv, fp[c, c])
        a_cvcv -= cp.einsum('ij,ab->iajb', ic, fz[v, v]) / s
        a_cvcv -= cp.einsum('ab,ji->iajb', iv, fz[c, c]) / s
        a_cvcv += cp.einsum('aibj->iajb', kref[v, c, v, c])

        # CV-CO and CV-OV
        a_cvco = cp.einsum('ij,av->iajv', ic, fm[v, o])
        a_cvco += cp.einsum('aivj->iajv', kref[v, c, o, c])
        a_cvco *= np.sqrt((2 * s + 1) / (2 * s))
        a_cvov = -cp.einsum('ab,vi->iavb', iv, fp[o, c])
        a_cvov += cp.einsum('aibv->iavb', kref[v, c, v, o])
        a_cvov *= np.sqrt((2 * s + 1) / (2 * s))

        # CO-CO
        a_coco = cp.einsum('ij,uv->iujv', ic, fm[o, o])
        a_coco -= cp.einsum('uv,ji->iujv', io, fp[c, c])
        a_coco -= 2 / (2 * s - 1) * cp.einsum('uv,ji->iujv', io, fz[c, c])
        a_coco += cp.einsum('uivj->iujv', kref[o, c, o, c])
        a_coco += cp.einsum('uvij->iujv', kref[o, o, c, c]) / (2 * s - 1)

        # CO-OV
        a_coov = -cp.einsum('ubiv->iuvb', kref[o, v, c, o]) / (2 * s - 1)
        a_coov += 2 * s / (2 * s - 1) * cp.einsum('uibv->iuvb', kref[o, c, v, o])

        # OV-OV
        a_ovov = cp.einsum('uv,ab->uavb', io, fm[v, v])
        a_ovov -= cp.einsum('ab,vu->uavb', iv, fp[o, o])
        a_ovov -= 2 / (2 * s - 1) * cp.einsum('uv,ab->uavb', io, fz[v, v])
        a_ovov += cp.einsum('aubv->uavb', kref[v, o, v, o])
        a_ovov += cp.einsum('abuv->uavb', kref[v, v, o, o]) / (2 * s - 1)

        # OO-CV: row indices are (u,t), corresponding to A_tu,bj.
        a_oocv = -cp.einsum('ut,jb->utjb', io, fz[c, v]) / s
        a_oocv += cp.einsum('tubj->utjb', kref[o, o, v, c])
        a_oocv *= np.sqrt((2 * s + 1) / (2 * s - 1))

        # OO-CO
        a_ooco = -np.sqrt(2 * s / (2 * s - 1)) * cp.einsum('vt,ju->utjv', io, fp[c, o])
        a_ooco += cp.einsum('ut,jv->utjv', io, fm[c, o]) / np.sqrt(2 * s * (2 * s - 1))
        a_ooco += np.sqrt(2 * s / (2 * s - 1)) * cp.einsum('tuvj->utjv', kref[o, o, o, c])

        # OO-OV
        a_ooov = np.sqrt(2 * s / (2 * s - 1)) * cp.einsum('uv,tb->utvb', io, fm[o, v])
        a_ooov -= cp.einsum('tu,vb->utvb', io, fp[o, v]) / np.sqrt(2 * s * (2 * s - 1))
        a_ooov += np.sqrt(2 * s / (2 * s - 1)) * cp.einsum('tubv->utvb', kref[o, o, v, o])

        # OO-OO
        a_oooo = cp.einsum('wu,tv->utwv', io, fm[o, o])
        a_oooo -= cp.einsum('tv,wu->utwv', io, fp[o, o])
        a_oooo += cp.einsum('tuvw->utwv', kref[o, o, o, o])

        # Native [[CO, CV], [OO, OV]] layout; transpose the reverse blocks.
        nocc, nvir = nc + no, no + nv
        a = cp.empty((nocc, nvir, nocc, nvir))
        a[:nc, :no, :nc, :no] = a_coco
        a[:nc, :no, :nc, no:] = a_cvco.transpose(2, 3, 0, 1)
        a[:nc, :no, nc:, :no] = a_ooco.transpose(2, 3, 0, 1)
        a[:nc, :no, nc:, no:] = a_coov
        a[:nc, no:, :nc, :no] = a_cvco
        a[:nc, no:, :nc, no:] = a_cvcv
        a[:nc, no:, nc:, :no] = a_oocv.transpose(2, 3, 0, 1)
        a[:nc, no:, nc:, no:] = a_cvov
        a[nc:, :no, :nc, :no] = a_ooco
        a[nc:, :no, :nc, no:] = a_oocv
        a[nc:, :no, nc:, :no] = a_oooo
        a[nc:, :no, nc:, no:] = a_ooov
        a[nc:, no:, :nc, :no] = a_coov.transpose(2, 3, 0, 1)
        a[nc:, no:, :nc, no:] = a_cvov.transpose(2, 3, 0, 1)
        a[nc:, no:, nc:, :no] = a_ooov.transpose(2, 3, 0, 1)
        a[nc:, no:, nc:, no:] = a_ovov
        return a.reshape(nocc * nvir, nocc * nvir)

    # deltaS=0: CV0-CV0
    a_cv0cv0 = cp.einsum('ij,ab->iajb', ic, f0[v, v])
    a_cv0cv0 -= cp.einsum('ab,ji->iajb', iv, f0[c, c])
    a_cv0cv0 += cp.einsum('aibj->iajb', kref[v, c, v, c])
    a_cv0cv0 -= 2 * cp.einsum('abij->iajb', kref[v, v, c, c])

    # CV0-CV
    a_cv0cv = -cp.einsum('ij,ab->iajb', ic, fz[v, v])
    a_cv0cv += cp.einsum('ab,ji->iajb', iv, fz[c, c])
    a_cv0cv *= np.sqrt((s + 1) / s)

    # CV0-CO
    a_cv0co = cp.einsum('ij,av->iajv', ic, fm[v, o]) / np.sqrt(2)
    a_cv0co += cp.einsum('aivj->iajv', kref[v, c, o, c]) / np.sqrt(2)
    a_cv0co -= np.sqrt(2) * cp.einsum('avij->iajv', kref[v, o, c, c])

    # CV0-OV
    a_cv0ov = cp.einsum('ab,vi->iavb', iv, fp[o, c]) / np.sqrt(2)
    a_cv0ov -= cp.einsum('aibv->iavb', kref[v, c, v, o]) / np.sqrt(2)
    a_cv0ov += np.sqrt(2) * cp.einsum('abiv->iavb', kref[v, v, c, o])

    # CV-CV
    a_cvcv = cp.einsum('ij,ab->iajb', ic, f0[v, v] - fz[v, v] / s)
    a_cvcv -= cp.einsum('ab,ji->iajb', iv, f0[c, c] + fz[c, c] / s)
    a_cvcv += cp.einsum('aibj->iajb', kref[v, c, v, c])

    # CV-CO and CV-OV
    a_cvco = cp.einsum('ij,av->iajv', ic, fm[v, o])
    a_cvco += cp.einsum('aivj->iajv', kref[v, c, o, c])
    a_cvco *= np.sqrt((s + 1) / (2 * s))
    a_cvov = -cp.einsum('ab,vi->iavb', iv, fp[o, c])
    a_cvov += cp.einsum('aibv->iavb', kref[v, c, v, o])
    a_cvov *= np.sqrt((s + 1) / (2 * s))

    # CO-CO and CO-OV
    a_coco = cp.einsum('ij,uv->iujv', ic, fm[o, o])
    a_coco -= cp.einsum('uv,ji->iujv', io, fm[c, c])
    a_coco += cp.einsum('uivj->iujv', kref[o, c, o, c])
    a_coco -= cp.einsum('uvij->iujv', kref[o, o, c, c])
    a_coov = cp.einsum('ubiv->iuvb', kref[o, v, c, o])

    # OV-OV
    a_ovov = cp.einsum('uv,ab->uavb', io, fp[v, v])
    a_ovov -= cp.einsum('ab,vu->uavb', iv, fp[o, o])
    a_ovov += cp.einsum('aubv->uavb', kref[v, o, v, o])
    a_ovov -= cp.einsum('abuv->uavb', kref[v, v, o, o])

    # The OO block is a scalar in this sector.
    a_oocv0 = -np.sqrt(2) * f0[c, v].reshape(1, -1)
    a_oocv = 2 * np.sqrt((s + 1) / (2 * s)) * fz[c, v].reshape(1, -1)
    a_ooco = -fm[c, o].reshape(1, -1)
    a_ooov = fp[o, v].reshape(1, -1)

    # Native order: CO, CV, scalar OO, OV, CV0.
    nco, ncv, nov = nc * no, nc * nv, no * nv
    a_coco = a_coco.reshape(nco, nco)
    a_cvcv = a_cvcv.reshape(ncv, ncv)
    a_ovov = a_ovov.reshape(nov, nov)
    a_cvco = a_cvco.reshape(ncv, nco)
    a_cvov = a_cvov.reshape(ncv, nov)
    a_coov = a_coov.reshape(nco, nov)
    a_cv0cv0 = a_cv0cv0.reshape(ncv, ncv)
    a_cv0cv = a_cv0cv.reshape(ncv, ncv)
    a_cv0co = a_cv0co.reshape(ncv, nco)
    a_cv0ov = a_cv0ov.reshape(ncv, nov)
    blocks = [
        [a_coco,   a_cvco.T, a_ooco.T,         a_coov,   a_cv0co.T],
        [a_cvco,   a_cvcv,   a_oocv.T,         a_cvov,   a_cv0cv.T],
        [a_ooco,   a_oocv,   cp.zeros((1, 1)), a_ooov,   a_oocv0],
        [a_coov.T, a_cvov.T, a_ooov.T,         a_ovov,   a_cv0ov.T],
        [a_cv0co,  a_cv0cv,  a_oocv0.T,        a_cv0ov,  a_cv0cv0],
    ]
    return cp.vstack([cp.hstack(row) for row in blocks])
