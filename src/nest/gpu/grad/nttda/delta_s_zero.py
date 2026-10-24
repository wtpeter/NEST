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

"""Analytic gradient for current NTTDA ``deltaS=0``.

The public ``grad_elec`` function exposes the complete channel data flow.
Shared ROKS and AO derivative primitives live in ``common``.
"""

from dataclasses import dataclass

import cupy as cp

from nest.gpu.nttda.nttda import gen_rohf_response_sc

from .common import (
    FockProjection,
    ResponseTerm,
    apply_hfx_responses,
    fock_response_q,
    orbital_spaces,
    pair_density,
)
from .roks import assemble_gradient


# Native amplitudes


@dataclass(frozen=True)
class SameSpinAmplitudes:
    """Five amplitude blocks used by ``NTTDA(deltaS=0)``."""

    co: cp.ndarray
    cv: cp.ndarray
    oo: float
    ov: cp.ndarray
    cv0: cp.ndarray


def same_spin_slices(spaces):
    """Return canonical slices for ``CO/CV/OO/OV/CV0`` amplitudes."""
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    nco = nc * no
    ncv = nc * nv
    nov = no * nv
    i1 = nco
    i2 = i1 + ncv
    i3 = i2 + 1
    i4 = i3 + nov
    return {
        'CO': slice(0, i1),
        'CV': slice(i1, i2),
        'OO': slice(i2, i3),
        'OV': slice(i3, i4),
        'CV0': slice(i4, i4 + ncv),
    }


def split_same_spin(tdobj, xy):
    """Split one packed ``deltaS=0`` vector into its five native blocks."""
    spaces = orbital_spaces(tdobj)
    if spaces.spin < 0.5:
        raise ValueError('NTTDA deltaS=0 requires at least one open orbital')
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = cp.asarray(vector).reshape(-1)
    slices = same_spin_slices(spaces)
    expected = slices['CV0'].stop
    if vector.size != expected:
        raise ValueError('deltaS=0 amplitude has size %d; expected %d' % (vector.size, expected))
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    return spaces, SameSpinAmplitudes(
        co=vector[slices['CO']].reshape(nc, no),
        cv=vector[slices['CV']].reshape(nc, nv),
        oo=float(vector[slices['OO']][0]),
        ov=vector[slices['OV']].reshape(no, nv),
        cv0=vector[slices['CV0']].reshape(nc, nv),
    )


def same_spin_transition_densities(tdobj, xy):
    """Return directed AO transition densities for the four response blocks."""
    spaces, amp = split_same_spin(tdobj, xy)
    return (
        spaces,
        amp,
        {
            'CO': pair_density(spaces.c_open, amp.co.T, spaces.c_closed),
            'CV': pair_density(spaces.c_virtual, amp.cv.T, spaces.c_closed),
            'OV': pair_density(spaces.c_virtual, amp.ov.T, spaces.c_open),
            'CV0': pair_density(spaces.c_virtual, amp.cv0.T, spaces.c_closed),
        },
    )


# Explicit F0/Fz ledger


def fock0_fockz(tdobj, max_memory=None):
    """Build exactly the ``F0`` and ``Fz`` matrices used by ``gen_vind_sc``."""
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sc(
        mf,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        hermi=0,
        max_memory=max_memory,
    )
    if tdobj.nobeta:
        dm_alpha, dm_beta = mf.make_rdm1()
        dm0 = 0.5 * (dm_alpha + dm_beta)
        fock = mf.get_fock(dm=cp.stack((dm0, dm0)))
    else:
        fock = mf.get_fock()
    fock0 = 0.5 * (fock.focka + fock.fockb)
    return fock0, fockz


def same_spin_fock_projections(tdobj, xy):
    """Return the complete five-block explicit-Fock ledger."""
    spaces, x = split_same_spin(tdobj, xy)
    spin = spaces.spin
    c = spaces.c_closed
    o = spaces.c_open
    v = spaces.c_virtual
    a = cp.sqrt((spin + 1.0) / (2.0 * spin))
    b = cp.sqrt(2.0 * (spin + 1.0) / spin)
    d = cp.sqrt((spin + 1.0) / spin)
    h = cp.sqrt(0.5)

    terms = []

    def indices(orbitals):
        if orbitals is c:
            return spaces.closed
        if orbitals is o:
            return spaces.open
        if orbitals is v:
            return spaces.virtual
        raise ValueError('Fock projection uses an unknown orbital space')

    def add(name, left, coefficient, right, f0, fz):
        coefficient = cp.asarray(coefficient)
        if coefficient.size:
            terms.append(
                FockProjection(
                    name,
                    indices(left),
                    left,
                    coefficient,
                    indices(right),
                    right,
                    float(f0),
                    float(fz),
                )
            )

    # CO row/column and its couplings.
    add('co-oo', o, x.co.T @ x.co, o, 1.0, -1.0)
    add('co-cc', c, -x.co @ x.co.T, c, 1.0, -1.0)
    add('co-cv', o, 2.0 * a * (x.co.T @ x.cv), v, 1.0, -1.0)
    add('co-oo1', o, -2.0 * x.oo * x.co.T, c, 1.0, -1.0)
    add('co-cv0', o, 2.0 * h * (x.co.T @ x.cv0), v, 1.0, -1.0)

    # CV block and its OO/OV/CV0 couplings.
    add('cv-vv', v, x.cv.T @ x.cv, v, 1.0, -1.0 / spin)
    add('cv-cc', c, -x.cv @ x.cv.T, c, 1.0, 1.0 / spin)
    add('cv-oo1', v, 2.0 * b * x.oo * x.cv.T, c, 0.0, 1.0)
    add('cv-ov', o, -2.0 * a * (x.ov @ x.cv.T), c, 1.0, 1.0)
    add(
        'cv-cv0-vv',
        v,
        -d * (x.cv.T @ x.cv0 + x.cv0.T @ x.cv),
        v,
        0.0,
        1.0,
    )
    add(
        'cv-cv0-cc',
        c,
        d * (x.cv0 @ x.cv.T + x.cv @ x.cv0.T),
        c,
        0.0,
        1.0,
    )

    # OV and CV0 diagonal/coupling terms.
    add('ov-vv', v, x.ov.T @ x.ov, v, 1.0, 1.0)
    add('ov-oo', o, -x.ov @ x.ov.T, o, 1.0, 1.0)
    add('ov-oo1', v, 2.0 * x.oo * x.ov.T, o, 1.0, 1.0)
    add('ov-cv0', c, 2.0 * h * (x.cv0 @ x.ov.T), o, 1.0, 1.0)
    add('cv0-vv', v, x.cv0.T @ x.cv0, v, 1.0, 0.0)
    add('cv0-cc', c, -x.cv0 @ x.cv0.T, c, 1.0, 0.0)
    add('cv0-oo1', v, -2.0 * cp.sqrt(2.0) * x.oo * x.cv0.T, c, 1.0, 0.0)
    return tuple(terms)


def same_spin_fock_probes(tdobj, xy):
    """Return AO probes ``(P0, Pz)`` generated from the Fock ledger."""
    nao = tdobj.mol.nao_nr()
    p0 = cp.zeros((nao, nao))
    pz = cp.zeros_like(p0)
    for term in same_spin_fock_projections(tdobj, xy):
        density = term.density()
        p0 += term.weight_f0 * density
        pz += term.weight_fz * density
    return p0, pz


def same_spin_fock_q(tdobj, xy, max_memory=None):
    """Return the explicit-Fock contribution to ``(Q_alpha,Q_beta)``.

    For HF the complete ``Fz`` response is exactly represented by the
    spin-resolved probes.  DFT callers add the independent ``Fz`` and
    ``nobeta`` response ledgers after this common ``F0`` contribution.
    """
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    fock0, fockz = fock0_fockz(tdobj, max_memory=max_memory)
    fock0_mo = mo.conj().T @ fock0 @ mo
    fockz_mo = mo.conj().T @ fockz @ mo
    q_alpha = cp.zeros((nmo, nmo))
    q_beta = cp.zeros_like(q_alpha)

    for term in same_spin_fock_projections(tdobj, xy):
        left = term.left_indices
        right = term.right_indices
        coeff = term.coefficient

        def project(target, operator, scale):
            if scale == 0.0:
                return
            target[:, left] += scale * operator[:, right] @ coeff.T
            target[:, right] += scale * operator[:, left] @ coeff

        project(q_alpha, fock0_mo, 0.5 * term.weight_f0)
        project(q_beta, fock0_mo, 0.5 * term.weight_f0)
        if mf._numint._xc_type(mf.xc) == 'HF':
            project(q_alpha, fockz_mo, 0.5 * term.weight_fz)
            project(q_beta, fockz_mo, 0.5 * term.weight_fz)
        else:
            project(q_alpha, fockz_mo, term.weight_fz)

    p0, pz = same_spin_fock_probes(tdobj, xy)
    p_alpha = 0.5 * p0
    p_beta = 0.5 * p0
    if mf._numint._xc_type(mf.xc) == 'HF':
        p_alpha = p_alpha + 0.5 * pz
        p_beta = p_beta - 0.5 * pz
    response_alpha, response_beta = fock_response_q(
        tdobj,
        p_alpha,
        p_beta,
    )
    q_alpha += response_alpha
    q_beta += response_beta
    return q_alpha, q_beta


# vref0/vref1 response ledger


def same_spin_response_terms(spin):
    """Directed coefficients transcribed from ``gen_rohf_response_sc``."""
    a = cp.sqrt((spin + 1.0) / (2.0 * spin))
    h = cp.sqrt(0.5)
    r2 = cp.sqrt(2.0)
    return (
        ResponseTerm('CO', 'CO', 1.0, -1.0),
        ResponseTerm('CO', 'CV', a, 0.0),
        ResponseTerm('CO', 'OV', 0.0, 1.0),
        ResponseTerm('CO', 'CV0', h, -r2),
        ResponseTerm('CV', 'CO', a, 0.0),
        ResponseTerm('CV', 'CV', 1.0, 0.0),
        ResponseTerm('CV', 'OV', a, 0.0),
        ResponseTerm('OV', 'CO', 0.0, 1.0),
        ResponseTerm('OV', 'CV', a, 0.0),
        ResponseTerm('OV', 'OV', 1.0, -1.0),
        ResponseTerm('OV', 'CV0', -h, r2),
        ResponseTerm('CV0', 'CO', h, -r2),
        ResponseTerm('CV0', 'OV', -h, r2),
        ResponseTerm('CV0', 'CV0', 1.0, -2.0),
    )


def _derivative_potentials(spaces, densities, vref0, vref1):
    potentials = {label: cp.zeros_like(dm) for label, dm in densities.items()}
    for term in same_spin_response_terms(spaces.spin):
        if term.vref0:
            potentials[term.target] += term.vref0 * vref0[term.source]
            potentials[term.source] += term.vref0 * vref0[term.target]
        if term.vref1:
            potentials[term.target] += term.vref1 * vref1[term.source]
            potentials[term.source] += term.vref1 * vref1[term.target]
    return potentials


def same_spin_hfx_projection_q(tdobj, xy):
    """Transition-factor derivative of only the hybrid/RSH response scalar."""
    spaces, amp, densities = same_spin_transition_densities(tdobj, xy)
    vref0, vref1 = apply_hfx_responses(tdobj, densities)
    potentials = _derivative_potentials(
        spaces,
        densities,
        vref0,
        vref1,
    )
    mo = cp.asarray(tdobj._scf.mo_coeff)
    q_alpha = cp.zeros((mo.shape[1], mo.shape[1]))
    q_beta = cp.zeros_like(q_alpha)
    block_data = {
        'CO': (spaces.open, spaces.closed, amp.co.T),
        'CV': (spaces.virtual, spaces.closed, amp.cv.T),
        'OV': (spaces.virtual, spaces.open, amp.ov.T),
        'CV0': (spaces.virtual, spaces.closed, amp.cv0.T),
    }
    for label, (target, source, coefficient) in block_data.items():
        potential_mo = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential_mo[:, source] @ coefficient.T
        q_alpha[:, source] += potential_mo[target, :].T @ coefficient
    return q_alpha, q_beta


# Channel assembly


def grad_elec(gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None):
    """Assemble the same-spin excitation-energy derivative."""
    spaces, amplitudes, densities = same_spin_transition_densities(tdobj, xy)
    blocks = {
        'CO': (spaces.open, spaces.closed, amplitudes.co.T),
        'CV': (spaces.virtual, spaces.closed, amplitudes.cv.T),
        'OV': (spaces.virtual, spaces.open, amplitudes.ov.T),
        'CV0': (spaces.virtual, spaces.closed, amplitudes.cv0.T),
    }
    channel_data = (spaces, amplitudes, densities, blocks, same_spin_response_terms(spaces.spin), 'beta')
    probes = same_spin_fock_probes(tdobj, xy)
    fock_q = same_spin_fock_q(tdobj, xy)
    response_q = same_spin_hfx_projection_q(tdobj, xy)
    return assemble_gradient(
        gradient_driver,
        tdobj,
        channel_data,
        probes,
        fock_q,
        response_q,
        atmlst,
        tolerance,
        max_cycle,
    )
