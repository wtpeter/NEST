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

"""Analytic gradient for NTTDA ``deltaS=+1``."""

import cupy as cp

from nest.gpu.nttda.nttda import gen_rohf_response_sfu

from .common import (
    FockProjection,
    ResponseTerm,
    fock_response_q,
    orbital_spaces,
    pair_density,
)
from .roks import assemble_gradient


def split_spin_raising(tdobj, xy):
    """Validate and return the native ``C->V`` amplitude."""
    spaces = orbital_spaces(tdobj)
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = cp.asarray(vector)
    expected = (len(spaces.closed), len(spaces.virtual))
    if vector.size != expected[0] * expected[1]:
        raise ValueError('deltaS=+1 amplitude has size %d; expected %d' % (vector.size, expected[0] * expected[1]))
    return spaces, vector.reshape(expected)


def spin_raising_transition_densities(tdobj, xy):
    """Return the alpha-target/beta-source transition density."""
    spaces, amplitudes = split_spin_raising(tdobj, xy)
    return (
        spaces,
        amplitudes,
        {
            'CV': pair_density(spaces.c_virtual, amplitudes.T, spaces.c_closed),
        },
    )


def spin_raising_fock0_fockz(tdobj, max_memory=None):
    """Build the F0/Fz operators used by ``gen_vind_sfu``."""
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sfu(
        mf,
        mo_coeff=mf.mo_coeff,
        mo_occ=mf.mo_occ,
        hermi=0,
        max_memory=max_memory,
    )
    if tdobj.nobeta:
        density_alpha, density_beta = mf.make_rdm1()
        density0 = 0.5 * (density_alpha + density_beta)
        fock = mf.get_fock(dm=cp.stack((density0, density0)))
    else:
        fock = mf.get_fock()
    return 0.5 * (fock.focka + fock.fockb), fockz


def spin_raising_fock_projections(tdobj, xy):
    """Return the particle and hole explicit-Fock terms."""
    spaces, amplitudes = split_spin_raising(tdobj, xy)
    x = amplitudes
    return (
        FockProjection(
            name='particle-vv',
            left_indices=spaces.virtual,
            left_orbitals=spaces.c_virtual,
            coefficient=x.T @ x,
            right_indices=spaces.virtual,
            right_orbitals=spaces.c_virtual,
            weight_f0=1.0,
            weight_fz=1.0,
        ),
        FockProjection(
            name='hole-cc',
            left_indices=spaces.closed,
            left_orbitals=spaces.c_closed,
            coefficient=-(x @ x.T),
            right_indices=spaces.closed,
            right_orbitals=spaces.c_closed,
            weight_f0=1.0,
            weight_fz=-1.0,
        ),
    )


def spin_raising_fock_probes(tdobj, xy):
    """Return ``P0=Dv-Dc`` and ``Pz=Dv+Dc``."""
    nao = tdobj.mol.nao_nr()
    p0 = cp.zeros((nao, nao))
    pz = cp.zeros_like(p0)
    for term in spin_raising_fock_projections(tdobj, xy):
        density = term.density()
        p0 += term.weight_f0 * density
        pz += term.weight_fz * density
    return p0, pz


def _transverse_potential(tdobj, density):
    """Apply the hybrid/RSH part of the transverse response."""
    mf = tdobj._scf
    potential = cp.zeros_like(density)
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return potential
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(
        mf.xc,
        mf.mol.spin,
    )
    potential -= hybrid * mf.get_k(mf.mol, density, hermi=0)
    if omega != 0:
        potential -= (alpha - hybrid) * mf.get_k(
            mf.mol,
            density,
            hermi=0,
            omega=omega,
        )
    return potential


def spin_raising_fock_q(tdobj, xy, max_memory=None):
    """Explicit-Fock projection and reference response M matrices."""
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    fock0, fockz = spin_raising_fock0_fockz(
        tdobj,
        max_memory=max_memory,
    )
    fock0_mo = mo.conj().T @ fock0 @ mo
    fockz_mo = mo.conj().T @ fockz @ mo
    q_alpha = cp.zeros((nmo, nmo))
    q_beta = cp.zeros_like(q_alpha)
    is_hf = mf._numint._xc_type(mf.xc) == 'HF'

    for term in spin_raising_fock_projections(tdobj, xy):
        indices = term.left_indices
        coefficient = term.coefficient

        def project(target, operator, scale):
            if scale:
                target[:, indices] += scale * operator[:, indices] @ coefficient.T
                target[:, indices] += scale * operator[:, indices] @ coefficient

        project(q_alpha, fock0_mo, 0.5 * term.weight_f0)
        project(q_beta, fock0_mo, 0.5 * term.weight_f0)
        if is_hf:
            project(q_alpha, fockz_mo, 0.5 * term.weight_fz)
            project(q_beta, fockz_mo, 0.5 * term.weight_fz)
        else:
            project(q_alpha, fockz_mo, term.weight_fz)

    p0, pz = spin_raising_fock_probes(tdobj, xy)
    p_alpha = 0.5 * p0
    p_beta = 0.5 * p0
    if is_hf:
        p_alpha = p_alpha + 0.5 * pz
        p_beta = p_beta - 0.5 * pz
    response_alpha, response_beta = fock_response_q(
        tdobj,
        p_alpha,
        p_beta,
    )
    return q_alpha + response_alpha, q_beta + response_beta


def spin_raising_response_projection_q(tdobj, xy):
    """Transition-factor derivative of ``T:K_perp[T]``."""
    spaces, amplitudes, densities = spin_raising_transition_densities(
        tdobj,
        xy,
    )
    potential = 2.0 * _transverse_potential(
        tdobj,
        densities['CV'],
    )
    mo = cp.asarray(tdobj._scf.mo_coeff)
    potential_mo = mo.conj().T @ potential @ mo
    q_alpha = cp.zeros_like(potential_mo)
    q_beta = cp.zeros_like(potential_mo)
    q_alpha[:, spaces.virtual] += potential_mo[:, spaces.closed] @ amplitudes
    q_beta[:, spaces.closed] += potential_mo[spaces.virtual, :].T @ amplitudes.T
    return q_alpha, q_beta


def grad_elec(gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None):
    """Assemble the spin-raising excitation-energy derivative."""
    spaces, amplitudes, densities = spin_raising_transition_densities(tdobj, xy)
    blocks = {'CV': (spaces.virtual, spaces.closed, amplitudes.T)}
    channel_data = (spaces, amplitudes, densities, blocks, (ResponseTerm('CV', 'CV', 1.0, 0.0),), 'alpha')
    probes = spin_raising_fock_probes(tdobj, xy)
    fock_q = spin_raising_fock_q(tdobj, xy)
    response_q = spin_raising_response_projection_q(tdobj, xy)
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
