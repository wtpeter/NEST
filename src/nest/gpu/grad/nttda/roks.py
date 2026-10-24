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

"""ROKS transpose-Hessian adjoint and final NTTDA gradient assembly."""

from dataclasses import dataclass

import cupy as cp
from gpu4pyscf.df.df_jk import _make_factorized_dm

from . import xc as xc_backend
from .common import (
    gen_roks_response,
    JKDerivativeLedger,
    response_direct_hfx,
    spin_fock_direct_hf,
    spin_fock_direct_dft,
    spin_fockz_hfx_terms,
)


@dataclass(frozen=True)
class GradientComponents:
    """Excitation-gradient pieces and the solved ROKS adjoint."""

    m_matrix: cp.ndarray
    direct: cp.ndarray
    orbital: cp.ndarray
    total: cp.ndarray
    zvector: cp.ndarray
    residual: float


@dataclass(frozen=True)
class CanonicalPairs:
    """GPU indices and spin weights for nonredundant ROKS rotations."""

    p: cp.ndarray
    q: cp.ndarray
    alpha_weight: cp.ndarray
    beta_weight: cp.ndarray

    def unpack(self, vector, nmo):
        source_alpha = cp.zeros((nmo, nmo))
        source_beta = cp.zeros_like(source_alpha)
        source_alpha[self.p, self.q] = vector * self.alpha_weight
        source_beta[self.p, self.q] = vector * self.beta_weight
        return source_alpha, source_beta

    def pack(self, matrix):
        return (matrix - matrix.T)[self.p, self.q]

    def preconditioner(self, epsilon_alpha, epsilon_beta):
        return self.alpha_weight * (epsilon_alpha[self.p] - epsilon_alpha[self.q]) + self.beta_weight * (
            epsilon_beta[self.p] - epsilon_beta[self.q]
        )


def assemble_gradient(
    driver, tdobj, channel_data, probes, fock_q, response_q, atmlst=None, tolerance=1e-12, max_cycle=None
):
    """Explicit derivative + ROKS orbital response, shared by all three channels.

    The channel supplies the spin coefficients and transition densities.
    This routine adds XC and J/K derivatives, solves H.T z = dE/dkappa,
    and contracts the Z-vector and overlap terms in the same orbital frame.
    """
    atmlst = tuple(range(tdobj.mol.natm) if atmlst is None else atmlst)
    spaces, _, densities, _, response_terms, _ = channel_data
    p0, pz = probes
    m_matrix = fock_q[0] + fock_q[1] + response_q[0] + response_q[1]
    ledger = JKDerivativeLedger()
    slots = ('direct', 'zvector')
    direct = response_direct_hfx(
        driver,
        tdobj,
        densities,
        response_terms,
        atmlst,
        jk_ledger=ledger,
        output_slot='direct',
    )
    xctype = tdobj._scf._numint._xc_type(tdobj._scf.xc)
    if xctype == 'HF':
        direct_alpha, direct_beta = 0.5 * (p0 + pz), 0.5 * (p0 - pz)
    else:
        response_builder, fockz_builder, nobeta_builder = {
            'LDA': (xc_backend.lda_response_terms, xc_backend.lda_fockz_terms, xc_backend.lda_nobeta_reference_q),
            'GGA': (xc_backend.gga_response_terms, xc_backend.gga_fockz_terms, xc_backend.gga_nobeta_reference_q),
            'MGGA': (xc_backend.mgga_response_terms, xc_backend.mgga_fockz_terms, xc_backend.mgga_nobeta_reference_q),
        }[xctype]
        response_xc = response_builder(driver, tdobj, channel_data, atmlst=atmlst)
        fockz_xc = fockz_builder(driver, tdobj, spaces, pz, atmlst=atmlst)
        fockz_hfx = spin_fockz_hfx_terms(
            driver,
            tdobj,
            pz,
            atmlst,
            jk_ledger=ledger,
            output_slot='direct',
        )
        for terms in (response_xc, fockz_xc, fockz_hfx):
            m_matrix += terms.q_alpha + terms.q_beta
            direct += terms.direct
        common_alpha, common_beta = nobeta_builder(tdobj, p0)
        m_matrix += common_alpha + common_beta
        direct_alpha, direct_beta = 0.5 * p0, 0.5 * p0

    action, pairs = make_hessian_transpose_action(tdobj)
    rhs = pairs.pack(m_matrix)
    zvector = solve_zvector(action, pairs, tdobj, rhs, tolerance, max_cycle)
    adjoint = zvector_adjoint_matrix(tdobj, pairs, zvector)
    residual = float(cp.max(cp.abs(pairs.pack(adjoint) - rhs))) if rhs.size else 0.0
    probe_alpha, probe_beta = zvector_probe_densities(tdobj, pairs, zvector)
    probe_alpha = cp.stack((direct_alpha, probe_alpha))
    probe_beta = cp.stack((direct_beta, probe_beta))
    if xctype == 'HF':
        fock_derivatives = spin_fock_direct_hf(
            driver,
            tdobj,
            probe_alpha,
            probe_beta,
            atmlst,
            jk_ledger=ledger,
            output_slots=slots,
        )
    else:
        fock_derivatives = spin_fock_direct_dft(
            driver,
            tdobj,
            probe_alpha,
            probe_beta,
            atmlst,
            nobeta_p0=p0,
            jk_ledger=ledger,
            output_slots=slots,
        )
    contractions = ledger.contract(driver, tdobj.mol, atmlst, slots=slots)
    direct += fock_derivatives[0] + contractions['direct']
    orbital = _orbital_gradient(
        driver,
        tdobj,
        m_matrix,
        adjoint,
        fock_derivatives[1] + contractions['zvector'],
        atmlst,
    )
    return GradientComponents(m_matrix, direct, orbital, direct + orbital, zvector, residual)


def canonical_pairs(tdobj):
    """Return nonredundant CO/CV/OV ROKS rotations as GPU arrays."""
    occ = cp.asarray(tdobj._scf.mo_occ)
    closed = cp.flatnonzero(occ == 2)
    open_ = cp.flatnonzero(occ == 1)
    virtual = cp.flatnonzero(occ == 0)
    nco = closed.size * open_.size
    ncv = closed.size * virtual.size
    nov = open_.size * virtual.size
    return CanonicalPairs(
        p=cp.concatenate(
            (
                cp.repeat(open_, closed.size),
                cp.repeat(virtual, closed.size),
                cp.repeat(virtual, open_.size),
            )
        ),
        q=cp.concatenate(
            (
                cp.tile(closed, open_.size),
                cp.tile(closed, virtual.size),
                cp.tile(open_, virtual.size),
            )
        ),
        alpha_weight=cp.concatenate(
            (
                cp.zeros(nco),
                cp.ones(ncv),
                cp.ones(nov),
            )
        ),
        beta_weight=cp.concatenate(
            (
                cp.ones(nco),
                cp.ones(ncv),
                cp.zeros(nov),
            )
        ),
    )


def _spin_focks_mo(mf):
    fock = mf.get_fock()
    mo = cp.asarray(mf.mo_coeff)
    return mo.conj().T @ fock.focka @ mo, mo.conj().T @ fock.fockb @ mo


def make_hessian_transpose_action(tdobj, pairs=None):
    """Return a matrix-free action for the transpose ROKS Hessian."""
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    nmo = mo.shape[1]
    if pairs is None:
        pairs = canonical_pairs(tdobj)
    fock_alpha, fock_beta = _spin_focks_mo(mf)
    occupation_alpha = (occ > 0).astype(float)
    occupation_beta = (occ == 2).astype(float)
    response = gen_roks_response(mf, hermi=1)
    occupied = cp.flatnonzero(occ > 0)
    occupied_coeff = cp.stack((mo[:, occupied], mo[:, occupied]))

    def unpack(vector):
        return pairs.unpack(vector, nmo)

    def apply_one(vector):
        source_alpha, source_beta = unpack(vector)
        gradient = fock_alpha @ (source_alpha + source_alpha.T)
        gradient += fock_beta @ (source_beta + source_beta.T)
        # Canonical rotation sources have occupied columns only. Keep their
        # exact factors through native DF J/K, avoiding its absolute SVD cutoff.
        left = 0.5 * cp.stack((mo @ source_alpha[:, occupied], mo @ source_beta[:, occupied]))
        density = _make_factorized_dm(left, occupied_coeff, symmetrize=1)
        potential_alpha, potential_beta = response(density)
        potential_alpha = mo.conj().T @ potential_alpha @ mo
        potential_beta = mo.conj().T @ potential_beta @ mo
        gradient += potential_alpha * occupation_alpha[None, :]
        gradient += potential_alpha.T * occupation_alpha[None, :]
        gradient += potential_beta * occupation_beta[None, :]
        gradient += potential_beta.T * occupation_beta[None, :]
        return pairs.pack(gradient)

    def apply(vector):
        vector = cp.asarray(vector)
        if vector.ndim == 1:
            return apply_one(vector)
        return cp.stack([apply_one(row) for row in vector])

    return apply, pairs


def _preconditioner(tdobj, pairs):
    fock_alpha, fock_beta = _spin_focks_mo(tdobj._scf)
    epsilon_alpha = cp.diag(fock_alpha)
    epsilon_beta = cp.diag(fock_beta)
    diagonal = pairs.preconditioner(epsilon_alpha, epsilon_beta)
    small = cp.abs(diagonal) < 1e-8
    diagonal[small] = cp.where(diagonal[small] < 0.0, -1e-8, 1e-8)
    return diagonal


def solve_zvector(action, pairs, tdobj, rhs, tolerance=1e-12, max_cycle=None):
    """Solve H.T z = rhs, enforcing the CPU gradient's true residual criterion."""
    diagonal = _preconditioner(tdobj, pairs)
    if max_cycle is None:
        max_cycle = len(rhs)
    try:
        solution, _, _ = _gmres(action, rhs, diagonal, tolerance, max_cycle)
    except RuntimeError as error:
        raise RuntimeError('NTTDA gradient Z-vector GMRES failed: %s' % error) from error
    return solution


def zvector_adjoint_matrix(tdobj, pairs, zvector):
    """Full MO adjoint matrix satisfying ``z.H(kappa)=Tr(G.T kappa)``."""
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    fock_alpha, fock_beta = _spin_focks_mo(mf)
    occupation_alpha = (occ > 0).astype(float)
    occupation_beta = (occ == 2).astype(float)
    source_alpha, source_beta = pairs.unpack(zvector, mo.shape[1])
    gradient = fock_alpha @ (source_alpha + source_alpha.T)
    gradient += fock_beta @ (source_beta + source_beta.T)
    # Use the same exact factors as the Hessian action, including small modes.
    occupied = cp.flatnonzero(occ > 0)
    occupied_coeff = cp.stack((mo[:, occupied], mo[:, occupied]))
    left = 0.5 * cp.stack((mo @ source_alpha[:, occupied], mo @ source_beta[:, occupied]))
    density = _make_factorized_dm(left, occupied_coeff, symmetrize=1)
    potential_alpha, potential_beta = gen_roks_response(
        mf,
        hermi=1,
    )(density)
    potential_alpha = mo.conj().T @ potential_alpha @ mo
    potential_beta = mo.conj().T @ potential_beta @ mo
    gradient += potential_alpha * occupation_alpha[None, :]
    gradient += potential_alpha.T * occupation_alpha[None, :]
    gradient += potential_beta * occupation_beta[None, :]
    gradient += potential_beta.T * occupation_beta[None, :]
    return gradient


def zvector_probe_densities(tdobj, pairs, zvector):
    mo = cp.asarray(tdobj._scf.mo_coeff)
    source_alpha, source_beta = pairs.unpack(zvector, mo.shape[1])
    return (
        mo @ source_alpha @ mo.conj().T,
        mo @ source_beta @ mo.conj().T,
    )


def _orbital_gradient(gradient_driver, tdobj, m_matrix, adjoint, fock_contraction, atmlst=None):
    mol = tdobj.mol
    mf = tdobj._scf
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    mo = cp.asarray(mf.mo_coeff)
    overlap_derivative = gradient_driver.get_ovlp(mol)
    offsets = mol.offset_nr_by_atom()
    result = cp.zeros((len(atmlst), 3))
    for k, atom in enumerate(atmlst):
        p0, p1 = offsets[atom][2:]
        for xyz in range(3):
            overlap = cp.zeros((mol.nao_nr(), mol.nao_nr()))
            overlap[p0:p1] += overlap_derivative[xyz, p0:p1]
            overlap[:, p0:p1] += overlap_derivative[xyz, p0:p1].T
            symmetric_kappa = -0.5 * (mo.conj().T @ overlap @ mo)
            result[k, xyz] = (
                -fock_contraction[k, xyz] - cp.trace(adjoint.T @ symmetric_kappa) + cp.trace(m_matrix @ symmetric_kappa)
            )
    return result


def _gmres(action, rhs, diagonal, tolerance, max_cycle):
    """Restarted, diagonally preconditioned GMRES with an explicit residual check.

    max_cycle limits inner iterations. At most 40 Krylov vectors are retained.
    No dense operator or inverse is constructed. The diagonal regularization
    affects only the preconditioner.
    """
    rhs = cp.asarray(rhs)
    size = rhs.size
    if size == 0 or cp.linalg.norm(rhs) <= tolerance:
        return cp.zeros_like(rhs), float(cp.linalg.norm(rhs)), 0
    diagonal = cp.asarray(diagonal)
    safe = cp.where(abs(diagonal) < 1e-4, cp.where(diagonal < 0, -1e-4, 1e-4), diagonal)
    solution = cp.zeros_like(rhs)
    restart = min(40, size)
    iterations = 0
    residual = float('inf')
    for start in range(0, max_cycle, restart):
        steps = min(restart, max_cycle - start)
        r = rhs - action(solution)
        beta = cp.linalg.norm(r)
        residual = float(beta)
        if residual <= tolerance:
            return solution, residual, iterations
        basis = cp.zeros((size, steps + 1))
        hessenberg = cp.zeros((steps + 1, steps))
        target = cp.zeros(steps + 1)
        target[0] = beta
        basis[:, 0] = r / beta
        for j in range(steps):
            vector = action(basis[:, j] / safe)
            scale = float(cp.linalg.norm(vector))
            # Reorthogonalization keeps nearly invariant subspaces accurate.
            for _ in range(2):
                projection = basis[:, : j + 1].T @ vector
                hessenberg[: j + 1, j] += projection
                vector -= basis[:, : j + 1] @ projection
            norm = cp.linalg.norm(vector)
            hessenberg[j + 1, j] = norm
            iterations += 1
            h = hessenberg[: j + 2, : j + 1]
            weights = cp.linalg.lstsq(h, target[: j + 2], rcond=None)[0]
            error = cp.linalg.norm(h @ weights - target[: j + 2])
            # A happy Arnoldi breakdown must terminate before dividing by zero.
            if error <= tolerance or float(norm) <= 1e-14 * max(1.0, scale) or j + 1 == steps:
                solution += (basis[:, : j + 1] @ weights) / safe
                break
            basis[:, j + 1] = vector / norm
    residual = float(cp.linalg.norm(action(solution) - rhs))
    if not cp.isfinite(residual) or residual > tolerance:
        raise RuntimeError('GPU response GMRES failed: residual=%.3g, tolerance=%.3g' % (residual, tolerance))
    return solution, residual, iterations
