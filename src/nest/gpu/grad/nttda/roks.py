"""ROKS transpose-Hessian adjoint and final NTTDA gradient assembly."""

from dataclasses import dataclass

import cupy as cp

from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import krylov

from .common import gen_roks_response



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
        return (
            self.alpha_weight
            * (epsilon_alpha[self.p] - epsilon_alpha[self.q])
            + self.beta_weight
            * (epsilon_beta[self.p] - epsilon_beta[self.q])
        )


def finish_gradient(
        gradient_driver, tdobj, m_matrix, direct, atmlst,
        tolerance, max_cycle, fock_direct, direct_fock_probes=None):
    """Solve the common ROKS Z-vector equation and assemble ``d omega/dR``.

    ``direct_fock_probes`` enables one batched Fock-derivative evaluation: its
    contraction is the first result and the Z-vector contraction is the second.
    """
    transpose_action, pairs = make_hessian_transpose_action(tdobj)
    rhs = pack_m_matrix(m_matrix, pairs)
    zvector = solve_zvector(
        transpose_action,
        pairs,
        tdobj,
        rhs,
        tolerance=tolerance,
        max_cycle=max_cycle,
    )
    adjoint = zvector_adjoint_matrix(tdobj, pairs, zvector)
    residual = float(cp.max(cp.abs(pack_m_matrix(adjoint, pairs) - rhs)))
    residual_scale = max(float(cp.max(cp.abs(rhs))), 1.0)
    residual_limit = max(100.0 * tolerance, 1e-7) * residual_scale
    if residual > residual_limit:
        logger.warn(
            gradient_driver,
            "NTTDA Z-vector residual %.3e exceeds %.3e. "
            "The Krylov tolerance %.3e controls the preconditioned "
            "subspace, not this unpreconditioned equation residual.",
            residual,
            residual_limit,
            tolerance,
        )
    probe_alpha, probe_beta = zvector_probe_densities(
        tdobj, pairs, zvector,
    )
    if direct_fock_probes is None:
        fock_contraction = fock_direct(
            gradient_driver,
            tdobj,
            probe_alpha,
            probe_beta,
            atmlst=atmlst,
        )
        direct_total = direct
    else:
        direct_alpha, direct_beta = direct_fock_probes
        fock_contractions = fock_direct(
            gradient_driver,
            tdobj,
            cp.stack((direct_alpha, probe_alpha)),
            cp.stack((direct_beta, probe_beta)),
            atmlst=atmlst,
        )
        direct_total = direct + fock_contractions[0]
        fock_contraction = fock_contractions[1]
    orbital = _orbital_gradient(
        gradient_driver,
        tdobj,
        m_matrix,
        adjoint,
        fock_contraction,
        atmlst=atmlst,
    )
    return GradientComponents(
        m_matrix=m_matrix,
        direct=direct_total,
        orbital=orbital,
        total=direct_total + orbital,
        zvector=zvector,
        residual=residual,
    )


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
        p=cp.concatenate((
            cp.repeat(open_, closed.size),
            cp.repeat(virtual, closed.size),
            cp.repeat(virtual, open_.size),
        )),
        q=cp.concatenate((
            cp.tile(closed, open_.size),
            cp.tile(closed, virtual.size),
            cp.tile(open_, virtual.size),
        )),
        alpha_weight=cp.concatenate((
            cp.zeros(nco),
            cp.ones(ncv),
            cp.ones(nov),
        )),
        beta_weight=cp.concatenate((
            cp.ones(nco),
            cp.ones(ncv),
            cp.zeros(nov),
        )),
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

    def unpack(vector):
        return pairs.unpack(vector, nmo)

    def apply_one(vector):
        source_alpha, source_beta = unpack(vector)
        gradient = fock_alpha @ (source_alpha + source_alpha.T)
        gradient += fock_beta @ (source_beta + source_beta.T)
        density_alpha = mo @ source_alpha @ mo.conj().T
        density_beta = mo @ source_beta @ mo.conj().T
        density_alpha = 0.5 * (density_alpha + density_alpha.T)
        density_beta = 0.5 * (density_beta + density_beta.T)
        potential_alpha, potential_beta = response(
            cp.stack((density_alpha, density_beta))
        )
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


def pack_m_matrix(matrix, pairs):
    return pairs.pack(matrix)


def _preconditioner(tdobj, pairs):
    fock_alpha, fock_beta = _spin_focks_mo(tdobj._scf)
    epsilon_alpha = cp.diag(fock_alpha)
    epsilon_beta = cp.diag(fock_beta)
    diagonal = pairs.preconditioner(epsilon_alpha, epsilon_beta)
    small = cp.abs(diagonal) < 1e-8
    diagonal[small] = cp.where(diagonal[small] < 0.0, -1e-8, 1e-8)
    return diagonal


def solve_zvector(action, pairs, tdobj, rhs, tolerance=1e-12, max_cycle=None):
    """Solve ``H.T z = rhs`` using the PySCF CPHF Krylov pattern."""
    diagonal = _preconditioner(tdobj, pairs)
    initial = rhs / diagonal
    if max_cycle is None:
        max_cycle = len(rhs)

    def operator(vector):
        vector = cp.asarray(vector)
        if vector.ndim == 1:
            return action(vector) / diagonal - vector
        return cp.asarray([action(row) / diagonal - row for row in vector])

    solution = krylov(
        operator,
        initial,
        tol=tolerance,
        max_cycle=max_cycle,
        lindep=1e-22,
        hermi=False,
        verbose=0,
    )
    return cp.asarray(solution).reshape(-1)


def _unpack_zvector_source(tdobj, pairs, zvector):
    nmo = tdobj._scf.mo_coeff.shape[1]
    return pairs.unpack(zvector, nmo)


def zvector_adjoint_matrix(tdobj, pairs, zvector):
    """Full MO adjoint matrix satisfying ``z.H(kappa)=Tr(G.T kappa)``."""
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    fock_alpha, fock_beta = _spin_focks_mo(mf)
    occupation_alpha = (occ > 0).astype(float)
    occupation_beta = (occ == 2).astype(float)
    source_alpha, source_beta = _unpack_zvector_source(
        tdobj, pairs, zvector,
    )
    gradient = fock_alpha @ (source_alpha + source_alpha.T)
    gradient += fock_beta @ (source_beta + source_beta.T)
    density_alpha = mo @ source_alpha @ mo.conj().T
    density_beta = mo @ source_beta @ mo.conj().T
    density_alpha = 0.5 * (density_alpha + density_alpha.T)
    density_beta = 0.5 * (density_beta + density_beta.T)
    potential_alpha, potential_beta = gen_roks_response(
        mf, hermi=1,
    )(cp.stack((density_alpha, density_beta)))
    potential_alpha = mo.conj().T @ potential_alpha @ mo
    potential_beta = mo.conj().T @ potential_beta @ mo
    gradient += potential_alpha * occupation_alpha[None, :]
    gradient += potential_alpha.T * occupation_alpha[None, :]
    gradient += potential_beta * occupation_beta[None, :]
    gradient += potential_beta.T * occupation_beta[None, :]
    return gradient


def zvector_probe_densities(tdobj, pairs, zvector):
    mo = cp.asarray(tdobj._scf.mo_coeff)
    source_alpha, source_beta = _unpack_zvector_source(
        tdobj, pairs, zvector,
    )
    return (
        mo @ source_alpha @ mo.conj().T,
        mo @ source_beta @ mo.conj().T,
    )


def _orbital_gradient(
        gradient_driver, tdobj, m_matrix, adjoint, fock_contraction,
        atmlst=None):
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
                -fock_contraction[k, xyz]
                - cp.trace(adjoint.T @ symmetric_kappa)
                + cp.trace(m_matrix @ symmetric_kappa)
            )
    return result
