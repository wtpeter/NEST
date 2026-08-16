"""Analytic gradient for NTTDA ``deltaS=+1``."""

from dataclasses import dataclass

import cupy as cp

from nest.gpu.nttda.nttda import gen_rohf_response_sfu

from . import xc as xc_backend
from .common import (
    FockProjection,
    JKDerivativeLedger,
    ResponseTerm,
    fock_response_q,
    orbital_spaces,
    response_direct_hfx,
    spin_fock_direct_dft,
    spin_fock_direct_hf,
    spin_fockz_hfx_terms,
)
from .roks import finish_gradient


@dataclass(frozen=True)
class SpinRaisingAmplitudes:
    """The sole closed-to-virtual spin-raising amplitude block."""

    cv: cp.ndarray


def split_spin_raising(tdobj, xy):
    """Validate and return the native ``C->V`` amplitude."""
    spaces = orbital_spaces(tdobj)
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = cp.asarray(vector)
    expected = (len(spaces.closed), len(spaces.virtual))
    if vector.size != expected[0] * expected[1]:
        raise ValueError(
            "deltaS=+1 amplitude has size %d; expected %d" %
            (vector.size, expected[0] * expected[1])
        )
    return spaces, SpinRaisingAmplitudes(vector.reshape(expected))


def spin_raising_transition_densities(tdobj, xy):
    """Return the alpha-target/beta-source transition density."""
    spaces, amplitudes = split_spin_raising(tdobj, xy)
    return spaces, amplitudes, {
        "CV": (
            spaces.c_virtual
            @ amplitudes.cv.T
            @ spaces.c_closed.conj().T
        ),
    }


def spin_raising_block_data(spaces, amplitudes):
    return {
        "CV": (spaces.virtual, spaces.closed, amplitudes.cv.T),
    }


def spin_raising_response_terms():
    return (ResponseTerm("CV", "CV", 1.0, 0.0),)


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
    x = amplitudes.cv
    return (
        FockProjection(
            name="particle-vv",
            left_indices=spaces.virtual,
            left_orbitals=spaces.c_virtual,
            coefficient=x.T @ x,
            right_indices=spaces.virtual,
            right_orbitals=spaces.c_virtual,
            weight_f0=1.0,
            weight_fz=1.0,
        ),
        FockProjection(
            name="hole-cc",
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


def _transverse_potential(
        tdobj, density, max_memory=None, hfx_only=False):
    """Apply the full or exact-exchange-only transverse kernel."""
    mf = tdobj._scf
    density = cp.asarray(density)
    if max_memory is None:
        max_memory = tdobj.max_memory
    if not hfx_only:
        response, _fockz = gen_rohf_response_sfu(
            mf,
            mo_coeff=mf.mo_coeff,
            mo_occ=mf.mo_occ,
            hermi=0,
            max_memory=max_memory,
        )
        return response(density[None])[0]

    potential = cp.zeros_like(density)
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return potential
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(
        mf.xc, mf.mol.spin,
    )
    potential -= hybrid * mf.get_k(mf.mol, density, hermi=0)
    if omega != 0:
        potential -= (alpha - hybrid) * mf.get_k(
            mf.mol, density, hermi=0, omega=omega,
        )
    return potential


def spin_raising_fock_q(tdobj, xy, max_memory=None):
    """Explicit-Fock projection and reference response M matrices."""
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    fock0, fockz = spin_raising_fock0_fockz(
        tdobj, max_memory=max_memory,
    )
    fock0_mo = mo.conj().T @ fock0 @ mo
    fockz_mo = mo.conj().T @ fockz @ mo
    q_alpha = cp.zeros((nmo, nmo))
    q_beta = cp.zeros_like(q_alpha)
    is_hf = mf._numint._xc_type(mf.xc) == "HF"

    for term in spin_raising_fock_projections(tdobj, xy):
        indices = term.left_indices
        coefficient = term.coefficient

        def project(target, operator, scale):
            if scale:
                target[:, indices] += (
                    scale * operator[:, indices] @ coefficient.T
                )
                target[:, indices] += (
                    scale * operator[:, indices] @ coefficient
                )

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
        tdobj, p_alpha, p_beta,
    )
    return q_alpha + response_alpha, q_beta + response_beta


def spin_raising_response_projection_q(
        tdobj, xy, max_memory=None, hfx_only=False):
    """Transition-factor derivative of ``T:K_perp[T]``."""
    spaces, amplitudes, densities = spin_raising_transition_densities(
        tdobj, xy,
    )
    potential = 2.0 * _transverse_potential(
        tdobj,
        densities["CV"],
        max_memory=max_memory,
        hfx_only=hfx_only,
    )
    mo = cp.asarray(tdobj._scf.mo_coeff)
    potential_mo = mo.conj().T @ potential @ mo
    q_alpha = cp.zeros_like(potential_mo)
    q_beta = cp.zeros_like(potential_mo)
    q_alpha[:, spaces.virtual] += (
        potential_mo[:, spaces.closed] @ amplitudes.cv
    )
    q_beta[:, spaces.closed] += (
        potential_mo[spaces.virtual, :].T @ amplitudes.cv.T
    )
    return q_alpha, q_beta


def grad_elec(
        gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12,
        max_cycle=None):
    """Build the complete analytic excitation gradient for deltaS=+1."""
    if tdobj.deltaS != 1:
        raise ValueError("deltaS=+1 gradient received a different spin channel")
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)

    spaces, amplitudes, densities = spin_raising_transition_densities(
        tdobj, xy,
    )
    blocks = spin_raising_block_data(spaces, amplitudes)
    response_terms = spin_raising_response_terms()
    channel_data = (
        spaces, amplitudes, densities, blocks, response_terms, "alpha",
    )
    p0, pz = spin_raising_fock_probes(tdobj, xy)
    jk_ledger = JKDerivativeLedger()
    direct_slot = "direct"
    zvector_slot = "zvector"
    fock_alpha, fock_beta = spin_raising_fock_q(tdobj, xy)

    if xctype == "HF":
        response_alpha, response_beta = (
            spin_raising_response_projection_q(tdobj, xy)
        )
        m_matrix = (
            fock_alpha + fock_beta + response_alpha + response_beta
        )
        direct_fock_probes = (
            0.5 * (p0 + pz),
            0.5 * (p0 - pz),
        )
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            densities,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )

        def fock_direct(driver, obj, p_alpha, p_beta, atmlst=None):
            local = spin_fock_direct_hf(
                driver,
                obj,
                p_alpha,
                p_beta,
                atmlst=atmlst,
                jk_ledger=jk_ledger,
                output_slots=(direct_slot, zvector_slot),
            )
            contractions = jk_ledger.contract(
                driver,
                obj.mol,
                atmlst,
                slots=(direct_slot, zvector_slot),
            )
            local[0] += contractions[direct_slot]
            local[1] += contractions[zvector_slot]
            return local
    else:
        try:
            response_builder, fockz_builder, nobeta_q_builder = {
                "LDA": (
                    xc_backend.lda_response_terms,
                    xc_backend.lda_fockz_terms,
                    xc_backend.lda_nobeta_reference_q,
                ),
                "GGA": (
                    xc_backend.gga_response_terms,
                    xc_backend.gga_fockz_terms,
                    xc_backend.gga_nobeta_reference_q,
                ),
                "MGGA": (
                    xc_backend.mgga_response_terms,
                    xc_backend.mgga_fockz_terms,
                    xc_backend.mgga_nobeta_reference_q,
                ),
            }[xctype]
        except KeyError as error:
            raise NotImplementedError(
                "NTTDA deltaS=+1 gradient does not support XC type %s" %
                xctype
            ) from error

        hfx_alpha, hfx_beta = spin_raising_response_projection_q(
            tdobj, xy, hfx_only=True,
        )
        response_xc = response_builder(
            gradient_driver,
            tdobj,
            channel_data,
            atmlst=atmlst,
        )
        fockz_xc = fockz_builder(
            gradient_driver,
            tdobj,
            spaces,
            pz,
            atmlst=atmlst,
        )
        fockz_hfx = spin_fockz_hfx_terms(
            gradient_driver,
            tdobj,
            pz,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        common_alpha, common_beta = nobeta_q_builder(tdobj, p0)
        m_matrix = (
            fock_alpha + fock_beta
            + hfx_alpha + hfx_beta
            + response_xc.q_alpha + response_xc.q_beta
            + fockz_xc.q_alpha + fockz_xc.q_beta
            + fockz_hfx.q_alpha + fockz_hfx.q_beta
            + common_alpha + common_beta
        )

        direct_fock_probes = (0.5 * p0, 0.5 * p0)
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            densities,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        direct += response_xc.direct
        direct += fockz_xc.direct
        direct += fockz_hfx.direct

        def fock_direct(driver, obj, p_alpha, p_beta, atmlst=None):
            local = spin_fock_direct_dft(
                driver,
                obj,
                p_alpha,
                p_beta,
                atmlst=atmlst,
                nobeta_p0=p0,
                jk_ledger=jk_ledger,
                output_slots=(direct_slot, zvector_slot),
            )
            contractions = jk_ledger.contract(
                driver,
                obj.mol,
                atmlst,
                slots=(direct_slot, zvector_slot),
            )
            local[0] += contractions[direct_slot]
            local[1] += contractions[zvector_slot]
            return local

    return finish_gradient(
        gradient_driver,
        tdobj,
        m_matrix,
        direct,
        atmlst,
        tolerance,
        max_cycle,
        fock_direct,
        direct_fock_probes=direct_fock_probes,
    )


__all__ = ["grad_elec"]
