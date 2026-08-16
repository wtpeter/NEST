"""Shared gpu4pyscf primitives for GPU NTTDA gradients."""

from dataclasses import dataclass

import cupy as cp
import numpy as np

from gpu4pyscf.df import int3c2e
from gpu4pyscf.grad import rhf as rhf_grad

from nest.gpu.nttda import nttda as nttda_mod

from . import xc as xc_backend


def gen_roks_response(mf, hermi):
    """Build the UKS response around the alpha/beta ROKS occupations."""
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    mo_coeff = cp.stack((mo, mo))
    mo_occ = cp.stack(((occ > 0).astype(float), (occ == 2).astype(float)))
    return mf.gen_response(
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        hermi=hermi,
        max_memory=mf.max_memory,
    )


@dataclass(frozen=True)
class OrbitalSpaces:
    """Closed, open, and virtual spatial-orbital partitions."""

    closed: cp.ndarray
    open: cp.ndarray
    virtual: cp.ndarray
    c_closed: cp.ndarray
    c_open: cp.ndarray
    c_virtual: cp.ndarray

    @property
    def spin(self):
        return 0.5 * len(self.open)


def orbital_spaces(tdobj):
    """Return the ROKS C/O/V orbital partition used by NTTDA."""
    mf = tdobj._scf
    occupation = cp.asarray(mf.mo_occ)
    if occupation.ndim != 1:
        raise ValueError("NTTDA gradients require spatial ROKS orbitals")
    closed = cp.flatnonzero(occupation == 2)
    open_ = cp.flatnonzero(occupation == 1)
    virtual = cp.flatnonzero(occupation == 0)
    coefficient = cp.asarray(mf.mo_coeff)
    return OrbitalSpaces(
        closed=closed,
        open=open_,
        virtual=virtual,
        c_closed=coefficient[:, closed],
        c_open=coefficient[:, open_],
        c_virtual=coefficient[:, virtual],
    )


def pair_density(c_left, coefficient, c_right):
    """Build ``C_left coefficient C_right.T`` without symmetrizing it."""
    return c_left @ cp.asarray(coefficient) @ c_right.conj().T


@dataclass(frozen=True)
class FockProjection:
    """One scalar term ``Tr[P (weight_f0 F0 + weight_fz Fz)]``."""

    name: str
    left_indices: cp.ndarray
    left_orbitals: cp.ndarray
    coefficient: cp.ndarray
    right_indices: cp.ndarray
    right_orbitals: cp.ndarray
    weight_f0: float
    weight_fz: float

    def density(self):
        return pair_density(
            self.left_orbitals, self.coefficient, self.right_orbitals,
        )


@dataclass(frozen=True)
class ResponseTerm:
    """Directed response term from one source density to one target block."""

    target: str
    source: str
    vref0: float
    vref1: float


def fxc_reference(tdobj):
    mf = tdobj._scf
    ni = mf._numint
    fxc = ni.cache_xc_kernel(
        mf.mol, mf.grids, mf.xc, mf.mo_coeff, mf.mo_occ, 1,
    )[2]
    return 0.5 * (
        fxc[0, :, 0] - fxc[0, :, 1]
        - fxc[1, :, 0] + fxc[1, :, 1]
    )


def apply_reference_responses(tdobj, densities, max_memory=None):
    """Apply the full reference ``vref0`` and ``vref1`` kernels."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if max_memory is None:
        max_memory = tdobj.max_memory
    labels = tuple(densities)
    dms = cp.asarray([densities[label] for label in labels])
    xctype = ni._xc_type(mf.xc)
    if xctype == "HF":
        vref0 = cp.zeros_like(dms)
        vref1 = cp.zeros_like(dms)
    else:
        fxc_ref = fxc_reference(tdobj)
        vref0 = ni.nr_rks_fxc(
            mol, mf.grids, mf.xc, None, dms, 0, 0,
            None, None, fxc_ref, max_memory=max_memory,
        )
        if xctype == "LDA":
            vref1 = ni.nr_rks_fxc(
                mol, mf.grids, mf.xc, None, dms, 0, 0,
                None, None, fxc_ref, max_memory=max_memory,
            )
        elif xctype == "GGA":
            vref1 = nttda_mod.nr_rks_fxc1_gga(
                ni, mol, mf.grids, mf.xc, dms, fxc_ref,
                max_memory=max_memory,
            )
        elif xctype == "MGGA":
            vref1 = nttda_mod.nr_rks_fxc1_mgga(
                ni, mol, mf.grids, mf.xc, dms, fxc_ref,
                max_memory=max_memory,
            )
        else:
            raise NotImplementedError(
                "NTTDA response does not support XC type %s" % xctype
            )

    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    if ni.libxc.is_hybrid_xc(mf.xc):
        vref0 -= hybrid * mf.get_k(mol, dms, hermi=0)
        vref1 -= hybrid * mf.get_j(mol, dms, hermi=0)
        if omega != 0:
            scale = alpha - hybrid
            vref0 -= scale * mf.get_k(mol, dms, hermi=0, omega=omega)
            vref1 -= scale * mf.get_j(mol, dms, hermi=0, omega=omega)
    return (
        {label: value for label, value in zip(labels, vref0)},
        {label: value for label, value in zip(labels, vref1)},
    )


def apply_hfx_responses(tdobj, densities):
    """Apply only the hybrid/RSH portions of ``vref0`` and ``vref1``."""
    mf = tdobj._scf
    labels = tuple(densities)
    dms = cp.asarray([densities[label] for label in labels])
    vref0 = cp.zeros_like(dms)
    vref1 = cp.zeros_like(dms)
    ni = mf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
    if ni.libxc.is_hybrid_xc(mf.xc):
        vref0 -= hybrid * mf.get_k(mf.mol, dms, hermi=0)
        vref1 -= hybrid * mf.get_j(mf.mol, dms, hermi=0)
        if omega != 0:
            scale = alpha - hybrid
            vref0 -= scale * mf.get_k(
                mf.mol, dms, hermi=0, omega=omega,
            )
            vref1 -= scale * mf.get_j(
                mf.mol, dms, hermi=0, omega=omega,
            )
    return (
        {label: value for label, value in zip(labels, vref0)},
        {label: value for label, value in zip(labels, vref1)},
    )


def fock_response_q(tdobj, p_alpha, p_beta):
    """Reference-density derivative of a spin-resolved Fock scalar."""
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    occupied_alpha = cp.flatnonzero(mf.mo_occ > 0)
    occupied_beta = cp.flatnonzero(mf.mo_occ == 2)
    v_alpha, v_beta = gen_roks_response(mf, hermi=0)(
        cp.stack((p_alpha.T, p_beta.T))
    )
    q_alpha = cp.zeros((mo.shape[1], mo.shape[1]))
    q_beta = cp.zeros_like(q_alpha)
    q_alpha[:, occupied_alpha] = (
        mo.conj().T @ (v_alpha + v_alpha.T) @ mo[:, occupied_alpha]
    )
    q_beta[:, occupied_beta] = (
        mo.conj().T @ (v_beta + v_beta.T) @ mo[:, occupied_beta]
    )
    return q_alpha, q_beta


def hcore_derivative(gradient_driver, probes, atmlst=None):
    """Differentiate ``Tr[probe hcore]`` for one or more AO probes."""
    probes = cp.asarray(probes)
    single_probe = probes.ndim == 2
    if single_probe:
        probes = probes[None]
    mol = gradient_driver.mol
    h1 = rhf_grad.get_hcore(gradient_driver.base._scf, mol)
    output = []
    for probe in probes:
        value = rhf_grad.contract_h1e_dm(mol, h1, probe, hermi=0)
        value += cp.asnumpy(int3c2e.get_dh1e(mol, probe))
        output.append(value)
    output = cp.asarray(np.asarray(output))
    if atmlst is not None:
        output = output[:, tuple(atmlst)]
    return output[0] if single_probe else output


@dataclass(frozen=True)
class JKDerivativeTerm:
    left: cp.ndarray
    right: cp.ndarray
    scale: float
    omega: float | None
    slot: object


class JKDerivativeLedger:
    """Batch fixed-AO J/K derivatives as gpu4pyscf density-matrix pairs."""

    def __init__(self):
        self._terms = {"j": [], "k": []}

    def add(self, operator, slot, terms):
        self._terms[operator].extend(
            JKDerivativeTerm(
                cp.asarray(left),
                cp.asarray(right),
                float(scale),
                None if omega is None else float(omega),
                slot,
            )
            for left, right, scale, omega in terms
            if scale != 0.0
        )

    def contract(self, gradient_driver, mol, atoms, slots=()):
        atoms = tuple(atoms)
        gradients = {
            slot: cp.zeros((len(atoms), 3), dtype=cp.float64)
            for slot in slots
        }
        terms_by_omega = {}
        for operator in ("j", "k"):
            for term in self._terms[operator]:
                gradients.setdefault(
                    term.slot,
                    cp.zeros((len(atoms), 3), dtype=cp.float64),
                )
                terms_by_omega.setdefault(term.omega, []).append(
                    (operator, term)
                )

        for omega, terms in terms_by_omega.items():
            dm_pairs = []
            j_factors = []
            k_factors = []
            for operator, term in terms:
                if operator == "j":
                    dm_pairs.append(cp.stack((term.left, term.right)))
                    j_factors.append(2.0 * term.scale)
                    k_factors.append(0.0)
                else:
                    dm_pairs.append(cp.stack((term.left, term.right.T)))
                    j_factors.append(0.0)
                    k_factors.append(-4.0 * term.scale)
            values = gradient_driver.jk_energies_per_atom(
                cp.stack(dm_pairs),
                j_factor=j_factors,
                k_factor=k_factors,
                omega=0.0 if omega is None else omega,
                sum_results=False,
            )
            values = cp.asarray(values)
            if atoms != tuple(range(mol.natm)):
                values = values[:, atoms]
            for value, (_operator, term) in zip(values, terms):
                gradients[term.slot] += value
        return gradients


def _reference_spin_densities(tdobj):
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    return (
        mo[:, mf.mo_occ > 0] @ mo[:, mf.mo_occ > 0].T,
        mo[:, mf.mo_occ == 2] @ mo[:, mf.mo_occ == 2].T,
    )


def _spin_probe_stacks(p_alpha, p_beta):
    p_alpha = cp.asarray(p_alpha)
    p_beta = cp.asarray(p_beta)
    single_probe = p_alpha.ndim == 2
    if single_probe:
        p_alpha = p_alpha[None]
        p_beta = p_beta[None]
    return p_alpha, p_beta, single_probe


def spin_fock_direct_dft(
        gradient_driver, tdobj, p_alpha, p_beta, atmlst=None,
        nobeta_p0=None, jk_ledger=None, output_slots=None):
    """Differentiate one or more ordinary UKS Fock scalar probes."""
    mf = tdobj._scf
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    p_alpha, p_beta, single_probe = _spin_probe_stacks(
        p_alpha, p_beta,
    )
    if output_slots is None:
        output_slots = tuple(range(len(p_alpha)))
    p_total = p_alpha + p_beta
    density_alpha, density_beta = _reference_spin_densities(tdobj)
    gradient = cp.zeros((len(p_alpha), len(atmlst), 3))
    gradient += hcore_derivative(
        gradient_driver, p_total, atmlst=atmlst,
    )
    ni = mf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    local_ledger = JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    for probe in range(len(p_alpha)):
        j_terms = [
            (p_total[probe], density_alpha, 1.0, None),
            (p_total[probe], density_beta, 1.0, None),
        ]
        k_terms = []
        if ni.libxc.is_hybrid_xc(mf.xc):
            k_terms.extend((
                (p_alpha[probe], density_alpha, -hybrid, None),
                (p_beta[probe], density_beta, -hybrid, None),
            ))
            if omega != 0:
                long_range = -(alpha - hybrid)
                k_terms.extend((
                    (p_alpha[probe], density_alpha, long_range, omega),
                    (p_beta[probe], density_beta, long_range, omega),
                ))
        ledger.add("j", output_slots[probe], j_terms)
        ledger.add("k", output_slots[probe], k_terms)
    try:
        derivative_contractor = {
            "LDA": xc_backend.contract_lda_vxc_derivative,
            "GGA": xc_backend.contract_gga_vxc_derivative,
            "MGGA": xc_backend.contract_mgga_vxc_derivative,
        }[ni._xc_type(mf.xc)]
    except KeyError as error:
        raise NotImplementedError(
            "ordinary Fock direct derivative is not implemented for %s" %
            ni._xc_type(mf.xc)
        ) from error
    if nobeta_p0 is not None and tdobj.nobeta:
        density0 = 0.5 * (density_alpha + density_beta)
        actual_probe_alpha = cp.array(p_alpha, copy=True)
        actual_probe_beta = cp.array(p_beta, copy=True)
        actual_probe_alpha[0] -= 0.5 * nobeta_p0
        actual_probe_beta[0] -= 0.5 * nobeta_p0
    else:
        density0 = None
        actual_probe_alpha = p_alpha
        actual_probe_beta = p_beta
    gradient += derivative_contractor(
        mf,
        density_alpha,
        density_beta,
        actual_probe_alpha,
        actual_probe_beta,
        atmlst=atmlst,
        max_memory=gradient_driver.max_memory,
    )
    if density0 is not None:
        gradient[0] += derivative_contractor(
            mf,
            density0,
            density0,
            0.5 * nobeta_p0,
            0.5 * nobeta_p0,
            atmlst=atmlst,
            max_memory=gradient_driver.max_memory,
        )
    if jk_ledger is None:
        contractions = local_ledger.contract(
            gradient_driver, mol, atmlst, slots=output_slots,
        )
        for probe, slot in enumerate(output_slots):
            gradient[probe] += contractions[slot]
    return gradient[0] if single_probe else gradient


def spin_fock_direct_hf(
        gradient_driver, tdobj, p_alpha, p_beta, atmlst=None,
        jk_ledger=None, output_slots=None):
    """Differentiate one or more spin-resolved HF Fock scalar probes."""
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    p_alpha, p_beta, single_probe = _spin_probe_stacks(
        p_alpha, p_beta,
    )
    if output_slots is None:
        output_slots = tuple(range(len(p_alpha)))
    p_total = p_alpha + p_beta
    density_alpha, density_beta = _reference_spin_densities(tdobj)
    gradient = cp.zeros((len(p_alpha), len(atmlst), 3))
    gradient += hcore_derivative(
        gradient_driver, p_total, atmlst=atmlst,
    )
    local_ledger = JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    for probe in range(len(p_alpha)):
        ledger.add(
            "j", output_slots[probe], (
                (p_total[probe], density_alpha, 1.0, None),
                (p_total[probe], density_beta, 1.0, None),
            ),
        )
        ledger.add(
            "k", output_slots[probe], (
                (p_alpha[probe], density_alpha, -1.0, None),
                (p_beta[probe], density_beta, -1.0, None),
            ),
        )
    if jk_ledger is None:
        contractions = local_ledger.contract(
            gradient_driver, mol, atmlst, slots=output_slots,
        )
        for probe, slot in enumerate(output_slots):
            gradient[probe] += contractions[slot]
    return gradient[0] if single_probe else gradient


def response_direct_hfx(
        gradient_driver, tdobj, densities, response_terms, atmlst=None,
        jk_ledger=None, output_slot=0):
    """Differentiate the hybrid/RSH response-term ledger."""
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    gradient = cp.zeros((len(atmlst), 3))
    ni = tdobj._scf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(
        tdobj._scf.xc, mol.spin,
    )
    if not ni.libxc.is_hybrid_xc(tdobj._scf.xc):
        return gradient

    scales = [(hybrid, None)]
    if omega != 0:
        scales.append((alpha - hybrid, omega))
    j_terms = []
    k_terms = []
    for term in response_terms:
        target = densities[term.target]
        source = densities[term.source]
        for coefficient, range_omega in scales:
            if term.vref0:
                k_terms.append((
                    target,
                    source,
                    -coefficient * term.vref0,
                    range_omega,
                ))
            if term.vref1:
                j_terms.append((
                    target,
                    source,
                    -coefficient * term.vref1,
                    range_omega,
                ))
    local_ledger = JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    ledger.add("j", output_slot, j_terms)
    ledger.add("k", output_slot, k_terms)
    if jk_ledger is None:
        gradient += local_ledger.contract(
            gradient_driver, mol, atmlst, slots=(output_slot,),
        )[output_slot]
    return gradient


def spin_fockz_hfx_terms(
        gradient_driver, tdobj, pz, atmlst=None, with_direct=True,
        jk_ledger=None, output_slot=0):
    """Differentiate ``-1/2 Pz:K(D_open)`` excluding the Pz projection."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    mo = cp.asarray(mf.mo_coeff)
    q_alpha = cp.zeros((mo.shape[1], mo.shape[1]))
    q_beta = cp.zeros_like(q_alpha)
    direct = cp.zeros((len(atmlst), 3))
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return xc_backend.XCGradientTerms(q_alpha, q_beta, direct)

    spaces = orbital_spaces(tdobj)
    density_open = spaces.c_open @ spaces.c_open.T
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    scales = [(hybrid, None)]
    if omega != 0:
        scales.append((alpha - hybrid, omega))
    k_terms = []
    for coefficient, range_omega in scales:
        if coefficient == 0.0:
            continue
        if range_omega is None:
            potential = mf.get_k(mol, pz, hermi=0)
        else:
            potential = mf.get_k(
                mol, pz, hermi=0, omega=range_omega,
            )
        q_alpha[:, spaces.open] -= 0.5 * coefficient * (
            mo.conj().T @ (potential + potential.T) @ spaces.c_open
        )
        if with_direct:
            k_terms.append((
                pz,
                density_open,
                -0.5 * coefficient,
                range_omega,
            ))
    if with_direct:
        local_ledger = JKDerivativeLedger()
        ledger = jk_ledger if jk_ledger is not None else local_ledger
        ledger.add("k", output_slot, k_terms)
        if jk_ledger is None:
            direct += local_ledger.contract(
                gradient_driver, mol, atmlst, slots=(output_slot,),
            )[output_slot]
    return xc_backend.XCGradientTerms(q_alpha, q_beta, direct)
