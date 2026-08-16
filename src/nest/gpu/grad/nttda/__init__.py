"""GPU analytic gradients for all NTTDA spin channels."""

import cupy as cp

from pyscf import lib
from gpu4pyscf.df.df_jk import _DFHF
from gpu4pyscf.df.grad import tduks_sf as df_tduks_sf_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tduks_sf as tduks_sf_grad
from gpu4pyscf.grad import uks as uks_grad
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array

from . import delta_s_minus_one, delta_s_plus_one, delta_s_zero
from .common import JKDerivativeLedger, hcore_derivative


def _reference_densities(mf):
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    occupied_alpha = mo[:, occ > 0]
    occupied_beta = mo[:, occ == 2]
    density_alpha = occupied_alpha @ occupied_alpha.T
    density_beta = occupied_beta @ occupied_beta.T
    return density_alpha, density_beta


def _energy_weighted_density(mf, density_alpha, density_beta):
    fock = mf.get_fock()
    return (
        density_alpha @ fock.focka @ density_alpha
        + density_beta @ fock.fockb @ density_beta
    )


def reference_gradient(gradient_driver):
    """Return the complete ground-state ROKS gradient."""
    mf = gradient_driver.base._scf
    mol = mf.mol
    if mf.do_nlc():
        raise NotImplementedError("GPU NTTDA gradients do not support NLC")
    if mf.do_disp():
        raise NotImplementedError("GPU NTTDA gradients do not support dispersion")

    density_alpha, density_beta = _reference_densities(mf)
    density_total = density_alpha + density_beta
    energy_weighted = _energy_weighted_density(
        mf, density_alpha, density_beta,
    )
    electronic = hcore_derivative(gradient_driver, density_total)
    overlap = gradient_driver.get_ovlp(mol)
    electronic -= cp.asarray(
        rhf_grad.contract_h1e_dm(
            mol, overlap, energy_weighted, hermi=1,
        )
    )

    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    spin_mo = cp.stack((mo, mo))
    spin_occ = cp.stack(((occ > 0).astype(float), (occ == 2).astype(float)))
    spin_density = tag_array(
        cp.stack((density_alpha, density_beta)),
        mo_coeff=spin_mo,
        mo_occ=spin_occ,
    )
    _exc, xc_gradient = uks_grad.get_exc(
        mf._numint,
        mol,
        mf.grids,
        mf.xc,
        spin_density,
        max_memory=gradient_driver.max_memory,
        verbose=gradient_driver.verbose,
    )
    electronic += 2.0 * cp.asarray(xc_gradient)

    omega, alpha, hybrid = mf._numint.rsh_and_hybrid_coeff(
        mf.xc, mol.spin,
    )
    ledger = JKDerivativeLedger()
    ledger.add(
        "j", "reference", (
            (density_total, density_total, 0.5, None),
        ),
    )
    ledger.add(
        "k", "reference", (
            (density_alpha, density_alpha, -0.5 * hybrid, None),
            (density_beta, density_beta, -0.5 * hybrid, None),
        ),
    )
    if omega != 0:
        long_range = -0.5 * (alpha - hybrid)
        ledger.add(
            "k", "reference", (
                (density_alpha, density_alpha, long_range, omega),
                (density_beta, density_beta, long_range, omega),
            ),
        )
    electronic += ledger.contract(
        gradient_driver,
        mol,
        range(mol.natm),
        slots=("reference",),
    )["reference"]
    nuclear = rhf_grad.GradientsBase.grad_nuc(gradient_driver, mol)
    return electronic + cp.asarray(nuclear)


class Gradients(tduks_sf_grad.Gradients):
    """Total GPU NTTDA gradients for excited states."""

    _keys = tduks_sf_grad.Gradients._keys | {
        "nttda_details",
        "auxbasis_response",
    }
    auxbasis_response = True

    def __init__(self, tdobj):
        super().__init__(tdobj)
        self.nttda_details = None

    def jk_energies_per_atom(
            self, dm_list, j_factor=None, k_factor=None, omega=0,
            hermi=0, sum_results=False, verbose=None):
        if isinstance(self.base._scf, _DFHF):
            method = df_tduks_sf_grad.Gradients.jk_energies_per_atom
        else:
            method = tduks_sf_grad.Gradients.jk_energies_per_atom
        return method(
            self,
            dm_list,
            j_factor=j_factor,
            k_factor=k_factor,
            omega=omega,
            hermi=hermi,
            sum_results=sum_results,
            verbose=verbose,
        )

    def grad_elec(self, xy, atmlst=None, verbose=None):
        del verbose
        if atmlst is not None:
            raise NotImplementedError("GPU NTTDA gradients do not support atmlst")
        options = {
            "tolerance": self.cphf_conv_tol,
            "max_cycle": self.cphf_max_cycle,
        }
        if self.base.deltaS == 0:
            details = delta_s_zero.grad_elec(self, self.base, xy, **options)
        elif self.base.deltaS == -1:
            details = delta_s_minus_one.grad_elec(
                self, self.base, xy, **options,
            )
        elif self.base.deltaS == 1:
            details = delta_s_plus_one.grad_elec(
                self, self.base, xy, **options,
            )
        else:
            raise ValueError("deltaS must be -1, 0, or 1")
        self.nttda_details = details
        return details.total

    def kernel(self, state=None, atmlst=None):
        if atmlst is not None:
            raise NotImplementedError("GPU NTTDA gradients do not support atmlst")
        if state is not None:
            self.state = state
        if self.state == 0:
            raise NotImplementedError("GPU NTTDA gradients require state >= 1")
        if self.base.xy is None:
            self.base.run()
        if not 1 <= self.state <= len(self.base.xy):
            raise ValueError("state must be in [1, %d]" % len(self.base.xy))
        if self.verbose >= logger.INFO:
            self.dump_flags()

        excitation = self.grad_elec(self.base.xy[self.state - 1])
        self.de = cp.asnumpy(reference_gradient(self) + excitation)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name="grad")


Grad = Gradients

__all__ = ["Grad", "Gradients"]
