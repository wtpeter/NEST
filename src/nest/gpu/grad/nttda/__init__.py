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

"""GPU analytic gradients for all NTTDA spin channels."""

import cupy as cp
import numpy as np

from pyscf import lib
from gpu4pyscf.df.df_jk import _DFHF
from gpu4pyscf.df.grad import tdrhf as df_tdrhf_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.grad import uks as uks_grad
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import get_avail_mem, tag_array
from nest.grad.nttda import Gradients as CPUGradients

from . import delta_s_minus_one, delta_s_plus_one, delta_s_zero
from .common import JKDerivativeLedger, hcore_derivative, _reference_spin_densities


def reference_gradient(gradient_driver):
    """Return the complete ground-state ROKS gradient."""
    mf = gradient_driver.base._scf
    mol = mf.mol
    if mf.do_nlc():
        raise NotImplementedError('GPU NTTDA gradients do not support NLC')

    density_alpha, density_beta = _reference_spin_densities(gradient_driver.base)
    density_total = density_alpha + density_beta
    fock = mf.get_fock()
    energy_weighted = density_alpha @ fock.focka @ density_alpha + density_beta @ fock.fockb @ density_beta
    electronic = hcore_derivative(gradient_driver, density_total)
    overlap = gradient_driver.get_ovlp(mol)
    electronic -= cp.asarray(
        rhf_grad.contract_h1e_dm(
            mol,
            overlap,
            energy_weighted,
            hermi=1,
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
        mf.xc,
        mol.spin,
    )
    ledger = JKDerivativeLedger()
    ledger.add(
        'j',
        'reference',
        ((density_total, density_total, 0.5, None),),
    )
    ledger.add(
        'k',
        'reference',
        (
            (density_alpha, density_alpha, -0.5 * hybrid, None),
            (density_beta, density_beta, -0.5 * hybrid, None),
        ),
    )
    if omega != 0:
        long_range = -0.5 * (alpha - hybrid)
        ledger.add(
            'k',
            'reference',
            (
                (density_alpha, density_alpha, long_range, omega),
                (density_beta, density_beta, long_range, omega),
            ),
        )
    electronic += ledger.contract(
        gradient_driver,
        mol,
        range(mol.natm),
        slots=('reference',),
    )['reference']
    nuclear = rhf_grad.GradientsBase.grad_nuc(gradient_driver, mol)
    if mf.do_disp():
        # The reference energy already contains this state-independent term.
        electronic += cp.asarray(rhf_grad.GradientsBase(mf).get_dispersion())
    return electronic + cp.asarray(nuclear)


class Gradients(tdrhf_grad.Gradients):
    """Analytic total-energy gradients for ROKS GPU NTTDA.

    Includes the auxiliary-basis response for DF references. DFT derivatives
    use fixed quadrature grids; moving-grid and X2C responses are not supported.
    """

    device = 'gpu'
    cphf_conv_tol = 1e-12
    cphf_max_cycle = None
    auxbasis_response = True
    grid_response = False
    _keys = tdrhf_grad.Gradients._keys | {'nttda_details', 'auxbasis_response', 'grid_response'}
    as_scanner = CPUGradients.as_scanner
    symmetrize = CPUGradients.symmetrize

    def __init__(self, tdobj):
        super().__init__(tdobj)
        self.nttda_details = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** GPU NTTDA nuclear gradients ********')
        log.info('state = %s, deltaS = %s, nobeta = %s', self.state, self.base.deltaS, self.base.nobeta)
        log.info('cphf_conv_tol = %g, cphf_max_cycle = %s', self.cphf_conv_tol, self.cphf_max_cycle)
        return self

    def jk_energies_per_atom(
        self, dm_list, j_factor=None, k_factor=None, omega=None, lr_factor=None, sr_factor=None,
        hermi=0, sum_results=False, verbose=None,
    ):
        if isinstance(self.base._scf, _DFHF):
            if omega:
                if j_factor is not None and any(j_factor):
                    # The DF wrapper rejects long-range J, but its factorized
                    # derivative kernel supports both J and K with this operator.
                    with_df = self.base._scf.with_df
                    mol, auxmol = with_df.mol, with_df.auxmol
                    with_df.reset()
                    with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
                        sorted_mol = df_tdrhf_grad.SortedMole.from_mol(mol, decontract=True)
                        intopt = df_tdrhf_grad.Int3c2eOpt(sorted_mol, auxmol).build()
                    factors = [df_tdrhf_grad._factorize_multiple_dm(sorted_mol, pair, hermi) for pair in dm_list]
                    if k_factor is None:
                        k_factor = [0.0] * len(dm_list)
                    # Retain the native wrapper's memory bound when bypassing
                    # its long-range J restriction. Each pair stores two
                    # auxiliary-by-orbital-pair intermediates and two AO DMs.
                    mem_avail = get_avail_mem(exclude_memory_pool=True) * .6
                    splits = [0]
                    cost = 0
                    for i, (left, _, right, _) in enumerate(factors):
                        pair_cost = (2 * left.shape[1] * right.shape[1] * intopt.auxmol.nao
                                     + 2 * sorted_mol.nao ** 2) * 8
                        if i > splits[-1] and cost + pair_cost > mem_avail:
                            splits.append(i)
                            cost = 0
                        cost += pair_cost
                    splits.append(len(factors))
                    results = []
                    for p0, p1 in zip(splits[:-1], splits[1:]):
                        results.append(df_tdrhf_grad._jk_energies_by_dm_factors(
                            intopt, factors[p0:p1], j_factor[p0:p1], k_factor[p0:p1],
                            sum_results, verbose, omega=omega, lr_factor=lr_factor, sr_factor=sr_factor,
                        ))
                    return np.sum(results, axis=0) if sum_results else np.concatenate(results)
                j_factor = None
            method = df_tdrhf_grad.Gradients.jk_energies_per_atom
        else:
            method = tdrhf_grad.Gradients.jk_energies_per_atom
        return method(
            self, dm_list, j_factor, k_factor, omega=omega,
            lr_factor=lr_factor, sr_factor=sr_factor,
            hermi=hermi, sum_results=sum_results, verbose=verbose,
        )

    def grad_elec(self, xy, atmlst=None, verbose=None):
        """Return the excitation-energy derivative on the GPU."""
        if self.base.deltaS == -1:
            channel = delta_s_minus_one
        elif self.base.deltaS == 0:
            channel = delta_s_zero
        elif self.base.deltaS == 1:
            channel = delta_s_plus_one
        else:
            raise ValueError('deltaS must be -1, 0, or 1')
        self.nttda_details = channel.grad_elec(
            self,
            self.base,
            xy,
            atmlst=atmlst,
            tolerance=self.cphf_conv_tol,
            max_cycle=self.cphf_max_cycle,
        )
        return self.nttda_details.total

    def kernel(self, state=None, atmlst=None):
        """Return d(E_ROKS + omega_state)/dR in Eh/Bohr as a NumPy array."""
        if state is not None:
            self.state = state
        if atmlst is not None:
            self.atmlst = list(atmlst)
        mf = self.base._scf
        for attribute, label in (('with_x2c', 'X2C'), ('with_solvent', 'Solvent-response')):
            if getattr(mf, attribute, None) is not None:
                raise NotImplementedError('%s GPU NTTDA gradients are not implemented' % label)
        if mf.do_nlc():
            raise NotImplementedError('NLC GPU NTTDA gradients are not implemented')
        if self.grid_response:
            raise NotImplementedError('GPU NTTDA gradients require grid_response=False')
        if isinstance(mf, _DFHF) and (mf.only_dfj or not self.auxbasis_response):
            raise NotImplementedError('DF NTTDA gradients require full J/K fitting and auxiliary-basis response')
        if self.state != 0:
            if self.base.xy is None:
                self.base.run()
            if not 1 <= self.state <= len(self.base.xy):
                raise ValueError('state must be in [0, %d]' % len(self.base.xy))
            if not bool(self.base.converged[self.state - 1]):
                raise RuntimeError('Converge the selected NTTDA state before computing a gradient')
        if not mf.converged:
            raise RuntimeError('Converge the ROKS reference before computing a gradient')
        if self.verbose >= logger.INFO:
            self.dump_flags()

        # Symmetrize the complete gradient before selecting atoms.
        result = reference_gradient(self)
        if self.state != 0:
            result += self.grad_elec(self.base.xy[self.state - 1])
        self.de = cp.asnumpy(result)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de)
        if self.atmlst is not None:
            self.de = self.de[list(self.atmlst)]
        self._finalize()
        return self.de

    def to_cpu(self):
        out = utils.to_cpu(self, out=CPUGradients(self.base.to_cpu()))
        out.nttda_details = None
        return out

    @classmethod
    def from_cpu(cls, method):
        if method.method != 'analytic':
            raise NotImplementedError('GPU NTTDA gradients implement method="analytic" only')
        out = cls(method.base.to_gpu())
        for name in ('state', 'cphf_conv_tol', 'cphf_max_cycle', 'max_memory', 'verbose', 'atmlst', 'de'):
            setattr(out, name, getattr(method, name))
        return out

    grad = lib.alias(kernel, alias_name='grad')


Grad = Gradients
