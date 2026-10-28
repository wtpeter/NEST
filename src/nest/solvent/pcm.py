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

"""State-specific PCM with fixed reference orbitals and a fixed TDA kernel.

The variational model is G[x] = E_ref + x.T A_vacuum x + E_PCM[D[x]].
Only the physical state's amplitudes and the reaction field are iterated.
Vertical ptSS energies use the quadratic fast-polarization correction
    1/2 (D_k - D_A) : B(eps_optical) : (D_k - D_A)
about the equilibrated physical state A.
"""

import numpy as np
from pyscf import lib, scf
from pyscf.lib import logger
from pyscf.solvent import pcm

from nest._lr_eig import eigh
from nest.nttda.nttda import NTTDA, _orbital_indices, _project_sfd, _sc_vector_slices, transition_dm
from nest.sftda.uhf_sf import TDA_SF, TDDFT_SF


class _FrozenFieldTDA:
    """Mixin marking a result from one fixed-field solve.

    ``_solve`` attaches this before the original TDA class and installs the
    combined matrix-vector product. The outer PCM loop may update the field
    between solves. These overrides limit inherited derivative/matrix APIs
    to the capabilities implemented for the solvent model.
    """

    def nuc_grad_method(self):
        raise NotImplementedError('Physical-state PCM gradients are not implemented.')

    Gradients = nuc_grad_method

    def nac_method(self):
        raise NotImplementedError('Physical-state PCM derivative couplings are not implemented.')

    NAC = nac_method

    def as_scanner(self):
        raise NotImplementedError('Create a new physical-state PCM driver for each geometry.')

    def get_ab_sf(self, *args, **kwargs):
        raise NotImplementedError('Use the frozen-field matrix-vector product for PCM-TDA.')

    get_ab = get_ab_sf


def _validate_td(td):
    if not isinstance(td, (NTTDA, TDA_SF)) or isinstance(td, TDDFT_SF):
        raise TypeError('StateSpecificPCM requires SF-TDA or NT-TDA.')
    mf = td._scf
    if (not isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF))
            or any(cls.__module__.startswith('nest.aocscf') for cls in type(mf).__mro__)):
        raise TypeError('Use an ordinary UKS/UHF or ROKS/ROHF reference.')
    if getattr(mf, 'with_solvent', None) is not None:
        raise ValueError('Start from a vacuum reference; StateSpecificPCM supplies the physical-state reaction field.')
    if not mf.converged:
        raise ValueError('Converge the vacuum reference before constructing StateSpecificPCM.')
    if np.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError('StateSpecificPCM currently supports real reference orbitals.')
    if isinstance(td, NTTDA) and td.deltaS not in (-1, 0, 1):
        raise ValueError('deltaS must be -1, 0, or 1.')
    if isinstance(td, NTTDA):
        nopen = np.count_nonzero(mf.mo_occ == 1)
        if nopen == 0 or (td.deltaS == -1 and nopen < 2):
            raise ValueError('NT-TDA requires an open-shell reference; deltaS=-1 requires S >= 1.')
    if isinstance(td, TDA_SF) and td.extype not in (0, 1):
        raise ValueError('extype must be 0 or 1.')


def state_density(td, state=1):
    """Spin-summed AO density of a normalized TDA root (1-based index).

    This is the fixed-orbital wavefunction density, including the reference
    occupation. It is not an orbital-relaxed energy-derivative density.
    """
    _validate_td(td)
    assert 1 <= state <= len(td.xy)
    x, y = td.xy[state - 1]
    if np.any(y):
        raise ValueError('A normalized TDA amplitude with Y=0 is required.')
    mf = td._scf
    norm = np.vdot(x, x)
    if isinstance(td, NTTDA):
        # transition_dm contains the normal-ordered E_pq matrix elements.
        gamma = transition_dm(td, state - 1, state - 1) + norm * np.diag(mf.mo_occ)
        # gamma[p,q] = <a_p^+ a_q>; the AO density uses its transpose.
        return mf.mo_coeff @ gamma.T @ mf.mo_coeff.conj().T

    spin_o, spin_v = (1, 0) if td.extype == 0 else (0, 1)
    co = mf.mo_coeff[spin_o][:, mf.mo_occ[spin_o] > 0]
    cv = mf.mo_coeff[spin_v][:, mf.mo_occ[spin_v] == 0]
    hole = x.conj() @ x.T
    particle = x.T @ x.conj()
    dm = norm * np.asarray(mf.make_rdm1()).sum(axis=0)
    return dm - co @ hole @ co.conj().T + cv @ particle @ cv.conj().T


def _potential_action(td, v_ao):
    """Normal-ordered spin-free one-electron potential in the TDA space.

    Returns its matrix-vector product and diagonal. The contractions are the
    dual of <I|E_pq|J>, including all NT CO/CV/OO/OV/CV(0) couplings.
    No AO response kernel is built here.
    """
    mf = td._scf
    if isinstance(td, TDA_SF):
        so, sv = (1, 0) if td.extype == 0 else (0, 1)
        co = mf.mo_coeff[so][:, mf.mo_occ[so] > 0]
        cv = mf.mo_coeff[sv][:, mf.mo_occ[sv] == 0]
        voo, vvv = co.T @ v_ao @ co, cv.T @ v_ao @ cv
        shape = (co.shape[1], cv.shape[1])

        def action(z):
            z = np.asarray(z).reshape(-1, *shape)
            out = np.einsum('ab,xib->xia', vvv, z) - np.einsum('ji,xja->xia', voo, z)
            return out.reshape(len(z), -1)

        return action, (vvv.diagonal()[None, :] - voo.diagonal()[:, None]).ravel()

    cs, os, vs = _orbital_indices(td)
    nc, no, nv = len(cs), len(os), len(vs)
    s = no / 2
    v = mf.mo_coeff.T @ v_ao @ mf.mo_coeff
    cc, oo, vv = v[np.ix_(cs, cs)], v[np.ix_(os, os)], v[np.ix_(vs, vs)]
    co, ov, cv = v[np.ix_(cs, os)], v[np.ix_(os, vs)], v[np.ix_(cs, vs)]
    dco = oo.diagonal()[None, :] - cc.diagonal()[:, None]
    dcv = vv.diagonal()[None, :] - cc.diagonal()[:, None]
    dov = vv.diagonal()[None, :] - oo.diagonal()[:, None]

    if td.deltaS == 1:
        def action(z):
            z = np.asarray(z).reshape(-1, nc, nv)
            out = np.einsum('ab,xib->xia', vv, z) - np.einsum('ji,xja->xia', cc, z)
            return out.reshape(len(z), -1)
        return action, dcv.ravel()

    if td.deltaS == 0:
        slices = _sc_vector_slices(nc, no, nv)
        f = np.sqrt((s + 1) / (2 * s))

        def action(z):
            z = np.atleast_2d(z)
            zco = z[:, slices['CO(1)']].reshape(-1, nc, no)
            zcv = z[:, slices['CV(1)']].reshape(-1, nc, nv)
            zoo = z[:, slices['OO(1)']].ravel()
            zov = z[:, slices['OV(1)']].reshape(-1, no, nv)
            zcv0 = z[:, slices['CV(0)']].reshape(-1, nc, nv)
            rco = np.einsum('uv,xiv->xiu', oo, zco) - np.einsum('ji,xju->xiu', cc, zco)
            rco += f * np.einsum('ub,xib->xiu', ov, zcv)
            rco -= np.einsum('iu,x->xiu', co, zoo)
            rco += np.einsum('ub,xib->xiu', ov, zcv0) / np.sqrt(2)
            rcv = np.einsum('ab,xib->xia', vv, zcv) - np.einsum('ji,xja->xia', cc, zcv)
            rcv += f * (np.einsum('ua,xiu->xia', ov, zco) - np.einsum('iu,xua->xia', co, zov))
            rov = np.einsum('ab,xub->xua', vv, zov) - np.einsum('vu,xva->xua', oo, zov)
            rov -= f * np.einsum('ju,xja->xua', co, zcv)
            rov += np.einsum('ua,x->xua', ov, zoo)
            rov += np.einsum('ju,xja->xua', co, zcv0) / np.sqrt(2)
            rcv0 = np.einsum('ab,xib->xia', vv, zcv0) - np.einsum('ji,xja->xia', cc, zcv0)
            rcv0 += (np.einsum('ua,xiu->xia', ov, zco) + np.einsum('iu,xua->xia', co, zov)) / np.sqrt(2)
            rcv0 -= np.sqrt(2) * np.einsum('ia,x->xia', cv, zoo)
            roo = -np.einsum('iu,xiu->x', co, zco) + np.einsum('ua,xua->x', ov, zov)
            roo -= np.sqrt(2) * np.einsum('ia,xia->x', cv, zcv0)
            return np.concatenate((rco.reshape(len(z), -1), rcv.reshape(len(z), -1), roo[:, None],
                                   rov.reshape(len(z), -1), rcv0.reshape(len(z), -1)), axis=1)

        diag = np.concatenate((dco.ravel(), dcv.ravel(), [0.0], dov.ravel(), dcv.ravel()))
        return action, diag

    a = np.sqrt(2 * s / (2 * s - 1))
    b = 1 / np.sqrt(2 * s * (2 * s - 1))
    f = np.sqrt((2 * s + 1) / (2 * s))

    def action(z):
        z = np.asarray(z).reshape(-1, nc + no, no + nv)
        zco, zcv, zoo, zov = z[:, :nc, :no], z[:, :nc, no:], z[:, nc:, :no], z[:, nc:, no:]
        trace = np.trace(zoo, axis1=1, axis2=2)
        rco = np.einsum('uv,xiv->xiu', oo, zco) - np.einsum('ji,xju->xiu', cc, zco)
        rco += f * np.einsum('ub,xib->xiu', ov, zcv)
        rco -= a * np.einsum('iw,xwu->xiu', co, zoo)
        rco += b * np.einsum('iu,x->xiu', co, trace)
        rcv = np.einsum('ab,xib->xia', vv, zcv) - np.einsum('ji,xja->xia', cc, zcv)
        rcv += f * (np.einsum('ua,xiu->xia', ov, zco) - np.einsum('iu,xua->xia', co, zov))
        roo = np.einsum('tv,xuv->xut', oo, zoo) - np.einsum('wu,xwt->xut', oo, zoo)
        roo -= a * np.einsum('ju,xjt->xut', co, zco)
        roo += a * np.einsum('tb,xub->xut', ov, zov)
        dtrace = b * (np.einsum('jv,xjv->x', co, zco) - np.einsum('vb,xvb->x', ov, zov))
        indices = np.arange(no)
        roo[:, indices, indices] += dtrace[:, None]
        rov = np.einsum('ab,xub->xua', vv, zov) - np.einsum('vu,xva->xua', oo, zov)
        rov -= f * np.einsum('ju,xja->xua', co, zcv)
        rov += a * np.einsum('va,xuv->xua', ov, zoo)
        rov -= b * np.einsum('ua,x->xua', ov, trace)
        return np.concatenate((np.concatenate((rco, rcv), axis=2),
                               np.concatenate((roo, rov), axis=2)), axis=1).reshape(len(z), -1)

    doo = oo.diagonal()[None, :] - oo.diagonal()[:, None]
    return action, np.block([[dco, dcv], [doo, dov]]).ravel()


class StateSpecificPCM(lib.StreamObject):
    """Equilibrate PCM with a physical SF/NT root, then compute vertical ptSS gaps.

    Args:
        td: Vacuum SF-TDA or NT-TDA template, with the desired number of roots.
            Its SCF reference must be converged. The driver solves the vacuum
            TDA eigenproblem internally, using compatible supplied amplitudes
            as initial guesses.
        solvent: A PySCF C-PCM object on the same molecule. It is copied.
        eps_optical: Optical dielectric constant (squared refractive index).

    ``base`` stores a copy of the input template, sharing its SCF reference.
    ``gas_td`` stores the initial vacuum spectrum obtained by ``_solve`` with
    zero reaction field, including ``e``, ``xy`` and ``converged``.
    ``td`` stores the most recent fixed-field result; ``ss_td`` stores the
    final target-field result from non-equilibrium SS. Their eigenvalues
    are fixed-field quantities; use this driver's returned energies/gaps.

    ``kernel(state=1)`` performs equilibrium SS and returns the total model energy of the
    selected physical root. ``vertical()`` returns ptSS energy gaps from that
    root. Both use Hartree. A different spin sector on the *same SCF object*
    may be supplied to ``vertical(td)``. Results are stored in ``td`` (frozen
    field amplitudes), ``e_frozen``, ``e_ptss`` and ``ptss_correction``.
    ``state_specific(state, td=None)`` relaxes one target in the source's
    fixed slow-polarization environment and returns its SS gap. Set
    ``diis_space=6`` to accelerate either outer loop; the default 0 uses
    direct iteration. DIIS uses the fixed-point residual D_out - D_in.
    ``conv_tol`` inherits the PCM energy tolerance (default 1e-7 Hartree).
    ``conv_tol_dm`` is the density-change norm threshold (default 1e-4).

    Both SS loops select the energy-ordered ``state`` at every iteration.
    Set ``follow_state=True`` on either call to track a root by amplitude
    overlap instead, for example when following a CT state through crossings.
    Each geometry requires a new driver and converged vacuum reference.
    Gradients, TDDFT X/Y amplitudes and orbital relaxation are not implemented.
    """

    def __init__(self, td, solvent, *, eps_optical):
        _validate_td(td)
        if isinstance(td, _FrozenFieldTDA):
            raise ValueError('Supply a vacuum TDA object, not a frozen-field result.')
        if type(solvent) is not pcm.PCM or solvent.method.upper() not in ('C-PCM', 'CPCM'):
            raise TypeError('The initial implementation requires a PySCF C-PCM object.')
        if solvent.mol is not td.mol:
            raise ValueError('PCM and TDA must use the same molecule object.')
        if solvent.frozen:
            raise ValueError('Supply an unfrozen PCM object for physical-state solvation.')
        if not np.isfinite(solvent.eps) or solvent.eps < 1 or not np.isfinite(eps_optical) or eps_optical < 1:
            raise ValueError('Static and optical dielectric constants must be finite and at least one.')
        self.base = td.copy()
        self.base.chkfile = None
        self.mol = td.mol
        self.stdout, self.verbose, self.max_memory = td.stdout, td.verbose, td.max_memory
        self.with_solvent = solvent.copy().reset()
        self.eps_optical = float(eps_optical)
        self.max_cycle = 30
        self.conv_tol = solvent.conv_tol
        self.conv_tol_dm = 1e-4
        self.diis_space = 0
        self.converged = False
        self.cycles = 0
        self.e_tot = None
        self.dm = None
        self.v = None
        self.td = None
        self.gas_td = None
        self.e_frozen = None
        self.e_ptss = None
        self.ptss_correction = None
        self.state = None
        self.history = []
        self.ss_converged = False
        self.e_ss = self.ss_dm = self.ss_td = self.ss_state = None
        self.ss_cycles = 0
        self.ss_history = []
        dm_ref = np.asarray(td._scf.make_rdm1())
        self._dm_ref = dm_ref.sum(axis=0) if dm_ref.ndim == 3 else dm_ref
        self._vacuum_response = self._build_response(self.base)

    def _build_response(self, td):
        if isinstance(td, NTTDA):
            return {-1: td.gen_vind_sfd, 0: td.gen_vind_sc, 1: td.gen_vind_sfu}[td.deltaS]()
        return td.gen_vind()

    def _solve(self, vacuum_td, vacuum_response, potential, x0=None):
        """Solve in one fixed reaction field; retain the original amplitude layout."""
        vacuum_vind, vacuum_diag = vacuum_response
        solvent_vind, solvent_diag = _potential_action(vacuum_td, potential)
        diagonal = vacuum_diag + solvent_diag
        base_precond = vacuum_td.get_precond(diagonal)
        nt_down = isinstance(vacuum_td, NTTDA) and vacuum_td.deltaS == -1
        nroots = min(vacuum_td.nstates, len(diagonal) - int(nt_down))
        if nt_down:
            cs, os, vs = _orbital_indices(vacuum_td)
            ncs, nos, nvs = len(cs), len(os), len(vs)

            def project(z):
                return _project_sfd(z, ncs, nos, nvs)

            def vind(z):
                z = project(z)
                return project(vacuum_vind(z) + solvent_vind(z))

            def precond(residual, energy):
                # Diagonal preconditioning can reintroduce the OO trace.
                return project(base_precond(residual, energy))
        else:
            def vind(z):
                return vacuum_vind(z) + solvent_vind(z)

            precond = base_precond

        indices = np.argsort(diagonal)[:min(nroots + 3, len(diagonal))]
        guesses = np.zeros((len(indices), len(diagonal)))
        guesses[np.arange(len(indices)), indices] = 1
        if x0 is not None and x0.shape[1] == len(diagonal):
            # Keep diagonal guesses as well: a previously converged root can
            # miss a newly lower state in another invariant subspace.
            guesses = np.vstack((x0, guesses))
        if nt_down:
            guesses = project(guesses)

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        # Use the TDA tolerance, tightening it if the outer density threshold
        # requires a more accurate eigenvector.
        tolerance = min(vacuum_td.conv_tol, self.conv_tol_dm * 0.1)
        converged, energies, vectors = eigh(
            vind, guesses, precond, nroots=nroots, pick=all_eigs,
            tol_residual=tolerance, lindep=vacuum_td.lindep,
            max_cycle=vacuum_td.max_cycle, max_memory=vacuum_td.max_memory,
            verbose=logger.new_logger(vacuum_td))
        field_td = lib.set_class(vacuum_td.copy(), (_FrozenFieldTDA, vacuum_td.__class__))
        field_td.chkfile = None
        field_td.e, field_td.converged = np.asarray(energies), np.asarray(converged)
        field_td.nstates = len(energies)
        if isinstance(vacuum_td, NTTDA):
            cs, os, vs = _orbital_indices(vacuum_td)
            shape = {-1: (len(cs)+len(os), len(os)+len(vs)), 0: (len(diagonal),),
                     1: (len(cs), len(vs))}[vacuum_td.deltaS]
        else:
            so, sv = (1, 0) if vacuum_td.extype == 0 else (0, 1)
            shape = (np.count_nonzero(vacuum_td._scf.mo_occ[so]),
                     np.count_nonzero(vacuum_td._scf.mo_occ[sv] == 0))
        field_td.xy = [(x.reshape(shape), 0) for x in vectors]

        def gen_vind():
            return vind, diagonal

        if isinstance(field_td, NTTDA):
            setattr(field_td, {-1: 'gen_vind_sfd', 0: 'gen_vind_sc', 1: 'gen_vind_sfu'}[field_td.deltaS], gen_vind)
        else:
            field_td.gen_vind = gen_vind
        if len(field_td.e) != nroots or not np.all(field_td.converged):
            x = np.asarray(vectors)
            residuals = np.linalg.norm(vind(x) - field_td.e[:, None] * x, axis=1)
            raise RuntimeError(f'The TDA eigensolver did not converge in the PCM reaction field: {residuals}.')
        return field_td

    def kernel(self, state=1, *, follow_state=False):
        """Equilibrium SS: relax static solvent polarization with one physical root.

        By default, ``state`` selects the 1-based energy-ordered root in every
        iteration. With ``follow_state=True``, it selects the initial vacuum
        root, then amplitude overlap tracks it. ``self.state`` is the final
        energy-ordered index. Reference orbitals and the vacuum kernel stay fixed.
        """
        if not isinstance(state, (int, np.integer)) or not 1 <= state <= self.base.nstates:
            raise ValueError('state must be between 1 and base.nstates.')
        self.converged = False
        self.history = []
        self.e_frozen = self.e_ptss = None
        self.ptss_correction = None
        self.ss_converged = False
        self.e_ss = self.ss_dm = self.ss_td = self.ss_state = None
        self.ss_cycles = 0
        self.ss_history = []
        x0 = None
        if self.base.xy:
            x0 = np.array([np.asarray(x).ravel() for x, _ in self.base.xy])
        field_td = self._solve(self.base, self._vacuum_response, np.zeros_like(self._dm_ref), x0=x0)
        self.gas_td = field_td
        if state > len(field_td.xy):
            raise ValueError('The requested vacuum root is outside the computed spectrum.')
        previous = np.asarray(field_td.xy[state - 1][0]).ravel()
        dm = state_density(field_td, state).real
        e_last = None
        extrapolator = lib.diis.DIIS(self, incore=True) if self.diis_space >= 2 else None
        if extrapolator is not None:
            extrapolator.space = self.diis_space
        for cycle in range(1, self.max_cycle + 1):
            _, v = self.with_solvent.kernel(dm)
            x0 = np.array([np.asarray(x).ravel() for x, _ in field_td.xy])
            field_td = self._solve(self.base, self._vacuum_response, v, x0=x0)
            overlaps = np.array([abs(np.vdot(previous, x)) for x, _ in field_td.xy])
            root = int(overlaps.argmax()) if follow_state else state - 1
            if follow_state and overlaps[root] < 0.5:
                raise RuntimeError('Lost the physical root during PCM iteration; compute more roots or inspect mixing.')
            dm_new = state_density(field_td, root + 1).real
            e_pcm, v_new = self.with_solvent.kernel(dm_new)
            e_vacuum = field_td.e[root] - np.einsum('ij,ji->', v, dm_new - self._dm_ref)
            energy = float(self.base._scf.e_tot + e_vacuum + e_pcm)
            dm_error = np.linalg.norm(dm_new - dm)
            de = np.inf if e_last is None else energy - e_last
            self.history.append((energy, dm_error, float(overlaps[root])))
            logger.info(self, 'PCM cycle %d: root=%d G=%.12f dG=%.3g |dD|=%.3g overlap=%.6f',
                        cycle, root + 1, energy, de, dm_error, overlaps[root])
            self.cycles = cycle
            self.state = root + 1
            self.dm, self.v, self.e_tot, self.td = dm_new, v_new, energy, field_td
            if abs(de) < self.conv_tol and dm_error < self.conv_tol_dm:
                self.converged = True
                break
            e_last = energy
            dm = dm_new if extrapolator is None else extrapolator.update(dm_new, xerr=dm_new-dm)
            previous = np.asarray(field_td.xy[root][0]).ravel()
        if not self.converged:
            raise RuntimeError(f'Physical-state PCM did not converge in {self.max_cycle} cycles.')
        return self.e_tot

    def state_specific(self, state, td=None, *, follow_state=False):
        """Return one non-equilibrium SS gap, keeping the source slow field fixed.

        Run ``kernel`` first to equilibrate the source's fast and slow field.
        ``td`` optionally supplies another vacuum spin-sector template on the
        same SCF object. By default, ``state`` selects the 1-based energy-ordered
        root in every iteration. With ``follow_state=True``, track the initial
        common-field root by amplitude overlap. Only fast polarization and
        target amplitudes relax; reference orbitals and the source slow field stay fixed.
        The source ``dm``, ``v``, ``e_tot`` and ``state`` are preserved.
        Target results are stored in ``e_ss``, ``ss_dm``, ``ss_td``,
        ``ss_state``, ``ss_cycles``, ``ss_history`` and ``ss_converged``.
        ``ss_td`` contains the final target-field spectrum; its other roots
        share the target's field and are available for state analysis only.
        Initialization refreshes ``vertical(method='frozen')`` results;
        save previously computed ptSS gaps before calling this method.
        """
        self.ss_converged = False
        self.e_ss = self.ss_dm = self.ss_td = self.ss_state = None
        self.ss_cycles = 0
        self.ss_history = []
        vacuum_td = self.base if td is None else td
        if not isinstance(state, (int, np.integer)) or not 1 <= state <= vacuum_td.nstates:
            raise ValueError('state must index a common-field root (1-based).')
        self.vertical(td, method='frozen')
        target_td = self.td
        if state > len(target_td.xy):
            raise ValueError('The requested root is outside the common-field spectrum.')
        vacuum_response = self._vacuum_response if td is None else self._build_response(td)
        fast = self.with_solvent.copy().reset()
        fast.eps = self.eps_optical
        e_pcm, _ = self.with_solvent.kernel(self.dm)
        previous = np.asarray(target_td.xy[state - 1][0]).ravel()
        dm = state_density(target_td, state).real
        e_last = None
        extrapolator = lib.diis.DIIS(self, incore=True) if self.diis_space >= 2 else None
        if extrapolator is not None:
            extrapolator.space = self.diis_space
        for cycle in range(1, self.max_cycle + 1):
            potential = self.v + fast._B_dot_x(dm - self.dm)
            x0 = np.array([np.asarray(x).ravel() for x, _ in target_td.xy])
            target_td = self._solve(vacuum_td, vacuum_response, potential, x0=x0)
            overlaps = np.array([abs(np.vdot(previous, x)) for x, _ in target_td.xy])
            root = int(overlaps.argmax()) if follow_state else state - 1
            if follow_state and overlaps[root] < 0.5:
                raise RuntimeError('Lost the target root during non-equilibrium SS iteration; inspect more roots.')
            dm_new = state_density(target_td, root + 1).real
            delta = dm_new - self.dm
            e_vacuum = target_td.e[root] - np.einsum('ij,ji', potential, dm_new - self._dm_ref)
            energy = float(self.base._scf.e_tot + e_vacuum + e_pcm + np.einsum('ij,ji', self.v, delta)
                           + 0.5 * np.einsum('ij,ji', delta, fast._B_dot_x(delta)))
            error = np.linalg.norm(dm_new - dm)
            de = np.inf if e_last is None else energy - e_last
            self.ss_history.append((energy, error, float(overlaps[root])))
            self.e_ss, self.ss_dm, self.ss_td = energy, dm_new, target_td
            self.ss_state, self.ss_cycles = root + 1, cycle
            logger.info(self, 'Non-equilibrium SS cycle %d: root=%d G=%.12f dG=%.3g |dD|=%.3g overlap=%.6f',
                        cycle, root + 1, energy, de, error, overlaps[root])
            if abs(de) < self.conv_tol and error < self.conv_tol_dm:
                self.ss_converged = True
                return energy - self.e_tot
            e_last = energy
            dm = dm_new if extrapolator is None else extrapolator.update(dm_new, xerr=dm_new-dm)
            previous = np.asarray(target_td.xy[root][0]).ravel()
        raise RuntimeError(f'Non-equilibrium SS did not converge in {self.max_cycle} cycles.')

    def vertical(self, td=None, method='ptss'):
        """Return vertical gaps (Hartree) in this state's environment.

        ``td`` defaults to the original spin sector. A supplied sector must
        share the original mean-field object, orbitals and energy reference.
        This method preserves the equilibrated density and reaction field.
        ``method`` selects ``'ptss'`` (the default difference-density correction)
        or ``'frozen'`` (common-field baseline).
        """
        if not self.converged:
            raise RuntimeError('Run kernel(state=...) to equilibrate a physical state first.')
        if method not in ('ptss', 'frozen'):
            raise ValueError("method must be 'ptss' or 'frozen'.")
        vacuum_td = self.base if td is None else td
        if td is None:
            vacuum_response = self._vacuum_response
        else:
            _validate_td(vacuum_td)
            if isinstance(vacuum_td, _FrozenFieldTDA):
                raise ValueError('Supply a vacuum TDA object for the target sector.')
            if vacuum_td._scf is not self.base._scf:
                raise ValueError('All sectors must share the same vacuum SCF object.')
            vacuum_response = self._build_response(vacuum_td)
        same_previous_sector = ((isinstance(self.td, NTTDA) and isinstance(vacuum_td, NTTDA)
                                 and self.td.deltaS == vacuum_td.deltaS)
                                or (isinstance(self.td, TDA_SF) and isinstance(vacuum_td, TDA_SF)
                                    and self.td.extype == vacuum_td.extype))
        x0 = None
        if same_previous_sector:
            x0 = np.array([np.asarray(x).ravel() for x, _ in self.td.xy])
        self.td = self._solve(vacuum_td, vacuum_response, self.v, x0=x0)
        e_pcm, _ = self.with_solvent.kernel(self.dm)
        offset = self.base._scf.e_tot + e_pcm - np.einsum('ij,ji->', self.v, self.dm - self._dm_ref)
        self.e_frozen = self.td.e + offset
        self.e_ptss = self.ptss_correction = None
        if method == 'frozen':
            return self.e_frozen - self.e_tot
        # Reset the copied cavity before changing eps so cached PCM matrices
        # always correspond to the requested optical dielectric constant.
        fast = self.with_solvent.copy().reset()
        fast.eps = self.eps_optical
        delta = np.array([state_density(self.td, k + 1).real - self.dm for k in range(len(self.td.e))])
        v_delta = fast._B_dot_x(delta)
        self.ptss_correction = 0.5 * np.einsum('kij,kji->k', delta, v_delta)
        self.e_ptss = self.e_frozen + self.ptss_correction
        return self.e_ptss - self.e_tot
