"""Analytic NTTDA Hessian with iterative responses and direct integral actions.

The integral derivatives are analytic libcint derivatives.  Orbital and state
responses are solved at one geometry; no displaced SCF/gradient is used here.
See ``DERIVATION.md`` for the constrained second-derivative formula.
"""

import copy
import itertools
import types

import numpy as np
from pyscf import dft, lib
from pyscf.hessian import rhf as rhf_hess
from pyscf.lib import logger
from pyscf.scf import hf

from nest.grad.nttda.roks import canonical_pairs, pack_m_matrix
from .response import orbital_actions, solve as solve_response
from .eri import DirectERI


def _transform(tensor, coefficients):
    """Transform each AO slot, retaining chemists' order for ERIs."""
    result = tensor
    for slot, coefficient in enumerate(coefficients):
        result = np.moveaxis(np.tensordot(result, coefficient, axes=(slot, 0)), -1, slot)
    return result


def _first_transform(tensor, derivative, c, ca):
    rank = tensor.ndim
    result = _transform(derivative, [c] * rank)
    for slot in range(rank):
        coefficients = [c] * rank
        coefficients[slot] = ca
        result += _transform(tensor, coefficients)
    return result


def _second_transform(tensor, ta, tb, tab, c, ca, cb, cab):
    """Product rule including both ordered cross-slot orbital derivatives."""
    rank = tensor.ndim
    result = _transform(tab, [c] * rank)
    for slot in range(rank):
        for integral, orbital in ((ta, cb), (tb, ca), (tensor, cab)):
            coefficients = [c] * rank
            coefficients[slot] = orbital
            result += _transform(integral, coefficients)
        for other in range(rank):
            if other != slot:
                coefficients = [c] * rank
                coefficients[slot] = ca
                coefficients[other] = cb
                result += _transform(tensor, coefficients)
    return result


def _rotation(vector, pairs, nmo):
    result = np.zeros((nmo, nmo))
    for value, (p, q, _) in zip(vector, pairs):
        result[p, q] = value
        result[q, p] = -value
    return result


def _mo_operators(tdobj, h, eri, pairs, xc=None):
    """Linear maps from MO integrals to E_ref, the ROKS residual, and A.

    Calling the existing HF NTTDA action in an identity MO basis preserves all
    channel coefficients and amplitude layouts.  This also works for integral
    derivatives because all three outputs are linear in h and (pq|rs).
    Optional XC values/derivatives supply contracted semilocal kernel actions.
    """
    occ = np.asarray(tdobj._scf.mo_occ)
    da = np.diag((occ > 0).astype(float))
    db = np.diag((occ == 2).astype(float))

    hybrid = tdobj._scf._numint.rsh_and_hybrid_coeff(tdobj._scf.xc, tdobj.mol.spin)[2]
    if isinstance(eri, np.ndarray):  # Explicit tensor oracle used by small tests.
        def integral_action(dm, exchange=False):
            script = 'prsq,...rs->...pq' if exchange else 'pqrs,...sr->...pq'
            return np.einsum(script, eri, dm, optimize=True)
    else:
        integral_action = eri.apply
    j = integral_action(da + db)
    ka, kb = integral_action(np.asarray((da, db)), exchange=True) if hybrid else (0., 0.)
    fa = h + j - hybrid * ka
    fb = h + j - hybrid * kb
    energy = 0.5 * (np.einsum('pq,pq', da, h + fa) + np.einsum('pq,pq', db, h + fb))
    kernel = None
    if xc is not None:
        energy_xc, potential, kernel, common = xc
        energy += energy_xc
        fa = fa + potential[0]
        fb = fb + potential[1]
    residual_focks = {'co': fb, 'cv': fa + fb, 'ov': fa}
    residual = np.asarray([
        residual_focks[name][p, q]
        for p, q, name in pairs
    ])
    if xc is not None and tdobj.nobeta:
        correction = common - 0.5 * (potential[0] + potential[1])
        fa = fa + correction
        fb = fb + correction

    def get_j(mol, dm, hermi=0, **kwargs):
        out = hybrid * integral_action(dm) if hybrid else np.zeros_like(dm)
        if kernel is not None:
            out -= kernel.apply(dm, exchange=True)
        return out

    def get_k(mol, dm, hermi=0, **kwargs):
        out = hybrid * integral_action(dm, exchange=True) if hybrid else np.zeros_like(dm)
        if kernel is not None:
            out -= kernel.apply(dm)
        return out

    mf = copy.copy(tdobj._scf)
    # Native channel algebra consumes only J/K actions and Fock matrices.
    mf.xc = 'HF'
    mf.mo_coeff = np.eye(len(occ))
    mf.get_j = get_j
    mf.get_k = get_k
    mf.get_fock = lambda **kwargs: lib.tag_array(0.5 * (fa + fb), focka=fa, fockb=fb)
    td = copy.copy(tdobj)
    td._scf = mf
    td.verbose = 0
    action, _ = {1: td.gen_vind_sfu, 0: td.gen_vind_sc, -1: td.gen_vind_sfd}[td.deltaS]()
    return energy, residual, action


def _overlap_second(saa, sab, mask_a, mask_b, xyz_a, xyz_b):
    same = mask_a * mask_b
    aa = saa[xyz_a, xyz_b]
    ab = sab[xyz_a, xyz_b]
    return (aa * same[:, None] + aa.T * same[None, :]
            + ab * mask_a[:, None] * mask_b[None, :]
            + ab.T * mask_b[:, None] * mask_a[None, :])


class Hessian(lib.StreamObject):
    """Analytic HF / fixed-grid DFT total-energy Hessian with direct integrals.

    ``kernel(state=1)`` uses one-based NTTDA roots and returns
    ``(natm, natm, 3, 3)`` in Eh/Bohr**2.  ``atmlst`` selects both atom axes.
    Only isolated states, real full-rank MOs, and conventional all-electron
    integrals are supported. DFT requires analytic XC fourth derivatives and
    holds grid coordinates and weights fixed. Responses default to matrix-free
    GMRES. AO integrals and their derivatives are contracted by shell batches.
    """

    _keys = {'base', 'mol', 'state', 'atmlst', 'de', 'response_residual', 'state_gap',
             'solver', 'conv_tol', 'max_cycle', 'response_iterations',
             'reference_residual', 'state_residual'}

    def __init__(self, tdobj):
        self.base = tdobj
        self.mol = tdobj.mol
        self.verbose = tdobj.verbose
        self.stdout = tdobj.stdout
        self.max_memory = tdobj.max_memory
        self.state = 1
        self.atmlst = None
        self.de = None
        self.response_residual = None
        self.state_gap = None
        self.solver = 'iterative'
        self.conv_tol = 1e-10
        self.max_cycle = 100
        self.response_iterations = None
        self.reference_residual = None
        self.state_residual = None

    def kernel(self, state=None, atmlst=None):
        td = self.base
        mf = td._scf
        mol = self.mol
        if state is not None:
            self.state = state
        if atmlst is not None:
            self.atmlst = tuple(atmlst)
        atoms = tuple(range(mol.natm)) if self.atmlst is None else self.atmlst
        if self.solver not in ('iterative', 'dense'):
            raise ValueError("solver must be 'iterative' or 'dense'")
        if self.conv_tol <= 0 or self.max_cycle < 1:
            raise ValueError('response tolerance and max_cycle must be positive')
        self.response_iterations = {'adjoint': 0, 'orbital': 0, 'state': 0}
        if not isinstance(mf, dft.roks.ROKS) or mf._numint._xc_type(mf.xc) not in ('HF', 'LDA', 'GGA', 'MGGA'):
            raise NotImplementedError('NTTDA analytic Hessian requires HF or semilocal ROKS')
        xctype = mf._numint._xc_type(mf.xc)
        coefficients = mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        if xctype == 'HF' and coefficients != (0, 1, 1):
            raise NotImplementedError('NTTDA Hessian requires full-range 100% HF exchange')
        if coefficients[0] != 0 or mf.do_nlc():
            raise NotImplementedError('NTTDA Hessian does not support range separation or nonlocal correlation')
        if xctype != 'HF':
            mf._numint.libxc.test_deriv_order(mf.xc, 4, raise_error=True)
            if mf._numint.libxc.needs_laplacian(mf.xc):
                raise NotImplementedError('NTTDA Hessian does not support Laplacian-dependent functionals')
        if (getattr(mf, 'with_df', None) is not None or getattr(mf, 'with_x2c', None) is not None
                or mol.has_ecp() or mol.pseudo):
            raise NotImplementedError('NTTDA Hessian requires conventional all-electron, nonrelativistic integrals')
        if td.deltaS not in (-1, 0, 1):
            raise ValueError('deltaS must be -1, 0, or 1')
        if not mf.converged:
            raise RuntimeError('ROKS reference is not converged')
        # RHF.get_jk may allocate incore ERIs even with direct_scf=True.
        # Use the direct SCF method on a private reference, including automatic
        # TD setup and the gradient/response routines reused by the Hessian.
        mf = copy.copy(mf)
        mf._eri = None
        mf._opt = {}
        mf.direct_scf = True
        mf.get_jk = types.MethodType(hf.SCF.get_jk, mf)
        td = copy.copy(td)
        td._scf = mf
        if td.xy is None:
            td.run()
            for name in ('e', 'xy', 'converged', 'nstates'):
                setattr(self.base, name, getattr(td, name))
        if not isinstance(self.state, (int, np.integer)) or not 1 <= self.state <= len(td.xy):
            raise ValueError('state must select an existing one-based NTTDA root')
        if not atoms or len(set(atoms)) != len(atoms) or any(a < 0 or a >= mol.natm for a in atoms):
            raise ValueError('atmlst must contain distinct valid atom indices')
        c = np.asarray(mf.mo_coeff)
        nao = mol.nao_nr()
        if np.iscomplexobj(c) or c.shape != (nao, nao):
            raise NotImplementedError('NTTDA Hessian requires real, full-rank spatial MOs')
        if not np.allclose(c.T @ mf.get_ovlp() @ c, np.eye(nao), atol=1e-9, rtol=0):
            raise ValueError('reference MOs are not overlap-orthonormal')
        pairs = canonical_pairs(td)
        nrot = len(pairs)
        vector = np.asarray(td.xy[self.state - 1][0]).ravel()
        ndim = len(vector)
        ncoord = 3 * len(atoms)
        # Density/potential batches, orbital derivatives and response vectors.
        response_size = 5 * ndim**2 + 3 * nrot**2 if self.solver == 'dense' else 45 * (ndim + nrot)
        required_mb = 8e-6 * ((160 + 6*ncoord) * nao**2 + response_size)
        if xctype != 'HF':
            nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
            # Compact factors for base/first/second kernel actions, plus one
            # block of uncontracted XC derivatives. No G*N**2 pair tensors.
            required_mb += 8e-6 * (
                len(mf.grids.weights) * (48 * nao + 32 * nvar**2)
                + min(1024, len(mf.grids.weights)) * (3 * (2*nvar)**4 + 100*nao))
        if required_mb > self.max_memory - lib.current_memory()[0]:
            raise MemoryError('NTTDA Hessian needs approximately %.0f MB of additional memory' % required_mb)
        log = logger.new_logger(self)
        log.info('Analytic %s NTTDA Hessian: state %d, %d rotations, %d amplitudes', xctype, self.state, nrot, ndim)
        xc_grid = None
        xc0 = None
        if xctype != 'HF':
            from .xc import Semilocal
            log.info('DFT Hessian holds quadrature coordinates and weights fixed')
            xc_grid = Semilocal(mf)
            xc0 = xc_grid.terms()
        h = mf.get_hcore()
        hm = c.T @ h @ c
        gm = DirectERI(mol, c, max_memory=self.max_memory)
        _, residual, action = _mo_operators(td, hm, gm, pairs, xc0)
        self.reference_residual = float(np.linalg.norm(residual))
        # Respect the reference's declared accuracy, including PySCF's factor
        # of three in its final SCF convergence check. Tight thresholds are
        # useful for finite differences, but are not a requirement for use.
        scf_tolerance = mf.conv_tol_grad
        if scf_tolerance is None:
            scf_tolerance = np.sqrt(mf.conv_tol)
        if self.reference_residual > max(1e-7, 3 * scf_tolerance):
            raise RuntimeError('ROKS orbital residual exceeds the reference convergence tolerance')
        state_tolerance = max(1e-7, td.conv_tol)
        vector = vector / np.linalg.norm(vector)
        if self.solver == 'dense':
            a = action(np.eye(ndim)).T
            if np.max(np.abs(a - a.T)) > 1e-9:
                raise RuntimeError('NTTDA matrix is not symmetric')
            av = a @ vector
            self.state_residual = float(np.linalg.norm(av - (vector @ av) * vector))
            energies, states = np.linalg.eigh(a)
            overlaps = states.T @ vector
            root = int(np.argmax(np.abs(overlaps)))
            if abs(overlaps[root]) < 0.99 or abs(energies[root] - td.e[self.state - 1]) > 1e-6:
                raise RuntimeError('selected NTTDA root does not match a converged dense eigenstate')
            x = states[:, root]
            gaps = energies[root] - energies
            gaps[root] = np.inf
            self.state_gap = float(np.min(np.abs(gaps)))
        else:
            # The direct integral action uses the same native channel algebra.
            # Reuse its factored XC kernel instead of reevaluating LibXC in
            # every state-response iteration. Only the preconditioner diagonal
            # is needed from the original AO/grid action.
            _, state_diagonal = {1: td.gen_vind_sfu, 0: td.gen_vind_sc, -1: td.gen_vind_sfd}[td.deltaS]()
            x = vector
            ax = action(x[None])[0]
            omega = x @ ax
            self.state_residual = float(np.linalg.norm(ax - omega * x))
            if self.state_residual > state_tolerance or abs(omega - td.e[self.state - 1]) > state_tolerance:
                raise RuntimeError('selected NTTDA root exceeds the TD convergence tolerance')
            # Only already computed roots are available; this is not a full-spectrum gap.
            other_energies = np.delete(td.e, self.state - 1)
            if td.deltaS == -1:
                other_energies = np.append(other_energies, 0.)
            self.state_gap = float(np.min(abs(omega - other_energies))) if len(other_energies) else None

            def projected_action(v):
                perpendicular = v - x * (x @ v)
                out = action(perpendicular[None])[0] - omega * perpendicular
                return out - x * (x @ out) + x * (x @ v)

        log.info('Input SCF residual %.3g; input TD residual %.3g',
                 self.reference_residual, self.state_residual)
        if self.state_gap is not None and self.state_gap < 1e-7:
            raise NotImplementedError('an isolated NTTDA state is required (gap must exceed 1e-7 Eh)')

        # R_q and E_q in the orthonormal MO chart C(q,R)=C0 S_m(R)^(-1/2) exp(K(q)).
        identity = np.eye(nao)
        zeros_h = np.zeros_like(hm)
        if self.solver == 'dense':
            jacobian = np.zeros((nrot, nrot))
            energy_q = np.zeros(nrot)
            for index in range(nrot):
                k = _rotation(np.eye(1, nrot, index).ravel(), pairs, nao)
                hq = _first_transform(hm, zeros_h, identity, k)
                gq = gm.derivative((None, 0, c @ k))
                xcq = None if xc_grid is None else xc_grid.terms((None, 0, c @ k))
                eq, rq, aq = _mo_operators(td, hq, gq, pairs, xcq)
                jacobian[:, index] = rq
                energy_q[index] = eq + x @ aq(x[None])[0]
            if nrot and np.linalg.cond(jacobian) > 1e12:
                raise RuntimeError('ROKS orbital response is singular or ill-conditioned')
            def forward(v):
                return jacobian @ v

            def transpose(v):
                return jacobian.T @ v
            z = np.linalg.solve(jacobian.T, energy_q)
        else:
            forward, transpose, orbital_diagonal = orbital_actions(td, pairs)
            # Reuse the gradient's orbital derivative, not its Z-vector solve.
            xy = (x.reshape(td.xy[self.state - 1][0].shape), 0)
            m = td.Gradients()._analytic_components(xy, (), with_response=False)
            energy_q = pack_m_matrix(m, pairs) + 2 * residual
            z, _, cycles = solve_response(transpose, energy_q, orbital_diagonal, self.conv_tol, self.max_cycle)
            self.response_iterations['adjoint'] += cycles

        # Analytic skeleton derivatives at the single reference geometry.
        grad = mf.nuc_grad_method()
        h1gen = grad.hcore_generator(mol)
        h2gen = rhf_hess.Hessian(mf).hcore_generator(mol)
        saa, sab, s1 = rhf_hess.get_ovlp(mol)
        masks = {}
        for atom in atoms:
            p0, p1 = mol.aoslice_by_atom()[atom, 2:]
            masks[atom] = (np.arange(nao) >= p0) & (np.arange(nao) < p1)
        first = []
        a1x = []
        state_response = []
        response_errors = [np.linalg.norm(transpose(z) - energy_q)]
        for atom in atoms:
            ha = h1gen(atom)
            mask = masks[atom]
            for xyz in range(3):
                sa = c.T @ (s1[xyz] * mask[:, None] + s1[xyz].T * mask[None, :]) @ c
                ca_metric = -0.5 * c @ sa
                hsk = _first_transform(h, ha[xyz], c, ca_metric)
                gsk = gm.derivative((atom, xyz, ca_metric))
                xc_sk = None
                if xc_grid is not None:
                    xc_sk = xc_grid.terms((atom, xyz, ca_metric))
                _, rsk, _ = _mo_operators(td, hsk, gsk, pairs, xc_sk)
                if self.solver == 'dense':
                    q = np.linalg.solve(jacobian, -rsk)
                else:
                    q, _, cycles = solve_response(forward, -rsk, orbital_diagonal, self.conv_tol, self.max_cycle)
                    self.response_iterations['orbital'] += cycles
                response_errors.append(np.linalg.norm(forward(q) + rsk))
                k = _rotation(q, pairs, nao)
                ca = c @ (k - 0.5 * sa)
                ht = hsk + _first_transform(hm, zeros_h, identity, k)
                gt = gm.derivative((atom, xyz, ca))
                xc_direction = None if xc_grid is None else (atom, xyz, ca)
                xc_a = None if xc_grid is None else xc_grid.terms(xc_direction)
                _, _, at = _mo_operators(td, ht, gt, pairs, xc_a)
                ax = at(x[None])[0]
                if self.solver == 'dense':
                    xa = states @ ((states.T @ ax) / gaps)
                else:
                    rhs = -(ax - x * (x @ ax))
                    xa, error, cycles = solve_response(
                        projected_action, rhs, state_diagonal - omega + x*x,
                        self.conv_tol, self.max_cycle,
                    )
                    xa -= x * (x @ xa)
                    response_errors.append(error)
                    self.response_iterations['state'] += cycles
                a1x.append(ax)
                state_response.append(xa)
                first.append((ha[xyz], sa, k, ca, xc_direction))
        self.response_residual = float(max(response_errors))
        tolerance = self.conv_tol if self.solver == 'iterative' else 1e-8
        if self.response_residual > tolerance:
            raise RuntimeError('NTTDA Hessian response solve did not converge')
        log.info('Response residual %.3g; available-state gap %s Eh', self.response_residual, self.state_gap)
        log.info('GMRES iterations: %s', self.response_iterations)

        result = np.zeros((ncoord, ncoord))
        for ia, atom_a in enumerate(atoms):
            for ib in range(ia + 1):
                atom_b = atoms[ib]
                # Retain all Cartesian components of contracted J/K matrices
                # only for this atom pair, never four-index integral tensors.
                gm.clear_potential_cache()
                hab = h2gen(atom_a, atom_b)
                for xyz_a, xyz_b in itertools.product(range(3), repeat=2):
                    index_a, index_b = 3 * ia + xyz_a, 3 * ib + xyz_b
                    if index_a < index_b:
                        continue
                    ha, sa, ka, ca, xc_a_direction = first[index_a]
                    hb, sb, kb, cb, xc_b_direction = first[index_b]
                    sab_mo = c.T @ _overlap_second(saa, sab, masks[atom_a], masks[atom_b], xyz_a, xyz_b) @ c
                    # Mixed derivative of S_m^(-1/2) exp(K), with q_AB set to zero.
                    uab = (-0.5 * sab_mo + 0.375 * (sa @ sb + sb @ sa)
                           - 0.5 * (sa @ kb + sb @ ka) + 0.5 * (ka @ kb + kb @ ka))
                    cab = c @ uab
                    htotal = _second_transform(h, ha, hb, hab[xyz_a, xyz_b], c, ca, cb, cab)
                    gtotal = gm.derivative((atom_a, xyz_a, ca), (atom_b, xyz_b, cb), cab)
                    xc_ab = None
                    if xc_grid is not None:
                        xc_ab = xc_grid.terms(xc_a_direction, xc_b_direction, cab)
                    eab, rab, aab = _mo_operators(td, htotal, gtotal, pairs, xc_ab)
                    value = (eab + x @ aab(x[None])[0] - z @ rab
                             + state_response[index_a] @ a1x[index_b]
                             + state_response[index_b] @ a1x[index_a])
                    result[index_a, index_b] = result[index_b, index_a] = value
            log.info('NTTDA Hessian atom %d/%d complete', ia + 1, len(atoms))
        self.de = result.reshape(len(atoms), 3, len(atoms), 3).transpose(0, 2, 1, 3)
        self.de += rhf_hess.hess_nuc(mol, atmlst=list(atoms))
        return self.de


__all__ = ['Hessian']
