"""Analytic NTTDA Hessian with iterative responses and dense integral tensors.

The integral derivatives are analytic libcint derivatives.  Orbital and state
responses are solved at one geometry; no displaced SCF/gradient is used here.
See ``DERIVATION.md`` for the constrained second-derivative formula.
"""

import copy
import itertools

import numpy as np
from pyscf import dft, lib
from pyscf.hessian import rhf as rhf_hess
from pyscf.lib import logger

from nest.grad.nttda.roks import canonical_pairs, pack_m_matrix
from .response import orbital_actions, solve as solve_response


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


def _second_mo_transform(tensor, ta, tb, tab, ua, ub, uab):
    """Product rule after transforming each skeleton tensor to MOs once.

    C_a=C u_a and C_ab=C u_ab. Only differentiated MO slots need another
    contraction; repeatedly transforming the unchanged slots costs 4x more
    for a four-index tensor. The dense solver retains the AO expression as
    an independent oracle.
    """
    result = tab.copy()
    for slot in range(tensor.ndim):
        result += np.moveaxis(np.tensordot(ta, ub, axes=(slot, 0)), -1, slot)
        result += np.moveaxis(np.tensordot(tb, ua, axes=(slot, 0)), -1, slot)
        result += np.moveaxis(np.tensordot(tensor, uab, axes=(slot, 0)), -1, slot)
        first = np.moveaxis(np.tensordot(tensor, ua, axes=(slot, 0)), -1, slot)
        for other in range(tensor.ndim):
            if other != slot:
                result += np.moveaxis(np.tensordot(first, ub, axes=(other, 0)), -1, other)
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
    Optional XC values/derivatives supply the nonlinear semilocal contributions.
    """
    occ = np.asarray(tdobj._scf.mo_occ)
    da = np.diag((occ > 0).astype(float))
    db = np.diag((occ == 2).astype(float))

    hybrid = tdobj._scf._numint.rsh_and_hybrid_coeff(tdobj._scf.xc, tdobj.mol.spin)[2]
    j = np.einsum('pqrs,sr->pq', eri, da + db)
    fa = h + j - hybrid * np.einsum('prsq,rs->pq', eri, da)
    fb = h + j - hybrid * np.einsum('prsq,rs->pq', eri, db)
    energy = 0.5 * (np.einsum('pq,pq', da, h + fa) + np.einsum('pq,pq', db, h + fb))
    response_eri = hybrid * eri
    if xc is not None:
        energy_xc, potential, kernel, common = xc
        energy += energy_xc
        fa = fa + potential[0]
        fb = fb + potential[1]
        # -K(response_eri) is K_ref, while -J(response_eri) is its
        # recoupled contraction.  The semilocal tensor is (pq|rs)_kernel.
        response_eri = response_eri - kernel.transpose(0, 2, 3, 1)
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
        return np.einsum('pqrs,...sr->...pq', response_eri, dm, optimize=True)

    def get_k(mol, dm, hermi=0, **kwargs):
        return np.einsum('prsq,...rs->...pq', response_eri, dm, optimize=True)

    mf = copy.copy(tdobj._scf)
    # All semilocal terms have already been supplied as MO tensors.
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


# Eight chemists' ERI symmetries, used to place differentiated AO slots.
_ERI_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _eri_first(ip1, mask, xyz):
    result = np.zeros_like(ip1[0])
    for slot in range(4):
        permutation = next(p for p in _ERI_PERMUTATIONS if p[slot] == 0)
        shape = [1] * 4
        shape[slot] = len(mask)
        # Nuclear displacement is minus the AO electronic-coordinate derivative.
        result -= ip1[xyz].transpose(permutation) * mask.reshape(shape)
    return result


def _eri_second(ipip1, ipvip1, ip1ip2, mask_a, mask_b, xyz_a, xyz_b):
    result = np.zeros_like(ipip1[0, 0])
    for first, second in itertools.product(range(4), repeat=2):
        if first == second:
            primitive = ipip1
            permutation = next(p for p in _ERI_PERMUTATIONS if p[first] == 0)
        else:
            partner = 1 if first // 2 == second // 2 else 2
            primitive = ipvip1 if partner == 1 else ip1ip2
            permutation = next(
                p for p in _ERI_PERMUTATIONS if p[first] == 0 and p[second] == partner
            )
        shape_a = [1] * 4
        shape_b = [1] * 4
        shape_a[first] = len(mask_a)
        shape_b[second] = len(mask_b)
        result += (primitive[xyz_a, xyz_b].transpose(permutation)
                   * mask_a.reshape(shape_a) * mask_b.reshape(shape_b))
    return result


def _overlap_second(saa, sab, mask_a, mask_b, xyz_a, xyz_b):
    same = mask_a * mask_b
    aa = saa[xyz_a, xyz_b]
    ab = sab[xyz_a, xyz_b]
    return (aa * same[:, None] + aa.T * same[None, :]
            + ab * mask_a[:, None] * mask_b[None, :]
            + ab.T * mask_b[:, None] * mask_a[None, :])


class Hessian(lib.StreamObject):
    """Small-system analytic HF / fixed-grid DFT total-energy Hessian.

    ``kernel(state=1)`` uses one-based NTTDA roots and returns
    ``(natm, natm, 3, 3)`` in Eh/Bohr**2.  ``atmlst`` selects both atom axes.
    Only isolated states, real full-rank MOs, and conventional all-electron
    integrals are supported. DFT requires analytic XC fourth derivatives and
    holds grid coordinates and weights fixed. Responses default to matrix-free
    GMRES, but dense integral storage still scales steeply with system size.
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
        if td.xy is None:
            td.run()
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
        # AO derivative buffers, stored first derivatives, and dense response matrices.
        response_size = 5 * ndim**2 + 3 * nrot**2 if self.solver == 'dense' else 45 * (ndim + nrot)
        required_mb = 8e-6 * ((40 + ncoord) * nao**4 + response_size)
        if xctype != 'HF':
            nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
            required_mb += 8e-6 * len(mf.grids.weights) * (
                4 * (20 + ncoord) * nao + 5 * nvar * nao**2 + 3 * (2 * nvar)**4
            )
        if required_mb > self.max_memory - lib.current_memory()[0]:
            raise MemoryError('dense NTTDA Hessian needs approximately %.0f MB of additional memory' % required_mb)
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
        eri = mol.intor('int2e', aosym='s1')
        hm = c.T @ h @ c
        gm = _transform(eri, [c] * 4)
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
            # The MO-integral action above is the same native channel algebra.
            # Reuse its cached XC kernel instead of reintegrating the grid in
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
        zeros_g = np.zeros_like(gm)
        if self.solver == 'dense':
            jacobian = np.zeros((nrot, nrot))
            energy_q = np.zeros(nrot)
            for index in range(nrot):
                k = _rotation(np.eye(1, nrot, index).ravel(), pairs, nao)
                hq = _first_transform(hm, zeros_h, identity, k)
                gq = _first_transform(gm, zeros_g, identity, k)
                xcq = None if xc_grid is None else xc_grid.terms(xc_grid.phi @ k)
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
        ip1 = mol.intor('int2e_ip1', comp=3, aosym='s1')
        ipip1 = mol.intor('int2e_ipip1', comp=9, aosym='s1').reshape(3, 3, *eri.shape)
        ipvip1 = mol.intor('int2e_ipvip1', comp=9, aosym='s1').reshape(3, 3, *eri.shape)
        ip1ip2 = mol.intor('int2e_ip1ip2', comp=9, aosym='s1').reshape(3, 3, *eri.shape)
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
                ga = _eri_first(ip1, mask, xyz)
                ca_metric = -0.5 * c @ sa
                hsk = _first_transform(h, ha[xyz], c, ca_metric)
                gsk = _first_transform(eri, ga, c, ca_metric)
                phi_sk = None
                xc_sk = None
                if xc_grid is not None:
                    phi_sk = xc_grid.ao_derivative(mask, xyz) @ c + xc_grid.ao0 @ ca_metric
                    xc_sk = xc_grid.terms(phi_sk)
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
                gt = gsk + _first_transform(gm, zeros_g, identity, k)
                phi_a = None if xc_grid is None else phi_sk + xc_grid.phi @ k
                xc_a = None if xc_grid is None else xc_grid.terms(phi_a)
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
                # The iterative path reuses MO skeleton derivatives in every
                # mixed derivative instead of repeating four AO transforms.
                stored_ga = ga if self.solver == 'dense' else _transform(ga, [c] * 4)
                first.append((ha[xyz], stored_ga, sa, k, ca, phi_a))
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
                hab = h2gen(atom_a, atom_b)
                for xyz_a, xyz_b in itertools.product(range(3), repeat=2):
                    index_a, index_b = 3 * ia + xyz_a, 3 * ib + xyz_b
                    if index_a < index_b:
                        continue
                    ha, ga, sa, ka, ca, phi_a = first[index_a]
                    hb, gb, sb, kb, cb, phi_b = first[index_b]
                    sab_mo = c.T @ _overlap_second(saa, sab, masks[atom_a], masks[atom_b], xyz_a, xyz_b) @ c
                    # Mixed derivative of S_m^(-1/2) exp(K), with q_AB set to zero.
                    uab = (-0.5 * sab_mo + 0.375 * (sa @ sb + sb @ sa)
                           - 0.5 * (sa @ kb + sb @ ka) + 0.5 * (ka @ kb + kb @ ka))
                    cab = c @ uab
                    gab = _eri_second(ipip1, ipvip1, ip1ip2, masks[atom_a], masks[atom_b], xyz_a, xyz_b)
                    htotal = _second_transform(h, ha, hb, hab[xyz_a, xyz_b], c, ca, cb, cab)
                    if self.solver == 'dense':
                        gtotal = _second_transform(eri, ga, gb, gab, c, ca, cb, cab)
                    else:
                        gtotal = _second_mo_transform(
                            gm, ga, gb, _transform(gab, [c] * 4),
                            ka - 0.5*sa, kb - 0.5*sb, uab,
                        )
                    xc_ab = None
                    if xc_grid is not None:
                        phi_ab = (xc_grid.ao_derivative(masks[atom_a] * masks[atom_b], xyz_a, xyz_b) @ c
                                  + xc_grid.ao_derivative(masks[atom_a], xyz_a) @ cb
                                  + xc_grid.ao_derivative(masks[atom_b], xyz_b) @ ca
                                  + xc_grid.ao0 @ cab)
                        xc_ab = xc_grid.terms(phi_a, phi_b, phi_ab)
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
