"""Dense analytic NTTDA Hessian for an all-electron ROKS(xc='HF') reference.

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

from nest.grad.nttda.roks import canonical_pairs


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


def _mo_operators(tdobj, h, eri, pairs):
    """Linear maps from MO integrals to E_ref, the ROKS residual, and A.

    Calling the existing HF NTTDA action in an identity MO basis preserves all
    channel coefficients and amplitude layouts.  This also works for integral
    derivatives because all three outputs are linear in h and (pq|rs).
    """
    occ = np.asarray(tdobj._scf.mo_occ)
    da = np.diag((occ > 0).astype(float))
    db = np.diag((occ == 2).astype(float))

    def get_j(mol, dm, hermi=0, **kwargs):
        return np.einsum('pqrs,...sr->...pq', eri, dm, optimize=True)

    def get_k(mol, dm, hermi=0, **kwargs):
        return np.einsum('prsq,...rs->...pq', eri, dm, optimize=True)

    j = get_j(None, da + db)
    fa = h + j - get_k(None, da)
    fb = h + j - get_k(None, db)
    energy = 0.5 * (np.einsum('pq,pq', da, h + fa) + np.einsum('pq,pq', db, h + fb))
    residual_focks = {'co': fb, 'cv': fa + fb, 'ov': fa}
    residual = np.asarray([
        residual_focks[name][p, q]
        for p, q, name in pairs
    ])
    mf = copy.copy(tdobj._scf)
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
    """Small-system, dense analytic HF NTTDA total-energy Hessian.

    ``kernel(state=1)`` uses one-based NTTDA roots and returns
    ``(natm, natm, 3, 3)`` in Eh/Bohr**2.  ``atmlst`` selects both atom axes.
    Only isolated states, real full-rank MOs, and conventional all-electron
    HF integrals are supported.  Memory and CPU scale steeply with system size.
    """

    _keys = {'base', 'mol', 'state', 'atmlst', 'de', 'response_residual', 'state_gap'}

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

    def kernel(self, state=None, atmlst=None):
        td = self.base
        mf = td._scf
        mol = self.mol
        if state is not None:
            self.state = state
        if atmlst is not None:
            self.atmlst = tuple(atmlst)
        atoms = tuple(range(mol.natm)) if self.atmlst is None else self.atmlst
        if (not isinstance(mf, dft.roks.ROKS) or mf._numint._xc_type(mf.xc) != 'HF'):
            raise NotImplementedError("NTTDA analytic Hessian currently requires ROKS(xc='HF')")
        if mf._numint.rsh_and_hybrid_coeff(mf.xc, mol.spin) != (0, 1, 1):
            raise NotImplementedError('NTTDA Hessian requires full-range 100% HF exchange')
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
        required_mb = 8e-6 * ((40 + ncoord) * nao**4 + 5 * ndim**2 + 3 * nrot**2)
        if required_mb > self.max_memory - lib.current_memory()[0]:
            raise MemoryError('dense NTTDA Hessian needs approximately %.0f MB of additional memory' % required_mb)
        log = logger.new_logger(self)
        log.info('Analytic HF NTTDA Hessian: state %d, %d rotations, %d amplitudes', self.state, nrot, ndim)
        h = mf.get_hcore()
        eri = mol.intor('int2e', aosym='s1')
        hm = c.T @ h @ c
        gm = _transform(eri, [c] * 4)
        _, residual, action = _mo_operators(td, hm, gm, pairs)
        if np.max(np.abs(residual), initial=0) > 1e-7:
            raise RuntimeError('ROKS orbital residual exceeds 1e-7; tighten SCF convergence')
        a = action(np.eye(ndim)).T
        if np.max(np.abs(a - a.T)) > 1e-9:
            raise RuntimeError('NTTDA matrix is not symmetric')
        energies, states = np.linalg.eigh(a)
        vector = vector / np.linalg.norm(vector)
        overlaps = states.T @ vector
        root = int(np.argmax(np.abs(overlaps)))
        if abs(overlaps[root]) < 0.99 or abs(energies[root] - td.e[self.state - 1]) > 1e-6:
            raise RuntimeError('selected NTTDA root does not match a converged dense eigenstate')
        x = states[:, root]
        gaps = energies[root] - energies
        gaps[root] = np.inf
        self.state_gap = float(np.min(np.abs(gaps)))
        if self.state_gap < 1e-7:
            raise NotImplementedError('an isolated NTTDA state is required (gap must exceed 1e-7 Eh)')

        # R_q and E_q in the orthonormal MO chart C(q,R)=C0 S_m(R)^(-1/2) exp(K(q)).
        identity = np.eye(nao)
        zeros_h = np.zeros_like(hm)
        zeros_g = np.zeros_like(gm)
        jacobian = np.zeros((nrot, nrot))
        energy_q = np.zeros(nrot)
        for index in range(nrot):
            k = _rotation(np.eye(1, nrot, index).ravel(), pairs, nao)
            hq = _first_transform(hm, zeros_h, identity, k)
            gq = _first_transform(gm, zeros_g, identity, k)
            eq, rq, aq = _mo_operators(td, hq, gq, pairs)
            jacobian[:, index] = rq
            energy_q[index] = eq + x @ aq(x[None])[0]
        if nrot and np.linalg.cond(jacobian) > 1e12:
            raise RuntimeError('ROKS orbital response is singular or ill-conditioned')
        z = np.linalg.solve(jacobian.T, energy_q)

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
        response_errors = [np.max(np.abs(jacobian.T @ z - energy_q), initial=0)]
        for atom in atoms:
            ha = h1gen(atom)
            mask = masks[atom]
            for xyz in range(3):
                sa = c.T @ (s1[xyz] * mask[:, None] + s1[xyz].T * mask[None, :]) @ c
                ga = _eri_first(ip1, mask, xyz)
                ca_metric = -0.5 * c @ sa
                hsk = _first_transform(h, ha[xyz], c, ca_metric)
                gsk = _first_transform(eri, ga, c, ca_metric)
                _, rsk, _ = _mo_operators(td, hsk, gsk, pairs)
                q = np.linalg.solve(jacobian, -rsk)
                response_errors.append(np.max(np.abs(jacobian @ q + rsk), initial=0))
                k = _rotation(q, pairs, nao)
                ca = c @ (k - 0.5 * sa)
                ht = hsk + _first_transform(hm, zeros_h, identity, k)
                gt = gsk + _first_transform(gm, zeros_g, identity, k)
                _, _, at = _mo_operators(td, ht, gt, pairs)
                ax = at(x[None])[0]
                xa = states @ ((states.T @ ax) / gaps)
                a1x.append(ax)
                state_response.append(xa)
                first.append((ha[xyz], ga, sa, k, ca))
        self.response_residual = float(max(response_errors))
        if self.response_residual > 1e-8:
            raise RuntimeError('NTTDA Hessian response solve did not converge')
        log.info('ROKS response residual %.3g; nearest NTTDA gap %.6g Eh', self.response_residual, self.state_gap)

        result = np.zeros((ncoord, ncoord))
        for ia, atom_a in enumerate(atoms):
            for ib in range(ia + 1):
                atom_b = atoms[ib]
                hab = h2gen(atom_a, atom_b)
                for xyz_a, xyz_b in itertools.product(range(3), repeat=2):
                    index_a, index_b = 3 * ia + xyz_a, 3 * ib + xyz_b
                    if index_a < index_b:
                        continue
                    ha, ga, sa, ka, ca = first[index_a]
                    hb, gb, sb, kb, cb = first[index_b]
                    sab_mo = c.T @ _overlap_second(saa, sab, masks[atom_a], masks[atom_b], xyz_a, xyz_b) @ c
                    # Mixed derivative of S_m^(-1/2) exp(K), with q_AB set to zero.
                    uab = (-0.5 * sab_mo + 0.375 * (sa @ sb + sb @ sa)
                           - 0.5 * (sa @ kb + sb @ ka) + 0.5 * (ka @ kb + kb @ ka))
                    cab = c @ uab
                    gab = _eri_second(ipip1, ipvip1, ip1ip2, masks[atom_a], masks[atom_b], xyz_a, xyz_b)
                    htotal = _second_transform(h, ha, hb, hab[xyz_a, xyz_b], c, ca, cb, cab)
                    gtotal = _second_transform(eri, ga, gb, gab, c, ca, cb, cab)
                    eab, rab, aab = _mo_operators(td, htotal, gtotal, pairs)
                    value = (eab + x @ aab(x[None])[0] - z @ rab
                             + state_response[index_a] @ a1x[index_b]
                             + state_response[index_b] @ a1x[index_a])
                    result[index_a, index_b] = result[index_b, index_a] = value
            log.info('NTTDA Hessian atom %d/%d complete', ia + 1, len(atoms))
        self.de = result.reshape(len(atoms), 3, len(atoms), 3).transpose(0, 2, 1, 3)
        self.de += rhf_hess.hess_nuc(mol, atmlst=list(atoms))
        return self.de


__all__ = ['Hessian']
