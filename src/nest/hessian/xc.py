"""Fixed-grid XC derivatives with blocked quadrature and contracted kernels.

Only orbital values and already contracted density derivatives are retained.
No grid-by-orbital-pair array or four-index MO kernel is constructed.
"""

import numpy as np
from pyscf import lib


_AO_INDEX = {
    (x, y, order - x - y): index
    for index, (order, x, y) in enumerate(
        (order, x, y) for order in range(4)
        for x in range(order, -1, -1) for y in range(order - x, -1, -1)
    )
}
# Each density feature is a short sum of left[a,i] * right[b,j].
_FEATURES = (((0, 0, 1.),), ((1, 0, 1.), (0, 1, 1.)),
             ((2, 0, 1.), (0, 2, 1.)), ((3, 0, 1.), (0, 3, 1.)),
             ((1, 1, .5), (2, 2, .5), (3, 3, .5)))


def _density(dm, factors, nvar):
    """Contract directed density matrices before forming grid features."""
    result = 0
    for left, right in factors:
        product = np.matmul(left, np.asarray(dm)[..., None, :, :])
        fields = [sum(weight * np.einsum('...gi,gi->...g', product[..., a, :, :], right[b])
                      for a, b, weight in feature) for feature in _FEATURES[:nvar]]
        result = result + np.stack(fields, axis=-2)
    return result


def _potential(weights, factors):
    """Contract weighted density features into (..., nmo, nmo)."""
    result = 0
    for left, right in factors:
        for x, feature in enumerate(_FEATURES[:weights.shape[-2]]):
            for a, b, weight in feature:
                result = result + np.matmul(
                    left[a].T, weight * weights[..., x, :, None] * right[b])
    return result


class _Kernel:
    """Quadrature factors for K_ijkl D_kl and K_ijkl D_jl, without K_ijkl.

    Blocks retain O(G*N) orbital data and O(G*nvar**2) coefficients, shared
    between product-rule terms. All higher XC tensors are block-local.
    """

    def __init__(self, nmo):
        self.nmo = nmo
        self.blocks = []

    def apply(self, dm, exchange=False):
        dm = np.asarray(dm)
        shape = dm.shape
        matrices = dm.reshape(-1, self.nmo, self.nmo)
        result = np.zeros_like(matrices)
        # Bound temporary memory even for the explicit solver's many probes.
        for start in range(0, len(matrices), 8):
            batch = matrices[start:start + 8]
            out = result[start:start + 8]
            for terms in self.blocks:
                for coefficient, left_factors, right_factors in terms:
                    nvar = len(coefficient)
                    if not exchange:
                        rho = _density(batch, right_factors, nvar)
                        weights = np.einsum('xyg,syg->sxg', coefficient, rho)
                        out += _potential(weights, left_factors)
                        continue
                    # K_ijkl D_jl: contract the two inner orbitals with D;
                    # only then reconstruct the exposed (i,k) matrix.
                    for left, inner_left in left_factors:
                        product = np.matmul(inner_left, batch[:, None])
                        for right, inner_right in right_factors:
                            field = np.einsum('sbgi,dgi->sbdg', product, inner_right)
                            weights = np.zeros((len(batch), len(left), len(right), field.shape[-1]))
                            for x, fx in enumerate(_FEATURES[:nvar]):
                                for y, fy in enumerate(_FEATURES[:nvar]):
                                    for a, b, wx in fx:
                                        for c, d, wy in fy:
                                            weights[:, a, c] += wx * wy * coefficient[x, y] * field[:, b, d]
                            for a in range(len(left)):
                                for c in range(len(right)):
                                    out += np.matmul(left[a].T, weights[:, a, c, :, None] * right[c])
        return result.reshape(shape)


class Semilocal:
    """Analytic value/first/mixed-second XC terms at fixed quadrature.

    A direction is (atom, xyz, C_A), with atom=None for a pure MO variation.
    C_A and C_AB are AO-to-MO coefficient derivatives, including the metric.
    AO-center motion is evaluated only within the current quadrature block.
    """

    def __init__(self, mf):
        self.mf = mf
        self.xctype = mf._numint._xc_type(mf.xc)
        mf._numint.libxc.test_deriv_order(mf.xc, 4, raise_error=True)
        self.nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[self.xctype]
        self.occ = np.asarray((mf.mo_occ > 0, mf.mo_occ == 2), dtype=float)
        self.masks = []
        for p0, p1 in mf.mol.aoslice_by_atom()[:, 2:]:
            self.masks.append((np.arange(mf.mol.nao_nr()) >= p0) & (np.arange(mf.mol.nao_nr()) < p1))
        # Reserve room for LibXC's expanded fourth derivatives and conversion
        # temporaries; compact MO factors survive the block, raw lxc does not.
        available = max(1., mf.max_memory - lib.current_memory()[0])
        per_point = 8 * (3 * (2*self.nvar)**4 + 100 * mf.mo_coeff.shape[1])
        self.block_size = max(1, min(1024, int(available * 1e6 * .2 / per_point)))

    def _ao_derivative(self, ao, atom, xyz, second=None, other=None):
        mask = self.masks[atom]
        if second is not None:
            mask = mask * self.masks[other]
        indices = []
        for component in range(1 if self.nvar == 1 else 4):
            powers = [0, 0, 0]
            if component:
                powers[component - 1] += 1
            powers[xyz] += 1
            if second is not None:
                powers[second] += 1
            indices.append(_AO_INDEX[tuple(powers)])
        return ao[indices] * mask * (-1 if second is None else 1)

    def terms(self, a=None, b=None, ab=None):
        mf, nvar = self.mf, self.nvar
        c = mf.mo_coeff
        nmo = c.shape[1]
        energy, potential, common = 0., np.zeros((2, nmo, nmo)), np.zeros((nmo, nmo))
        kernel = _Kernel(nmo)
        ni = mf._numint
        order = 2 if b is not None else 1 if a is not None else 0
        dm_occ = np.array([np.diag(occ) for occ in self.occ])
        for start in range(0, len(mf.grids.weights), self.block_size):
            end = start + self.block_size
            w = mf.grids.weights[start:end]
            ao = ni.eval_ao(mf.mol, mf.grids.coords[start:end], deriv=2 if nvar == 1 else 3)
            ao0 = ao[:1 if nvar == 1 else 4]
            phi = ao0 @ c
            p = ((phi, phi),)
            rho = _density(dm_occ, p, nvar)
            actual = ni.eval_xc_eff(mf.xc, rho, deriv=1+order, xctype=self.xctype, spin=1)
            v = actual[1].reshape(2*nvar, -1)
            equal = np.repeat((.5*rho.sum(axis=0))[None], 2, axis=0)
            ref = ni.eval_xc_eff(mf.xc, equal, deriv=2+order, xctype=self.xctype, spin=1)
            spin = np.array([1., -1.])
            fref = .5 * np.einsum('a,b,axbyg->xyg', spin, spin, ref[2])
            cv = .5 * ref[1].sum(axis=0)
            if order == 0:
                energy += np.dot(w, actual[0] * rho[:, 0].sum(axis=0))
                potentials, kernels, commons = ((v, p),), ((fref, p, p),), ((cv, p),)
            else:
                atom_a, xyz_a, ca = a
                aod_a = 0 if atom_a is None else self._ao_derivative(ao, atom_a, xyz_a)
                phi_a = ao0 @ ca if atom_a is None else aod_a @ c + ao0 @ ca
                pa = ((phi_a, phi), (phi, phi_a))
                ra = _density(dm_occ, pa, nvar).reshape(2*nvar, -1)
                ta = ra.reshape(2, nvar, -1).sum(axis=0)
                f = actual[2].reshape(2*nvar, 2*nvar, -1)
                kref = .25 * np.einsum('a,b,axbyczg->xyzg', spin, spin, ref[3])
                cf = .25 * ref[2].sum(axis=(0, 2))
                va = np.einsum('stg,tg->sg', f, ra)
                fa = np.einsum('xyzg,zg->xyg', kref, ta)
                cva = np.einsum('xyg,yg->xg', cf, ta)
                if order == 1:
                    energy += np.einsum('g,sg,sg->', w, v, ra)
                    potentials = ((va, p), (v, pa))
                    kernels = ((fa, p, p), (fref, pa, p), (fref, p, pa))
                    commons = ((cva, p), (cv, pa))
                else:
                    atom_b, xyz_b, cb = b
                    aod_b = 0 if atom_b is None else self._ao_derivative(ao, atom_b, xyz_b)
                    phi_b = ao0 @ cb if atom_b is None else aod_b @ c + ao0 @ cb
                    phi_ab = ao0 @ ab
                    if atom_a is not None:
                        phi_ab += aod_a @ cb
                    if atom_b is not None:
                        phi_ab += aod_b @ ca
                    if atom_a is not None and atom_b is not None:
                        phi_ab += self._ao_derivative(ao, atom_a, xyz_a, xyz_b, atom_b) @ c
                    pb = ((phi_b, phi), (phi, phi_b))
                    pab = ((phi_ab, phi), (phi, phi_ab), (phi_a, phi_b), (phi_b, phi_a))
                    rb = _density(dm_occ, pb, nvar).reshape(2*nvar, -1)
                    rab = _density(dm_occ, pab, nvar).reshape(2*nvar, -1)
                    tb = rb.reshape(2, nvar, -1).sum(axis=0)
                    tab = rab.reshape(2, nvar, -1).sum(axis=0)
                    k = actual[3].reshape(2*nvar, 2*nvar, 2*nvar, -1)
                    lref = .125 * np.einsum('a,b,axbyczdwg->xyzwg', spin, spin, ref[4])
                    ck = .125 * ref[3].sum(axis=(0, 2, 4))
                    vb = np.einsum('stg,tg->sg', f, rb)
                    vab = np.einsum('stg,tg->sg', f, rab) + np.einsum('stug,tg,ug->sg', k, ra, rb)
                    fb = np.einsum('xyzg,zg->xyg', kref, tb)
                    fab = np.einsum('xyzg,zg->xyg', kref, tab) + np.einsum('xyzwg,zg,wg->xyg', lref, ta, tb)
                    cvb = np.einsum('xyg,yg->xg', cf, tb)
                    cvab = np.einsum('xyg,yg->xg', cf, tab) + np.einsum('xyzg,yg,zg->xg', ck, ta, tb)
                    energy += np.einsum('g,sg,sg->', w, v, rab) + np.einsum('g,stg,sg,tg->', w, f, ra, rb)
                    potentials = ((vab, p), (va, pb), (vb, pa), (v, pab))
                    kernels = ((fab, p, p), (fa, pb, p), (fa, p, pb), (fb, pa, p), (fb, p, pa),
                               (fref, pab, p), (fref, p, pab), (fref, pa, pb), (fref, pb, pa))
                    commons = ((cvab, p), (cva, pb), (cvb, pa), (cv, pab))
            for value, factors in potentials:
                potential += _potential(w * value.reshape(2, nvar, -1), factors)
            for value, factors in commons:
                common += _potential(w * value, factors)
            kernel.blocks.append(tuple((w * value, left, right) for value, left, right in kernels))
            # Do not retain LibXC's uncontracted high-order tensors between blocks.
            del actual, ref
        return energy, potential, kernel, common
