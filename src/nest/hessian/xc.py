"""Fixed-grid semilocal XC contributions through analytic mixed second order.

Grid orbitals are (1 or 4, ngrids, nmo). Pair features are (nvar, ngrids,
nmo, nmo), with nvar=1 for LDA, 4 for GGA, 5 for tau-dependent MGGA.
"""

import numpy as np


# PySCF AO derivative order through cubic: 0,x,y,z,xx,xy,xz,yy,yz,zz,...
_AO_INDEX = {
    (x, y, order - x - y): index
    for index, (order, x, y) in enumerate(
        (order, x, y) for order in range(4)
        for x in range(order, -1, -1) for y in range(order - x, -1, -1)
    )
}


class Semilocal:
    """Cache XC density derivatives at one geometry and apply the chain rule."""

    def __init__(self, mf):
        ni = mf._numint
        xctype = ni._xc_type(mf.xc)
        ni.libxc.test_deriv_order(mf.xc, 4, raise_error=True)
        self.nvar = nvar = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
        self.weights = mf.grids.weights
        self.ao = ni.eval_ao(mf.mol, mf.grids.coords, deriv=2 if nvar == 1 else 3)
        self.ao0 = self.ao[:1 if nvar == 1 else 4]
        self.phi = self.ao0 @ mf.mo_coeff
        self.pairs = self._pair(self.phi, self.phi)
        self.occ = np.asarray((mf.mo_occ > 0, mf.mo_occ == 2), dtype=float)
        rho = np.einsum('si,xgii->sxg', self.occ, self.pairs)
        exc, v, f, k = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype, spin=1)
        self.energy = np.dot(self.weights, exc * rho[:, 0].sum(axis=0))
        self.v = v.reshape(2 * nvar, -1)
        self.f = f.reshape(2 * nvar, 2 * nvar, -1)
        self.k = k.reshape(2 * nvar, 2 * nvar, 2 * nvar, -1)
        equal = np.repeat((0.5 * rho.sum(axis=0))[None], 2, axis=0)
        _, v, f, k, l = ni.eval_xc_eff(mf.xc, equal, deriv=4, xctype=xctype, spin=1)
        spin = np.array([1., -1.])
        self.fref = 0.5 * np.einsum('a,b,axbyg->xyg', spin, spin, f)
        # Charge-feature derivatives: rho_equal_sx = rho_total_x / 2.
        self.kref = 0.25 * np.einsum('a,b,axbyczg->xyzg', spin, spin, k)
        self.lref = 0.125 * np.einsum('a,b,axbyczdwg->xyzwg', spin, spin, l)
        self.common = 0.5 * v.sum(axis=0)
        self.common_f = 0.25 * f.sum(axis=(0, 2))
        self.common_k = 0.125 * k.sum(axis=(0, 2, 4))

    def _pair(self, left, right):
        """Bilinear density/gradient/tau features, with tau = sum grad^2 / 2."""
        result = np.empty((self.nvar, *left.shape[1:], right.shape[-1]))
        result[0] = np.einsum('gi,gj->gij', left[0], right[0])
        if self.nvar > 1:
            result[1:4] = (np.einsum('xgi,gj->xgij', left[1:4], right[0])
                           + np.einsum('gi,xgj->xgij', left[0], right[1:4]))
        if self.nvar == 5:
            result[4] = 0.5 * np.einsum('xgi,xgj->gij', left[1:4], right[1:4])
        return result

    def ao_derivative(self, mask, first, second=None):
        """One/two nuclear-center derivatives of AO values and AO gradients."""
        indices = []
        for component in range(len(self.ao0)):
            powers = [0, 0, 0]
            if component:
                powers[component - 1] += 1
            powers[first] += 1
            if second is not None:
                powers[second] += 1
            indices.append(_AO_INDEX[tuple(powers)])
        return self.ao[indices] * mask * (-1 if second is None else 1)

    def terms(self, phi_a=None, phi_b=None, phi_ab=None):
        """Return value, first derivative, or mixed derivative in the MO chart.

        phi_a/b include AO-center motion and reference MO response. phi_ab has
        second orbital response set to zero, as required by the adjoint formula.
        The tuple contains reference XC energy, two actual spin potentials,
        reconstructed kernel (ij,kl), and the equal-spin common potential.
        """
        p = self.pairs
        w = self.weights
        nvar = self.nvar
        if phi_a is None:
            energy = self.energy
            potentials = ((self.v, p),)
            kernels = ((self.fref, p, p),)
            common = ((self.common, p),)
        else:
            pa = self._pair(phi_a, self.phi)
            pa = pa + pa.transpose(0, 1, 3, 2)
            ra = np.einsum('si,xgii->sxg', self.occ, pa).reshape(2 * nvar, -1)
            ta = ra.reshape(2, nvar, -1).sum(axis=0)
            va = np.einsum('stg,tg->sg', self.f, ra)
            fa = np.einsum('xyzg,zg->xyg', self.kref, ta)
            ca = np.einsum('xyg,yg->xg', self.common_f, ta)
            if phi_b is None:
                energy = np.einsum('g,sg,sg->', w, self.v, ra)
                potentials = ((va, p), (self.v, pa))
                kernels = ((fa, p, p), (self.fref, pa, p), (self.fref, p, pa))
                common = ((ca, p), (self.common, pa))
            else:
                pb = self._pair(phi_b, self.phi)
                pb = pb + pb.transpose(0, 1, 3, 2)
                pab = self._pair(phi_ab, self.phi) + self._pair(phi_a, phi_b)
                pab = pab + pab.transpose(0, 1, 3, 2)
                rb = np.einsum('si,xgii->sxg', self.occ, pb).reshape(2 * nvar, -1)
                rab = np.einsum('si,xgii->sxg', self.occ, pab).reshape(2 * nvar, -1)
                tb = rb.reshape(2, nvar, -1).sum(axis=0)
                tab = rab.reshape(2, nvar, -1).sum(axis=0)
                vb = np.einsum('stg,tg->sg', self.f, rb)
                vab = (np.einsum('stg,tg->sg', self.f, rab)
                       + np.einsum('stug,tg,ug->sg', self.k, ra, rb))
                fb = np.einsum('xyzg,zg->xyg', self.kref, tb)
                fab = (np.einsum('xyzg,zg->xyg', self.kref, tab)
                       + np.einsum('xyzwg,zg,wg->xyg', self.lref, ta, tb))
                cb = np.einsum('xyg,yg->xg', self.common_f, tb)
                cab = (np.einsum('xyg,yg->xg', self.common_f, tab)
                       + np.einsum('xyzg,yg,zg->xg', self.common_k, ta, tb))
                energy = (np.einsum('g,sg,sg->', w, self.v, rab)
                          + np.einsum('g,stg,sg,tg->', w, self.f, ra, rb))
                potentials = ((vab, p), (va, pb), (vb, pa), (self.v, pab))
                kernels = ((fab, p, p), (fa, pb, p), (fa, p, pb), (fb, pa, p), (fb, p, pa),
                           (self.fref, pab, p), (self.fref, p, pab),
                           (self.fref, pa, pb), (self.fref, pb, pa))
                common = ((cab, p), (ca, pb), (cb, pa), (self.common, pab))
        nmo = self.phi.shape[-1]
        v = np.zeros((2, nmo, nmo))
        kernel = np.zeros((nmo,) * 4)
        v_common = np.zeros((nmo, nmo))
        for potential, pair in potentials:
            v += np.einsum('g,sxg,xgij->sij', w, potential.reshape(2, nvar, -1), pair, optimize=True)
        for coefficient, left, right in kernels:
            kernel += np.einsum('xyg,xgij,ygkl->ijkl', w * coefficient, left, right, optimize=True)
        for potential, pair in common:
            v_common += np.einsum('xg,xgij->ij', w * potential, pair, optimize=True)
        return energy, v, kernel, v_common
