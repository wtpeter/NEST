# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

"""Direct MO J/K actions and their first/mixed second nuclear derivatives.

Differentiate the four AO-to-MO factors, then move them onto the input
density and output potential. Libcint derivative integrals are contracted
in PySCF's shell driver; no AO or MO four-index tensor is allocated here.
"""

import ctypes

import numpy as np
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from pyscf.scf import jk, _vhf


_ERI_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


class DirectERI:
    """J[D]_pq=(pq|rs)D_sr and K[D]_pq=(prsq)D_rs in a moving MO basis.

    Directions are (atom, xyz, C_A), as in xc.Semilocal. atom=None denotes
    an orbital-only variation. Terms store only four coefficient matrices,
    derivative component, and AO-center restrictions, never integral arrays.
    """

    def __init__(self, mol, c, a=None, b=None, ab=None, max_memory=4000, _root=None):
        self.mol, self.nmo, self.max_memory = mol, c.shape[1], max_memory
        self.c = c
        self._root = _root
        if _root is None:
            self.optimizers = {}
            self.potential_cache = {}
            self.cache_bytes = 0
            self.cache_limit = min(32e6, .05*max_memory*1e6)
        else:
            self.optimizers = _root.optimizers
        self.cache_potentials = b is not None
        self.groups = {}

        def add(coefficients, da=None, db=None):
            if any(not np.any(coefficient) for coefficient in coefficients):
                return
            if da is None:
                terms = [('int2e', 0, (0, 1, 2, 3), [None]*4, 1.)]
            elif da[0] is None or (db is not None and db[0] is None):
                return
            else:
                terms = []
                for first in range(4):
                    for second in (range(4) if db is not None else (None,)):
                        centers = [None]*4
                        centers[first] = da[0]
                        if second is None:
                            intor, component, sign = 'int2e_ip1', da[1], -1.
                            permutation = next(p for p in _ERI_PERMUTATIONS if p[first] == 0)
                        else:
                            if first == second and da[0] != db[0]:
                                continue
                            centers[second] = db[0]
                            component, sign = 3*da[1] + db[1], 1.
                            if first == second:
                                intor = 'int2e_ipip1'
                                permutation = next(p for p in _ERI_PERMUTATIONS if p[first] == 0)
                            else:
                                partner = 1 if first//2 == second//2 else 2
                                intor = 'int2e_ipvip1' if partner == 1 else 'int2e_ip1ip2'
                                permutation = next(p for p in _ERI_PERMUTATIONS
                                                   if p[first] == 0 and p[second] == partner)
                        terms.append((intor, component, permutation, centers, sign))
            for intor, component, permutation, centers, sign in terms:
                canonical_centers = tuple(centers[permutation.index(i)] for i in range(4))
                self.groups.setdefault((intor, canonical_centers), []).append(
                    (component, permutation, tuple(coefficients), sign))

        if a is None:
            add([c]*4)
        elif b is None:
            add([c]*4, a)
            for slot in range(4):
                coefficients = [c]*4
                coefficients[slot] = a[2]
                add(coefficients)
        else:
            add([c]*4, a, b)
            for slot in range(4):
                for direction, orbital in ((a, b[2]), (b, a[2]), (None, ab)):
                    coefficients = [c]*4
                    coefficients[slot] = orbital
                    add(coefficients, direction)
                for other in range(4):
                    if other != slot:
                        coefficients = [c]*4
                        coefficients[slot], coefficients[other] = a[2], b[2]
                        add(coefficients)

    def derivative(self, a, b=None, ab=None):
        """Reuse geometry-only integral optimizers and shell Schwarz bounds."""
        root = self if self._root is None else self._root
        return DirectERI(self.mol, self.c, a, b, ab, self.max_memory, root)

    def clear_potential_cache(self):
        """Release contracted matrices when moving to the next atom pair."""
        root = self if self._root is None else self._root
        root.potential_cache.clear()
        root.cache_bytes = 0

    def _optimizer(self, intor):
        if intor not in self.optimizers:
            mol = self.mol
            if intor == 'int2e':
                opt = _vhf._VHFOpt(mol, intor, 'CVHFnrs8_prescreen', 'CVHFnr_int2e_q_cond')
            else:
                key = intor.removeprefix('int2e_')
                prescreen = 'CVHFgrad_jk_prescreen' if key == 'ip1' else 'CVHF'+key+'_prescreen'
                opt = _vhf._VHFOpt(mol, intor, prescreen)
                if key in ('ip1', 'ip1ip2'):
                    opt.q_cond = rhf_grad._calc_q_cond(mol, opt)
                else:
                    # Same derivative Schwarz bounds as pyscf.hessian.rhf.
                    # Resolve the spherical/Cartesian libcint symbol explicitly.
                    libcvhf = _vhf.libcvhf
                    q_cond = np.empty((2, mol.nbas, mol.nbas))
                    ao_loc = mol.ao_loc_nr()
                    bound_intor = 'int2e_ipip1ipip2' if key == 'ipip1' else 'int2e_ipvip1ipvip2'
                    with mol.with_integral_screen(opt.direct_scf_tol**2):
                        for index, name, builder in (
                                (0, bound_intor, libcvhf.CVHFnr_int2e_pppp_q_cond),
                                (1, 'int2e', libcvhf.CVHFnr_int2e_q_cond)):
                            builder(getattr(libcvhf, mol._add_suffix(name)), lib.c_null_ptr(),
                                    q_cond[index].ctypes, ao_loc.ctypes, mol._atm.ctypes,
                                    ctypes.c_int(mol.natm), mol._bas.ctypes,
                                    ctypes.c_int(mol.nbas), mol._env.ctypes)
                    opt.q_cond = q_cond
            self.optimizers[intor] = opt
        return self.optimizers[intor]

    def apply(self, dm, exchange=False):
        dm = np.asarray(dm)
        matrices = dm.reshape(-1, self.nmo, self.nmo)
        result = np.zeros_like(matrices)
        # Original chemists' slots: J contracts (3,2), K contracts (1,2).
        inner, outer = ((1, 2), (0, 3)) if exchange else ((3, 2), (0, 1))
        mol = self.mol
        aoslices = mol.aoslice_by_atom()
        nao = mol.nao_nr()
        available = max(1., self.max_memory - lib.current_memory()[0])
        for (intor, centers), terms in self.groups.items():
            comp = 1 if intor == 'int2e' else 3 if intor == 'int2e_ip1' else 9
            # The shell driver also accumulates thread-local potential blocks.
            bytes_per_density = 8*(comp+2)*nao**2*(lib.num_threads()+1)
            limit = max(1, min(128, int(.1*available*1e6 / bytes_per_density)))
            shls, rows = [], []
            for atom in centers:
                sh0, sh1, p0, p1 = (0, mol.nbas, 0, nao) if atom is None else aoslices[atom]
                shls.extend((int(sh0), int(sh1)))
                rows.append(slice(p0, p1))
            densities, scripts, outputs = [], [], []
            opt = self._optimizer(intor)

            def contract():
                # PySCF's derivative prescreens assume particular J/K layouts.
                # A uniform upper bound is conservative for ALL directed slot
                # permutations, including restricted rectangular density blocks.
                # Keep dmcondname=None so direct_bindm does not reinterpret them
                # as full square AO densities when updating its screening data.
                root = self if self._root is None else self._root
                keys = [(intor, tuple(shls), script, density.tobytes())
                        for script, density in zip(scripts, densities)] if self.cache_potentials else []
                missing = [i for i in range(len(densities)) if not keys or keys[i] not in root.potential_cache]
                potentials = [None]*len(densities)
                if missing:
                    dm_max = max(np.max(np.abs(densities[i])) for i in missing)
                    opt.dm_cond = np.full((mol.nbas, mol.nbas), max(1., dm_max))
                    aosym = 's8' if intor == 'int2e' else 's1' if intor == 'int2e_ip1ip2' else 's2kl'
                    direct_dms, direct_scripts, transpose = [], [], []
                    for i in missing:
                        density, script = densities[i], scripts[i]
                        flip = False
                        if aosym == 's2kl':
                            # Only the undifferentiated (k,l) pair is symmetric.
                            # Relabel that pair, then orient D/output to one of
                            # the four scripts supported by PySCF's s2kl driver.
                            inner_labels, outer_labels = script[5:].split('->')
                            if set(inner_labels) in ({'i', 'k'}, {'j', 'l'}):
                                inner_labels = inner_labels.translate(str.maketrans('kl', 'lk'))
                                outer_labels = outer_labels.translate(str.maketrans('kl', 'lk'))
                            target = next((d, v) for d, v in (('ji', 'kl'), ('lk', 'ij'),
                                                             ('jk', 'il'), ('li', 'kj'))
                                          if set(d) == set(inner_labels))
                            if inner_labels != target[0]:
                                density = density.T
                            flip = outer_labels != target[1]
                            script = 'ijkl,%s->%s' % target
                        direct_dms.append(density)
                        direct_scripts.append(script)
                        transpose.append(flip)
                    values = jk.get_jk(mol, direct_dms, scripts=direct_scripts, intor=intor,
                                       comp=comp, aosym=aosym, hermi=0, shls_slice=tuple(shls), vhfopt=opt)
                    for i, value, flip in zip(missing, values, transpose):
                        if flip:
                            value = value.swapaxes(-1, -2)
                        potentials[i] = value
                        if keys and keys[i] not in root.potential_cache:
                            size = value.nbytes + len(keys[i][-1]) + 256
                            if root.cache_bytes + size <= root.cache_limit:
                                root.potential_cache[keys[i]] = value
                                root.cache_bytes += size
                for i, potential in enumerate(potentials):
                    if potential is None:
                        potentials[i] = root.potential_cache[keys[i]]
                for potential, (index, component, left, right, sign) in zip(potentials, outputs):
                    if comp > 1:
                        potential = potential[component]
                    result[index] += sign * (left.T @ potential @ right)
                densities.clear()
                scripts.clear()
                outputs.clear()

            for component, permutation, coefficients, sign in terms:
                cs = [coefficient[rows[permutation[i]]] for i, coefficient in enumerate(coefficients)]
                labels = ['ijkl'[p] for p in permutation]
                script = 'ijkl,%s%s->%s%s' % tuple(labels[i] for i in (*inner, *outer))
                for index, matrix in enumerate(matrices):
                    densities.append(cs[inner[0]] @ matrix @ cs[inner[1]].T)
                    scripts.append(script)
                    outputs.append((index, component, cs[outer[0]], cs[outer[1]], sign))
                    if len(densities) == limit:
                        contract()
            if densities:
                contract()
        return result.reshape(dm.shape)
