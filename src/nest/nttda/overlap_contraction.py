"""Exact cubic contractions for two-open-orbital NT-TDA AWF overlaps.

Integrate out the doubly occupied core by a Schur complement. Each AWF
determinant has at most one core hole and one virtual particle. The remaining
bordered determinant has dimension <=4. Its symbolic permutation expansion
becomes vectors/matrices contracted with root amplitudes by GEMM; no
determinant-pair matrix or rank-four orbital tensor is constructed.

NumPy is an independent CPU execution backend. The GPU backend uses CuPy dot
throughout and keeps orbitals, amplitudes and contractions on device. Only
the cross-geometry AO metric and small checks/outputs cross the host boundary.
"""

from functools import lru_cache
import itertools
import math

import numpy as np
from pyscf import gto
from nest.nttda.overlap import _configurations


class SingularCoreOverlapError(RuntimeError):
    """Core elimination is ill-conditioned; use the determinant backend."""


@lru_cache(maxsize=8)
def _compression_plan(occupations, delta_s):
    """Orbital/spin metadata only: map amplitudes to core/active/virtual sectors.

    Group (active bit pattern, hole spin or None, particle spin or None).
    Core-first determinant ordering and deleted-core cofactor signs are
    absorbed here. Cached plans contain no wavefunction values or CUDA data.
    """
    nmo = len(occupations)
    closed = [i for i, occ in enumerate(occupations) if occ == 2]
    opened = [i for i, occ in enumerate(occupations) if occ == 1]
    virtual = [i for i, occ in enumerate(occupations) if occ == 0]
    nc, nv = len(closed), len(virtual)
    core_mask = sum(1 << (i+s*nmo) for s in range(2) for i in closed)
    virtual_mask = sum(1 << (i+s*nmo) for s in range(2) for i in virtual)
    hole_index = {i+s*nmo: (s, k) for s in range(2) for k, i in enumerate(closed)}
    particle_index = {i+s*nmo: (s, k) for s in range(2) for k, i in enumerate(virtual)}
    contiguous = closed+opened+virtual == list(range(nmo))
    if not contiguous:
        core_order = [i+s*nmo for s in range(2) for i in closed]
        extra_order = [i+s*nmo for orbitals in (opened, virtual) for s in range(2) for i in orbitals]
        ranks = {orbital: index for index, orbital in enumerate(core_order+extra_order)}
        def inverted(a, b):
            return int((a < b) != (ranks[a] < ranks[b]))
        core_inversions = sum(inverted(a, b) for index, a in enumerate(core_order) for b in core_order[index+1:])
        removed_inversions = {a: sum(inverted(a, b) for b in core_order if b != a) for a in core_order}
        added_inversions = {a: sum(inverted(a, b) for b in core_order) for a in extra_order}
    groups = {}
    configurations = _configurations(occupations, delta_s)
    for origin, config in enumerate(configurations):
        for determinant, coefficient in config.items():
            holes, particles = core_mask & ~determinant, virtual_mask & determinant
            if holes.bit_count() > 1 or particles.bit_count() > 1:
                raise ValueError('AWF exceeds the supported single-hole/single-particle structure')
            hs, hi = hole_index[holes.bit_length()-1] if holes else (None, 0)
            ps, pa = particle_index[particles.bit_length()-1] if particles else (None, 0)
            active = sum(int(bool(determinant & (1 << (i+s*nmo)))) << (2*s+k)
                         for s in range(2) for k, i in enumerate(opened))
            na, nb = (active & 3).bit_count(), (active >> 2).bit_count()
            # alpha-first -> core-alpha/core-beta/active/virtual ordering.
            inversions = (nc-int(hs == 1))*(na+int(ps == 0)) + nb*int(ps == 0)
            if not contiguous:
                extras = [i+s*nmo for s in range(2) for k, i in enumerate(opened)
                          if active & (1 << (2*s+k))]
                if particles:
                    extras.append(particles.bit_length()-1)
                inversions = core_inversions+sum(added_inversions[a] for a in extras)
                inversions += sum(inverted(a, b) for index, a in enumerate(extras) for b in extras[index+1:])
                if holes:
                    hole = holes.bit_length()-1
                    inversions -= removed_inversions[hole]+sum(inverted(a, hole) for a in extras)
            if holes:
                inversions += hi+nc*hs
            phase = (-1)**inversions
            key = active, hs, ps
            target = hi*(nv if particles else 1)+pa
            groups.setdefault(key, []).append((target, origin, phase*coefficient))
    plan = {}
    for key, entries in groups.items():
        target, origin, weights = zip(*entries)
        plan[key] = (np.array(target, dtype=np.int64), np.array(origin, dtype=np.int64),
                     np.array(weights))
    return plan, nc, nv, len(configurations)


def _host(array, xp):
    return np.asarray(array) if xp is np else xp.asnumpy(array)


def _coefficients(td, xp):
    occ = tuple(_host(xp.asarray(td._scf.mo_occ), xp).tolist())
    plan, nc, nv, size = _compression_plan(occ, int(td.deltaS))
    x = xp.stack([xp.asarray(xy[0]).ravel() for xy in td.xy])
    if x.shape[1] != size:
        raise ValueError('Amplitude length does not match NT-TDA orbital partition')
    groups = {}
    gram = xp.zeros((len(x), len(x)), dtype=x.dtype)
    for key, (target, origin, weights) in plan.items():
        _, hs, ps = key
        shape = (len(x),) + ((nc,) if hs is not None else ()) + ((nv,) if ps is not None else ())
        values = xp.zeros((len(x), math.prod(shape[1:])), dtype=x.dtype)
        # Coefficients stay on the device. Repeated destinations (e.g. CV(1)
        # and CV(0), or redundant OO trace) must be added, not overwritten.
        xp.add.at(values, (slice(None), xp.asarray(target)),
                  x[:, xp.asarray(origin)]*xp.asarray(weights)[None])
        gram += values.conj().dot(values.T)
        groups[key] = values.reshape(shape)
    if np.max(abs(_host(gram, xp)-np.eye(len(x)))) > 1e-7:
        raise ValueError('AWF roots must be normalized/orthogonal; check the OO projection')
    return groups, occ, nc, nv


def _active(pattern):
    return [('fixed', k//2, k % 2) for k in range(4) if pattern & (1 << k)]


@lru_cache(maxsize=1024)
def _pair_terms(left, right):
    """Compile bordered determinant permutations into two-index tensor factors."""
    al, hl, pl = left
    ar, hr, pr = right
    rows = ([('j', hr, None)] if hr is not None else []) + _active(al)
    columns = ([('i', hl, None)] if hl is not None else []) + _active(ar)
    if pl is not None:
        rows.append(('a', pl, None))
    if pr is not None:
        columns.append(('b', pr, None))
    if len(rows) != len(columns):
        raise ValueError('Electron counts differ between AWF sectors')
    terms = {}
    for permutation in itertools.permutations(range(len(rows))):
        sign = (-1)**sum(permutation[i] > permutation[j]
                         for i in range(len(rows)) for j in range(i+1, len(rows)))
        # One right core deletion alone has an extra cofactor minus sign;
        # the total closed core has even dimension 2*nc.
        if hr is not None and hl is None:
            sign = -sign
        factors = []
        for row, column_index in zip(rows, permutation):
            column = columns[column_index]
            rl, rs, ri = row
            cl, cs, ci = column
            if rs != cs:
                break  # spin-independent spatial overlap: exact structural zero
            if rl == 'j':
                matrix = 'K' if cl == 'i' else 'R'
            else:
                matrix = 'L' if cl == 'i' else 'Q'
            if matrix == 'L':
                sign = -sign
            factors.append((matrix, rl, ri, cl, ci))
        else:
            key = tuple(sorted(factors, key=repr))
            terms[key] = terms.get(key, 0)+sign
    return tuple((sign, factors) for factors, sign in terms.items() if sign)


@lru_cache(maxsize=4096)
def _gemm_plan(ll, left_shape, rl, right_shape):
    """Static reshape/GEMM metadata; never cache electronic array values."""
    shared = [label for label in ll if label in rl]
    la, ra = [label for label in ll if label not in shared], [label for label in rl if label not in shared]
    lshape = dict(zip(ll, left_shape))
    rshape = dict(zip(rl, right_shape))
    if any(lshape[label] != rshape[label] for label in shared):
        raise ValueError('Contraction index dimensions disagree')
    k = math.prod(lshape[label] for label in shared)
    m = math.prod(lshape[label] for label in la)
    n = math.prod(rshape[label] for label in ra)
    out_shape = tuple([lshape[label] for label in la]+[rshape[label] for label in ra])
    return (tuple(ll.index(label) for label in la+shared),
            tuple(rl.index(label) for label in shared+ra), m, k, n, out_shape, ''.join(la+ra))


@lru_cache(maxsize=1024)
def _network_plan(signature):
    nodes = list(signature)
    steps = []
    while len(nodes) > 1:
        choices = []
        for i, (al, ashape) in enumerate(nodes):
            for j in range(i+1, len(nodes)):
                bl, bshape = nodes[j]
                shared = set(al) & set(bl)
                if shared:
                    k = math.prod(ashape[al.index(label)] for label in shared)
                    choices.append((math.prod(ashape)*math.prod(bshape)//k, i, j))
        _, i, j = min(choices) if choices else (0, 0, 1)
        al, ashape = nodes[i]
        bl, bshape = nodes[j]
        plan = _gemm_plan(al, ashape, bl, bshape)
        steps.append((i, j, plan))
        nodes = [node for k, node in enumerate(nodes) if k not in (i, j)]+[(plan[-1], plan[-2])]
    return tuple(steps), nodes[0][0]


def _network(nodes, xp, intermediates=None):
    """Greedy connected contractions, avoiding four orbital indices at once.

    Every orbital index has degree two. The only cycle is
    X_left[i,a]--K[j,i]--X_right[j,b]--Q[a,b], handled by two GEMMs.
    Other components reduce to root vectors and their small outer product.
    """
    plan, labels = _network_plan(tuple((labels, tuple(tensor.shape)) for tensor, labels in nodes))
    arrays = [tensor for tensor, _ in nodes]
    for i, j, (at, bt, m, k, n, shape, _) in plan:
        key = (id(arrays[i]), id(arrays[j]), at, bt, m, k, n, shape)
        if intermediates is not None and key in intermediates:
            result = intermediates[key]
        else:
            a = arrays[i].transpose(at).reshape(m, k)
            b = arrays[j].transpose(bt).reshape(k, n)
            result = a.dot(b).reshape(shape)
            if intermediates is not None:
                intermediates[key] = result
        arrays = [array for index, array in enumerate(arrays) if index not in (i, j)]+[result]
    array = arrays[0]
    if set(labels) != {'s', 't'}:
        raise RuntimeError(f'Uncontracted overlap indices: {labels}')
    return array if labels == 'st' else array.T


def overlap_from_mo(previous, current, mo_overlap, xp=np):
    """Exact CPU/GPU contractions for one spin sector and a cross-MO metric.

    For a fixed two-open-orbital reference and fixed number of roots, work
    is O(N^3), storage O(N^2). A singular closed-core cross overlap is rejected
    rather than inverted unstably; the determinant backend remains an oracle.
    """
    if previous.deltaS != current.deltaS:
        raise ValueError('Spin sector changed between steps')
    left, occ_l, nc, nv = _coefficients(previous, xp)
    right, occ_r, nc_r, nv_r = _coefficients(current, xp)
    if occ_l != occ_r or (nc, nv) != (nc_r, nv_r):
        raise ValueError('Reference orbital partition changed between steps')
    mo = xp.asarray(mo_overlap)
    closed = xp.asarray([i for i, occ in enumerate(occ_l) if occ == 2], dtype=xp.int64)
    extra = xp.asarray([i for value in (1, 0) for i, occ in enumerate(occ_l) if occ == value], dtype=xp.int64)
    d = mo[xp.ix_(closed, closed)]
    if nc:
        singular = _host(xp.linalg.svd(d, compute_uv=False), xp)
        if singular.min() < 1e-8 or singular.max()/singular.min() > 1e8:
            raise SingularCoreOverlapError('Closed-core overlap is ill-conditioned; use WFoverlap')
        inverse = xp.linalg.inv(d)
        sign, logarithm = xp.linalg.slogdet(d)
        prefactor = (sign*xp.exp(logarithm))**2
    else:
        inverse, prefactor = xp.zeros((0, 0)), xp.asarray(1.)
    ce, ec = mo[xp.ix_(closed, extra)], mo[xp.ix_(extra, closed)]
    r = inverse.dot(ce)
    l = ec.dot(inverse)
    q = mo[xp.ix_(extra, extra)]-ec.dot(r)
    matrices = dict(K=inverse, R=r, L=l, Q=q)
    views = {}
    def factor(description):
        if description not in views:
            matrix, row, ri, column, ci = description
            rows = ri if row == 'fixed' else slice(2, None) if row == 'a' else slice(None)
            columns = ci if column == 'fixed' else slice(2, None) if column == 'b' else slice(None)
            tensor = matrices[matrix][rows, columns]
            labels = ('' if row == 'fixed' else row)+('' if column == 'fixed' else column)
            views[description] = tensor, labels
        return views[description]
    result = xp.zeros((len(previous.xy), len(current.xy)), dtype=xp.result_type(mo, *left.values(), *right.values()))
    # Frame-local common subexpressions: neither amplitudes nor metrics are
    # reused across geometries. Keep every primitive alive while identity keys
    # are used, to prevent Python object-id reuse and stale numerical values.
    left = {key: value.conj() for key, value in left.items()}
    intermediates, scales = {}, {}
    for lk, a in left.items():
        _, hl, pl = lk
        ll = 's'+('i' if hl is not None else '')+('a' if pl is not None else '')
        for rk, b in right.items():
            _, hr, pr = rk
            rl = 't'+('j' if hr is not None else '')+('b' if pr is not None else '')
            for sign, descriptions in _pair_terms(lk, rk):
                nodes, scalar_descriptions = [(a, ll), (b, rl)], []
                for description in descriptions:
                    tensor, labels = factor(description)
                    if labels:
                        nodes.append((tensor, labels))
                    else:
                        scalar_descriptions.append(description)
                scale_key = sign, tuple(scalar_descriptions)
                if scale_key not in scales:
                    scale = sign
                    for description in scalar_descriptions:
                        scale = scale*factor(description)[0]
                    scales[scale_key] = scale
                result += scales[scale_key]*_network(nodes, xp, intermediates)
    return prefactor*result


def gpu_overlap_matrix(previous, current, ao_overlap=None, mo_overlap=None):
    """CuPy-native cross-geometry AWF overlaps; return a device root matrix.

    PySCF/libcint evaluates the two-center cross-geometry AO overlap on CPU.
    It is transferred once; current GPU orbitals/amplitudes are never copied
    to host. Previous CPU snapshot arrays are uploaded explicitly once.
    """
    import cupy as cp
    if mo_overlap is None:
        if ao_overlap is None:
            ao_overlap = gto.intor_cross('int1e_ovlp', previous.mol, current.mol)
        c_left, c_right = cp.asarray(previous._scf.mo_coeff), cp.asarray(current._scf.mo_coeff)
        mo_overlap = c_left.conj().T.dot(cp.asarray(ao_overlap)).dot(c_right)
    return overlap_from_mo(previous, current, mo_overlap, xp=cp)
