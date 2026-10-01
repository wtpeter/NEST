"""Spin-adapted auxiliary-wavefunction overlaps for triplet-reference NT-TDA.

The determinant convention is alpha-first. Only deltaS=-1 and 0 from a
two-open-orbital restricted reference are implemented. These are auxiliary
NT-TDA wavefunctions, not correlated eigenstates. The GPU backend uses cubic
Schur-complement contractions; external WFoverlap and the small-system Python
determinant backend remain exact references. No frozen core or amplitude-dot
approximation is used.
"""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from pyscf import gto
from nest.nttda.nttda import transition_dm


def _combine(parts):
    result = {}
    for scale, state in parts:
        for det, value in state.items():
            result[det] = result.get(det, 0) + scale * value
    return {det: value for det, value in result.items() if abs(value) > 1e-14}


def _one_body(state, p, q):
    result = {}
    for det, value in state.items():
        if not det & (1 << q):
            continue
        phase = (-1)**((det & ((1 << q)-1)).bit_count())
        removed = det ^ (1 << q)
        if removed & (1 << p):
            continue
        phase *= (-1)**((removed & ((1 << p)-1)).bit_count())
        result[removed | (1 << p)] = phase * value
    return result


def _tensor(state, p, q, projection, nmo):
    if projection == -1:
        return _one_body(state, nmo+p, q)
    if projection == 1:
        return _combine([(-1, _one_body(state, p, nmo+q))])
    return _combine([(1/np.sqrt(2), _one_body(state, p, q)),
                     (-1/np.sqrt(2), _one_body(state, nmo+p, nmo+q))])


@lru_cache(maxsize=8)
def _configurations(occupations, delta_s):
    nmo = len(occupations)
    closed = [i for i, occ in enumerate(occupations) if occ == 2]
    opened = [i for i, occ in enumerate(occupations) if occ == 1]
    virtual = [i for i, occ in enumerate(occupations) if occ == 0]
    if len(opened) != 2 or any(occ not in (0, 1, 2) for occ in occupations):
        raise NotImplementedError('AWF overlaps require an integer triplet reference (two open orbitals)')
    reference = {sum(1 << i for i in closed+opened) +
                 sum(1 << (nmo+i) for i in closed): 1.0}
    lowered = [reference]
    for _ in range(2):
        state = _combine([(1, _one_body(lowered[-1], nmo+i, i)) for i in opened])
        norm = np.sqrt(sum(abs(c)**2 for c in state.values()))
        lowered.append({d: c/norm for d, c in state.items()})
    configurations = []
    if delta_s == -1:
        for q in closed+opened:
            for p in opened+virtual:
                state = _combine([(c, _tensor(ref, p, q, m, nmo))
                                  for c, ref, m in zip((1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)),
                                                      lowered, (-1, 0, 1))])
                scale = np.sqrt(2/3) if (q in closed) == (p in opened) else 1
                if q in opened and p in opened:
                    scale = 1/np.sqrt(3)
                configurations.append(_combine([(scale, state)]))
    elif delta_s == 0:
        # Coupled rank-one tensor |S=1,M=1>. NEST uses the negative
        # coupled-tensor phase, OO=-reference, and positive scalar CV(0).
        def rank_one(p, q):
            state = _combine([(-1/np.sqrt(2), _tensor(reference, p, q, 0, nmo)),
                              (1/np.sqrt(2), _tensor(lowered[1], p, q, 1, nmo))])
            norm = np.sqrt(sum(abs(c)**2 for c in state.values()))
            return {d: c/norm for d, c in state.items()}
        configurations.extend(rank_one(p, q) for q in closed for p in opened)
        configurations.extend(rank_one(p, q) for q in closed for p in virtual)
        configurations.append(_combine([(-1, reference)]))
        configurations.extend(rank_one(p, q) for q in opened for p in virtual)
        configurations.extend(_combine([(1/np.sqrt(2), _one_body(reference, p, q)),
                                        (1/np.sqrt(2), _one_body(reference, nmo+p, nmo+q))])
                              for q in closed for p in virtual)
    else:
        raise NotImplementedError('AWF overlaps support deltaS=-1 and 0')
    return tuple(configurations)


@dataclass(frozen=True)
class AuxiliaryWavefunction:
    nmo: int
    nalpha: int
    nbeta: int
    coefficients: dict

    @property
    def norm(self):
        return float(sum(abs(c)**2 for c in self.coefficients.values()))


def build_awf(td, root):
    """Expand a zero-based root of a CPU NEST solution (explicit GPU copy first)."""
    occ = tuple(np.asarray(td._scf.mo_occ).tolist())
    x = np.asarray(td.xy[root][0]).ravel()
    configs = _configurations(occ, td.deltaS)
    if len(x) != len(configs):
        raise ValueError('NT-TDA amplitude length does not match orbital partition')
    state = _combine(zip(x, configs))
    nc = occ.count(2)
    wf = AuxiliaryWavefunction(len(occ), nc+2+td.deltaS, nc-td.deltaS, state)
    if not np.isclose(wf.norm, 1, atol=1e-7, rtol=0):
        raise ValueError(f'AWF root {root} is not normalized: {wf.norm}; check OO trace/projection')
    return wf


def determinant_overlap(left, right, mo):
    """Exact small-system reference, with both orbital motion and CI amplitudes."""
    if (left.nalpha, left.nbeta) != (right.nalpha, right.nbeta):
        return 0.0
    def occupied(det, nmo, offset):
        return [i for i in range(nmo) if det & (1 << (i+offset))]
    result = 0.0
    for dl, cl in left.coefficients.items():
        al, bl = occupied(dl, left.nmo, 0), occupied(dl, left.nmo, left.nmo)
        for dr, cr in right.coefficients.items():
            ar, br = occupied(dr, right.nmo, 0), occupied(dr, right.nmo, right.nmo)
            result += cl.conjugate()*cr*np.linalg.det(mo[np.ix_(al, ar)])*np.linalg.det(mo[np.ix_(bl, br)])
    return result


def determinant_rdm(left, right, projection=None):
    """Small-system oracle: <left|E_pq|right> or rank-one T_pq,projection."""
    nmo = left.nmo
    result = np.zeros((nmo, nmo), dtype=complex)
    for p in range(nmo):
        for q in range(nmo):
            state = (_combine([(1, _one_body(right.coefficients, p, q)),
                               (1, _one_body(right.coefficients, nmo+p, nmo+q))])
                     if projection is None else _tensor(right.coefficients, p, q, projection, nmo))
            result[p, q] = sum(left.coefficients.get(d, 0).conjugate()*c for d, c in state.items())
    return result


def state_density(td, bra, ket):
    """Polynomial-cost scalar density, including the reference term on diagonal.

    Converged roots must be normalized/orthogonal and deltaS=-1 OO traceless.
    The analytic excitation correction is NEST's transition_dm expression.
    """
    correction = transition_dm(td, bra, ket)
    if bra == ket:
        correction += np.diag(td._scf.mo_occ)
    return correction


def _write_dets(path, wavefunctions):
    nmo = wavefunctions[0].nmo
    determinants = sorted(set().union(*(w.coefficients for w in wavefunctions)))
    with Path(path).open('w') as stream:
        stream.write(f'{len(wavefunctions)} {nmo} {len(determinants)}\n')
        for det in determinants:
            alpha = [bool(det & (1 << i)) for i in range(nmo)]
            beta = [bool(det & (1 << (nmo+i))) for i in range(nmo)]
            # Native file strings use interleaved alpha/beta ordering;
            # WFoverlap converts back to alpha-first when reading them.
            phase = (-1)**sum(sum(alpha[i+1:]) for i in range(nmo) if beta[i])
            occupation = ''.join('d' if a and b else 'a' if a else 'b' if b else 'e'
                                 for a, b in zip(alpha, beta))
            coefficients = np.array([phase*w.coefficients.get(det, 0) for w in wavefunctions])
            if np.max(abs(coefficients.imag)) > 1e-13:
                raise NotImplementedError('WFoverlap native CI files require real amplitudes')
            stream.write(occupation+' '+' '.join(f'{c:.17e}' for c in coefficients.real)+'\n')


def _write_identity_mos(path, nmo):
    with Path(path).open('w') as stream:
        stream.write(f'2mocoef\nheader\n1\nNT-TDA cross-MO overlap basis\n1\n{nmo} {nmo}\na\nmocoef\n(*)\n')
        for column in np.eye(nmo):
            stream.write(' '.join(f'{c:.17e}' for c in column)+'\n')


def wfoverlap(left, right, mo, directory, executable=None, memory_mb=8000):
    """Run SHARC WFoverlap on spin-adapted determinants and cross-MO metric.

    The output's raw overlap is returned without state renormalization.
    All input/output/scratch files stay in the caller-provided directory.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = executable or shutil.which('wfoverlap.x')
    if executable is None and os.environ.get('SHARC'):
        executable = str(Path(os.environ['SHARC'])/'wfoverlap.x')
    if executable is None or not Path(executable).is_file():
        raise FileNotFoundError('Set wfoverlap_executable or activate SHARC to find wfoverlap.x')
    for name, states in (('a', left), ('b', right)):
        _write_dets(directory/f'det.{name}', states)
        _write_identity_mos(directory/f'mo.{name}', states[0].nmo)
    np.savetxt(directory/'mix_ovl', mo, header=f'{mo.shape[0]} {mo.shape[1]}', comments='', fmt='%.17e')
    (directory/'cioverlap.input').write_text('a_mo=mo.a\nb_mo=mo.b\na_det=det.a\nb_det=det.b\nmix_aoovl=mix_ovl\n')
    with (directory/'wfoverlap.out').open('w') as out, (directory/'wfoverlap.err').open('w') as err:
        completed = subprocess.run([str(Path(executable).resolve()), '-m', str(memory_mb)],
                                   cwd=directory, stdout=out, stderr=err, check=False)
    text = (directory/'wfoverlap.out').read_text()
    if completed.returncode or 'Overlap matrix <PsiA_i|PsiB_j>' not in text:
        raise RuntimeError(f'WFoverlap failed; inspect {directory}/wfoverlap.out and .err')
    block = text.split('Overlap matrix <PsiA_i|PsiB_j>', 1)[1].split('Renormalized overlap matrix', 1)[0]
    rows = re.findall(r'<PsiA\s+\d+\|([^\n]+)', block)
    result = np.array([[float(value.replace('D', 'E')) for value in row.split()] for row in rows])
    if result.shape != (len(left), len(right)) or not np.all(np.isfinite(result)):
        raise RuntimeError(f'Invalid/truncated WFoverlap output in {directory}')
    return result


def overlap_matrix(previous, current, directory=None, backend='wfoverlap', ao_overlap=None, **kwargs):
    """Cross-geometry root overlap in one spin sector, in energy/root order."""
    if backend == 'gpu':
        import cupy as cp
        from nest.nttda.overlap_contraction import gpu_overlap_matrix
        # File-interface boundary: only the small root matrix returns to CPU.
        return cp.asnumpy(gpu_overlap_matrix(previous, current, ao_overlap=ao_overlap))
    if previous.deltaS != current.deltaS or not np.array_equal(previous._scf.mo_occ, current._scf.mo_occ):
        raise ValueError('Spin sector or reference occupations changed between steps')
    ao = ao_overlap if ao_overlap is not None else gto.intor_cross('int1e_ovlp', previous.mol, current.mol)
    mo = previous._scf.mo_coeff.conj().T @ ao @ current._scf.mo_coeff
    left = [build_awf(previous, i) for i in range(len(previous.xy))]
    right = [build_awf(current, i) for i in range(len(current.xy))]
    if backend == 'python':
        return np.array([[determinant_overlap(a, b, mo) for b in right] for a in left])
    if backend != 'wfoverlap' or directory is None:
        raise ValueError('Choose python or wfoverlap with an explicit working directory')
    return wfoverlap(left, right, mo, directory, **kwargs)
