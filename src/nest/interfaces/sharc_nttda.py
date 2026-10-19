"""Experimental SHARC file interface: GPU ROKS/NT-TDA + CPU NEST SOC.

Run ``python -m nest.interfaces.sharc_nttda QM.in --config nttda.json``
from the trajectory's QM directory. Supports singlet deltaS=-1 and triplet
deltaS=0 roots with AWF overlaps, or the legacy reference-triplet DDR path. See
``docs/gpu-nttda-sharc.md`` in the workspace for the supported scope.
"""

import argparse
import itertools
import json
from pathlib import Path
import time
import warnings

import numpy as np
from pyscf import gto

from nest.soc.gpu_nttda import from_gpu_nttda, to_nest
from nest.nttda.overlap import overlap_matrix, state_density


def read_qmin(path):
    lines = Path(path).read_text().splitlines()
    natom = int(lines[0])
    atoms = [(fields[0], tuple(map(float, fields[1:4])))
             for fields in (line.split() for line in lines[2:2+natom])]
    if len(atoms) != natom:
        raise ValueError('Incomplete geometry in QM.in')
    keywords = {}
    selecting = False
    supported = {'states', 'charge', 'unit', 'step', 'savedir', 'dt', 'retain',
                 'h', 'soc', 'dm', 'grad', 'nacdr', 'overlap', 'phases',
                 'init', 'samestep', 'restart'}
    for line in lines[2+natom:]:
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        key = fields[0].lower()
        if selecting:
            if key == 'end':
                selecting = False
            continue
        if key not in supported:
            raise NotImplementedError(f'Unsupported QM.in request: {key}')
        keywords[key] = fields[1:]
        if key == 'nacdr' and [v.lower() for v in fields[1:]] == ['select']:
            selecting = True
    if selecting:
        raise ValueError('Unterminated NACDR SELECT block')
    states = list(map(int, keywords['states']))
    if len(states) != 3 or states[0] < 1 or states[1] != 0 or states[2] < 1:
        raise NotImplementedError('Requires states Nsinglets 0 Ntriplets, both positive')
    charges = list(map(int, keywords.get('charge', ['0'])))
    if len(charges) not in (1, 3):
        raise ValueError('Specify one common charge or three multiplicity charges')
    if len(charges) > 1 and charges[0] != charges[2]:
        raise ValueError('Singlets and triplet must have the same charge')
    unit = keywords.get('unit', ['angstrom'])[0].lower()
    if unit not in ('angstrom', 'bohr'):
        raise ValueError('Geometry unit must be angstrom or bohr')
    return atoms, keywords, states[0], charges[0], unit


def canonical_order(soc):
    """NEST root/Ms order -> SHARC multiplicity/Ms/root order."""
    entries = []
    for state, block in zip(soc.states, soc.state_slices):
        for offset, ms in enumerate(soc._m_values(state.spin)):
            # Root order inside each multiplicity follows NEST's input order.
            entries.append((int(round(2*state.spin+1)), ms, block.start+offset))
    return np.array([index for _, _, index in sorted(entries)], dtype=int)


def _complex_matrix(flag, title, matrix):
    nrow, ncol = matrix.shape
    lines = [f'! {flag} {title}', f'{nrow} {ncol}']
    lines.extend(' '.join(f'{z.real:.14e} {z.imag:.14e}' for z in row) for row in matrix)
    return '\n'.join(lines)+'\n\n'


def write_qmout(path, result, requests, elapsed):
    h, grad, nac, dm = (result[key] for key in ('h', 'grad', 'nac', 'dm'))
    nstate, natom = len(h), grad.shape[1]
    text = _complex_matrix(1, 'Hamiltonian Matrix', h)
    if 'dm' in requests:
        text += '! 2 Dipole Moment Matrices\n'
        for component in dm:
            text += _complex_matrix(0, '', component).split('\n', 1)[1].rstrip()+'\n'
        text += '\n'
    if 'grad' in requests:
        text += '! 3 Gradient Vectors\n'
        for vector in grad:
            text += f'{natom} 3\n'+''.join(' '.join(f'{v:.14e}' for v in row)+'\n' for row in vector)
    if 'nacdr' in requests:
        text += '! 5 Non-adiabatic couplings (ddr)\n'
        for i, j in itertools.product(range(nstate), repeat=2):
            text += f'{natom} 3\n'+''.join(' '.join(f'{v:.14e}' for v in row)+'\n' for row in nac[i, j])
    if 'overlap' in requests:
        text += _complex_matrix(6, 'Overlap matrix', result['overlap'])
    if 'phases' in requests:
        text += f'! 7 Wave function phases\n{nstate}\n'
        text += ''.join(f'{p:.14e} 0.0\n' for p in result['phases'])
    text += f'! 8 Runtime\n{elapsed:.8f}\n'
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def _snapshot(td):
    cpu = to_nest(td)
    return dict(mol=np.array(cpu.mol.dumps()), mo_coeff=cpu._scf.mo_coeff,
                mo_occ=cpu._scf.mo_occ, mo_energy=cpu._scf.mo_energy,
                e_tot=np.array(cpu._scf.e_tot), e=cpu.e,
                x=np.array([xy[0] for xy in cpu.xy]), deltaS=np.array(cpu.deltaS))


def _load_snapshot(path):
    from nest.nttda import NTTDA
    from pyscf import dft
    with np.load(path, allow_pickle=False) as data:
        mf = dft.ROKS(gto.loads(str(data['mol'])))
        for key in ('mo_coeff', 'mo_occ', 'mo_energy', 'e_tot'):
            setattr(mf, key, data[key].copy())
        mf.converged = True
        td = NTTDA(mf)
        td.deltaS = int(data['deltaS']) if 'deltaS' in data else -1
        td.e, td.xy = data['e'].copy(), [(x.copy(), 0) for x in data['x']]
        td.converged = np.ones(len(td.e), dtype=bool)
    return td


def project_reference(previous, mol):
    """Project previous spatial MOs into the new AO basis and orthonormalize.

    This small CPU boundary precedes the GPU SCF. Keeping the previous
    occupations in a Newton solve avoids Aufbau/DIIS switching ROKS solutions.
    """
    from pyscf.scf.addons import project_mo_nr2nr
    projected = project_mo_nr2nr(previous.mol, previous._scf.mo_coeff, mol)
    metric = projected.T @ mol.intor_symmetric('int1e_ovlp') @ projected
    eigenvalues, vectors = np.linalg.eigh(metric)
    if eigenvalues.min() < 1e-9:
        raise RuntimeError('Projected MO space is linearly dependent; cannot continue reference')
    return projected @ (vectors * eigenvalues**-.5) @ vectors.T


def reference_overlap(previous, current):
    """Highest-Ms determinant overlap, used to detect SCF branch jumps."""
    ao = gto.intor_cross('int1e_ovlp', previous.mol, current.mol)
    mo = previous._scf.mo_coeff.T @ ao @ current._scf.mo_coeff
    occ = previous._scf.mo_occ
    if not np.array_equal(occ, current._scf.mo_occ):
        raise RuntimeError('Reference orbital occupations changed between steps')
    alpha, beta = np.flatnonzero(occ > 0), np.flatnonzero(occ == 2)
    return np.linalg.det(mo[np.ix_(alpha, alpha)])*np.linalg.det(mo[np.ix_(beta, beta)])


def overlaps(previous, current, previous_triplet=None, current_triplet=None,
             directory=None, backend='python', **kwargs):
    """Exact determinant AWF overlaps, including spatial MO/AO motion.

    Singlet-triplet overlaps vanish by spin symmetry. All three triplet
    projections have the same root overlap block because orbital motion is
    spin independent. Legacy calls without a triplet solution use its reference.
    """
    if backend != 'gpu' and type(current).__module__.startswith('gpu4pyscf'):
        current = to_nest(current)
    ao = gto.intor_cross('int1e_ovlp', previous.mol, current.mol)
    n = len(current.e)
    nt = len(current_triplet.e) if current_triplet is not None else 1
    result = np.zeros((n+3*nt, n+3*nt), dtype=complex)
    result[:n, :n] = overlap_matrix(previous, current,
                                   directory=Path(directory)/'singlets' if directory else None,
                                   backend=backend, ao_overlap=ao, **kwargs)
    if current_triplet is not None:
        current_triplet = (to_nest(current_triplet) if backend != 'gpu' and type(current_triplet).__module__.startswith('gpu4pyscf')
                           else current_triplet)
        block = overlap_matrix(previous_triplet, current_triplet,
                               directory=Path(directory)/'triplets' if directory else None,
                               backend=backend, ao_overlap=ao, **kwargs)
        for ms in range(3):
            offset = n+ms*nt
            result[offset:offset+nt, offset:offset+nt] = block
        return result
    if backend == 'gpu':
        import cupy as cp
        mo = cp.asarray(previous._scf.mo_coeff).T.dot(cp.asarray(ao)).dot(cp.asarray(current._scf.mo_coeff))
        current_occ = cp.asnumpy(cp.asarray(current._scf.mo_occ))
        linalg = cp.linalg
    else:
        mo = previous._scf.mo_coeff.T @ ao @ current._scf.mo_coeff
        current_occ = current._scf.mo_occ
        linalg = np.linalg
    occ = previous._scf.mo_occ
    if not np.array_equal(occ, current_occ):
        raise RuntimeError('Reference orbital occupations changed between steps')
    alpha, beta = np.flatnonzero(occ > 0), np.flatnonzero(occ == 2)
    value = linalg.det(mo[np.ix_(alpha, alpha)])*linalg.det(mo[np.ix_(beta, beta)])
    if backend == 'gpu':
        value = float(cp.asnumpy(value))
    result[n:, n:] = np.eye(3)*value
    return result


def _dipoles(td, triplet=None):
    td = to_nest(td) if type(td).__module__.startswith('gpu4pyscf') else td
    if triplet is not None and type(triplet).__module__.startswith('gpu4pyscf'):
        triplet = to_nest(triplet)
    coeff, occ = td._scf.mo_coeff, td._scf.mo_occ
    ao = td.mol.intor('int1e_r', comp=3)
    mo = np.einsum('up,xuv,vq->xpq', coeff, ao, coeff)
    n = len(td.e)
    nt = len(triplet.e) if triplet is not None else 1
    dm = np.zeros((3, n+3*nt, n+3*nt), dtype=complex)
    def block(solution):
        values = np.zeros((3, len(solution.e), len(solution.e)), dtype=complex)
        for i in range(len(solution.e)):
            for j in range(i+1):
                value = -np.einsum('xpq,pq->x', mo, state_density(solution, i, j))
                values[:, i, j], values[:, j, i] = value, value.conj()
        return values
    dm[:, :n, :n] = block(td)
    tblock = block(triplet) if triplet is not None else -np.einsum('xpp,p->x', mo, occ)[:, None, None]
    for ms in range(3):
        offset = n+ms*nt
        dm[:, offset:offset+nt, offset:offset+nt] = tblock
    nuclear = np.einsum('a,ax->x', td.mol.atom_charges(), td.mol.atom_coords())
    dm += nuclear[:, None, None]*np.eye(n+3*nt)
    return dm


def _triplet_gradients(td, cphf_conv_tol=1e-10, backend='serial', cphf_max_cycle=None):
    """All NT-TDA triplet gradients with one geometry-local cache.

    The reference term is common to all roots; ``grad_elec`` supplies the
    excitation derivative using the public driver and its response cache.
    Never retain this driver after replacing orbitals or coordinates.
    """
    if backend not in ('serial', 'batch'):
        raise ValueError('triplet_gradient_backend must be serial or batch')
    driver = td.Gradients().set(cphf_conv_tol=cphf_conv_tol,
                                cphf_max_cycle=cphf_max_cycle)
    if backend == 'serial':
        if cphf_max_cycle is not None and driver._context.method.reference_kind == 'ensemble_roks':
            # The standalone selected-reference solve has its own iteration
            # limit; the public NT-TDA driver does not forward this option.
            reference_driver = td._scf.nuc_grad_method().set(max_cycle=cphf_max_cycle)
            reference_driver.verbose = 0
            if driver._with_df:
                reference_driver.nttda_xc_backend = driver._context.xc_backend
            reference = reference_driver.kernel()
        else:
            reference = driver.grad_nuc()
        values = [reference+driver.grad_elec(xy) for xy in td.xy]
    else:
        # The fork's public compute_frame covers spin-flip gradients/NACs.
        # Its channel preparations also allow same-spin gradient RHSs to
        # share a solve without requesting unsupported same-spin NACs.
        from dataclasses import replace
        import cupy as cp
        from gpu4pyscf.sftda import nttda_methods as methods
        from gpu4pyscf.grad._nttda.response import finish_prepared_gradients

        context = driver._context
        atoms = tuple(range(td.mol.natm))
        prepared = [methods.prepare_gradient(
            context.method, driver, td, xy, atmlst=atoms,
            tolerance=cphf_conv_tol, max_cycle=driver.cphf_max_cycle,
            cache=context) for xy in td.xy]
        if context.method.reference_kind == 'ensemble_roks':
            pairs = methods.orbital_backend(context.method).canonical_pairs(td, compact=True)
            reference_driver = td._scf.nuc_grad_method()
            reference_driver.verbose = 0
            reference_prepared = reference_driver.prepare_nttda_fusion(
                pairs, atmlst=atoms, response_cache=context)
            if reference_prepared.pairs != pairs:
                raise RuntimeError('Selected-reference preparation changed orbital pair order')
            prepared = [replace(item,
                                orbital_rhs_shift=reference_prepared.orbital_rhs_shift,
                                tolerance=min(float(cphf_conv_tol), 1e-12))
                        for item in prepared]
            reference = reference_prepared.unrelaxed_gradient
        else:
            reference = driver.grad_nuc()
        components = finish_prepared_gradients(prepared)
        values = [cp.asnumpy(cp.asarray(reference)+item.total) for item in components]
    if td.mol.symmetry:
        values = [driver.symmetrize(value) for value in values]
    return values


def _newton_reference(mf, max_cycle):
    # AH uses sqrt(ah_conv_tol) as a residual criterion. Keep that below
    # the requested orbital gradient; the usual 1e-12 AH tolerance can
    # stall at ~1e-6 when a tighter SCF gradient is requested.
    return mf.newton().set(max_cycle=max_cycle, conv_tol=mf.conv_tol,
                           conv_tol_grad=mf.conv_tol_grad,
                           ah_conv_tol=min(1e-12, .01*mf.conv_tol_grad**2))


def run(qmin, config_path, output=None):
    start = time.monotonic()
    atoms, requests, nroots, charge, unit = read_qmin(qmin)
    config = json.loads(Path(config_path).read_text())
    allowed = {'basis', 'xc', 'grid_level', 'soctype', 'scf_conv_tol', 'scf_conv_tol_grad', 'td_conv_tol',
               'cphf_conv_tol', 'cphf_max_cycle', 'min_overlap', 'max_cycle', 'triplets', 'density_fit',
               'auxbasis', 'scf_diis_cycles', 'scf_newton_cycles', 'overlap_backend',
               'wfoverlap_executable', 'overlap_memory_mb', 'min_reference_overlap', 'initial_reference',
               'reference_model', 'triplet_gradient_backend', 'reference_overlap_action'}
    unknown = set(config)-allowed
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    reference_overlap_action = config.get('reference_overlap_action', 'stop')
    if reference_overlap_action not in ('stop', 'warn'):
        raise ValueError('reference_overlap_action must be stop or warn')
    if config.get('triplet_gradient_backend', 'serial') not in ('serial', 'batch'):
        raise ValueError('triplet_gradient_backend must be serial or batch')
    if config.get('overlap_backend', 'gpu') not in ('gpu', 'wfoverlap', 'python'):
        raise ValueError('overlap_backend must be gpu, wfoverlap or python')
    reference_model = config.get('reference_model', 'roks')
    if reference_model not in ('roks', 'ensemble_roks'):
        raise ValueError('reference_model must be roks or ensemble_roks')
    ntriplets = int(requests['states'][2])
    triplet_mode = config.get('triplets', 'nttda')
    if triplet_mode not in ('nttda', 'reference') or (triplet_mode == 'reference' and ntriplets != 1):
        raise ValueError('triplets must be nttda or reference (requires one triplet)')
    if triplet_mode == 'reference' and config.get('triplet_gradient_backend', 'serial') == 'batch':
        raise ValueError('Batched triplet gradients require triplets=nttda')
    if triplet_mode == 'nttda' and 'nacdr' in requests:
        raise NotImplementedError('Excited triplet dynamics requires SHARC coupling overlap; deltaS=0 NAC is unavailable')
    if reference_model == 'ensemble_roks' and triplet_mode != 'nttda':
        raise NotImplementedError('EnsembleROKS integration requires both NT-TDA spin sectors and overlap coupling')
    dimension = nroots+3*ntriplets
    mol = gto.M(atom=atoms, unit=unit, basis=config.get('basis', 'sto-3g'),
                spin=2, charge=charge, verbose=0)
    step = int(requests.get('step', ['0'])[0])
    savedir = Path(' '.join(requests.get('savedir', ['savedir']))) / 'nttda'
    savedir.mkdir(parents=True, exist_ok=True)
    cache = savedir/f'frame-{step}.npz'
    signature = json.dumps(dict(config=config, symbols=[a[0] for a in atoms],
                                coordinates=mol.atom_coords().tolist(), charge=charge,
                                nroots=nroots, ntriplets=ntriplets), sort_keys=True)
    # Repeated calls in the same SHARC step must use the same wavefunction.
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data['signature']) != signature:
                raise RuntimeError('Geometry/configuration changed for an already saved step; use a fresh savedir')
            result = {k: data[k].copy() for k in ('h', 'grad', 'nac', 'dm', 'overlap', 'phases')}
            if 'nacdr' in requests and not bool(data['has_nac']):
                raise RuntimeError('This step was cached without DDR couplings; request NACDR in the first call')
    else:
        import cupy as cp
        from gpu4pyscf.dft.roks import ROKS
        from gpu4pyscf.sftda import NTTDA_ROKS
        from gpu4pyscf.grad.nttda import compute_frame
        if reference_model == 'ensemble_roks':
            from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS
            reference_class, td_class = EnsembleROKS, NTTDA_EnsembleROKS
        else:
            reference_class, td_class = ROKS, NTTDA_ROKS
        timings = {}
        diagnostics = {'step': step}
        stage_start = time.monotonic()

        def timed_stage(name):
            nonlocal stage_start
            cp.cuda.get_current_stream().synchronize()
            now = time.monotonic()
            timings[name] = now-stage_start
            print(f'Step {step}: {name} {timings[name]:.3f} s', flush=True)
            stage_start = now

        mf = reference_class(mol, xc=config.get('xc', 'PBE'))
        if config.get('density_fit', False):
            mf = mf.density_fit(auxbasis=config.get('auxbasis', 'def2-universal-jkfit'))
        mf.conv_tol = config.get('scf_conv_tol', 1e-10)
        mf.conv_tol_grad = config.get('scf_conv_tol_grad', 1e-8)
        mf.max_cycle = config.get('scf_diis_cycles', 30)
        mf.grids.level = config.get('grid_level', 3)
        mf.small_rho_cutoff = 0
        if step > 0:
            previous_path = savedir/f'wavefunction-{step-1}.npz'
            if not previous_path.exists():
                raise RuntimeError(f'Missing previous wavefunction: {previous_path}')
            previous = _load_snapshot(previous_path)
            previous_triplet = (_load_snapshot(savedir/f'wavefunction-triplet-{step-1}.npz')
                                if triplet_mode == 'nttda' else None)
            # A density-only restart may change which orbitals are singly
            # occupied. Newton keeps occupations while relaxing projected MOs.
            newton = _newton_reference(mf, config.get('scf_newton_cycles', 100))
            newton.kernel(cp.asarray(project_reference(previous, mol)), cp.asarray(previous._scf.mo_occ))
            mf = newton.undo_soscf()
        else:
            previous = previous_triplet = None
            if config.get('initial_reference'):
                seed = _load_snapshot(Path(config_path).resolve().parent / config['initial_reference'])
                if seed.mol.nelectron != mol.nelectron or seed.mol.spin != mol.spin or [seed.mol.atom_symbol(i) for i in range(seed.mol.natm)] != [mol.atom_symbol(i) for i in range(mol.natm)]:
                    raise ValueError('Initial reference has different atoms, charge or spin')
                newton = _newton_reference(mf, config.get('scf_newton_cycles', 100))
                newton.kernel(cp.asarray(project_reference(seed, mol)), cp.asarray(seed._scf.mo_occ))
                mf = newton.undo_soscf()
            else:
                mf.kernel()
        if not mf.converged:
            newton = _newton_reference(mf, config.get('scf_newton_cycles', 100))
            newton.kernel(mf.mo_coeff, mf.mo_occ)
            mf = newton.undo_soscf()
        if not mf.converged:
            raise RuntimeError(f'{reference_model} reference did not converge, including Newton fallback')
        timed_stage('reference')
        td = td_class(mf).set(deltaS=-1, nstates=nroots,
                                 conv_tol=config.get('td_conv_tol', 1e-10),
                                 max_cycle=config.get('max_cycle', 200)).run()
        if len(td.e) != nroots or not np.all(td.converged):
            raise RuntimeError('NT-TDA singlet roots are missing or unconverged')
        timed_stage('singlet_energies')
        if previous is not None:
            reference_value = reference_overlap(previous, to_nest(td))
            diagnostics['reference_overlap'] = float(reference_value)
            print(f'Step {step}: reference overlap {reference_value:.10f}', flush=True)
            if abs(reference_value) < config.get('min_reference_overlap', 0.9):
                message = (f'{reference_model} reference changed discontinuously '
                           f'(overlap={reference_value:.10f}); inspect the orbital branch and reference model')
                if reference_overlap_action == 'stop':
                    raise RuntimeError(message)
                diagnostics['reference_overlap_warning'] = message
                warnings.warn(message, RuntimeWarning)
        triplet = (td_class(mf).set(deltaS=0, nstates=ntriplets,
                                    conv_tol=config.get('td_conv_tol', 1e-10),
                                    max_cycle=config.get('max_cycle', 200)).run()
                   if triplet_mode == 'nttda' else None)
        if triplet is not None and (len(triplet.e) != ntriplets or not np.all(triplet.converged)):
            raise RuntimeError('NT-TDA triplet roots are missing or unconverged')
        timed_stage('triplet_energies')
        soc = from_gpu_nttda(td, *([triplet] if triplet is not None else []),
                             include_reference=triplet is None,
                             soctype=config.get('soctype', 'SOMF')).run()
        order = canonical_order(soc)
        reference_energy = td.reference_energy()
        h = soc.h_soc[np.ix_(order, order)] + reference_energy*np.eye(dimension)
        timed_stage('soc')
        overlap_start = time.monotonic()
        ovlp = (overlaps(previous, td, previous_triplet, triplet,
                         directory=savedir/f'overlap-{step}', backend=config.get('overlap_backend', 'gpu'),
                         executable=config.get('wfoverlap_executable'),
                         memory_mb=config.get('overlap_memory_mb', 8000))
                if previous is not None else np.eye(dimension))
        print(f'Step {step}: overlaps {time.monotonic()-overlap_start:.3f} s', flush=True)
        timed_stage('overlap')
        diagnostics['singlet_overlap_singular_values'] = np.linalg.svd(ovlp[:nroots, :nroots], compute_uv=False).tolist()
        diagnostics['triplet_overlap_singular_values'] = np.linalg.svd(ovlp[nroots:nroots+ntriplets, nroots:nroots+ntriplets], compute_uv=False).tolist()
        diagonal = ovlp.diagonal().real
        # With nophases_from_interface SHARC handles phases from the full
        # overlap matrix, including subspace rotations near degeneracies.
        # If scalar phases are explicitly requested, refuse ambiguous signs.
        if 'phases' in requests and np.min(abs(diagonal)) < config.get('min_overlap', 0.5):
            raise RuntimeError('Ambiguous root/reference phases: reduce the time step or enlarge the state space')
        phases = np.where(diagonal < 0, -1.0, 1.0)
        pairs = list(itertools.combinations(range(1, nroots+1), 2)) if 'nacdr' in requests else []
        frame = compute_frame(td, 1, pairs, gradient_states=list(range(1, nroots+1)),
                              use_etfs=False, cphf_conv_tol=config.get('cphf_conv_tol', 1e-10),
                              cphf_max_cycle=config.get('cphf_max_cycle'))
        timed_stage('singlet_gradients')
        if triplet is None:
            triplet_grads = [td.Gradients().kernel(state=0)]
        else:
            triplet_grads = _triplet_gradients(
                triplet, config.get('cphf_conv_tol', 1e-10),
                backend=config.get('triplet_gradient_backend', 'serial'),
                cphf_max_cycle=config.get('cphf_max_cycle'))
        timed_stage('triplet_gradients')
        grad = np.array([frame['gradients'][i] for i in range(1, nroots+1)] + triplet_grads*3)
        nac = np.zeros((dimension, dimension, mol.natm, 3))
        for (i, j), vector in frame['nac'].items():
            # compute_frame already returns full d_IJ in Bohr^-1, not gap*d_IJ.
            nac[i-1, j-1], nac[j-1, i-1] = vector, -vector
        result = dict(h=h, grad=grad, nac=nac, dm=_dipoles(td, triplet), overlap=ovlp, phases=phases)
        timed_stage('dipoles')
        for key, array in result.items():
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f'Nonfinite {key} in electronic result')
        np.savez_compressed(savedir/f'wavefunction-{step}.npz', **_snapshot(td))
        if triplet is not None:
            np.savez_compressed(savedir/f'wavefunction-triplet-{step}.npz', **_snapshot(triplet))
        np.savez_compressed(cache, signature=signature, has_nac='nacdr' in requests, **result)
        diagnostics.update(timings_seconds=timings,
                           reference_model=reference_model,
                           reference_energy_hartree=reference_energy,
                           orbital_objective_hartree=float(mf.e_tot),
                           singlet_energies_hartree=(reference_energy+np.asarray(td.e)).tolist(),
                           triplet_energies_hartree=(reference_energy+np.asarray(triplet.e)).tolist() if triplet is not None else [reference_energy])
        diagnostic_path = savedir/f'frame-{step}.json'
        diagnostic_tmp = diagnostic_path.with_suffix('.json.tmp')
        diagnostic_tmp.write_text(json.dumps(diagnostics, indent=2)+'\n')
        diagnostic_tmp.replace(diagnostic_path)
    target = Path(output) if output else Path(qmin).with_suffix('.out')
    write_qmout(target, result, requests, time.monotonic()-start)
    print(f'Step {step}: {nroots} singlets + {ntriplets} triplets ({triplet_mode}); QM.out -> {target}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qmin')
    parser.add_argument('--config', default='nttda.json')
    parser.add_argument('--output')
    args = parser.parse_args()
    run(args.qmin, args.config, args.output)


if __name__ == '__main__':
    main()
