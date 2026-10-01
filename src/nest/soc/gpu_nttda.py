"""Explicit CPU bridge from GPU NT-TDA solutions to NEST's SOC driver.

The electronic solution is copied, never recomputed. NEST evaluates SOC
integrals and reduced densities on the CPU. CUDA is an optional dependency.
"""

import numpy as np
from pyscf import dft

from nest.nttda import NTTDA
from nest.soc.nttda import SOC


def to_nest(td):
    """Copy a converged GPU NT-TDA solution into a CPU NEST NT-TDA object.

    This carrier is intended for SOC only. In particular, ensemble-reference
    gradients must still be obtained from the original GPU object.
    """
    import cupy as cp

    def host(value):
        return cp.asnumpy(value) if isinstance(value, cp.ndarray) else np.array(value, copy=True)

    if not td._scf.converged or not np.all(host(td.converged)):
        raise ValueError('SCF and all NT-TDA roots must be converged before SOC')
    if td.deltaS not in (-1, 0, 1):
        raise ValueError('Unsupported NT-TDA spin channel')
    mf = dft.ROKS(td.mol)
    mf.xc = td._scf.xc
    mf.mo_coeff = host(td._scf.mo_coeff)
    mf.mo_occ = host(td._scf.mo_occ)
    mf.mo_energy = host(td._scf.mo_energy)
    mf.e_tot = float(td._scf.e_tot)
    mf.converged = True
    mf.verbose = td.verbose
    if mf.mo_coeff.ndim != 2 or mf.mo_occ.ndim != 1:
        raise ValueError('SOC requires spatial restricted reference orbitals')
    if not np.all(np.isin(mf.mo_occ, (0, 1, 2))):
        raise ValueError('SOC requires integer closed/open/virtual occupations')
    carrier = NTTDA(mf)
    carrier.deltaS = td.deltaS
    carrier.nobeta = getattr(td, 'nobeta', False)
    carrier.e = host(td.e)
    carrier.xy = [(host(x), host(y)) for x, y in td.xy]
    carrier.nstates = len(carrier.e)
    carrier.converged = host(td.converged)
    return carrier


def from_gpu_nttda(td, *others, soctype='SOMF', include_reference=False):
    """Return a NEST SOC driver; call ``kernel()`` or ``run()`` to evaluate.

    NEST's Hamiltonian contains excitation energies relative to the common
    SCF reference. Add ``td._scf.e_tot * I`` for absolute energies in SHARC.
    """
    return SOC([to_nest(item) for item in (td, *others)],
               soctype=soctype, include_reference=include_reference)
