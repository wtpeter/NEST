"""Compare iterative and explicitly assembled response equations.

    OMP_NUM_THREADS=20 conda run --no-capture-output -n nest-soc python examples/hessian/02_compare_solvers.py

Both paths use direct derivative integral contractions. Timings are for one atom's
3x3 Hessian block and exclude SCF/TD setup; they are not large-system benchmarks.
"""

import time

import numpy as np
from pyscf import dft, gto

from nest.nttda import NTTDA


for basis in ('sto-3g', '6-31g', 'cc-pvdz'):
    mol = gto.M(atom='C 0 0 0; H 1.4 0.2 1.1; H -1.2 0 1.3',
                basis=basis, spin=2, unit='Bohr', verbose=0)
    mf = dft.ROKS(mol).set(xc='HF', conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200).run()
    assert mf.converged
    td = NTTDA(mf).set(deltaS=-1, nstates=3, conv_tol=1e-10, lindep=1e-20, max_cycle=200).run()
    assert td.converged[0]
    results = {}
    for solver in ('dense', 'iterative'):
        driver = td.Hessian().set(solver=solver)
        start = time.perf_counter()
        results[solver] = driver.kernel(atmlst=[1])
        print(basis, mol.nao_nr(), 'AOs', solver,
              'seconds:', time.perf_counter() - start,
              'response residual:', driver.response_residual,
              'iterations:', driver.response_iterations, flush=True)
    error = np.max(abs(results['dense'] - results['iterative']))
    print('Maximum Hessian difference / Eh Bohr^-2:', error, flush=True)
    np.testing.assert_allclose(results['dense'], results['iterative'], atol=2e-9, rtol=0)
