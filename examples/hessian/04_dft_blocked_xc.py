"""Fixed-grid PBE NTTDA Hessian with default SCF/TD/grid settings.

Requires analytic fourth derivatives from the XC library (see DERIVATION.md).
Run with OMP_NUM_THREADS=20 OMP_WAIT_POLICY=PASSIVE in the NEST environment.
Both ERI derivatives and XC kernels are contracted without four-index storage.
Change xc to 'B3LYP' for a hybrid; deltaS can be -1, 0, or +1.
"""

import time

from pyscf import dft, gto
from nest.nttda import NTTDA


mol = gto.M(atom='''
O  0.64372820  0.14077399 -0.04477253
O -0.64862595 -0.12779073 -0.05445498
H  1.16027512 -0.65947800  0.36730132
H -1.12109306  0.55561188  0.42651873
''', basis='6-31g*', spin=2, symmetry=False, max_memory=4000, verbose=3)
mf = dft.ROKS(mol).set(xc='PBE').run()
assert mf.converged
td = NTTDA(mf).set(deltaS=-1).run()
assert td.converged[0]

start = time.perf_counter()
driver = td.Hessian()  # Iterative orbital and state response by default.
# This example computes the first oxygen's 3x3 diagonal block. Omit atmlst
# for the complete (4,4,3,3) Hessian; neither call includes grid response.
hessian = driver.kernel(state=1, atmlst=[0])
print('Hessian / Eh Bohr^-2:', hessian[0, 0])
print('Seconds:', time.perf_counter() - start)
print('Response residual:', driver.response_residual)
