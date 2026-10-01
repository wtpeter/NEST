"""Run on a CUDA worker with the gpu4pyscf NT-TDA branch installed."""

from pyscf import gto
from gpu4pyscf.dft.roks import ROKS
from gpu4pyscf.sftda import NTTDA_ROKS
from nest.soc.gpu_nttda import from_gpu_nttda

mol = gto.M(atom='C .02 -.03 .01; H -.02 .8 .62; H .03 -.91 .5',
            basis='sto-3g', spin=2, verbose=0)
mf = ROKS(mol, xc='PBE').set(conv_tol=1e-12)
mf.grids.level = 0  # Small integration example; use converged grids for research.
mf.run()
td = NTTDA_ROKS(mf).set(deltaS=-1, nstates=3, conv_tol=1e-10).run()
soc = from_gpu_nttda(td, soctype='SOMF', include_reference=True).run()
soc.analyze(verbose=4)
# soc.h_soc contains relative energies. SHARC needs mf.e_tot on the diagonal.
