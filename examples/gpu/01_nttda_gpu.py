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

"""GPU NT-TDA energies, analytic gradients, and CPU/GPU conversion."""

from pyscf import gto
from nest.gpu.nttda import NTTDA

mol = gto.M(
    atom="""
    H  0.000000  0.934473 -0.588078
    H  0.000000 -0.934473 -0.588078
    C  0.000000  0.000000  0.000000
    O  0.000000  0.000000  1.221104
    """,
    basis="6-31g",
    spin=2
)
mf = mol.ROKS(xc='CAM-B3LYP').to_gpu().density_fit()  # currently, aocscf is not supported
mf.kernel()

td = NTTDA(mf).set(deltaS=-1, nstates=3).run()
print('Excitation energies / Eh:', td.e)
print('Total gradient / Eh/Bohr:', td.Gradients().kernel(state=1))

# Both directions preserve the reference settings and converged amplitudes.
td_cpu = td.to_cpu()
td_gpu = td_cpu.to_gpu()
print('Converted energy / Eh:', td_gpu.e_tot)
