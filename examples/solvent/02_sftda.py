#!/usr/bin/env python
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

"""SF-TDA in water: equilibrium solvation and vertical ptSS energy gaps."""

from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM

mol = gto.M(
    atom="C 0 0 0; O 0 0 1.221104; H 0 .934473 -.588078; H 0 -.934473 -.588078",
    basis='def2-svp', spin=2,
)
mf = mol.UKS(xc='CAM-B3LYP').density_fit().run()  # Vacuum triplet reference.
td = mf.TDA_SF().set(extype=1, nstates=5)

water = pcm.PCM(mol)
water.eps = 78.3553
sol = StateSpecificPCM(td, water, eps_optical=1.333**2)
energy = sol.kernel(state=1)  # The lowest physical root defines the solvent environment.
gaps = sol.vertical() * HARTREE2EV  # ptSS is the default; gaps are measured from root 1.

print('Root 1 equilibrium energy (Hartree):', energy)
print('Root    ptSS gap (eV)    <S^2>')
for root, (gap, spin_square) in enumerate(zip(gaps, sol.td.spin_square()), 1):
    print(f'{root:4d}    {gap:13.6f}    {spin_square:.6f}')
