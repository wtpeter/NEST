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

"""NT-TDA in water: equilibrate S0, then compute singlet and triplet gaps."""

from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM

mol = gto.M(
    atom="C 0 0 0; O 0 0 1.221104; H 0 .934473 -.588078; H 0 -.934473 -.588078",
    basis='def2-svp', spin=2,
)
# Converge the vacuum triplet reference; PCM updates the physical TDA states.
mf = mol.ROKS(xc='CAM-B3LYP').density_fit().run()
singlets = mf.NTTDA().set(deltaS=-1, nstates=5)

water = pcm.PCM(mol)
water.eps = 78.3553
sol = StateSpecificPCM(singlets, water, eps_optical=1.333**2)
energy = sol.kernel(state=1)  # Equilibrate both fast and slow polarization with S0.
print('S0 equilibrium energy (Hartree):', energy)
print('Singlet ptSS gaps from S0 (eV):', sol.vertical() * HARTREE2EV)

# Use the same reference to compute triplets in the S0 solvent environment.
triplets = mf.NTTDA().set(deltaS=0, nstates=5)
print('Triplet ptSS gaps from S0 (eV):', sol.vertical(triplets) * HARTREE2EV)

# Optionally relax S1 amplitudes and fast polarization, keeping the S0 slow field.
print('S1 non-equilibrium SS gap (eV):', sol.state_specific(state=2) * HARTREE2EV)
