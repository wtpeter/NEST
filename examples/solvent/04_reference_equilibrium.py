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

"""
The solvent is in equilibrium with the triplet reference, and the
excited states are calculated in a frozen way.

ORCA input file:
! UKS PBE0 DefGrid3 def2-SVP CPCM TightSCF NoAutoStart NoRI
%tddft
  LRCPCM true
  CPCMEQ false
  OrbWin[0] = 0,-1,-1,-1
  OrbWin[1] = 0,-1,-1,-1
  SF true
  TDA true
  NRoots 5
end
* xyz 0 3...*
%cpcm
 epsilon 78.3553
 refrac 1.333
end
"""

from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from nest import sftda

mol = gto.M(
    atom='C 0 0 0; O 0 0 1.221104; H 0 .934473 -.588078; H 0 -.934473 -.588078',
    basis='def2-svp', spin=2
)
mf = mol.UKS(xc='PBE0').PCM()
mf.with_solvent.eps = 78.3553
mf.kernel()
td = sftda.TDA_SF(mf, extype=1, collinear='col').set(nstates=5).run()

# Energies relative to the triplet reference, matching ORCA's STATE output.
orca = [-2.774712, 1.518905, 1.899950, 4.852406, 5.697502]
print('Root    NEST (eV)    ORCA (eV)')
for root, (energy, reference) in enumerate(zip(td.e * HARTREE2EV, orca), 1):
    print(f'{root:4d}    {energy:9.6f}    {reference:9.6f}')
