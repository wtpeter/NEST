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

"""Compare NT-TDA vacuum, water ptSS and water SS gaps at fixed geometries.

Water-minus-vacuum shifts: positive means blue, negative means red.
Experimental reference values:
  acetone n -> pi*: +1785 cm^-1 = +0.22131 eV (band-envelope center, 293 K).
    Renge, J. Phys. Chem. A 2009, 113, 10678-10686.
    https://doi.org/10.1021/jp9033886
  p-nitroaniline bright pi -> pi*: -0.98 eV (4.24 eV vapor -> 3.26 eV water),
    quoted in Table 2 of Sok et al., J. Phys. Chem. A 2011, 115, 9801-9809.
    https://doi.org/10.1021/jp2045564
These experimental band shifts include solvent structure and nuclear motion.
Here each molecule has one fixed geometry and the solvent is a continuum.
Both ptSS and SS keep the slow polarization equilibrated with S0.

Computed shifts with the settings below (eV):
                   acetone             p-nitroaniline
PCM method       ptSS        SS         ptSS        SS
C-PCM         +0.26634   +0.26422     -0.63118   -0.64476
COSMO         +0.27324   +0.27197     -0.57393   -0.58305
IEF-PCM       +0.27529   +0.27396     -0.59951   -0.61049
SS(V)PE       +0.26889   +0.26759     -0.58530   -0.59633
The script runs C-PCM by default; set water.method to select another row.
"""

import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM

for molecule in ('acetone', 'p-nitroaniline'):
    mol = gto.M(atom=f'geometries/{molecule}.xyz', basis='def2-svp', spin=2)
    mf = mol.ROKS(xc='CAM-B3LYP').density_fit().run()
    td = mf.NTTDA().set(deltaS=-1, nstates=5)
    td.kernel()
    # All roots are singlets. Root 1 is S0; e[k] is relative to the SCF reference.
    vacuum_gaps = td.e - td.e[0]
    if molecule == 'acetone':
        target = 1  # S1: the lowest n -> pi* excitation (zero-based index).
        expected = 'blue shift (positive)'
    else:
        target = int(np.argmax(td.oscillator_strength(ref=1))) + 1
        expected = 'red shift (negative)'

    water = pcm.PCM(mol)
    # water.method = 'IEF-PCM'  # Also accepts 'COSMO' and 'SS(V)PE'; default is 'C-PCM'.
    water.eps = 78.3553
    sol = StateSpecificPCM(td, water, eps_optical=1.333**2)
    sol.kernel(state=1)
    ptss_gaps = sol.vertical()

    # Match the same electronic state, allowing its energy ordering to change.
    overlaps = [abs(np.vdot(td.xy[target][0], x)) for x, _ in sol.td.xy]
    matched = int(np.argmax(overlaps))
    vacuum_ev = vacuum_gaps[target] * HARTREE2EV
    ptss_ev = ptss_gaps[matched] * HARTREE2EV
    ptss_x = sol.td.xy[matched][0]

    # Relax the same optical target's amplitudes and fast polarization.
    ss_ev = sol.state_specific(state=matched+1, follow_state=True) * HARTREE2EV
    ss_overlap = abs(np.vdot(ptss_x, sol.ss_td.xy[sol.ss_state-1][0]))
    print(f'\n{molecule}: expected {expected}')
    print(f'Vacuum root {target+1} -> water root {matched+1}; overlap = {overlaps[matched]:.4f}')
    print(f'SS root {sol.ss_state}; overlap with the ptSS state = {ss_overlap:.4f}')
    print(f'Vacuum gap = {vacuum_ev:.5f} eV')
    print(f'Water ptSS gap = {ptss_ev:.5f} eV; shift = {ptss_ev-vacuum_ev:+.5f} eV')
    print(f'Water SS gap   = {ss_ev:.5f} eV; shift = {ss_ev-vacuum_ev:+.5f} eV')
