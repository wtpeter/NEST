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

"""相同慢溶剂环境下比较 ptSS 与自洽 SS，并测量 DIIS 的作用。

甲醛使用 CAM-B3LYP/def2-SVP；SI 二聚体使用 HF/STO-3G 识别 CT 根。
所有根编号都指同一个出发态固定场谱，SS 迭代用振幅重叠跟踪目标根。
"""

import os

threads = min(20, int(os.environ.get('OMP_NUM_THREADS', '2')))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = str(threads)

from pathlib import Path
from time import perf_counter
import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM, state_density

formaldehyde = 'C 0 0 0; O 0 0 1.221104; H 0 .934473 -.588078; H 0 -.934473 -.588078'
ct_file = Path(__file__).resolve().parent / 'geometries' / 'c2h4_c2f4_5A.xyz'
ct_geometry = '\n'.join(ct_file.read_text().splitlines()[2:])

for name, atom, basis, xc, nstates in (
    ('formaldehyde', formaldehyde, 'def2-svp', 'CAM-B3LYP', 5),
    ('CT dimer', ct_geometry, 'sto-3g', 'HF', 16),
):
    mol = gto.M(atom=atom, basis=basis, spin=2, verbose=0)
    mf = mol.ROKS(xc=xc)
    if xc != 'HF':
        mf = mf.density_fit()
        mf.grids.level = 2
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    td = mf.NTTDA().set(deltaS=-1, nstates=nstates)
    field = pcm.PCM(mol)
    field.eps = 78.3553
    field.lebedev_order = 11
    reference = None

    print(f'\n{name}: {xc}/{basis}', flush=True)
    for diis_space in (0, 6):
        started = perf_counter()
        calculation = StateSpecificPCM(td, field, eps_optical=1.333**2)
        calculation.diis_space = diis_space
        calculation.verbose = 4
        calculation.kernel(state=1)
        gaps_ptss = calculation.vertical().copy()
        frozen_td = calculation.td.copy()
        if name == 'formaldehyde':
            targets = [2]  # 最低单重激发态候选
        else:
            end = mol.aoslice_by_atom()[5, 3]  # 前 6 个原子为 C2F4
            overlap = mf.get_ovlp()
            charges = []
            for k in range(len(frozen_td.e)):
                delta = state_density(frozen_td, k+1)-calculation.dm
                charges.append(-np.trace((delta @ overlap)[:end, :end]))
            targets = [k+1 for k, charge in enumerate(charges) if abs(charge) > 0.8]
            if not targets:
                raise RuntimeError('所求根中缺少 CT 候选，请增加 nstates。')

        print(f'DIIS space={diis_space}; source cycles={calculation.cycles}', flush=True)
        ss_gaps = []
        for state in targets:
            gap_ss = calculation.state_specific(state)
            ss_gaps.append(gap_ss)
            # SS 自洽后根可能重排；这里检查目标态与初始固定场根的重叠。
            x = calculation.ss_td.xy[calculation.ss_state-1][0]
            identity = abs(np.vdot(frozen_td.xy[state-1][0], x))
            print(f'root {state}: ptSS={gaps_ptss[state-1]*HARTREE2EV:.8f} eV, '
                  f'SS={gap_ss*HARTREE2EV:.8f} eV, '
                  f'ptSS-SS={(gaps_ptss[state-1]-gap_ss)*HARTREE2EV:+.8f} eV, '
                  f'cycles={calculation.ss_cycles}, overlap={identity:.6f}', flush=True)
            assert calculation.ss_converged and identity > 0.7
        result = np.r_[calculation.e_tot, ss_gaps]
        if reference is None:
            reference = result
        else:
            np.testing.assert_allclose(result, reference, atol=1e-8, rtol=0)
        print(f'total time={perf_counter()-started:.1f} s', flush=True)
