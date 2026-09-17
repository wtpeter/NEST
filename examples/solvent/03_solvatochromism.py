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

"""SF-TDA 定性验证：丙酮 n→π* 蓝移与对硝基苯胺 π→π* 红移。

使用固定几何和水的连续介质参数；先识别气相态，再用振幅重叠跟踪溶液态。
打印实际计算结果。实验的氢键、构型采样等贡献属于进一步的模型验证。
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
from nest.solvent import StateSpecificPCM

geometry_folder = Path(__file__).resolve().parent / 'geometries'
for molecule in ('acetone', 'p-nitroaniline'):
    started = perf_counter()
    geometry = (geometry_folder / f'{molecule}.xyz').read_text().splitlines()[2:]
    mol = gto.M(atom='\n'.join(geometry), basis='def2-svp', spin=2, verbose=0)
    mf = mol.UKS(xc='CAM-B3LYP').density_fit()
    mf.grids.level = 2
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged

    td = mf.TDA_SF().set(extype=1, collinear='mcol', collinear_samples=20, nstates=5)
    field = pcm.PCM(mol)
    field.eps = 78.3553
    field.lebedev_order = 11
    calculation = StateSpecificPCM(td, field, eps_optical=1.333**2)
    calculation.verbose = 4
    calculation.kernel(state=1)
    solution_gaps = calculation.vertical(method='ptss')

    gas = calculation.gas_td
    gas_gaps = gas.e - gas.e[0]
    spin_square = gas.spin_square()
    strengths = gas.oscillator_strength(ref=1)
    print(f'\n{molecule}: 气相根的指认')
    for k in range(1, len(gas.e)):
        print(f'root {k+1}: gap={gas_gaps[k]*HARTREE2EV:8.4f} eV, '
              f'<S^2>={spin_square[k]:7.4f}, f={strengths[k-1]:.6f}')

    singlet_roots = [k for k in range(1, len(gas.e)) if spin_square[k] < 0.5]
    if not singlet_roots:
        raise RuntimeError('所求根中缺少单重态候选，请增大 nstates 并检查振幅。')
    if molecule == 'acetone':
        target = min(singlet_roots, key=lambda k: gas_gaps[k])  # 最低单重激发态，n→π* 候选
    else:
        target = max(singlet_roots, key=lambda k: strengths[k-1])  # 强吸收 π→π* 候选

    overlaps = np.array([abs(np.vdot(gas.xy[target][0], x)) for x, _ in calculation.td.xy])
    matched = int(overlaps.argmax())
    if overlaps[matched] < 0.7:
        raise RuntimeError('溶液态混合较强，请检查更多根的振幅和电荷分布后进行态指认。')
    gas_ev = gas_gaps[target] * HARTREE2EV
    solution_ev = solution_gaps[matched] * HARTREE2EV
    shift = solution_ev - gas_ev
    print(f'选定气相 root {target+1} → 溶液 root {matched+1}，重叠={overlaps[matched]:.6f}')
    print(f'气相 {gas_ev:.5f} eV，水模型 ptSS {solution_ev:.5f} eV')
    print(f'位移 {shift:+.5f} eV（正值为蓝移，负值为红移）')
    print(f'自洽 {calculation.cycles} 轮，总耗时 {perf_counter()-started:.1f} s', flush=True)
