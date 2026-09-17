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

"""文章 SI 的 C2H4...C2F4：用片段电荷识别 CT 根，再比较 ptSS 位移。

使用 HF 泛函和小基组进行态搜索。SI 几何的间距是 5 Å。
与文章不同的电子结构方法在此作为趋势对照；定量研究应继续检查基组和泛函。
"""

import os

threads = min(20, int(os.environ.get('OMP_NUM_THREADS', '2')))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = str(threads)

from pathlib import Path
import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM, state_density

path = Path(__file__).resolve().parent / 'geometries' / 'c2h4_c2f4_5A.xyz'
mol = gto.M(atom='\n'.join(path.read_text().splitlines()[2:]), basis='sto-3g', spin=2, verbose=3)
mf = mol.ROKS(xc='HF').run(conv_tol=1e-10)
assert mf.converged

# CT 态可能高于最低几个局域激发态，因此这个专门的态搜索例子求更多根。
td = mf.NTTDA().set(deltaS=-1, nstates=16)
field = pcm.PCM(mol)
field.eps = 78.3553
field.lebedev_order = 11
calculation = StateSpecificPCM(td, field, eps_optical=1.333**2)
calculation.verbose = 4
calculation.kernel(state=1)
gaps = calculation.vertical()
gas = calculation.gas_td

# 坐标文件前 6 个原子是 C2F4。Mulliken 电荷用于这里的片段定性指认。
fragment_ao_end = mol.aoslice_by_atom()[5, 3]
fragment_nuclear_charge = mol.atom_charges()[:6].sum()
overlap = mf.get_ovlp()
ground = state_density(calculation.td, calculation.state)
ground_charge = fragment_nuclear_charge - np.trace((ground @ overlap)[:fragment_ao_end, :fragment_ao_end])

print('\nroot   溶液能隙/eV   C2F4电荷变化/e   对应气相根   振幅重叠   位移/eV')
for k, (x, _) in enumerate(calculation.td.xy):
    density = state_density(calculation.td, k + 1)
    charge = fragment_nuclear_charge - np.trace((density @ overlap)[:fragment_ao_end, :fragment_ao_end])
    matches = np.array([abs(np.vdot(x, y)) for y, _ in gas.xy])
    matched = int(matches.argmax())
    shift = (gaps[k] - (gas.e[matched] - gas.e[0])) * HARTREE2EV
    print(f'{k+1:3d}   {gaps[k]*HARTREE2EV:12.5f}   {(charge-ground_charge).real:14.5f}'
          f'   {matched+1:10d}   {matches[matched]:8.4f}   {shift:+9.5f}')

# 接近一个电子的片段电荷变化是 CT 态的重要指征。
# 对重叠较低的根，先检查态混合和相邻根，再解释溶剂位移。
