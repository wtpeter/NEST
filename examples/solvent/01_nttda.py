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

"""甲醛：在同一个 S0 溶剂环境中计算单重态和三重态垂直能隙。"""

import os

threads = min(20, int(os.environ.get('OMP_NUM_THREADS', '2')))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = str(threads)

from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM

mol = gto.M(
    atom='''
    C  0.000000  0.000000  0.000000
    O  0.000000  0.000000  1.221104
    H  0.000000  0.934473 -0.588078
    H  0.000000 -0.934473 -0.588078
    ''',
    basis='def2-svp', spin=2, verbose=3,
)
mf = mol.ROKS(xc='CAM-B3LYP').density_fit()
mf.grids.level = 2
mf.conv_tol = 1e-10
mf.kernel()
assert mf.converged

# 三重态参考，deltaS=-1 的物理根为单重态。这里以最低根作为 S0。
singlets = mf.NTTDA().set(deltaS=-1, nstates=5)
field = pcm.PCM(mol)
field.method = 'C-PCM'
field.eps = 78.3553
field.lebedev_order = 11  # 演示网格；正式计算需增加网格并检查能量收敛。
calculation = StateSpecificPCM(singlets, field, eps_optical=1.333**2)
calculation.verbose = 4
energy = calculation.kernel(state=1)
print(f'\nS0 平衡溶剂模型总能量: {energy:.10f} Hartree')
print(f'溶剂自洽轮数: {calculation.cycles}')

print('\nS0 环境中的单重态垂直能隙 / eV')
for method in ('frozen', 'ptss', 'ptlr'):
    gaps = calculation.vertical(method=method) * HARTREE2EV
    print(f'{method:>6}:', ' '.join(f'{value:9.5f}' for value in gaps))
print('ptLR 是实验性的孤立根一阶响应近似；当前验证限于固定轨道模型内部。')

# 同一个 mf 保证两个自旋分支使用共同轨道和能量零点。
triplets = mf.NTTDA().set(deltaS=0, nstates=5)
gaps = calculation.vertical(triplets, method='ptss') * HARTREE2EV
print('\nS0 环境中的三重态垂直能隙 / eV')
for root, gap in enumerate(gaps, 1):
    print(f'triplet root {root}: {gap:9.5f}')

# 要研究 T1 平衡环境，可另建驱动：
# triplet_environment = StateSpecificPCM(triplets, field, eps_optical=1.333**2)
# triplet_environment.kernel(state=1)
# gaps_from_t1 = triplet_environment.vertical(singlets)
