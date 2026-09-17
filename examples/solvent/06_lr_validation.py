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

"""在完整的小模型态空间中比较 ptLR 与耦合 LR。

水/HF/STO-3G 用于算法验证。完整空间的实部与虚部能量 Hessian
还在 test_nonequilibrium.py 中由有限差分独立检查。
最后的两能级例子人为缩小能隙间距，用来展示近简并混合。
"""

import os

threads = min(20, int(os.environ.get('OMP_NUM_THREADS', '2')))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = str(threads)

import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM
from nest.solvent.pcm import _transition_density  # 研究验证使用的内部 AO 跃迁密度

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587', basis='sto-3g', spin=2, verbose=0)
mf = mol.ROKS(xc='HF').run(conv_tol=1e-11)
# 18 维原始空间包含一个自旋旋转零模；驱动去掉它后得到 17 个物理根。
td = mf.NTTDA().set(deltaS=-1, nstates=18, conv_tol=1e-10)
field = pcm.PCM(mol)
field.eps = 10
field.lebedev_order = 7
calculation = StateSpecificPCM(td, field, eps_optical=2)
calculation.conv_tol_dm = 1e-9
calculation.kernel()
calculation.vertical(method='ptlr')
work = calculation.td
source = calculation.state-1
roots = [k for k in range(len(work.e)) if k != source]
gaps = work.e[roots]-work.e[source]
transitions = np.array([_transition_density(work, source, k).real for k in roots])
fast = field.copy().reset()
fast.eps = 2
response = fast._B_dot_x(transitions)
kernel = np.einsum('iab,jba->ij', transitions, response)


def coupled_lr(gaps, kernel):
    a = np.diag(gaps)+kernel
    b = kernel
    values, vectors = np.linalg.eig(np.block([[a, b], [-b, -a]]))
    if np.max(abs(values.imag)) > 1e-8:
        raise RuntimeError('耦合响应出现复频率，请检查所选态的稳定性。')
    selected = np.flatnonzero(values.real > 0)
    selected = selected[np.argsort(values[selected].real)]
    forward = vectors[:len(gaps), selected]
    weights = abs(forward)**2
    weights /= weights.sum(axis=0)
    return values[selected].real, weights


print('快响应缩放     最低 ptLR/eV     最低耦合 LR/eV     差值/eV       最大谱差/eV')
for strength in (1, 0.1, 0.01):
    exact, weights = coupled_lr(gaps, strength*kernel)
    perturbative = gaps+strength*kernel.diagonal()
    print(f'{strength:10.2g} {perturbative[0]*HARTREE2EV:16.9f} '
          f'{exact[0]*HARTREE2EV:18.9f} {(perturbative[0]-exact[0])*HARTREE2EV:14.9f} '
          f'{np.max(abs(np.sort(perturbative)-exact))*HARTREE2EV:14.9f}')

# 从实际核选出耦合最大的两个方向，把两个裸能级人为放到近简并位置。
i, j = np.unravel_index(np.argmax(abs(np.triu(kernel, 1))), kernel.shape)
small_kernel = kernel[np.ix_([i, j], [i, j])]
spacing = abs(small_kernel[0, 1])*0.1
center = max(gaps[i], gaps[j])
near_gaps = center+np.array([-spacing/2, spacing/2])
exact, weights = coupled_lr(near_gaps, small_kernel)
perturbative = near_gaps+small_kernel.diagonal()
print('\n人为近简并两能级模型：')
print('ptLR 的两个根/eV:', perturbative*HARTREE2EV)
print('耦合 LR 两个模/eV:', exact*HARTREE2EV)
print('耦合模前向分量在两个原始方向上的归一权重（每列一个模）:')
print(weights)
