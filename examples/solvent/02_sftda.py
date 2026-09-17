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

"""UKS/SF-TDA：按指定物理根自洽溶剂，检查态密度并计算 ptSS 能隙。"""

import os

threads = min(20, int(os.environ.get('OMP_NUM_THREADS', '2')))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = str(threads)

import numpy as np
from pyscf import gto
from pyscf.data.nist import HARTREE2EV
from pyscf.solvent import pcm
from nest.solvent import StateSpecificPCM, state_density

mol = gto.M(atom='C 0 0 0; O 0 0 1.221104; H 0 .934473 -.588078; H 0 -.934473 -.588078',
            basis='def2-svp', spin=2, verbose=3)
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
gaps = calculation.vertical(method='ptss') * HARTREE2EV

overlap = mf.get_ovlp()
spin_square = calculation.td.spin_square()
print('\n根   相对出发态能隙/eV    电子数       <S^2>')
for root, gap in enumerate(gaps, 1):
    density = state_density(calculation.td, root)
    electrons = np.einsum('ij,ji', density, overlap).real
    print(f'{root:2d}   {gap:14.6f}   {electrons:9.6f}   {spin_square[root-1]:9.6f}')

# UKS/SF-TDA 的根可能带有自旋混合；结合 <S^2>、振幅和跃迁性质进行态指认。
