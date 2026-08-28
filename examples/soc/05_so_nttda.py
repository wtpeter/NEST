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

"""Run direct spin-orbit-coupled NTTDA for triplet H2O2."""

import numpy as np
from pyscf import gto

from nest.soc import SONTTDA  # noqa: F401  # Registers mf.SONTTDA().


mol = gto.M(
    atom="""
    O   0.64372820   0.14077399  -0.04477253
    O  -0.64862595  -0.12779073  -0.05445498
    H   1.16027512  -0.65947800   0.36730132
    H  -1.12109306   0.55561188   0.42651873
    """,
    basis="sto-3g",
    charge=0,
    spin=2,
    symmetry=False,
    verbose=0,
)
mf = mol.ROKS(xc="SVWN").set(conv_tol=1e-12).run()

# The default deltaS=(-1, 0, 1) includes all three final-spin sectors. Each
# NTTDA amplitude space is copied for M_S=-S,...,+S. The deltaS=0 space already
# contains the reference, so direct SO-NTTDA has no include_reference option.
td = mf.SONTTDA(
    deltaS=(-1, 0, 1),
    soctype="SOMF",
).set(
    nstates=5,
    conv_tol=1e-6,
    max_cycle=300,
    verbose=0,
).run()

td.analyze(verbose=4)

# xy[root][0] is one complex vector containing every (deltaS, M_S) block;
# xy[root][1] is the zero-Y sentinel used by NEST's TDA interfaces.
print(f"\nStored amplitude length: {td.xy[0][0].size}")
print(f"All roots converged: {np.all(td.converged)}")

# A subset of spin sectors can be selected with a list or tuple, for example:
#
#     td = mf.SONTTDA(deltaS=[-1, 0], soctype="SOMF").set(nstates=10).run()
#
# If a selected sector has negative final spin, SONTTDA warns, skips that
# sector, and continues with the remaining physical sectors.
