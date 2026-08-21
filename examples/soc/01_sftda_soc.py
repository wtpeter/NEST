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

"""Run an SF-TDA SOC calculation and print the complete SOC analysis."""

from pyscf import gto
from nest import sftda  # noqa: F401  # Registers the SFTDA methods.

mol = gto.M(
    atom="""
    C  -0.14368842   0.50601076   0.00000016
    C   1.23930312  -0.17143621  -0.00000011
    C  -1.26848863  -0.24989368   0.00000010
    H   1.55960644  -0.32833431   1.00880551
    H   1.94392608   0.45624549  -0.50440288
    H   1.17528653  -1.11291349  -0.50440313
    H  -0.21651060   1.57352982   0.00000038
    H  -1.19566646  -1.31741273   0.00000059
    H  -2.22939834   0.22080000  -0.00000137
    """,
    basis="631g",
    charge=0,
    spin=2,
    symmetry=False,
)

# 1. High-spin scalar reference.  Both ROKS and UKS references are supported:
#
#     mf = mol.ROKS(xc="bhandhlyp").run()
#     mf = mol.UKS(xc="bhandhlyp").run()
#
# This example uses UKS and matches the Q-Chem BHHLYP calculation below.
mf = mol.UKS(xc="bhandhlyp").set(conv_tol=1e-12).run()

# 2. Spin-flip-down excited states.  Both SF-TDA and SF-TDDFT are supported:
#
#     td = mf.SFTDA()
#     td = mf.SFTDDFT()
#
# The XC response can use either collinear="col" or multicollinear="mcol".
# This example uses collinear SF-TDA to match Q-Chem.
# SOC currently supports extype=1 only.
td = mf.SFTDA().set(
    extype=1,
    nstates=3,
    collinear="col",
    conv_tol=1e-6,
    max_cycle=300,
).run()

# 3. Build and diagonalize the SOC Hamiltonian.
#
# The PySCF-style entry point below is recommended.  run() calls kernel() and
# returns the SOC driver itself, with the numerical results stored as
# attributes.
soc_driver = td.SOC(soctype="SOMF").run()
#
# Equivalent explicit construction:
#
#     from nest.soc import SFTDASOC
#     soc_driver = SFTDASOC(td, soctype="SOMF").run()
#
# Empty construction is also supported.  The TD object and all options are read
# when run()/kernel() starts, so attributes can be changed before calculation:
#
#     soc_driver = SFTDASOC()
#     soc_driver.tdobj = td
#     soc_driver.soctype = "SOMF"
#     e, v = soc_driver.kernel()
#
# The same setup can be written with the StreamObject interface:
#
#     soc_driver = SFTDASOC().run(tdobj=td, soctype="SOMF")
#
# The first argument is the converged SF-TDA/SF-TDDFT object ``td``, not the
# SCF object ``mf``.  Available soctype values are:
#
#   "SOMF"           one-electron SOC plus the two-electron SOMF contribution
#   "SOMF_AMFI"      one-electron SOC plus the one-center AMFI approximation to SOMF
#   "Zeff"           one-electron SOC with ORCA-style effective nuclear charges
#   "1e"             bare one-electron nuclear SOC
#   "X2C1E"          spin-dependent part of PySCF's spinor X2C1e Hamiltonian
#   "X2CAMF"         X2CAMF SOC; requires the socutils package
#   "X2CMP"           X2C molecular picture-change SOC; requires the socutils package

# 4. Print scalar states, SOC blocks, SOCCs, SOC energies, and compositions.
# verbose=6 enables the detailed block matrices and eigenstate compositions.
soc_driver.analyze(verbose=6)

# The numerical results remain available for further processing:
# soc_driver.states        spin-free input states
# soc_driver.state_slices  mapping from spin-free states to Hamiltonian slices
# soc_driver.h_soc         complex Hermitian SOC Hamiltonian, in Hartree
# soc_driver.e, .v         SOC eigenvalues and eigenvectors


r"""
Q-Chem input
============

$molecule
0 3
 C                 -0.14368842    0.50601076    0.00000016
 C                  1.23930312   -0.17143621   -0.00000011
 C                 -1.26848863   -0.24989368    0.00000010
 H                  1.55960644   -0.32833431    1.00880551
 H                  1.94392608    0.45624549   -0.50440288
 H                  1.17528653   -1.11291349   -0.50440313
 H                 -0.21651060    1.57352982    0.00000038
 H                 -1.19566646   -1.31741273    0.00000059
 H                 -2.22939834    0.22080000   -0.00000137
$end

$rem
   METHOD               BHHLYP
   BASIS                6-31G
   PRINT_ORBITALS       20
   SPIN_FLIP            TRUE
   CIS_N_ROOTS          3
   CIS_CONVERGENCE      8
   MAX_SCF_CYCLES       600
   MAX_CIS_CYCLES       100
   SCF_ALGORITHM        DIIS
   MEM_STATIC           300
   MEM_TOTAL            4000
   SYMMETRY             FALSE
   SYM_IGNORE           TRUE
   RPA                  FALSE
   CALC_SOC             2
   SET_ITER             300
$end


Q-Chem 6.2.1 output
            *********SPIN-ORBIT COUPLING JOB USING WIGNER–ECKART THEOREM BEGINS HERE*********

**************************************************
 State A: Root 1
 State B: Root 2

 Analysing Sz ans S^2 of the pair of states...
 Ket state:  Computed S^2 = 0.013047662 will be treated as 0.000000000 Sz = 0.000000000
 Bra state:  Computed S^2 = 2.017199406 will be treated as 2.000000000 Sz = 0.000000000
 Clebsh-Gordan coefficient: <0.000,0.000;1.000,0.000|1.000,0.000> = 1.000

_______________________________
 One-electron SO (cm-1)
 Reduced matrix elements:
 <S|| Hso(L-) ||S'> = (-0.023616,-0.105439)
 <S|| Hso(L0) ||S'> = (0.000000,0.334749)
 <S|| Hso(L+) ||S'> = (-0.023616,0.105439)

 SOCC = 0.367977

 Actual matrix elements:
       |Sz=0.00>
 <Sz=-1.00|(-0.023616,0.105439)
 <Sz=0.00|(0.000000,0.334749)
 <Sz=1.00|(-0.023616,-0.105439)
_______________________________
 Mean-field SO (cm-1)
 Reduced matrix elements:
 <S|| Hso(L-) ||S'> = (-0.003641,-0.061712)
 <S|| Hso(L0) ||S'> = (0.000000,-0.005961)
 <S|| Hso(L+) ||S'> = (-0.003536,0.061711)

 Singlet part of <S|| Hso(L0) ||S'> = (-0.000000,-0.000081) (excluded from all matrix elements)
 L-/L+ Averaged reduced matrix elements:
 <S|| Hso(L-) ||S'> = (-0.003588,-0.061711)
 <S|| Hso(L+) ||S'> = (-0.003588,0.061711)

 SOCC = 0.087624

 Actual matrix elements:
       |Sz=0.00>
 <Sz=-1.00|(-0.003588,0.061711)
 <Sz=0.00|(0.000000,-0.005961)
 <Sz=1.00|(-0.003588,-0.061711)
_______________________________

**************************************************
 State A: Root 2
 State B: Root 3

 Analysing Sz ans S^2 of the pair of states...
 Ket state:  Computed S^2 = 2.017 will be treated as 2.000 Sz = 0.000
 Bra state:  Computed S^2 = 0.048 will be treated as 0.000 Sz = 0.000
 Clebsh-Gordan coefficient: <1.000,0.000;1.000,0.000|0.000,0.000> = 0.577

_______________________________
 One-electron SO (cm-1)
 Reduced matrix elements:
 <S|| Hso(L-) ||S'> = (-0.231553,0.243679)
 <S|| Hso(L0) ||S'> = (0.000000,0.078807)
 <S|| Hso(L+) ||S'> = (-0.231553,-0.243679)

 SOCC = 0.278210

 Actual matrix elements:
       |Sz=-1.00>           |Sz=0.00>           |Sz=1.00>
 <Sz=0.00|(0.133687,-0.140688)(0.000000,0.045499)(0.133687,0.140688)
_______________________________
 Mean-field SO (cm-1)
 Reduced matrix elements:
 <S|| Hso(L-) ||S'> = (-0.111334,0.143864)
 <S|| Hso(L0) ||S'> = (0.000000,-0.004524)
 <S|| Hso(L+) ||S'> = (-0.114090,-0.144744)

 Singlet part of <S|| Hso(L0) ||S'> = (0.000000,0.002660) (excluded from all matrix elements)
 L-/L+ Averaged reduced matrix elements:
 <S|| Hso(L-) ||S'> = (-0.112712,0.144304)
 <S|| Hso(L+) ||S'> = (-0.112712,-0.144304)

 SOCC = 0.149528

 Actual matrix elements:
       |Sz=-1.00>           |Sz=0.00>           |Sz=1.00>
 <Sz=0.00|(0.065074,-0.083314)(0.000000,-0.002612)(0.065074,0.083314)
_______________________________

            *********SPIN-ORBIT COUPLING JOB ENDS HERE*********
"""
