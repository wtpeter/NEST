# GPU NT-TDA / NEST SOC / SHARC overlap example

NEST supplies the GPU-to-CPU SOC bridge, the SHARC file interface, and the
two-spin-sector overlap implementation. SCF, NT-TDA energies and gradients
run in gpu4pyscf on GPU; SOC and dipoles run in NEST on CPU. The default
overlap contractions run on GPU after CPU evaluation of the cross-geometry
AO overlap. SHARC reads the resulting Hamiltonian, gradients, dipoles and
overlaps from `QM.out` and propagates the trajectory.

## Install the matching code

Use this NEST branch and the following GPU dependency:

| Repository | Branch | Validated revision |
| --- | --- | --- |
| [NonDFT/NEST](https://github.com/NonDFT/NEST) | `feat/gpu-nttda-sharc` | This branch |
| [jingjjkkyj/gpu4pyscf-nttda](https://github.com/jingjjkkyj/gpu4pyscf-nttda) | `feat/nttda-gpu` | [`622be27b7ebd8d07d931805e4337bfb59149cf4a`](https://github.com/jingjjkkyj/gpu4pyscf-nttda/commit/622be27b7ebd8d07d931805e4337bfb59149cf4a) |

The GPU source used by this example has no local modifications: all 991
tracked files match that upstream revision, and all 323 installed Python
files match the source. No additional GPU patch or fork is needed. The GPU
branch has advanced since that revision; the interfaces and limitations
documented below refer to the pinned revision. A standard PyPI gpu4pyscf
installation does not supply this NT-TDA implementation.

The tested environment uses Linux, Python 3.11, CUDA Toolkit 12.6, an NVIDIA
A100, PySCF 2.14.0, NumPy 2.4.6, SciPy 1.17.1, CuPy 13.6.0 and cuTENSOR 2.2.0.
The source build needs `nvcc` on `PATH`, CMake, a CUDA-compatible C/C++ compiler,
and CUTLASS 3.4.0. Run these commands from a new working directory:

```bash
mkdir -p src files/tmp files/wheels
python3.11 -m venv envs/nttda
source envs/nttda/bin/activate
export TMPDIR="$PWD/files/tmp"

git clone --branch feat/gpu-nttda-sharc https://github.com/NonDFT/NEST.git src/NEST
git clone --branch feat/nttda-gpu https://github.com/jingjjkkyj/gpu4pyscf-nttda.git src/gpu4pyscf-nttda
git -C src/gpu4pyscf-nttda checkout 622be27b7ebd8d07d931805e4337bfb59149cf4a
git clone --depth 1 --branch v3.4.0 https://github.com/NVIDIA/cutlass.git src/cutlass-v3.4.0

python -m pip install 'setuptools>=69' wheel cmake \
  pyscf==2.14.0 numpy==2.4.6 scipy==1.17.1 \
  cupy-cuda12x==13.6.0 cutensor-cu12==2.2.0 gpu4pyscf-libxc-cuda12x==0.8.1
CMAKE_CONFIGURE_ARGS="-DCUTLASS_SRC_DIR=$PWD/src/cutlass-v3.4.0 -DCUDA_ARCHITECTURES=70-real;80-real;90-real;90-virtual" \
  python -m pip wheel --no-build-isolation --no-deps ./src/gpu4pyscf-nttda --wheel-dir files/wheels
python -m pip install files/wheels/gpu4pyscf_cuda12x-*.whl
python -m pip install -e ./src/NEST
```

The explicit CUTLASS path overrides a machine-specific default in the GPU
fork. Do not install its old `requirements.txt` on top of this environment:
it pins PySCF 2.8.0, whereas NEST requires PySCF >=2.13.0. The CUDA architecture
list above matches the existing build; adapt it if targeting another GPU.
NEST is editable, while gpu4pyscf is installed from the built wheel. Changes
to the GPU source require rebuilding and reinstalling that wheel.

On the GPU node, verify both the CUDA device and the actual import paths:

```bash
python - <<'PY'
import cupy as cp
import pyscf, gpu4pyscf, nest
from gpu4pyscf.sftda import NTTDA_ROKS
from nest.soc.gpu_nttda import from_gpu_nttda
from nest.interfaces import sharc_nttda
print('CUDA devices:', cp.cuda.runtime.getDeviceCount())
for module in (pyscf, gpu4pyscf, nest, sharc_nttda):
    print(module.__name__, module.__file__)
PY
```

Install and compile SHARC 4 separately using its installation instructions.
The commands below assume its source/install tree is `src/sharc4`; adjust
that path for your installation. Apply the control fixes in the next section
if your SHARC source is affected. WFoverlap is optional: the default `gpu`
overlap backend does not use its executable.

## SHARC control fixes

Some SHARC4 sources leave `ctrl%army_ants` uninitialized. The integrator only
evaluates hopping when this flag is 0 or 1; another integer silently skips both
hopping branches while the trajectory can finish normally. The accompanying
`sharc4-hopping-control.patch` initializes ordinary dynamics to 0 and explicitly
rejects invalid restart values. Apply it only to affected sources, rebuild the
`sharc` target, and install the new binary atomically. From the workspace:

```bash
patch -d src/sharc4 -p1 < src/NEST/examples/sharc/sharc4-hopping-control.patch
make -C src/sharc4/source -j4 sharc
install -m755 src/sharc4/source/sharc1.x src/sharc4/bin/sharc.x.next
mv src/sharc4/bin/sharc.x.next src/sharc4/bin/sharc.x
envs/nttda/bin/python src/NEST/examples/sharc/validate_hopping_control.py \
  --sharc src/sharc4/bin/sharc.x --output files/verification/sharc-hopping-control
```

The CPU synthetic regression checks 3S3T/5S3T/6S3T initialization, nonzero
random draws/hopping probabilities, and rejection of a corrupted restart flag.
It also checks that a valid restart requests dipoles. Legacy restart files do
not store `calc_dipole`; the patch reconstructs the ordinary dipole mode, or
the higher-multipole mode selected by saved laser flags, without changing the
positional restart format. An undefined value can omit `DM` from `QM.in` while
the engine attempts to read dipoles and fails.
It does not validate molecular energies, forces, SOC or physical yields.
Audit existing `restart.ctrl` files before interpreting affected runs; the
`army_ants` value is five lines after the `! hopping_procedure` marker. Invalid
runs must be recalculated from their initial conditions. Resetting their flag
cannot recover omitted hopping. Independent runs can also return opposite AWF
signs: align SOC/dipole matrices using same-frame AWF overlaps, and align both
endpoints when comparing cross-step overlaps or restart results.

## Run the small trajectory

From the working directory containing `src/` and `envs/`, activate the same
Python environment in the GPU job that starts SHARC. `runQM.sh` inherits that
environment and invokes its `python`:

```bash
source envs/nttda/bin/activate
source src/sharc4/bin/sharcvars.sh
export PATH="$SHARC:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
export TMPDIR="$PWD/files/tmp"
mkdir -p "$TMPDIR" files/trajectories/ch2-smoke/QM
cp src/NEST/examples/sharc/{input,geom,veloc} files/trajectories/ch2-smoke/
cp src/NEST/examples/sharc/{runQM.sh,nttda.json} files/trajectories/ch2-smoke/QM/
(cd files/trajectories/ch2-smoke && "$SHARC/sharc.x" input)
```

Use a fresh trajectory directory for each new run. Successful completion
produces the initial frame and two 0.1 fs steps in `output.lis`/`output.dat`.
Electronic logs are in `QM/QM.log` and `QM/QM.err`; saved wavefunctions,
per-frame diagnostics and timings are under `restart/nttda/` for this input.
SHARC invokes `QM/runQM.sh` with `/bin/sh`. Each call starts a Python process;
the interface saves previous wavefunctions and same-step results to disk.

This CH2 / PBE / STO-3G example computes three deltaS=-1 singlet roots and
three deltaS=0 triplet roots from a shared ordinary GPU ROKS reference, with
density fitting and CPU SOMF SOC. It propagates 12 spin components using
`coupling overlap`, `nogradcorrect`, and momentum adjustment `parallel_vel`.
No NAC is requested or evaluated. Grid level 0 and 0.2 fs are integration
checks; they are not suitable convergence settings for production dynamics.
Use `select_directly`: the adapter evaluates all requested states' gradients
and caches the result, rather than implementing SHARC's gradient-selection
subsequent-call protocol.

For another molecule, replace `geom` and `veloc`, set charge, state counts
and initial state in `input`, and set basis, XC functional and grid in
`QM/nttda.json`. The example's `ezero` belongs to CH2; update it to the desired
energy reference for the new system. Keep the coupling/phase options below
consistent with this interface. The current reference must have exactly two
open orbitals; this is not a general interface for arbitrary multiplicities.

## Interface conventions and options

The interface is `nest.interfaces.sharc_nttda`. Set SHARC's
`nstates Ns 0 Nt`; both Ns and Nt must be positive. For a closed-shell molecule,
the lowest singlet root is S0, so five singlets represent S0 through S4;
five triplet roots represent T1 through T5. `triplets: "nttda"` is the default.
`triplets: "reference"` retains the older one-triplet-reference path, including
its optional singlet-only DDR mode. Excited triplets require overlap coupling.

The GPU-to-CPU bridge is `nest.soc.gpu_nttda.from_gpu_nttda`; it copies
converged amplitudes/orbitals explicitly, without re-solving the SCF or TD
problem. NEST SOC stays on CPU; GPU SCF, both NT-TDA sectors and their analytic
gradients support optional density fitting. Initial DIIS is followed by GPU
Newton if needed. Later frames project the previous MOs into the new AO basis,
orthonormalize them and use fixed-occupation Newton directly. This avoids
Aufbau/DIIS switching ROKS reference solutions. `min_reference_overlap` (0.9)
stops discontinuous reference changes before propagation. SOMF is the default
SOC model; `soctype: "Zeff"` is also supported.
`reference_overlap_action: "stop"` is the default; the explicit `"warn"`
option records a discontinuity and continues for diagnostic runs. It does
not restore reference continuity.
NT-TDA excitation energies are shifted by the selected reference energy.
`scf_conv_tol_grad` defaults to 1e-8; the Newton inner AH tolerance is tightened
consistently with that orbital residual. For a controlled initial SCF guess,
`initial_reference` accepts a wavefunction snapshot from this interface,
resolved relative to the configuration file. Its orbitals are projected and
relaxed with fixed occupations; the initial overlap matrix remains identity.
The atom order, electron count and spin must agree. An initial guess does not
replace convergence checks or enforce a chosen electronic state.

The optional `cphf_max_cycle` sets the iteration ceiling for singlet and
triplet gradient responses, including the EnsembleROKS selected-reference
response. Omitting it preserves the GPU drivers' defaults. The NT-TDA
response default depends on the number of orbital pairs, so a fixed value
such as 500 can lower that default for a larger molecule. This option changes
only the iteration ceiling; it does not change `cphf_conv_tol`.

`reference_model: "roks"` is the default. The experimental
`reference_model: "ensemble_roks"` instead uses GPU `EnsembleROKS` and
`NTTDA_EnsembleROKS` in both spin sectors, with exactly two open orbitals.
This is fixed average-occupation orbital optimization, not state-weighted
CASSCF/GOK optimization. It changes the electronic model and its spectrum;
it is not a numerical recovery of the ordinary ROKS solution. Its NT-TDA
energy zero is the high-spin ROKS energy on the ensemble orbitals,
`td.reference_energy()`, rather than the ensemble objective `mf.e_tot`.
The GPU gradients include that selected reference's nonstationary orbital
response. SOC and overlap use the same ensemble orbitals/amplitudes.
Use this option only with `triplets: "nttda"` and overlap coupling;
the single-reference-triplet DDR path is not implemented for it.

The default `overlap_backend: "gpu"` uses exact spin-adapted contractions in
`nest.nttda.overlap_contraction`. The cross-geometry AO metric is evaluated
with CPU PySCF/libcint and uploaded once; MO transformations, amplitudes and
overlap contractions stay on GPU and use CuPy dot. Eliminating the closed core
reduces the remaining determinants to dimension <=4. For a fixed number of
roots and two open orbitals, work is cubic and memory quadratic in orbital
count. Frame-local intermediate reuse avoids repeated small GEMMs. Cached
plans contain only orbital/spin/shape metadata, never amplitudes or metrics.
The closed-core cross overlap must be well-conditioned; otherwise this backend
raises an error. The reference-continuation guard normally prevents this in
small-step dynamics. The determinant backends below provide independent
checks of these contractions.

`overlap_backend: "wfoverlap"` retains SHARC's external exact determinant
oracle. Activate SHARC or set `wfoverlap_executable` to its absolute path;
`overlap_memory_mb` defaults to 8000. The `python` backend is a slow small-system
determinant oracle. All backends use the same AWFs and actual AO/MO metric;
there is no coefficient truncation beyond numerical zeros, frozen core or
amplitude-dot approximation. Changing backend requires a fresh savedir.

Matrices use SHARC singlet / triplet(Ms=-1) / triplet(Ms=0) /
triplet(Ms=+1) root order. Triplet overlap and gradient blocks are repeated
for each Ms; cross-spin overlaps and scalar dipoles vanish. SOC couples the
spin sectors. SHARC transforms the spin-free Hamiltonian and overlap matrices
into its SOC eigenbasis for electronic propagation. The supplied input uses
`nophases_from_interface` and `phase_correction_algo zhou2020`, leaving phase
handling to SHARC's full overlap algorithm. The interface does not reorder
roots or change their signs independently of SOC/dipoles. Explicit scalar
`phases` requests stop if diagonal overlaps are below `min_overlap` (0.5).

Wavefunction snapshots for both sectors and output arrays are retained in
`savedir/nttda/`; the external backend also retains WFoverlap inputs/logs there.
Same-step repeated requests use the
cached wavefunctions/results; use a fresh savedir when changing configuration.
All electronic output uses atomic units.
Each completed new frame also writes `frame-N.json` beside its array cache,
with component wall times, absolute spin-free energies, reference overlap
and the singlet/triplet overlap singular values. Times synchronize CUDA and
include first-call setup in that Python process. All triplet gradients reuse
one geometry-local response cache and one common reference gradient.

`triplet_gradient_backend: "serial"` preserves the default triplet gradient
path. The optional `"batch"` backend prepares the same-spin gradient right-hand
sides with the installed NT-TDA fork and shares their orbital response solve.
For EnsembleROKS it fuses the selected-reference response into each gradient,
using the same high-spin energy zero. This does not enable triplet NACs.
Use a fresh savedir when changing this setting. Small ordinary/ensemble
regressions and a 2-thiouracil three-triplet comparison validate the optional
backend; measure first-call and repeated-call costs on the target system.

Validated: real restricted molecular ROKS with exactly two open orbitals,
deltaS=-1/0, DF gradients, phase covariance and small overlap SOC trajectories.
The ensemble option has a selected-energy-zero regression against an
independent CPU fixed-orbital ROKS evaluation, and a 2-thiouracil continuity
and one-component gradient probe. Full ensemble reference dynamics,
solvent/relativistic SCF, SOC derivatives, laser-driven dynamics, and production
crossing/long-time convergence require further validation. SHARC uses its usual
approximation omitting explicit SOC derivatives.
State-space completeness, timestep, SOC model and electronic convergence
remain system-dependent research choices.
