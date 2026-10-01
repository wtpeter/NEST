# GPU NT-TDA / NEST SOC / SHARC overlap example

Run on a CUDA worker with the gpu4pyscf NT-TDA branch, editable NEST, and
compiled SHARC + WFoverlap. Copy `input`, `geom`, `veloc` into a new trajectory
and `runQM.sh`, `nttda.json` into its `QM/` subdirectory. Activate the Python
and SHARC environment, then run `sharc.x input` in the trajectory directory.
SHARC invokes `QM/runQM.sh` with `/bin/sh`. Generated trajectories, logs and
scratch belong under the workspace's `files/` directory.

This CH2 / PBE / STO-3G example computes three deltaS=-1 singlet roots and
three deltaS=0 triplet roots from a shared ordinary GPU ROKS reference, with
density fitting and CPU SOMF SOC. It propagates 12 spin components using
`coupling overlap`, `nogradcorrect`, and momentum adjustment `parallel_vel`.
No NAC is requested or evaluated. Grid level 0 and 0.2 fs are integration
checks; they are not suitable convergence settings for production dynamics.
Use `select_directly`: the adapter evaluates all requested states' gradients
and caches the result, rather than implementing SHARC's gradient-selection
subsequent-call protocol.

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
NT-TDA excitation energies are shifted by the common reference SCF energy.
`scf_conv_tol_grad` defaults to 1e-8; the Newton inner AH tolerance is tightened
consistently with that orbital residual. For a controlled initial SCF guess,
`initial_reference` accepts a wavefunction snapshot from this interface,
resolved relative to the configuration file. Its orbitals are projected and
relaxed with fixed occupations; the initial overlap matrix remains identity.
The atom order, electron count and spin must agree. An initial guess does not
replace convergence checks or enforce a chosen electronic state.

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
small-step dynamics. See workspace `docs/gpu-nttda-overlap.md` for derivation,
validation and first-call versus repeated-call timing.

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

Validated: real restricted molecular ROKS with exactly two open orbitals,
deltaS=-1/0, DF gradients, phase covariance and small overlap SOC trajectories.
Ensemble reference dynamics, solvent/relativistic SCF, SOC derivatives,
laser-driven dynamics, and production crossing/long-time convergence are not
validated. SHARC uses its usual approximation omitting explicit SOC derivatives.
State-space completeness, timestep, SOC model and electronic convergence
remain system-dependent research choices.
