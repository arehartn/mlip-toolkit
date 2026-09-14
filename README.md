# mlip-toolkit

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![ASE](https://img.shields.io/badge/built%20on-ASE-orange.svg)](https://wiki.fysik.dtu.dk/ase/)
[![MACE](https://img.shields.io/badge/MLIP-MACE%20%7C%20CHGNet-green.svg)](https://github.com/ACEsuit/mace)

A configuration-driven Python package for running machine-learned interatomic potential (MLIP) molecular dynamics and quantitatively validating the resulting trajectories against ab initio molecular dynamics (AIMD) references.

Ab initio MD is the standard method for studying ion transport in candidate solid-state electrolytes, but it costs on the order of weeks of compute and hundreds of dollars per system. Foundation MLIPs such as MACE-MP-0 reproduce comparable dynamics in minutes on a single GPU with no system-specific training. `mlip-toolkit` provides the infrastructure to determine, quantitatively and reproducibly, where that substitution is valid — and where it is not.

The package was developed to benchmark MACE against AIMD on solid-state sodium ionic conductors, with particular attention to hydridic systems, where hydrogen's low mass and high vibrational frequencies make dynamics especially difficult to reproduce.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running on an HPC cluster](#running-on-an-hpc-cluster)
- [Validation metrics](#validation-metrics)
- [Implementation notes](#implementation-notes)
- [Benchmark results](#benchmark-results)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [Citation and references](#citation-and-references)

---

## Features

- **Three job types from one interface** — structure relaxation, MLIP molecular dynamics, and trajectory validation, all driven by JSON configuration files with no code changes required.
- **Multiple MLIP backends** — MACE (including multi-head foundation models) and CHGNet, selectable by a single config field. CHGNet is imported lazily and is not a hard dependency.
- **Distribution-aware validation** — agreement between MLIP and AIMD trajectories is scored with Earth Mover's Distance across nine independently toggleable observables, rather than pointwise error metrics that are sensitive to trajectory phase alignment.
- **Two independent diffusion estimators** — Einstein-relation (MSD) and Green–Kubo (VACF integral) diffusion coefficients, enabling cross-validation of transport properties that neither method establishes alone.
- **Reproducibility by construction** — every run is fully specified by its JSON config, with RNG seeds locked across `random`, `numpy`, `torch`, and CUDA.
- **HPC-ready** — a single batch-dispatchable entry point designed for Slurm-scheduled GPU jobs.

---

## Installation

```bash
git clone https://github.com/arehartn/mlip-toolkit.git
cd mlip-toolkit
pip install -e .
```

This installs `numpy`, `ase`, and `mace-torch`, along with three console entry points: `md-sim`, `md-relax`, and `md-validate`.

On a shared cluster, install into a dedicated conda environment rather than the base environment, as GPU-enabled PyTorch and MACE builds are large and version-sensitive:

```bash
module load miniconda3
conda create -n mace_env_gpu python=3.10 -y
conda activate mace_env_gpu
pip install -e .
```

CHGNet support requires `pip install chgnet` separately; it is only imported when `model_type` is set to `chgnet`.

---

## Quick start

Each job is defined by a single JSON file containing a top-level `"job"` field. Create a working directory, place one config in it, and invoke the dispatcher:

```bash
mkdir my_run && cd my_run
cp /path/to/mlip-toolkit/examples/nbh_700K.json .
# edit input_structure, temp_kelvin, model_variant, etc.
python /path/to/mlip-toolkit/examples/run_all.py
```

`run_all.py` collects every `*.json` in the current directory, reads each file's `"job"` field, and dispatches to the corresponding routine:

| `"job"` value | Routine invoked | Primary outputs |
|---|---|---|
| `"relaxation"` | `md_sim.relaxation.run` | Relaxed structure |
| `"md"` | `md_sim.run_simulation.run` | `.traj` trajectory, summary CSV, per-atom CSV |
| `"validation"` | `md_sim.validation.run` | Scalar metrics CSV, raw distributions CSV, plots |

Individual entry points are also available for single jobs:

```bash
md-relax   --config relax.json
md-sim     --config md.json
md-validate --config validation.json
```

or from Python:

```python
from md_sim.run_simulation import run
run(config_path="md.json")
```

### Typical directory structure

Molecular dynamics and validation are separate jobs, linked by relative paths. A validation config references the MD outputs from its parent directory:

```
run_2/
├── large-0b2_nbh.json           # "job": "md"
├── mace_large-0b2_traj.traj     # MD outputs
├── mace_large-0b2_summary.csv
├── mace_large-0b2_detailed.csv
└── validation/
    └── nbh_large0b2_val.json    # "job": "validation" → ../mace_large-0b2_traj.traj
```

Run the dispatcher once in `run_2/`, then again in `validation/` after the MD job completes.

---

## Configuration

All behavior is controlled through JSON configs loaded by `md_sim/config.py`. Keys designated as paths are coerced to `pathlib.Path` objects, and all `run_*`, `plot_*`, and `make_plots` flags are coerced to booleans. Use JSON-native `true`/`false`, not Python literals.

Configs are not schema-validated: an unrecognized key is silently ignored and the corresponding default is used. Verify key spelling against a known-good config if a setting does not appear to take effect.

The package ships default configs at `md_sim/configs/simulation.json` and `md_sim/configs/relaxation.json`, used only when an entry point is invoked without `--config`. These are generic fallbacks. Production runs should begin from a complete worked example such as `examples/nbh_700K.json`, copied and modified — the resulting file serves as a self-contained record of the parameters that produced a given result.

### Selected configuration keys

**Molecular dynamics** (`"job": "md"`)

| Key | Description |
|---|---|
| `input_structure` | Path to the input structure (e.g. POSCAR) |
| `model_type` | `"mace"` or `"chgnet"` |
| `model_variant` | Foundation model name (`"medium-0b2"`, `"large-0b2"`) or path to a fine-tuned model |
| `head` | Output head for multi-head MACE models (e.g. `"mp_pbe_refit_add"`) |
| `device` | `"cpu"` or `"cuda"` |
| `seed` | RNG seed, locked across `random`, `numpy`, `torch`, CUDA |
| `temp_kelvin`, `dt_fs`, `friction`, `n_steps` | Langevin dynamics parameters |
| `log_interval` | Frames between logged steps |
| `stationary`, `fix_com`, `zero_rotation` | Momentum and drift constraints |
| `run_validation` | Chain directly into validation on MD completion |

**Relaxation** (`"job": "relaxation"`)

| Key | Description |
|---|---|
| `fmax`, `max_steps` | Convergence criteria |
| `relax_cell` | Relax cell degrees of freedom via `FrechetCellFilter` |
| `optimizer` | `"lbfgs"` or `"precon_lbfgs"` |

**Validation** (`"job": "validation"`)

| Key | Description |
|---|---|
| `reference_traj_file` | AIMD reference trajectory |
| `trajectory_file`, `summary_csv`, `atoms_csv` | MLIP run outputs under evaluation |
| `validation_burn_in_frames` | Equilibration frames discarded from both trajectories |
| `run_energy`, `run_forces`, `run_rdf`, `run_adf`, `run_cv`, `run_msd`, `run_green_kubo`, `run_vacf`, `run_vdos` | Per-metric toggles |
| `aimd_dt_fs`, `aimd_log_interval` | Overrides for references logged at a different cadence than the MLIP run |
| `validation_output_csv` | Destination for scalar metrics |

Any config may set `make_plots: true` alongside individual `plot_*` flags to emit thermodynamic traces, RDF/ADF overlays, VACF/VDOS overlays, energy and force parity plots, and velocity histograms, styled via the `plot_style` block.

---

## Running on an HPC cluster

`run_all.py` is designed to be the sole command in a batch script. The example below targets the Ohio Supercomputer Center's Cardinal cluster; account, module, and environment names are installation-specific and should be substituted accordingly.

```bash
#!/bin/bash
#SBATCH --job-name=MD_job
#SBATCH --account=<your-project-account>
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --output=job_%j.log

module purge
module load miniconda3/24.1.2-py310
conda activate mace_env_gpu

python run_all.py > job.log
```

Submit from the directory containing the job's config:

```bash
cd run_2/
sbatch submit_osc_cardinal.sh
```

The same script serves both MD and validation stages without modification, since the dispatcher acts on whichever config is present in the working directory.

---

## Validation metrics

`md_sim/validation.py` computes structural, thermodynamic, and dynamical observables from a reference/prediction trajectory pair and scores agreement using Earth Mover's Distance (Wasserstein distance). Comparing full distributions rather than pointwise values makes the metrics robust to frame misalignment between independently generated trajectories.

| Category | Observables |
|---|---|
| Energetics | Potential and kinetic energy distributions (per-atom and system total); heat capacity error from energy variance |
| Forces | Force-magnitude distribution |
| Structure | Radial distribution function *g(r)*; angular distribution function |
| Transport | Mean squared displacement → Einstein diffusion coefficient; Green–Kubo diffusion coefficient |
| Vibrational | Velocity autocorrelation function; vibrational density of states |

Each metric is independently toggleable. In addition to scalar scores, the underlying distributions are written to a `*_RAW_DATA.csv` companion file for downstream analysis and custom plotting.

---

## Implementation notes

Several details in the validation pipeline address failure modes that are easy to encounter and difficult to detect:

- **Kinetic energy extraction.** AIMD trajectories frequently contain missing, zeroed, or unit-inconsistent velocities. Velocity-derived kinetic energy is cross-checked against trajectory metadata and an equipartition estimate (3/2 *N* k_B*T*) before being accepted, with explicit warnings on fallback rather than silent propagation of invalid values.
- **Velocity reconstruction.** Where stored velocities are unusable, velocities are recovered by central-difference differentiation of periodic-boundary-unwrapped positions, with center-of-mass drift removed prior to VACF and VDOS computation.
- **MSD fitting window.** Mean squared displacement is computed from unwrapped coordinates over multiple time origins. The diffusion coefficient is fit to the central 25–75% of the curve, excluding the short-time ballistic regime and the long-time tail where few time origins contribute.
- **Green–Kubo diffusion.** *D* = (1/3*N*) ∫₀^tc ⟨Σᵢ **v**ᵢ(0)·**v**ᵢ(*t*)⟩ d*t*, integrated to the first zero crossing of the VACF. Because this estimator derives from the velocity autocorrelation rather than the MSD slope, it provides an independent check on transport properties and is markedly less sensitive to fitting-window artifacts in short trajectories.
- **VDOS comparison.** The vibrational density of states is obtained from a Hann-windowed FFT of the normalized VACF and compared as a normalized spectral distribution, so that spectra differing in amplitude scale but not in shape are not penalized disproportionately.
- **Timestep reconciliation.** `aimd_dt_fs` and `aimd_log_interval` permit comparison of trajectories logged at differing intervals without manual resampling.
- **RDF normalization.** *g(r)* is normalized against ideal-gas pair counts only; additional normalization by number density is a common error that flattens the resulting distribution.

---

## Benchmark results

All simulations were performed on the Ohio Supercomputer Center's Cardinal cluster. MACE trajectories were compared against AIMD (VASP, PBE) references generated with identical MD parameters in fully independent runs.

Systems studied:

- **NBH** — Na₂B₁₀H₁₀, Na₂B₁₂H₁₂, and a mixed phase, at 700 K and 800 K. Selected as a demanding test case: hydrogen's low mass and high vibrational frequencies stress the MLIP's ability to reproduce dynamics.
- **NNOC** — NaNbOCl₄, experimentally synthesized and computationally derived lowest-energy structures, at 500 K. Serves as a hydrogen-free reference point.

### Initial survey (MACE-MP-0b2)

| System | Friction | *E*_pot EMD (eV/atom) | *E*_kin EMD (eV/atom) | RDF EMD (Å) | ADF EMD (°) | VDOS EMD (THz) | *C*_v error (eV/K/atom) |
|---|---|---|---|---|---|---|---|
| NBH 700 K, 20 ps | 0.01 | 0.00656 | 1.28 | 0.0042 | 0.26 | 0.64 | 0.00038 |
| NBH 700 K, 20 ps | 0.25 | 0.00444 | 1.28 | 0.0043 | 0.31 | 2.42 | 0.00038 |
| NBH 700 K, 1 ps | 0.01 | 0.01594 | 162.33 | 0.0116 | 0.11 | 131.43 | 0.00006 |
| NBH 700 K, 1 ps | 0.25 | 0.00460 | 162.32 | 0.0124 | 0.39 | 126.65 | 0.00006 |
| NNOC 500 K | 0.01 | 0.01200 | 0.016 | 0.0080 | 0.81 | 0.75 | 0.00014 |
| NNOC 500 K | 0.25 | 0.00430 | 0.00088 | 0.0100 | 0.50 | 11.28 | 0.00008 |

Principal observations from this survey:

- Potential energy is reproduced accurately across both systems and both friction settings.
- Kinetic energy error in the hydrogen-containing NBH system substantially exceeds that of hydrogen-free NNOC, consistent with hydrogen dynamics as the limiting factor.
- Thermostat friction materially affects which observables are reliable, and short (1 ps) trajectories exhibit large kinetic energy and VDOS errors relative to 20 ps runs of the same system.
- Structural agreement is uniformly good, with RDF EMD in the 0.004–0.012 Å range.

MSD-derived diffusion coefficients for NBH were not sufficiently converged to interpret, and no independent transport estimator was available at this stage.

### Follow-up: Na₂B₁₀H₁₀, MACE large-0b2, 700 K

Configuration: `model_variant="large-0b2"`, `head="mp_pbe_refit_add"`, `dt_fs=0.5`, `friction=0.01`, `n_steps=2000`, `validation_burn_in_frames=800`, `seed=1212`, CUDA.

| Metric | Value |
|---|---|
| *E*_pot EMD (eV/atom) | 1.09 × 10⁻³ |
| *E*_pot EMD (eV, system) | 0.766 |
| *E*_kin EMD (eV/atom) | 8.79 × 10⁻³ |
| *E*_kin EMD (eV, system) | 6.19 |
| Force EMD (eV/Å) | 0.233 |
| RDF EMD (Å) | 1.64 × 10⁻² |
| ADF EMD (°) | 8.35 × 10⁻² |
| VDOS EMD (THz) | 14.2 |
| *C*_v error (eV/K/atom) | 1.29 × 10⁻⁴ |
| *D* reference, MSD (m²/s) | 9.35 × 10⁻⁹ |
| *D* predicted, MSD (m²/s) | 4.47 × 10⁻⁹ |
| *D* reference, Green–Kubo (m²/s) | 1.27 × 10⁻⁸ |
| *D* predicted, Green–Kubo (m²/s) | 1.01 × 10⁻⁸ |

Findings:

- **Transport is reproduced at the correct order of magnitude.** Green–Kubo and Einstein estimators agree to within a factor of two on both the reference and predicted trajectories, establishing that the earlier inability to interpret NBH diffusion reflected MSD convergence at short trajectory lengths rather than a failure of the potential to capture ion transport.
- **MACE underestimates the diffusion coefficient** by approximately 20% under the Green–Kubo estimator and 52% under the Einstein estimator. Given the Green–Kubo estimator's lower sensitivity to fitting-window effects in short trajectories, the former is the more reliable figure.
- **Energetic agreement is substantially improved** relative to the initial survey, with per-atom potential and kinetic energy EMD reduced by roughly a factor of five and two orders of magnitude respectively. This reflects both the larger model variant and revisions to kinetic energy extraction and reference alignment in the validation pipeline.

These figures derive from a single production run rather than a systematic sweep, and should be treated as provisional pending replication across friction and trajectory-length conditions.

---

## Repository layout

```
mlip-toolkit/
├── md_sim/
│   ├── config.py              # JSON config loading, type coercion, CLI parsing
│   ├── md_config.py           # Backward-compatible PARAMS shim
│   ├── md_tools.py            # Calculator construction, Langevin dynamics, logging
│   ├── relaxation.py          # Structure relaxation (LBFGS, preconditioned LBFGS)
│   ├── run_simulation.py      # MD driver with seed locking
│   ├── md_plotting.py         # Thermodynamic, structural, and spectral plotting
│   ├── validation.py          # EMD-based validation engine
│   └── configs/
│       ├── simulation.json    # Default MD parameters
│       └── relaxation.json    # Default relaxation parameters
├── examples/
│   ├── nbh_700K.json          # Worked MD configuration
│   ├── run_all.py             # Config dispatcher (primary entry point)
│   └── submit_osc_cardinal.sh # Example Slurm batch script
├── docs/
│   └── Final_Poster.pdf       # Research poster
└── setup.py
```

---

## Roadmap

- Replicate the large-0b2 configuration across the full friction and trajectory-length grid to confirm the improvements observed in the single follow-up run.
- Extend Green–Kubo cross-validation to NNOC and the remaining NBH phases.
- Characterize the residual ~20% underestimation of the diffusion coefficient.
- Improve MSD fitting robustness for short trajectories.
- Extend benchmarking to additional sodium-conductor compositions.

---

## Citation and references

Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., & Csányi, G. (2022). MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. *Advances in Neural Information Processing Systems*.

The research poster describing this work — *Is There a Better, More Affordable Alternative to AIMD? Validating Foundational MLIPs on Solid-State Sodium Ionic Conductors* — is included at [`docs/Final_Poster.pdf`](docs/Final_Poster.pdf).

---

## Acknowledgments

Developed at The Ohio State University, Department of Materials Science and Engineering, with computational resources provided by the Ohio Supercomputer Center. Research advised by Chaitanya Kolluru.

## Contact

Nate Arehart — [arehart.29@osu.edu](mailto:arehart.29@osu.edu) — [github.com/arehartn](https://github.com/arehartn)