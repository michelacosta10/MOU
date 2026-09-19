# MOU

Simulation and estimation of multivariate Ornstein-Uhlenbeck (MOU) processes with random GOE or Ginibre drift matrices.

## Scripts

- `generate_data_MOU.py` — simulates MOU trajectories for random drift matrices `A` across a range of coupling strengths `c`, and saves one `.npz` file per `c` value.
- `estimate_MOU.py` — loads the simulated trajectories and estimates the drift matrix `A` (and related quantities) using the Bayes I estimator, saving one `.npz` file per `c` value.
- `process_MOU.py` — loads the estimation files, computes all derived analysis quantities (true matrices, spectra, Hessian eigenvalues, condition numbers, Frobenius errors, self-averaging statistics, projected soft-mode uncertainty), and saves everything into a single pickle.
- `plots_MOU.py` — reads the pickle from `process_MOU.py` and generates the analysis figures (spectra, error distributions, self-averaging, Hessian, condition numbers, soft-mode uncertainty).
- `phase_diagrams_MOU.py` — theoretical resolvability / stability-inference phase diagrams for the GOE ensemble. Purely theoretical: does not need simulated or estimated data. Generates the two boundaries from the paper (`reliable` = diagnosis boundary $\sigma_r/r=1$, `resolvability` = resolution boundary $\sigma_r/r=0.1$) as separate figures.

## Usage

```bash
pip install -r requirements.txt

python generate_data_MOU.py --data-dir ./data
python estimate_MOU.py --data-dir ./data
python process_MOU.py --data-dir ./data
python plots_MOU.py --dataset ./data/processed/dataset_GOE_lag1_6JUN.pkl --output-dir ./figures
python phase_diagrams_MOU.py --output-dir ./figures
```

By default, `--data-dir` points to a `data/` folder next to the scripts (not tracked by git). All simulation parameters (`--N`, `--q`, `--n-A`, `--n-sim`, `--delta-t`, `--dt-internal`, `--ensemble`, `--c-values`, `--tag`) are also CLI flags — see `--help` on each script. Defaults match the paper's Figure 1 configuration (`N=100`, `q=10`, `n_A=30`, `n_sim=30`, GOE, `c` up to `0.7`). `estimate_MOU.py` and `process_MOU.py` locate the matching files automatically (via `--ensemble`, `--c-values`, `--tag`), so make sure these match what you passed to the previous step. Output is organized as:

```
data/
├── sim_trajectories/   # from generate_data_MOU.py
├── estimates/          # from estimate_MOU.py
└── processed/          # from process_MOU.py (pickled dataset for plots_MOU.py)
figures/                 # from plots_MOU.py and phase_diagrams_MOU.py
```

Run the scripts in order — `generate_data_MOU.py` → `estimate_MOU.py` → `process_MOU.py` → `plots_MOU.py` — since each step reads the files produced by the previous one. `phase_diagrams_MOU.py` is independent and can be run at any time (only `--N`, `--delta-t`, `--q`, `--c-fixed` control it).

## Running on the cluster (SLURM)

For a full run (all 7 `c` values at `N=100, q=10, n_A=30, n_sim=30`), submit the pipeline as SLURM jobs instead of running directly on the login node:

```bash
./sbatch/submit_pipeline.sh
```

This submits `01_generate` → `02_estimate` → `03_process` → `04_plots` as a dependency chain (each stage starts only if the previous one succeeded), plus `05_phase_diagrams` independently. Track progress with `squeue -u $USER`; logs land in `logs/`.

Override the run configuration via environment variables, without editing the scripts:

```bash
TAG=myrun DATA_DIR=./data C_VALUES="0.1 0.2 0.3 0.4 0.5 0.6 0.7" ./sbatch/submit_pipeline.sh
```

Each stage can also be submitted individually (e.g. to re-run just one step): `sbatch --export=ALL,TAG=myrun sbatch/03_process.sbatch`. This account's QOS caps concurrent usage at 16 CPUs, which `01_generate`/`02_estimate` request in full (they parallelize over `n_A` realisations via `joblib`) — other stages will queue behind them rather than run in parallel.
