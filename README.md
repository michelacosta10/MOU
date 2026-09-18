# MOU

Simulation and estimation of multivariate Ornstein-Uhlenbeck (MOU) processes with random Ginibre/GOE drift matrices.

## Scripts

- `generate_data_MOU.py` — simulates MOU trajectories for random drift matrices `A` across a range of coupling strengths `c`, and saves one `.npz` file per `c` value.
- `estimate_MOU.py` — loads the simulated trajectories and estimates the drift matrix `A` (and related quantities) using the Bayes I estimator, saving one `.npz` file per `c` value.

## Usage

```bash
pip install -r requirements.txt

python generate_data_MOU.py --data-dir ./data
python estimate_MOU.py --data-dir ./data
```

By default, `--data-dir` points to a `data/` folder next to the scripts (not tracked by git). All simulation parameters (`--N`, `--q`, `--n-A`, `--n-sim`, `--delta-t`, `--dt-internal`, `--ensemble`, `--c-values`, `--tag`) are also CLI flags — see `--help` on either script. Defaults are `N=100, q=10, n_A=30, n_sim=30`, GOE ensemble, `c` from `0.1` to `0.7`. `estimate_MOU.py` locates the matching simulation files automatically (via `--ensemble`, `--c-values`, `--tag`), so make sure these match what you passed to `generate_data_MOU.py`. Output is organized as:

```
data/
├── sim_trajectories/   # from generate_data_MOU.py
└── estimates/          # from estimate_MOU.py
```

Run `generate_data_MOU.py` before `estimate_MOU.py`, since the estimation step reads the files produced by the simulation step.
