import argparse
import numpy as np
import os
import time
import gc
from scipy.linalg import logm, solve_discrete_lyapunov
from joblib import Parallel, delayed

import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)


####################
# ESTIMATOR: BAYES I
####################

def estimate_bayes_I(X, dt, lag=1):

    X = X.astype(np.float64)
    N, T = X.shape

    T1 = X[:, lag:]  @ X[:, lag:].T  / (T - lag)
    T2 = X[:, lag:]  @ X[:, :-lag].T / (T - lag)
    T3 = X[:, :-lag] @ X[:, :-lag].T / (T - lag)

    T3_inv = np.linalg.inv(T3)

    Q = T2 @ T3_inv

    Sigma_dt = T1 - T2 @ T3_inv @ T2.T
    Sigma_dt = (Sigma_dt + Sigma_dt.T) / 2.0

    A_comp = -logm(Q) / (dt * lag)
    A = np.real(A_comp)

    Sigma_inf = solve_discrete_lyapunov(Q, Sigma_dt)
    Sigma_inf = (Sigma_inf + Sigma_inf.T) / 2.0

    B = 0.5 * (A @ Sigma_inf + Sigma_inf @ A.T)

    return A, Q, Sigma_dt, Sigma_inf, B


####################
# ESTIMATE ONE A BLOCK
####################

def estimate_one_A_block(X_block, Delta_t, lag):

    n_sim, N, M_steps = X_block.shape

    A_block         = np.full((n_sim, N, N), np.nan, dtype=np.float32)
    Q_block         = np.full((n_sim, N, N), np.nan, dtype=np.float32)
    Sigma_dt_block  = np.full((n_sim, N, N), np.nan, dtype=np.float32)
    Sigma_inf_block = np.full((n_sim, N, N), np.nan, dtype=np.float32)
    B_block         = np.full((n_sim, N, N), np.nan, dtype=np.float32)

    fail_count = 0

    for sim in range(n_sim):

        try:

            A, Q, Sigma_dt, Sigma_inf, B = estimate_bayes_I(
                X_block[sim],
                Delta_t,
                lag
            )

            A_block[sim]         = np.real(A).astype(np.float32)
            Q_block[sim]         = np.real(Q).astype(np.float32)
            Sigma_dt_block[sim]  = np.real(Sigma_dt).astype(np.float32)
            Sigma_inf_block[sim] = np.real(Sigma_inf).astype(np.float32)
            B_block[sim]         = np.real(B).astype(np.float32)

        except Exception:

            fail_count += 1

    return {
        "A": A_block,
        "Q": Q_block,
        "Sigma_dt": Sigma_dt_block,
        "Sigma_inf": Sigma_inf_block,
        "B": B_block,
        "fail_count": fail_count
    }


######################################################################################################################
# CONFIG
######################################################################################################################

parser = argparse.ArgumentParser(description="Estimate MOU drift matrices from simulated trajectories.")
parser.add_argument(
    "--data-dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    help="Base directory containing 'sim_trajectories' and where 'estimates' is written (default: ./data next to this script).",
)
args = parser.parse_args()

lag = 1

sim_dir = os.path.join(args.data_dir, "sim_trajectories")
est_dir = os.path.join(args.data_dir, "estimates")

os.makedirs(est_dir, exist_ok=True)

print("\n############################################################")
print("ESTIMATION ONLY — BAYES I")
print("ONE ESTIMATION FILE PER c")
print("ONLY lag = 1")
print("PARALLEL OVER A REALIZATIONS")
print("NO T1, T2, T3 SAVED")
print("############################################################")

total_start = time.time()


#############################
# SELECT c VALUES TO ESTIMATE
#############################

c_values_to_estimate = np.array([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99 ])

sim_files = []

for c in c_values_to_estimate:

    c_tag = f"{c:g}"

    sim_file = (
        f"DATASIM"
        f"_N100"
        f"_q80"
        f"_Dt1.0"
        f"_nA25"
        f"_nsim25"
        f"_c{c_tag}"
        f"_GINIBRE_6JUN.npz"
    )

    sim_path = os.path.join(sim_dir, sim_file)

    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"Missing simulation file:\n{sim_path}")

    sim_files.append(sim_file)

print(f"Selected {len(sim_files)} simulation files:")
for f in sim_files:
    print(f)


#############################
# MAIN LOOP — ONE FILE PER c
#############################

for file_idx, sim_file in enumerate(sim_files):

    c_start = time.time()

    sim_path = os.path.join(sim_dir, sim_file)

    print("\n" + "=" * 60)
    print(f"Loading simulation file {file_idx + 1}/{len(sim_files)}")
    print(sim_file)
    print("=" * 60)

    data = np.load(sim_path)

    X_c = data["X_c"]
    A_true_c = data["A_true_c"]

    c_value = float(data["c_value"])

    N = int(data["N"])
    q = int(data["q"])
    Delta_t = float(data["Delta_t"])
    dt_internal = float(data["dt_internal"])
    M_steps = int(data["M_steps"])

    n_A = int(data["n_A"])
    n_sim = int(data["n_sim"])

    print(f"c = {c_value:.4f}")
    print(f"X_c.shape      = {X_c.shape}")
    print(f"A_true_c.shape = {A_true_c.shape}")
    print(f"N={N}, q={q}, n_A={n_A}, n_sim={n_sim}")

    shape = (n_A, n_sim, N, N)

    A_est_c         = np.full(shape, np.nan, dtype=np.float32)
    Q_est_c         = np.full(shape, np.nan, dtype=np.float32)
    Sigma_dt_est_c  = np.full(shape, np.nan, dtype=np.float32)
    Sigma_inf_est_c = np.full(shape, np.nan, dtype=np.float32)
    B_est_c         = np.full(shape, np.nan, dtype=np.float32)

    fail_per_A = np.zeros(n_A, dtype=int)

    results = Parallel(n_jobs=-1)(
        delayed(estimate_one_A_block)(
            X_c[i_A],
            Delta_t,
            lag
        )
        for i_A in range(n_A)
    )

    for i_A, res in enumerate(results):

        A_est_c[i_A]         = res["A"]
        Q_est_c[i_A]         = res["Q"]
        Sigma_dt_est_c[i_A]  = res["Sigma_dt"]
        Sigma_inf_est_c[i_A] = res["Sigma_inf"]
        B_est_c[i_A]         = res["B"]

        fail_per_A[i_A] = res["fail_count"]

    fail_total = int(np.sum(fail_per_A))

    c_tag = f"{c_value:g}"

    est_filename = (
        f"EST_BAYES_I"
        f"_N{N}"
        f"_q{q}"
        f"_Dt{Delta_t}"
        f"_nA{n_A}"
        f"_nsim{n_sim}"
        f"_lag{lag}"
        f"_c{c_tag}"
        f"_GINIBRE_6JUN.npz"
    )

    est_path = os.path.join(est_dir, est_filename)

    np.savez_compressed(
        est_path,

        A_est=A_est_c,
        Q_est=Q_est_c,
        B_est=B_est_c,
        Sigma_dt_est=Sigma_dt_est_c,
        Sigma_inf_est=Sigma_inf_est_c,

        A_true=A_true_c,

        c_value=c_value,

        lag=lag,

        fail_total=fail_total,
        fail_per_A=fail_per_A,

        N=N,
        q=q,
        Delta_t=Delta_t,
        dt_internal=dt_internal,
        M_steps=M_steps,

        n_A=n_A,
        n_sim=n_sim,

        source_sim_file=sim_file
    )

    print(f"Saved estimates for c = {c_value:.4f} at:")
    print(est_path)
    print(f"fail_total = {fail_total}")
    print(f"Done in {time.time() - c_start:.2f} sec")

    del data
    del X_c, A_true_c
    del A_est_c, Q_est_c, B_est_c
    del Sigma_dt_est_c, Sigma_inf_est_c
    del results
    gc.collect()


print(f"\nTOTAL TIME: {time.time() - total_start:.2f} sec")