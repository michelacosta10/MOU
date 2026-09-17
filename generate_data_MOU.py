import argparse
import numpy as np
import os
import time
import gc
from joblib import Parallel, delayed


####################
# DRIFT MATRIX
####################

def generate_drift_matrix(N, c):

    while True:

        G = np.random.randn(N, N) * (c / np.sqrt(N))
        A = np.eye(N) + G

        if np.min(np.real(np.linalg.eigvals(A))) > 0:
            return A


####################
# SIMULATION
####################

def simulate_mvou_jump(A, M_steps, k, dt_internal, X0=None):

    N = A.shape[0]

    B = np.eye(N)

    F1 = np.eye(N) - A * dt_internal
    Fk = np.linalg.matrix_power(F1, k)

    Sk = np.zeros((N, N))
    Fj = np.eye(N)

    for _ in range(k):

        Sk += 2 * dt_internal * (Fj @ B @ Fj.T)
        Fj = Fj @ F1

    Sk = (Sk + Sk.T) / 2.0

    try:

        L = np.linalg.cholesky(Sk)

    except np.linalg.LinAlgError:

        Sk += 1e-9 * np.eye(N)
        L = np.linalg.cholesky(Sk)

    Z = np.random.randn(N, M_steps)
    noise_jump = L @ Z

    X_sampled = np.zeros((N, M_steps), dtype=np.float32)

    if X0 is None:

        x_curr = 1e-3 * np.random.randn(N)

    else:

        x_curr = X0.copy()

    X_sampled[:, 0] = x_curr

    for t in range(1, M_steps):

        x_curr = Fk @ x_curr + noise_jump[:, t]
        X_sampled[:, t] = x_curr

    return X_sampled


####################
# ONE A REALIZATION + n_sim TRAJECTORIES
####################

def run_one_A_realization(
    N,
    c,
    M_steps,
    k,
    dt_internal,
    n_sim,
    seed_A,
    seeds_X
):

    np.random.seed(seed_A)

    A_true = generate_drift_matrix(N, c)

    X_block = np.zeros((n_sim, N, M_steps), dtype=np.float32)

    for sim in range(n_sim):

        np.random.seed(seeds_X[sim])

        X_block[sim] = simulate_mvou_jump(
            A_true,
            M_steps,
            k,
            dt_internal
        ).astype(np.float32)

    return A_true.astype(np.float32), X_block


#############################
# PARAMETERS
#############################

N = 100
q = 80

c_values = np.array([ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

Delta_t = 1.0
dt_internal = 0.1

n_A = 25
n_sim = 25


#############################
# CONFIG
#############################

parser = argparse.ArgumentParser(description="Generate MOU simulated trajectories.")
parser.add_argument(
    "--data-dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    help="Base directory where simulation output is saved (default: ./data next to this script).",
)
args = parser.parse_args()

mode_tag = "sim_trajectories"

save_dir = os.path.join(args.data_dir, mode_tag)

print("\n############################################################")
print(f"SIMULATION ONLY — mode: {mode_tag}")
print("DOUBLE MONTE CARLO")
print("ONE FILE PER c")
print("PARALLEL OVER A REALIZATIONS")
print("############################################################")

total_start = time.time()

os.makedirs(save_dir, exist_ok=True)

M_steps = int(q * N)
k = int(Delta_t / dt_internal)
n_c = len(c_values)


print(f"\n############################################")
print(f"RUNNING SIMULATION FOR N = {N}")
print(f"n_A   = {n_A}")
print(f"n_sim = {n_sim}")
print(f"M_steps = {M_steps}")
print("############################################")


#############################
# MAIN LOOP — SAVE ONE FILE PER c
#############################

for i_c, c in enumerate(c_values):

    c_start_time = time.time()

    print("\n" + "=" * 60)
    print(f"Running c = {c:.4f} ({i_c + 1}/{n_c})")
    print("=" * 60)

    # --------------------------------------------------
    # Allocate only current c
    # --------------------------------------------------

    A_true_c = np.zeros((n_A, N, N), dtype=np.float32)
    X_c = np.zeros((n_A, n_sim, N, M_steps), dtype=np.float32)

    # --------------------------------------------------
    # Seeds
    # --------------------------------------------------

    seeds_A = np.random.randint(0, 2**32 - 1, size=n_A)
    seeds_X = np.random.randint(0, 2**32 - 1, size=(n_A, n_sim))

    # --------------------------------------------------
    # Parallel over A realizations
    # --------------------------------------------------

    results = Parallel(n_jobs=-1)(
        delayed(run_one_A_realization)(
            N,
            c,
            M_steps,
            k,
            dt_internal,
            n_sim,
            seeds_A[i_A],
            seeds_X[i_A]
        )
        for i_A in range(n_A)
    )

    for i_A, (A_true, X_block) in enumerate(results):

        A_true_c[i_A] = A_true
        X_c[i_A] = X_block

    # --------------------------------------------------
    # Save current c
    # --------------------------------------------------

    c_tag = f"{c:g}"

    filename = (
        f"DATASIM"
        f"_N{N}"
        f"_q{q}"
        f"_Dt{Delta_t}"
        f"_nA{n_A}"
        f"_nsim{n_sim}"
        f"_c{c_tag}"
        f"_GINIBRE_6JUN.npz"
    )

    filepath = os.path.join(save_dir, filename)

    np.savez_compressed(
        filepath,

        X_c=X_c,
        A_true_c=A_true_c,

        c_value=c,
        i_c=i_c,
        c_values=c_values,

        N=N,
        q=q,
        Delta_t=Delta_t,
        dt_internal=dt_internal,
        M_steps=M_steps,
        k=k,

        n_A=n_A,
        n_sim=n_sim,

        seeds_A=seeds_A,
        seeds_X=seeds_X
    )

    print(f"Saved c = {c:.4f} at:")
    print(filepath)

    print(f"Finished c = {c:.4f} in {time.time() - c_start_time:.2f} sec")

    # --------------------------------------------------
    # Free RAM
    # --------------------------------------------------

    del A_true_c, X_c, results
    gc.collect()


print(f"\nTOTAL TIME: {time.time() - total_start:.2f} sec")