import argparse
import glob
import os
import time
import gc
import pickle

import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_continuous_lyapunov


######################################################################################################################
# ARGUMENTS
######################################################################################################################

parser = argparse.ArgumentParser(
    description="Load estimation files from estimate_MOU.py and compute all derived quantities "
                 "(true matrices, spectra, Hessian eigenvalues, errors, condition numbers, "
                 "soft-mode uncertainty). Saves everything into a single pickle for plots_MOU.py."
)
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--ensemble", type=str, default="GOE", choices=["GINIBRE", "GOE"])
parser.add_argument(
    "--c-values", type=float, nargs="+", dest="c_values",
    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    help="c values to load; must match files produced by estimate_MOU.py"
)
parser.add_argument("--tag", type=str, default="6JUN", help="run tag used in the estimation filenames")
parser.add_argument("--lag", type=int, default=1)
parser.add_argument(
    "--output", type=str, default=None,
    help="Output pickle path. Default: <data-dir>/processed/dataset_<ENSEMBLE>_lag<lag>_<tag>.pkl"
)
args = parser.parse_args()

lag = args.lag
ENSEMBLE = args.ensemble
RUN_TAG = args.tag
ensemble_tag = ENSEMBLE.upper()

est_dir = os.path.join(args.data_dir, "estimates")

if args.output is None:
    out_dir = os.path.join(args.data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"dataset_{ensemble_tag}_lag{lag}_{RUN_TAG}.pkl")
else:
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

total_start = time.time()

print("\n############################################################")
print("PROCESS ESTIMATES: load + derive all analysis quantities")
print(f"ENSEMBLE: {ensemble_tag}")
print("############################################################")


######################################################################################################################
# LOAD ONE FILE PER c
######################################################################################################################

c_values_to_load = np.array(args.c_values)

A_est_list = []
B_est_list = []
Q_est_list = []
Sigma_dt_est_list = []
Sigma_inf_est_list = []
A_true_list = []

c_loaded = []
fail_total_list = []
fail_per_A_list = []

for c in c_values_to_load:

    c_tag = f"{c:g}"

    pattern = os.path.join(
        est_dir,
        f"EST_*_lag{lag}_c{c_tag}_{ensemble_tag}_{RUN_TAG}.npz"
    )

    matches = sorted(glob.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No estimation file found matching:\n{pattern}")

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple estimation files match c={c_tag}, expected exactly one:\n"
            + "\n".join(matches)
        )

    filepath = matches[0]
    print(f"Loading: {os.path.basename(filepath)}")

    data = np.load(filepath)

    A_est_list.append(data["A_est"])
    B_est_list.append(data["B_est"])
    Q_est_list.append(data["Q_est"])
    Sigma_dt_est_list.append(data["Sigma_dt_est"])
    Sigma_inf_est_list.append(data["Sigma_inf_est"])

    A_true_list.append(data["A_true"])

    c_loaded.append(float(data["c_value"]))
    fail_total_list.append(int(data["fail_total"]))
    fail_per_A_list.append(data["fail_per_A"])


######################################################################################################################
# STACK INTO ARRAYS
######################################################################################################################

A_est_all = np.stack(A_est_list, axis=0)
B_est_all = np.stack(B_est_list, axis=0)
Q_est_all = np.stack(Q_est_list, axis=0)
Sigma_dt_est_all = np.stack(Sigma_dt_est_list, axis=0)
Sigma_inf_est_all = np.stack(Sigma_inf_est_list, axis=0)

A_true_all = np.stack(A_true_list, axis=0)

c_values = np.array(c_loaded)
fail_total = int(np.sum(fail_total_list))
fail_per_c_A = np.stack(fail_per_A_list, axis=0)


######################################################################################################################
# METADATA (from last loaded file)
######################################################################################################################

N = int(data["N"])
q = int(data["q"])
Delta_t = float(data["Delta_t"])
dt_internal = float(data["dt_internal"])
M_steps = int(data["M_steps"])

n_A = int(data["n_A"])
n_sim = int(data["n_sim"])
lag = int(data["lag"])

n_c = len(c_values)


######################################################################################################################
# DATASET DICTIONARY
######################################################################################################################

dataset = {
    "A_est_all": A_est_all,
    "B_est_all": B_est_all,
    "Q_est_all": Q_est_all,
    "Sigma_dt_est_all": Sigma_dt_est_all,
    "Sigma_inf_est_all": Sigma_inf_est_all,

    "A_true_all": A_true_all,

    "c_values": c_values,
    "lag": lag,

    "ENSEMBLE": ensemble_tag,

    "N": N,
    "q": q,
    "Delta_t": Delta_t,
    "dt_internal": dt_internal,
    "M_steps": M_steps,

    "n_c": n_c,
    "n_A": n_A,
    "n_sim": n_sim,

    "fail_total": fail_total,
    "fail_per_c_A": fail_per_c_A,
}

print("\n======================================================")
print("Loaded estimation dataset")
print(f"ENSEMBLE       = {ensemble_tag}")
print(f"N              = {N}")
print(f"q              = {q}")
print(f"Delta_t        = {Delta_t}")
print(f"lag            = {lag}")
print(f"n_c            = {n_c}")
print(f"n_A            = {n_A}")
print(f"n_sim          = {n_sim}")
print(f"c_values       = {c_values}")
print(f"Failures       = {fail_total} / {n_c * n_A * n_sim}")
print(f"Shape A_est    = {A_est_all.shape}")
print(f"Shape A_true   = {A_true_all.shape}")
print(f"Shape B_est    = {B_est_all.shape}")
print(f"Shape Q_est    = {Q_est_all.shape}")
print("======================================================")


######################################################################################################################
# COMPUTE TRUE MATRICES: Q_true, Sigma_inf_true, Sigma_dt_true
# A_true_all.shape = (n_c, n_A, N, N), B_true = I_N
######################################################################################################################

print(f"\nComputing TRUE matrices — N = {N}")

Q_true_all = np.full((n_c, n_A, N, N), np.nan, dtype=np.float32)
Sigma_inf_true_all = np.full((n_c, n_A, N, N), np.nan, dtype=np.float32)
Sigma_dt_true_all = np.full((n_c, n_A, N, N), np.nan, dtype=np.float32)

fail_nan = 0
fail_exception = 0

for i_c in range(n_c):

    c = c_values[i_c]

    for i_A in range(n_A):

        A_true = A_true_all[i_c, i_A]
        B_true = np.eye(N)

        if not np.isfinite(A_true).all():
            fail_nan += 1
            print(f"[NaN/Inf] skipped: c={c:.4f} i_A={i_A}")
            continue

        try:

            # Q_true = exp(-Delta_t A)
            Q_true = expm(-Delta_t * A_true)

            # A Sigma + Sigma A^T = 2B
            Sigma_inf_true = solve_continuous_lyapunov(A_true, 2.0 * B_true)
            Sigma_inf_true = 0.5 * (Sigma_inf_true + Sigma_inf_true.T)

            # Sigma_dt = Sigma_inf - Q Sigma_inf Q^T
            Sigma_dt_true = Sigma_inf_true - Q_true @ Sigma_inf_true @ Q_true.T
            Sigma_dt_true = 0.5 * (Sigma_dt_true + Sigma_dt_true.T)

            Q_true_all[i_c, i_A] = np.real(Q_true).astype(np.float32)
            Sigma_inf_true_all[i_c, i_A] = np.real(Sigma_inf_true).astype(np.float32)
            Sigma_dt_true_all[i_c, i_A] = np.real(Sigma_dt_true).astype(np.float32)

        except Exception as e:
            fail_exception += 1
            print(f"[EXCEPTION] c={c:.4f} i_A={i_A} -> {type(e).__name__}")
            continue

dataset["Q_true_all"] = Q_true_all
dataset["Sigma_inf_true_all"] = Sigma_inf_true_all
dataset["Sigma_dt_true_all"] = Sigma_dt_true_all

print("\nTRUE matrices computation complete.")
print(f"NaN/Inf failures   : {fail_nan}")
print(f"Exception failures : {fail_exception}")
print(f"Total failures     : {fail_nan + fail_exception}")


######################################################################################################################
# COMPUTE SPECTRA OF A, Q, SIGMA_DT, SIGMA_INF (true & estimated)
######################################################################################################################

print("\nComputing spectra of A, Q, Sigma_dt, Sigma_inf...")

spectra_start = time.time()


def compute_true_spectrum(matrix_all, name):

    eig_all = np.full((n_c, n_A, N), np.nan + 1j * np.nan, dtype=np.complex64)
    lambda_min_all = np.full((n_c, n_A), np.nan, dtype=np.float32)

    fail = 0

    for i_c, c in enumerate(c_values):

        print(f"Computing TRUE spectrum of {name} for c = {c:.4f}")

        for i_A in range(n_A):

            M = matrix_all[i_c, i_A]

            if not np.isfinite(M).all():
                fail += 1
                continue

            try:
                eigvals = np.linalg.eigvals(M)
                eig_all[i_c, i_A] = eigvals.astype(np.complex64)
                lambda_min_all[i_c, i_A] = np.min(np.real(eigvals)).astype(np.float32)
            except Exception:
                fail += 1

    return eig_all, lambda_min_all, fail


def compute_est_spectrum(matrix_all, name):

    eig_all = np.full((n_c, n_A, n_sim, N), np.nan + 1j * np.nan, dtype=np.complex64)
    lambda_min_all = np.full((n_c, n_A, n_sim), np.nan, dtype=np.float32)

    fail = 0

    for i_c, c in enumerate(c_values):

        print(f"Computing ESTIMATED spectrum of {name} for c = {c:.4f}")

        for i_A in range(n_A):

            for sim in range(n_sim):

                M = matrix_all[i_c, i_A, sim]

                if not np.isfinite(M).all():
                    fail += 1
                    continue

                try:
                    eigvals = np.linalg.eigvals(M)
                    eig_all[i_c, i_A, sim] = eigvals.astype(np.complex64)
                    lambda_min_all[i_c, i_A, sim] = np.min(np.real(eigvals)).astype(np.float32)
                except Exception:
                    fail += 1

    return eig_all, lambda_min_all, fail


eig_A_true_all, lambda_min_true_A, fail_true_A = compute_true_spectrum(A_true_all, "A")
eig_Q_true_all, lambda_min_true_Q, fail_true_Q = compute_true_spectrum(Q_true_all, "Q")
eig_Sigma_dt_true_all, lambda_min_true_Sigma_dt, fail_true_Sigma_dt = compute_true_spectrum(Sigma_dt_true_all, "Sigma_dt")
eig_Sigma_inf_true_all, lambda_min_true_Sigma_inf, fail_true_Sigma_inf = compute_true_spectrum(Sigma_inf_true_all, "Sigma_inf")

eig_A_est_all, lambda_min_est_A, fail_est_A = compute_est_spectrum(A_est_all, "A")
eig_Q_est_all, lambda_min_est_Q, fail_est_Q = compute_est_spectrum(Q_est_all, "Q")
eig_Sigma_dt_est_all, lambda_min_est_Sigma_dt, fail_est_Sigma_dt = compute_est_spectrum(Sigma_dt_est_all, "Sigma_dt")
eig_Sigma_inf_est_all, lambda_min_est_Sigma_inf, fail_est_Sigma_inf = compute_est_spectrum(Sigma_inf_est_all, "Sigma_inf")

dataset["eig_A_true_all"] = eig_A_true_all
dataset["eig_A_est_all"] = eig_A_est_all
dataset["lambda_min_true_A"] = lambda_min_true_A
dataset["lambda_min_est_A"] = lambda_min_est_A

dataset["eig_Q_true_all"] = eig_Q_true_all
dataset["eig_Q_est_all"] = eig_Q_est_all
dataset["lambda_min_true_Q"] = lambda_min_true_Q
dataset["lambda_min_est_Q"] = lambda_min_est_Q

dataset["eig_Sigma_dt_true_all"] = eig_Sigma_dt_true_all
dataset["eig_Sigma_dt_est_all"] = eig_Sigma_dt_est_all
dataset["lambda_min_true_Sigma_dt"] = lambda_min_true_Sigma_dt
dataset["lambda_min_est_Sigma_dt"] = lambda_min_est_Sigma_dt

dataset["eig_Sigma_inf_true_all"] = eig_Sigma_inf_true_all
dataset["eig_Sigma_inf_est_all"] = eig_Sigma_inf_est_all
dataset["lambda_min_true_Sigma_inf"] = lambda_min_true_Sigma_inf
dataset["lambda_min_est_Sigma_inf"] = lambda_min_est_Sigma_inf

print("\nSpectral computation complete.")
print(f"TRUE failures      : A={fail_true_A} Q={fail_true_Q} Sigma_dt={fail_true_Sigma_dt} Sigma_inf={fail_true_Sigma_inf}")
print(f"ESTIMATED failures : A={fail_est_A} Q={fail_est_Q} Sigma_dt={fail_est_Sigma_dt} Sigma_inf={fail_est_Sigma_inf}")
print(f"TOTAL TIME: {time.time() - spectra_start:.2f} sec")

gc.collect()


######################################################################################################################
# STABILITY MASK (permissive) — used for the "filtered spectra" plot
# Criterion: at least one finite eigenvalue, and min Re(lambda(A_est)) >= 0
######################################################################################################################

stable_mask = np.full((n_c, n_A, n_sim), False, dtype=bool)

table_rows = []

for i_c, c in enumerate(c_values):

    n_total = 0
    n_unstable = 0

    for i_A in range(n_A):

        for sim in range(n_sim):

            eigvals = eig_A_est_all[i_c, i_A, sim]
            eigvals = eigvals[np.isfinite(eigvals)]

            if len(eigvals) == 0:
                continue

            n_total += 1

            if np.min(np.real(eigvals)) >= 0:
                stable_mask[i_c, i_A, sim] = True
            else:
                n_unstable += 1

    table_rows.append({
        "c": c,
        "total": n_total,
        "unstable": n_unstable,
        "stable": n_total - n_unstable,
        "fraction_unstable": (n_unstable / n_total if n_total > 0 else np.nan)
    })

unstable_table = pd.DataFrame(table_rows)

dataset["stable_mask"] = stable_mask
dataset["unstable_table"] = unstable_table

print("\n===================================================")
print("UNSTABLE ESTIMATES (permissive mask)")
print("Criterion: min Re(lambda(A_est)) < 0")
print("===================================================")
print(unstable_table.to_string(index=False))


######################################################################################################################
# HESSIAN EIGENVALUES (strict stability mask: exactly N finite eigenvalues, all >= 0)
######################################################################################################################

print(f"\nComputing filtered Hessian and inverse-Hessian eigenvalues — N = {N}")

hess_start = time.time()

stable_A_est_mask = np.full((n_c, n_A, n_sim), False, dtype=bool)

unstable_rows = []

for i_c, c in enumerate(c_values):

    n_total = 0
    n_unstable = 0

    for i_A in range(n_A):

        for sim in range(n_sim):

            eig_A_est = eig_A_est_all[i_c, i_A, sim]
            eig_A_est = eig_A_est[np.isfinite(eig_A_est)]

            if len(eig_A_est) != N:
                continue

            n_total += 1

            if np.min(np.real(eig_A_est)) >= 0:
                stable_A_est_mask[i_c, i_A, sim] = True
            else:
                n_unstable += 1

    unstable_rows.append({
        "c": c,
        "valid_A_est": n_total,
        "stable_A_est": n_total - n_unstable,
        "unstable_A_est": n_unstable,
        "fraction_unstable": (n_unstable / n_total if n_total > 0 else np.nan)
    })

unstable_A_est_table = pd.DataFrame(unstable_rows)

print("\n===================================================")
print("UNSTABLE A_EST COUNTS (strict mask)")
print("Criterion: min Re(lambda(A_est)) < 0")
print("===================================================")
print(unstable_A_est_table.to_string(index=False))

eigens_H = {
    "HJJ_true": [], "HJJ_est": [],
    "HQQ_true": [], "HQQ_est": [],
    "HJJ_inv_true": [], "HJJ_inv_est": [],
    "HQQ_inv_true": [], "HQQ_inv_est": [],
}

fail_true = 0
fail_est = 0
skipped_unstable_est = 0


def compute_hessian_blocks(eig_Sdt, eig_Sinf):

    eig_Sdt = eig_Sdt[np.isfinite(eig_Sdt)]
    eig_Sinf = eig_Sinf[np.isfinite(eig_Sinf)]

    if len(eig_Sdt) != N or len(eig_Sinf) != N:
        return None

    eig_Sdt = np.real(eig_Sdt)
    eig_Sinf = np.real(eig_Sinf)

    if np.any(eig_Sdt == 0):
        return None

    eig_J = 1.0 / eig_Sdt

    eig_HJJ = np.outer(eig_Sdt, eig_Sdt).reshape(-1)
    eig_HQQ = 2.0 * np.outer(eig_J, eig_Sinf).reshape(-1)

    if np.any(eig_HJJ == 0) or np.any(eig_HQQ == 0):
        return None

    if not np.isfinite(eig_HJJ).all() or not np.isfinite(eig_HQQ).all():
        return None

    eig_HJJ_inv = 1.0 / eig_HJJ
    eig_HQQ_inv = 1.0 / eig_HQQ

    if not np.isfinite(eig_HJJ_inv).all() or not np.isfinite(eig_HQQ_inv).all():
        return None

    return (
        eig_HJJ.astype(np.float32),
        eig_HQQ.astype(np.float32),
        eig_HJJ_inv.astype(np.float32),
        eig_HQQ_inv.astype(np.float32),
    )


for i_c, c in enumerate(c_values):

    print(f"Processing c = {c:.4f} ... ", end="", flush=True)

    eig_HJJ_true_c, eig_HQQ_true_c = [], []
    eig_HJJ_inv_true_c, eig_HQQ_inv_true_c = [], []
    eig_HJJ_est_c, eig_HQQ_est_c = [], []
    eig_HJJ_inv_est_c, eig_HQQ_inv_est_c = [], []

    for i_A in range(n_A):

        out_true = compute_hessian_blocks(
            eig_Sigma_dt_true_all[i_c, i_A],
            eig_Sigma_inf_true_all[i_c, i_A]
        )

        if out_true is None:
            fail_true += 1
            continue

        eig_HJJ_true, eig_HQQ_true, eig_HJJ_inv_true, eig_HQQ_inv_true = out_true

        eig_HJJ_true_c.append(eig_HJJ_true)
        eig_HQQ_true_c.append(eig_HQQ_true)
        eig_HJJ_inv_true_c.append(eig_HJJ_inv_true)
        eig_HQQ_inv_true_c.append(eig_HQQ_inv_true)

    for i_A in range(n_A):

        for sim in range(n_sim):

            if not stable_A_est_mask[i_c, i_A, sim]:
                skipped_unstable_est += 1
                continue

            out_est = compute_hessian_blocks(
                eig_Sigma_dt_est_all[i_c, i_A, sim],
                eig_Sigma_inf_est_all[i_c, i_A, sim]
            )

            if out_est is None:
                fail_est += 1
                continue

            eig_HJJ_est, eig_HQQ_est, eig_HJJ_inv_est, eig_HQQ_inv_est = out_est

            eig_HJJ_est_c.append(eig_HJJ_est)
            eig_HQQ_est_c.append(eig_HQQ_est)
            eig_HJJ_inv_est_c.append(eig_HJJ_inv_est)
            eig_HQQ_inv_est_c.append(eig_HQQ_inv_est)

    eigens_H["HJJ_true"].append(np.array(eig_HJJ_true_c, dtype=np.float32))
    eigens_H["HQQ_true"].append(np.array(eig_HQQ_true_c, dtype=np.float32))
    eigens_H["HJJ_inv_true"].append(np.array(eig_HJJ_inv_true_c, dtype=np.float32))
    eigens_H["HQQ_inv_true"].append(np.array(eig_HQQ_inv_true_c, dtype=np.float32))

    eigens_H["HJJ_est"].append(np.array(eig_HJJ_est_c, dtype=np.float32))
    eigens_H["HQQ_est"].append(np.array(eig_HQQ_est_c, dtype=np.float32))
    eigens_H["HJJ_inv_est"].append(np.array(eig_HJJ_inv_est_c, dtype=np.float32))
    eigens_H["HQQ_inv_est"].append(np.array(eig_HQQ_inv_est_c, dtype=np.float32))

    gc.collect()

    print("Done")

dataset["eigens_H_stable_A_est"] = eigens_H
dataset["stable_A_est_mask"] = stable_A_est_mask
dataset["unstable_A_est_table"] = unstable_A_est_table

print("\nFiltered Hessian and inverse-Hessian eigenvalues computed.")
print(f"fail_true = {fail_true}, fail_est = {fail_est}, skipped_unstable_est = {skipped_unstable_est}")
print(f"TOTAL TIME: {time.time() - hess_start:.2f} sec")


######################################################################################################################
# CONDITION NUMBERS: kappa(A), kappa(H) — quartile summaries per c
######################################################################################################################


def condition_number_from_eigs(eigs, positive_only=True):

    eigs = np.asarray(eigs).reshape(-1)
    eigs = np.real(eigs)
    eigs = eigs[np.isfinite(eigs)]

    if positive_only:
        eigs = eigs[eigs > 0]

    if len(eigs) == 0:
        return np.nan

    lam_min = np.min(eigs)
    lam_max = np.max(eigs)

    if lam_min <= 0:
        return np.nan

    return lam_max / lam_min


def q25_med_q75(list_of_arrays):

    q25, med, q75 = [], [], []

    for arr in list_of_arrays:

        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            q25.append(np.nan)
            med.append(np.nan)
            q75.append(np.nan)
        else:
            q25.append(np.nanpercentile(arr, 25))
            med.append(np.nanmedian(arr))
            q75.append(np.nanpercentile(arr, 75))

    return np.array(q25), np.array(med), np.array(q75)


kappa_A_true_all = []
kappa_A_est_all = []

for i_c in range(n_c):

    kappa_A_true_c = []
    kappa_A_est_c = []

    for i_A in range(n_A):
        kappa_A_true_c.append(
            condition_number_from_eigs(eig_A_true_all[i_c, i_A], positive_only=True)
        )

    for i_A in range(n_A):
        for sim in range(n_sim):
            if not stable_A_est_mask[i_c, i_A, sim]:
                continue
            kappa_A_est_c.append(
                condition_number_from_eigs(eig_A_est_all[i_c, i_A, sim], positive_only=True)
            )

    kappa_A_true_all.append(np.array(kappa_A_true_c, dtype=float))
    kappa_A_est_all.append(np.array(kappa_A_est_c, dtype=float))

kappa_H_true_all = []
kappa_H_est_all = []

for i_c in range(n_c):

    kappa_H_true_c = []
    kappa_H_est_c = []

    for i in range(len(eigens_H["HJJ_true"][i_c])):
        H_true = np.concatenate([
            eigens_H["HJJ_true"][i_c][i].reshape(-1),
            eigens_H["HQQ_true"][i_c][i].reshape(-1)
        ])
        kappa_H_true_c.append(condition_number_from_eigs(H_true, positive_only=True))

    for i in range(len(eigens_H["HJJ_est"][i_c])):
        H_est = np.concatenate([
            eigens_H["HJJ_est"][i_c][i].reshape(-1),
            eigens_H["HQQ_est"][i_c][i].reshape(-1)
        ])
        kappa_H_est_c.append(condition_number_from_eigs(H_est, positive_only=True))

    kappa_H_true_all.append(np.array(kappa_H_true_c, dtype=float))
    kappa_H_est_all.append(np.array(kappa_H_est_c, dtype=float))

dataset["kappa_A_true_q25"], dataset["kappa_A_true_med"], dataset["kappa_A_true_q75"] = q25_med_q75(kappa_A_true_all)
dataset["kappa_A_est_q25"], dataset["kappa_A_est_med"], dataset["kappa_A_est_q75"] = q25_med_q75(kappa_A_est_all)
dataset["kappa_H_true_q25"], dataset["kappa_H_true_med"], dataset["kappa_H_true_q75"] = q25_med_q75(kappa_H_true_all)
dataset["kappa_H_est_q25"], dataset["kappa_H_est_med"], dataset["kappa_H_est_q75"] = q25_med_q75(kappa_H_est_all)

print("\nCondition number summaries (kappa(A), kappa(H)) computed.")


######################################################################################################################
# FROBENIUS ERROR FOR A (unfiltered)
######################################################################################################################

frob_A = np.full((n_c, n_A, n_sim), np.nan)

for i_c in range(n_c):
    for i_A in range(n_A):
        A_true = A_true_all[i_c, i_A]
        for sim in range(n_sim):
            A_est = A_est_all[i_c, i_A, sim]
            if np.isfinite(A_est).all():
                frob_A[i_c, i_A, sim] = np.linalg.norm(A_est - A_true, ord="fro") ** 2 / N

dataset["frob_A"] = frob_A

print("Frobenius error for A (unfiltered) computed.")


######################################################################################################################
# ABSOLUTE SQUARED FROBENIUS ERROR FOR SIGMA_INF (mean over sim, 90th-percentile clipped)
######################################################################################################################

frob_err_Sigma_inf_sq_over_N = np.full((n_c, n_A, n_sim), np.nan)

for i_c in range(n_c):
    for i_A in range(n_A):
        Sigma_inf_true = Sigma_inf_true_all[i_c, i_A]
        for sim in range(n_sim):
            Sigma_inf_est = Sigma_inf_est_all[i_c, i_A, sim]
            if not np.isfinite(Sigma_inf_est).all():
                continue
            frob_err_Sigma_inf_sq_over_N[i_c, i_A, sim] = (
                np.linalg.norm(Sigma_inf_est - Sigma_inf_true, ord="fro") ** 2 / N
            )

err_Sigma_inf_sq_over_N = np.nanmean(frob_err_Sigma_inf_sq_over_N, axis=2)

err_clean_Sigma_inf = err_Sigma_inf_sq_over_N.copy()
global_thr_Sigma_inf = np.nanpercentile(err_clean_Sigma_inf, 90)
err_clean_Sigma_inf[err_clean_Sigma_inf > global_thr_Sigma_inf] = np.nan

print(f"Sigma_inf error: removed values above 90th percentile = {global_thr_Sigma_inf:.4e}")

lambda_min_est_mean_A = np.nanmean(lambda_min_est_A, axis=2)

dataset["frob_err_Sigma_inf_sq_over_N"] = frob_err_Sigma_inf_sq_over_N
dataset["err_Sigma_inf_sq_over_N_mean_over_sim"] = err_Sigma_inf_sq_over_N
dataset["err_Sigma_inf_sq_over_N_clean"] = err_clean_Sigma_inf
dataset["lambda_min_est_mean_A"] = lambda_min_est_mean_A

SigmaInf_error_summary = pd.DataFrame({
    "c": c_values,
    "mean_lambda_min_true": np.nanmean(lambda_min_true_A, axis=1),
    "mean_lambda_min_est": np.nanmean(lambda_min_est_mean_A, axis=1),
    "mean_SigmaInf_error": np.nanmean(err_Sigma_inf_sq_over_N, axis=1),
    "median_SigmaInf_error": np.nanmedian(err_Sigma_inf_sq_over_N, axis=1),
    "std_SigmaInf_error": np.nanstd(err_Sigma_inf_sq_over_N, axis=1),
})

dataset["SigmaInf_error_summary"] = SigmaInf_error_summary

print("\n" + "=" * 90)
print("SIGMA_INF ERROR SUMMARY")
print("=" * 90)
print(SigmaInf_error_summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
print("=" * 90)


######################################################################################################################
# ABSOLUTE SQUARED FROBENIUS ERROR FOR A (mean over sim, 90th-percentile clipped)
######################################################################################################################

frob_err_A_sq_over_N = np.full((n_c, n_A, n_sim), np.nan)

for i_c in range(n_c):
    for i_A in range(n_A):
        A_true = A_true_all[i_c, i_A]
        for sim in range(n_sim):
            A_est = A_est_all[i_c, i_A, sim]
            if not np.isfinite(A_est).all():
                continue
            frob_err_A_sq_over_N[i_c, i_A, sim] = np.linalg.norm(A_est - A_true, ord="fro") ** 2 / N

err_A_sq_over_N = np.nanmean(frob_err_A_sq_over_N, axis=2)

err_clean_A = err_A_sq_over_N.copy()
global_thr_A = np.nanpercentile(err_clean_A, 90)
err_clean_A[err_clean_A > global_thr_A] = np.nan

print(f"A error: removed values above percentile threshold = {global_thr_A:.4e}")

dataset["frob_err_A_sq_over_N"] = frob_err_A_sq_over_N
dataset["err_A_sq_over_N_mean_over_sim"] = err_A_sq_over_N
dataset["err_A_sq_over_N_clean"] = err_clean_A

A_error_summary = pd.DataFrame({
    "c": c_values,
    "mean_lambda_min_true": np.nanmean(lambda_min_true_A, axis=1),
    "mean_lambda_min_est": np.nanmean(lambda_min_est_mean_A, axis=1),
    "mean_A_error": np.nanmean(err_A_sq_over_N, axis=1),
    "median_A_error": np.nanmedian(err_A_sq_over_N, axis=1),
    "std_A_error": np.nanstd(err_A_sq_over_N, axis=1),
})

dataset["A_error_summary"] = A_error_summary

print("\n" + "=" * 90)
print("A ERROR SUMMARY")
print("=" * 90)
print(A_error_summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
print("=" * 90)


######################################################################################################################
# SELF-AVERAGING OF FROBENIUS ERROR
# eps_F(A) = <||Ahat-A||_F>_s ,  CV_F(c) = Std_A(eps_F) / mean_A(eps_F)
######################################################################################################################

frob_err = np.full((n_c, n_A, n_sim), np.nan)

for i_c in range(n_c):
    for i_A in range(n_A):
        A_true = A_true_all[i_c, i_A]
        for sim in range(n_sim):
            A_est = A_est_all[i_c, i_A, sim]
            if not np.isfinite(A_est).all():
                continue
            frob_err[i_c, i_A, sim] = np.linalg.norm(A_est - A_true, ord="fro")

eps_F = np.nanmean(frob_err, axis=2)
mean_eps_F = np.nanmean(eps_F, axis=1)
std_eps_F = np.nanstd(eps_F, axis=1, ddof=1)
CV_F = std_eps_F / mean_eps_F

dataset["eps_F"] = eps_F
dataset["eps_F_bar_A"] = mean_eps_F
dataset["eps_F_std_A"] = std_eps_F
dataset["CV_F"] = CV_F

CV_F_table = pd.DataFrame({
    "c": c_values,
    "eps_F_bar_A": mean_eps_F,
    "eps_F_std_A": std_eps_F,
    "CV_F": CV_F,
})

dataset["CV_F_table"] = CV_F_table

print("\nSelf-averaging of Frobenius error (CV_F) computed.")
print(CV_F_table.to_string(index=False))


######################################################################################################################
# SELF-AVERAGING OF lambda_min(A)
# CV(c) = Std_A(lambda_min) / Mean_A(lambda_min)
######################################################################################################################

mean_lambda = np.nanmean(lambda_min_true_A, axis=1)
var_lambda = np.nanvar(lambda_min_true_A, axis=1, ddof=1)
std_lambda = np.sqrt(var_lambda)
CV_lambda = std_lambda / mean_lambda

dataset["lambda_min_mean_A"] = mean_lambda
dataset["lambda_min_std_A"] = std_lambda
dataset["lambda_min_CV_A"] = CV_lambda

lambda_table = pd.DataFrame({
    "c": c_values,
    "lambda_bar_A": mean_lambda,
    "lambda_std_A": std_lambda,
    "CV_lambda": CV_lambda,
})

dataset["lambda_min_CV_table"] = lambda_table

print("\nSelf-averaging of lambda_min(A) computed.")
print(lambda_table.to_string(index=False))


######################################################################################################################
# PROJECTED SOFT-MODE UNCERTAINTY
# MC estimate vs Wishart N-dim theory vs cW N-dim theory (tau-weighted effective sample size)
######################################################################################################################

r_true_proj = np.full((n_c, n_A), np.nan)
r_hat_proj = np.full((n_c, n_A, n_sim), np.nan)
M_eff_N_all = np.full((n_c, n_A), np.nan)

for i_c in range(n_c):
    for i_A in range(n_A):

        A = A_true_all[i_c, i_A]
        if not np.isfinite(A).all():
            continue

        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argmin(eigvals)
        r_true = eigvals[idx]
        u_star = eigvecs[:, idx]

        r_true_proj[i_c, i_A] = r_true

        # M_eff_N = M / <tau>_weighted ,  <tau>_weighted = mean(tau^2) / mean(tau)
        x = eigvals * Delta_t
        tau_all = (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))
        tau_mean_weighted = np.mean(tau_all ** 2) / np.mean(tau_all)
        M_eff_N_all[i_c, i_A] = M_steps / tau_mean_weighted

        for sim in range(n_sim):
            Q_hat = Q_est_all[i_c, i_A, sim]
            if not np.isfinite(Q_hat).all():
                continue
            q_hat_star = float(u_star.T @ Q_hat @ u_star)
            if q_hat_star <= 0:
                continue
            r_hat_proj[i_c, i_A, sim] = -np.log(q_hat_star) / Delta_t

sigma_proj_per_A = np.nanstd(r_hat_proj, axis=2, ddof=1)
rel_err_proj_per_A = sigma_proj_per_A / r_true_proj
r_mean = np.nanmean(r_true_proj, axis=1)
rel_err_emp = np.nanmean(rel_err_proj_per_A, axis=1)
rel_err_emp_std = np.nanstd(rel_err_proj_per_A, axis=1, ddof=1)

exp_factor = (np.exp(2.0 * r_true_proj * Delta_t) - 1.0) / (M_steps * r_true_proj ** 2 * Delta_t ** 2)

# THEORY 1: Wishart N-dim, kappa = (M-1)/(M-N-2)
kappa_W_N = (M_steps - 1) / (M_steps - N - 2)
rel_err_W_N = np.nanmean(np.sqrt(kappa_W_N * exp_factor), axis=1)

# THEORY 2: cW N-dim, kappa = (M_eff_N-1)/(M_eff_N-N-2)
kappa_cW_N_per_A = np.full_like(r_true_proj, np.nan)
valid_N = np.isfinite(M_eff_N_all) & (M_eff_N_all > N + 2)
kappa_cW_N_per_A[valid_N] = (M_eff_N_all[valid_N] - 1) / (M_eff_N_all[valid_N] - N - 2)

rel_err_cW_N_per_A = np.full_like(r_true_proj, np.nan)
rel_err_cW_N_per_A[valid_N] = np.sqrt(kappa_cW_N_per_A[valid_N] * exp_factor[valid_N])

rel_err_cW_N = np.nanmean(rel_err_cW_N_per_A, axis=1)

order = np.argsort(r_mean)

soft_mode_summary = pd.DataFrame({
    "c": c_values[order],
    "r_mean": r_mean[order],
    "sigma_r_over_r_MC": rel_err_emp[order],
    "sigma_r_over_r_MC_std": rel_err_emp_std[order],
    "M_eff_N": np.nanmean(M_eff_N_all, axis=1)[order],
    "kappa_W_N": np.full_like(r_mean, kappa_W_N),
    "kappa_cW_N": np.nanmean(kappa_cW_N_per_A, axis=1)[order],
    "theory_WN": rel_err_W_N[order],
    "theory_cWN": rel_err_cW_N[order],
})

dataset["r_true_proj"] = r_true_proj
dataset["r_hat_proj"] = r_hat_proj
dataset["M_eff_N_all"] = M_eff_N_all
dataset["soft_mode_order"] = order
dataset["soft_mode_r_plot"] = r_mean[order]
dataset["soft_mode_emp_plot"] = rel_err_emp[order]
dataset["soft_mode_emp_std_plot"] = rel_err_emp_std[order]
dataset["soft_mode_WN_plot"] = rel_err_W_N[order]
dataset["soft_mode_cWN_plot"] = rel_err_cW_N[order]
dataset["soft_mode_c_plot"] = c_values[order]
dataset["soft_mode_uncertainty_summary"] = soft_mode_summary

print("\nProjected soft-mode uncertainty computed.")
print(soft_mode_summary.round(6).to_string(index=False))


######################################################################################################################
# SAVE
######################################################################################################################

with open(output_path, "wb") as f:
    pickle.dump(dataset, f)

print(f"\nSaved processed dataset to:\n{output_path}")
print(f"\nTOTAL TIME: {time.time() - total_start:.2f} sec")
