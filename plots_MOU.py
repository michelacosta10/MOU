import argparse
import os
import pickle

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


######################################################################################################################
# ARGUMENTS
######################################################################################################################

parser = argparse.ArgumentParser(
    description="Generate all figures from the dataset produced by process_MOU.py."
)
parser.add_argument("--dataset", type=str, required=True, help="Path to the pickle saved by process_MOU.py")
parser.add_argument("--output-dir", type=str, default="./figures")
parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument(
    "--figures", type=str, nargs="+", default=["all"],
    help="Which figures to generate (default: all). See FIGURES dict in this script for names."
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


def savefig(fig, name):
    path = os.path.join(args.output_dir, f"{name}.{args.format}")
    fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {path}")


######################################################################################################################
# FIGURE 1: lambda_min(Sigma_inf_est) vs q_eff (symlog)
######################################################################################################################

def fig_lambda_min_vs_qeff(dataset):

    lambda_min_true_A = dataset["lambda_min_true_A"]          # (n_c, n_A)
    lambda_min_est = dataset["lambda_min_est_Sigma_inf"]       # (n_c, n_A, n_sim)

    N = dataset["N"]
    M = dataset["M_steps"]
    Delta_t = dataset["Delta_t"]
    c_values = dataset["c_values"]
    q = dataset["q"]

    n_c, n_A, n_sim = lambda_min_est.shape

    q_eff_all = (M / N) * (1 - np.exp(-2 * lambda_min_true_A * Delta_t)) / \
                          (1 + np.exp(-2 * lambda_min_true_A * Delta_t))
    q_eff_rep = np.repeat(q_eff_all[:, :, np.newaxis], n_sim, axis=2)

    colors = cm.magma(np.linspace(0.15, 0.85, n_c))

    fig, ax = plt.subplots(figsize=(7, 5))

    for i_c, c in enumerate(c_values):

        x = q_eff_rep[i_c].flatten()
        y = lambda_min_est[i_c].flatten()

        mask = np.isfinite(x) & np.isfinite(y)

        ax.scatter(x[mask], y[mask], s=8, alpha=0.35, color=colors[i_c],
                   label=f'$c={c:.1f}$', rasterized=True)

    ax.axvline(1, color='r', lw=1.2, ls='--', label=r'$q_{\rm eff}=1$')
    ax.axhline(0, color='k', lw=0.8, ls=':')

    ax.set_yscale('symlog', linthresh=1e-3)

    ax.set_xlabel(r'$q_{\rm eff}$', fontsize=13)
    ax.set_ylabel(r'$\lambda_{\min}(\hat{\Sigma}_{\infty})$', fontsize=13)
    ax.set_title(
        rf'$\lambda_{{\min}}(\hat{{\Sigma}}_{{\infty}})$ vs $q_{{\rm eff}}$  '
        rf'$(q={q},\, N={N},\, \Delta t={Delta_t})$',
        fontsize=12
    )

    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(markerscale=2, fontsize=9, frameon=False, loc='lower right')

    fig.tight_layout()
    savefig(fig, "lambda_min_Sigma_inf_vs_qeff")


######################################################################################################################
# FIGURE 2: eigenvalue spectra of A, Q, Sigma_dt, Sigma_inf — unfiltered
######################################################################################################################

def fig_spectra_unfiltered(dataset):

    spectra_dict = {
        "A": (dataset["eig_A_true_all"], dataset["eig_A_est_all"], r"$A$"),
        "Q": (dataset["eig_Q_true_all"], dataset["eig_Q_est_all"], r"$Q$"),
        "Sigma_dt": (dataset["eig_Sigma_dt_true_all"], dataset["eig_Sigma_dt_est_all"], r"$\Sigma_{\Delta t}$"),
        "Sigma_inf": (dataset["eig_Sigma_inf_true_all"], dataset["eig_Sigma_inf_est_all"], r"$\Sigma_{\infty}$"),
    }

    c_values = dataset["c_values"]
    n_c = dataset["n_c"]

    colors = cm.magma(np.linspace(0.15, 0.90, n_c))

    n_rows = len(spectra_dict)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4.2 * n_rows), squeeze=False)

    for row, (key, (eig_true_all, eig_est_all, label)) in enumerate(spectra_dict.items()):

        ax_true = axes[row, 0]
        ax_est = axes[row, 1]

        if key == "Sigma_inf":

            all_true = np.real(eig_true_all.reshape(-1))
            all_true = all_true[np.isfinite(all_true)]
            xmin_true, xmax_true = np.min(all_true), np.max(all_true)
            pad_true = 0.05 * (xmax_true - xmin_true)

            all_est = np.real(eig_est_all.reshape(-1))
            all_est = all_est[np.isfinite(all_est)]
            xmin_est, xmax_est = np.min(all_est), np.max(all_est)
            pad_est = 0.05 * (xmax_est - xmin_est)

        else:

            all_true = np.real(eig_true_all.reshape(-1))
            all_est = np.real(eig_est_all.reshape(-1))
            all_vals = np.concatenate([all_true, all_est])
            all_vals = all_vals[np.isfinite(all_vals)]
            xmin, xmax = np.min(all_vals), np.max(all_vals)
            pad = 0.05 * (xmax - xmin)

        for i_c, c in enumerate(c_values):

            eig_true = eig_true_all[i_c].reshape(-1)
            eig_est = eig_est_all[i_c].reshape(-1)

            eig_true = np.real(eig_true[np.isfinite(eig_true)])
            eig_est = np.real(eig_est[np.isfinite(eig_est)])

            ax_true.hist(eig_true, bins=80, density=True, histtype="step",
                        linewidth=2.0, color=colors[i_c], label=f"c = {c:g}")

            ax_est.hist(eig_est, bins=80, density=True, histtype="step",
                       linewidth=2.0, color=colors[i_c], label=f"c = {c:g}")

        ax_true.set_title(label + r" true spectrum")
        ax_true.set_xlabel(r"$\lambda$")
        ax_true.set_ylabel("density")

        ax_est.set_title(label + r" estimated spectrum")
        ax_est.set_xlabel(r"$\lambda$")
        ax_est.set_ylabel("density")

        if key == "Sigma_inf":
            ax_true.set_xlim(xmin_true - pad_true, xmax_true + pad_true)
            ax_est.set_xlim(xmin_est - pad_est, xmax_est + pad_est)
        else:
            ax_true.set_xlim(xmin - pad, xmax + pad)
            ax_est.set_xlim(xmin - pad, xmax + pad)

        ax_true.legend(frameon=False, fontsize=8)
        ax_est.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    savefig(fig, "spectra_unfiltered")


######################################################################################################################
# FIGURE 3: eigenvalue spectra — estimated filtered to stable A_est only (permissive mask)
######################################################################################################################

def fig_spectra_filtered(dataset):

    spectra_dict = {
        "A": (dataset["eig_A_true_all"], dataset["eig_A_est_all"], r"$A$"),
        "Q": (dataset["eig_Q_true_all"], dataset["eig_Q_est_all"], r"$Q$"),
        "Sigma_dt": (dataset["eig_Sigma_dt_true_all"], dataset["eig_Sigma_dt_est_all"], r"$\Sigma_{\Delta t}$"),
        "Sigma_inf": (dataset["eig_Sigma_inf_true_all"], dataset["eig_Sigma_inf_est_all"], r"$\Sigma_{\infty}$"),
    }

    c_values = dataset["c_values"]
    n_c = dataset["n_c"]
    stable_mask = dataset["stable_mask"]

    print("\nUnstable estimates table (used to filter this figure):")
    print(dataset["unstable_table"].to_string(index=False))

    colors = cm.magma(np.linspace(0.15, 0.90, n_c))

    n_rows = len(spectra_dict)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4.2 * n_rows), squeeze=False)

    for row, (key, (eig_true_all, eig_est_all, label)) in enumerate(spectra_dict.items()):

        ax_true = axes[row, 0]
        ax_est = axes[row, 1]

        if key == "Sigma_inf":

            all_true = eig_true_all.reshape(-1)
            all_true = all_true[np.isfinite(all_true)]

            all_est = []
            for i_c in range(n_c):
                tmp = eig_est_all[i_c][stable_mask[i_c]]
                if len(tmp) > 0:
                    all_est.append(tmp.reshape(-1))

            if len(all_est) > 0:
                all_est = np.concatenate(all_est)
                all_est = all_est[np.isfinite(all_est)]

            xmin_true, xmax_true = np.min(all_true), np.max(all_true)
            xmin_est, xmax_est = np.min(all_est), np.max(all_est)
            pad_true = 0.05 * (xmax_true - xmin_true)
            pad_est = 0.05 * (xmax_est - xmin_est)

        else:

            all_true = eig_true_all.reshape(-1)

            all_est = []
            for i_c in range(n_c):
                tmp = eig_est_all[i_c][stable_mask[i_c]]
                if len(tmp) > 0:
                    all_est.append(tmp.reshape(-1))

            all_true = all_true[np.isfinite(all_true)]

            if len(all_est) > 0:
                all_est = np.concatenate(all_est)
                all_est = all_est[np.isfinite(all_est)]
            else:
                all_est = np.array([0.0])

            all_vals = np.concatenate([all_true, all_est])
            xmin, xmax = np.min(all_vals), np.max(all_vals)
            pad = 0.05 * (xmax - xmin)

        for i_c, c in enumerate(c_values):

            eig_true = eig_true_all[i_c].reshape(-1)
            eig_true = np.real(eig_true[np.isfinite(eig_true)])

            eig_est = eig_est_all[i_c][stable_mask[i_c]].reshape(-1)
            eig_est = np.real(eig_est[np.isfinite(eig_est)])

            ax_true.hist(eig_true, bins=80, density=True, histtype="step",
                        linewidth=2, color=colors[i_c], label=f"c={c:g}")

            if len(eig_est) > 0:
                ax_est.hist(eig_est, bins=80, density=True, histtype="step",
                           linewidth=2, color=colors[i_c], label=f"c={c:g}")

        ax_true.set_title(label + " true spectrum")
        ax_est.set_title(label + " estimated spectrum\n(stable A only)")

        ax_true.set_xlabel(r"$\lambda$")
        ax_est.set_xlabel(r"$\lambda$")
        ax_true.set_ylabel("density")
        ax_est.set_ylabel("density")

        if key == "Sigma_inf":
            ax_true.set_xscale("symlog", linthresh=1e-2)
            ax_est.set_xscale("symlog", linthresh=1e-2)

        ax_true.legend(frameon=False, fontsize=8)
        ax_est.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    savefig(fig, "spectra_filtered_stable")


######################################################################################################################
# FIGURE 4: Frobenius error distribution for A, per c (99th percentile clip)
######################################################################################################################

def fig_frobenius_error_A_hist(dataset):

    frob_A = dataset["frob_A"]
    c_values = dataset["c_values"]
    n_c = dataset["n_c"]

    colors = cm.magma(np.linspace(0.15, 0.90, n_c))

    all_values = frob_A.reshape(-1)
    all_values = all_values[np.isfinite(all_values)]
    clip_max = np.nanpercentile(all_values, 99)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i_c, c in enumerate(c_values):

        values = frob_A[i_c].reshape(-1)
        values = values[np.isfinite(values)]
        values = values[values <= clip_max]

        ax.hist(values, bins=60, density=True, histtype="step",
               linewidth=2, color=colors[i_c], label=f"c={c:g}")

    ax.set_xlabel(r"$\|\hat{A}-A\|_F^2/N$", fontsize=13)
    ax.set_ylabel("density", fontsize=13)
    ax.set_title(r"Distribution of $A_{GOE}$ Frobenius error per $c$ ", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    savefig(fig, "frobenius_error_A_hist")


######################################################################################################################
# FIGURE 5: Sigma_inf error vs stability (true and estimated lambda_min)
######################################################################################################################

def fig_sigma_inf_error_vs_stability(dataset):

    print("\nSigma_inf error summary:")
    print(dataset["SigmaInf_error_summary"].to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    lambda_min_true_A = dataset["lambda_min_true_A"]
    lambda_min_est_mean_A = dataset["lambda_min_est_mean_A"]
    err_clean = dataset["err_Sigma_inf_sq_over_N_clean"]
    c_values = dataset["c_values"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for i_c, c in enumerate(c_values):
        mask = np.isfinite(err_clean[i_c])
        ax.scatter(lambda_min_true_A[i_c][mask], err_clean[i_c][mask], alpha=0.75, s=35, label=f"c={c:g}")
    ax.set_xlabel(r"$\lambda_{\min}(A)$", fontsize=14)
    ax.set_ylabel(
        r"$\left\langle\frac{\|\widehat{\Sigma}_{\infty}-\Sigma_{\infty}\|_F^2}{N}\right\rangle_s$",
        fontsize=14
    )
    ax.set_title(r"$\Sigma_{\infty}$ error vs true stability", fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i_c, c in enumerate(c_values):
        mask = np.isfinite(err_clean[i_c])
        ax.scatter(lambda_min_est_mean_A[i_c][mask], err_clean[i_c][mask], alpha=0.75, s=35, label=f"c={c:g}")
    ax.set_xlabel(r"$\left\langle\lambda_{\min}(\widehat A)\right\rangle_s$", fontsize=14)
    ax.set_ylabel(
        r"$\left\langle\frac{\|\widehat{\Sigma}_{\infty}-\Sigma_{\infty}\|_F^2}{N}\right\rangle_s$",
        fontsize=14
    )
    ax.set_title(r"$\Sigma_{\infty}$ error vs estimated stability", fontsize=14)
    ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(c_values), 5),
              fontsize=10, bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout()
    savefig(fig, "sigma_inf_error_vs_stability")


######################################################################################################################
# FIGURE 6: A error vs stability (true and estimated lambda_min)
######################################################################################################################

def fig_A_error_vs_stability(dataset):

    print("\nA error summary:")
    print(dataset["A_error_summary"].to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    lambda_min_true_A = dataset["lambda_min_true_A"]
    lambda_min_est_mean_A = dataset["lambda_min_est_mean_A"]
    err_clean = dataset["err_A_sq_over_N_clean"]
    c_values = dataset["c_values"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    for i_c, c in enumerate(c_values):
        mask = np.isfinite(err_clean[i_c])
        ax.scatter(lambda_min_true_A[i_c][mask], err_clean[i_c][mask], alpha=0.75, s=35, label=f"c={c:g}")
    ax.set_xlabel(r"$\lambda_{\min}(A)$", fontsize=14)
    ax.set_ylabel(r"$\left\langle\frac{\|\widehat A-A\|_F^2}{N}\right\rangle_s$", fontsize=14)
    ax.set_title(r"$A$ error vs true stability", fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i_c, c in enumerate(c_values):
        mask = np.isfinite(err_clean[i_c])
        ax.scatter(lambda_min_est_mean_A[i_c][mask], err_clean[i_c][mask], alpha=0.75, s=35, label=f"c={c:g}")
    ax.set_xlabel(r"$\left\langle\lambda_{\min}(\widehat A)\right\rangle_s$", fontsize=14)
    ax.set_ylabel(r"$\left\langle\frac{\|\widehat A-A\|_F^2}{N}\right\rangle_s$", fontsize=14)
    ax.set_title(r"$A$ error vs estimated stability", fontsize=14)
    ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(c_values), 5),
              fontsize=10, bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout()
    savefig(fig, "A_error_vs_stability")


######################################################################################################################
# FIGURE 7: self-averaging of Frobenius error (CV_F)
######################################################################################################################

def fig_self_averaging_frobenius(dataset):

    print("\nSelf-averaging of Frobenius error:")
    print(dataset["CV_F_table"].to_string(index=False))

    c_values = dataset["c_values"]
    CV_F = dataset["CV_F"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c_values, CV_F, "o-", linewidth=2, markersize=8)
    ax.set_xlabel(r"$c$", fontsize=14)
    ax.set_ylabel(
        r"$\frac{\mathrm{Std}_A\!\left(\langle\|\widehat A-A\|_F\rangle_s\right)}"
        r"{\overline{\langle\|\widehat A-A\|_F\rangle_s}^{\,A}}$",
        fontsize=13
    )
    ax.set_title(r"Self-averaging of Frobenius estimation error", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "self_averaging_frobenius_CV")


######################################################################################################################
# FIGURE 8: self-averaging of lambda_min(A)
######################################################################################################################

def fig_self_averaging_lambda_min(dataset):

    print("\nSelf-averaging of lambda_min(A):")
    print(dataset["lambda_min_CV_table"].to_string(index=False))

    c_values = dataset["c_values"]
    CV_lambda = dataset["lambda_min_CV_A"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(c_values, CV_lambda, "o-", linewidth=2, markersize=8)
    ax.set_xlabel(r"$c$", fontsize=14)
    ax.set_ylabel(
        r"$\frac{\mathrm{Std}_A(\lambda_{\min}(A))}{\overline{\lambda_{\min}(A)}^{\,A}}$",
        fontsize=14
    )
    ax.set_title(
        r"Self-averaging of $\lambda_{\min}(A)$"
        "\n"
        r"$CV_{\lambda}(c)=\frac{\mathrm{Std}_A(\lambda_{\min}(A))}{\overline{\lambda_{\min}(A)}^{\,A}}$",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, "self_averaging_lambda_min_CV")


######################################################################################################################
# FIGURE 9: Hessian spectra (HQQ, HJJ, H) — stable A_est only
######################################################################################################################

def fig_hessian_spectra(dataset):

    eigens_H = dataset["eigens_H_stable_A_est"]
    c_values = dataset["c_values"]
    n_c = dataset["n_c"]

    colors = cm.magma(np.linspace(0.15, 0.90, n_c))

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    for i_c, c in enumerate(c_values):

        color = colors[i_c]

        HQQ_true = eigens_H["HQQ_true"][i_c].reshape(-1)
        HQQ_est = eigens_H["HQQ_est"][i_c].reshape(-1)
        HQQ_true = HQQ_true[np.isfinite(HQQ_true)]
        HQQ_est = HQQ_est[np.isfinite(HQQ_est)]
        HQQ_true = HQQ_true[HQQ_true > 0]
        HQQ_est = HQQ_est[HQQ_est > 0]

        axes[0, 0].hist(HQQ_true, bins=100, density=True, histtype="step",
                       linewidth=2, color=color, label=f"c = {c:g}")
        axes[0, 1].hist(HQQ_est, bins=100, density=True, histtype="step",
                       linewidth=2, color=color, label=f"c = {c:g}")

        HJJ_true = eigens_H["HJJ_true"][i_c].reshape(-1)
        HJJ_est = eigens_H["HJJ_est"][i_c].reshape(-1)
        HJJ_true = HJJ_true[np.isfinite(HJJ_true)]
        HJJ_est = HJJ_est[np.isfinite(HJJ_est)]
        HJJ_true = HJJ_true[HJJ_true > 0]
        HJJ_est = HJJ_est[HJJ_est > 0]

        axes[1, 0].hist(HJJ_true, bins=100, density=True, histtype="step", linewidth=2, color=color)
        axes[1, 1].hist(HJJ_est, bins=100, density=True, histtype="step", linewidth=2, color=color)

        H_true = np.concatenate([HJJ_true, HQQ_true])
        H_est = np.concatenate([HJJ_est, HQQ_est])

        axes[2, 0].hist(H_true, bins=100, density=True, histtype="step", linewidth=2, color=color)
        axes[2, 1].hist(H_est, bins=100, density=True, histtype="step", linewidth=2, color=color)

    axes[0, 0].set_title(r"$H_{QQ}$ true")
    axes[0, 1].set_title(r"$H_{QQ}$ estimated")
    axes[1, 0].set_title(r"$H_{JJ}$ true")
    axes[1, 1].set_title(r"$H_{JJ}$ estimated")
    axes[2, 0].set_title(r"$H = H_{JJ}\cup H_{QQ}$ true")
    axes[2, 1].set_title(r"$H = H_{JJ}\cup H_{QQ}$ estimated")

    for ax in axes.flatten():
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("density")
        ax.set_xscale("log")

    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    savefig(fig, "hessian_spectra")


######################################################################################################################
# FIGURE 10: paper plot — A and H spectra + condition number vs c
######################################################################################################################

def fig_paper_plot(dataset):

    c_values = dataset["c_values"]
    n_c = dataset["n_c"]
    q = dataset["q"]

    colors = cm.magma(np.linspace(0.15, 0.90, n_c))
    eigens_H = dataset["eigens_H_stable_A_est"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    A_true_all = np.real(dataset["eig_A_true_all"].reshape(-1))
    A_est_all = np.real(dataset["eig_A_est_all"].reshape(-1))
    A_all = np.concatenate([A_true_all, A_est_all])
    A_all = A_all[np.isfinite(A_all)]
    xmin_A, xmax_A = np.min(A_all), np.max(A_all)
    pad_A = 0.05 * (xmax_A - xmin_A)

    H_all = []
    for i_c in range(n_c):
        H_true = np.concatenate([eigens_H["HJJ_true"][i_c].reshape(-1), eigens_H["HQQ_true"][i_c].reshape(-1)])
        H_est = np.concatenate([eigens_H["HJJ_est"][i_c].reshape(-1), eigens_H["HQQ_est"][i_c].reshape(-1)])
        H_true = H_true[np.isfinite(H_true)]
        H_est = H_est[np.isfinite(H_est)]
        H_true = H_true[H_true > 0]
        H_est = H_est[H_est > 0]
        if len(H_true) > 0:
            H_all.append(H_true)
        if len(H_est) > 0:
            H_all.append(H_est)
    H_all = np.concatenate(H_all)
    xmin_H = np.min(H_all)

    for i_c, c in enumerate(c_values):

        A_true = np.real(dataset["eig_A_true_all"][i_c].reshape(-1))
        A_est = np.real(dataset["eig_A_est_all"][i_c].reshape(-1))
        A_true = A_true[np.isfinite(A_true)]
        A_est = A_est[np.isfinite(A_est)]

        axes[0, 0].hist(A_true, bins=100, density=True, histtype="step",
                       linewidth=2, color=colors[i_c], label=f"c={c:g}")
        axes[0, 1].hist(A_est, bins=100, density=True, histtype="step",
                       linewidth=2, color=colors[i_c], label=f"c={c:g}")

    for i_c, c in enumerate(c_values):

        H_true = np.concatenate([eigens_H["HJJ_true"][i_c].reshape(-1), eigens_H["HQQ_true"][i_c].reshape(-1)])
        H_est = np.concatenate([eigens_H["HJJ_est"][i_c].reshape(-1), eigens_H["HQQ_est"][i_c].reshape(-1)])
        H_true = H_true[np.isfinite(H_true)]
        H_est = H_est[np.isfinite(H_est)]
        H_true = H_true[H_true > 0]
        H_est = H_est[H_est > 0]

        axes[1, 0].hist(H_true, bins=120, density=True, histtype="step",
                       linewidth=2, color=colors[i_c], label=f"c={c:g}")
        axes[1, 1].hist(H_est, bins=120, density=True, histtype="step",
                       linewidth=2, color=colors[i_c], label=f"c={c:g}")

    axes[0, 0].set_xlim(xmin_A - pad_A, 6)
    axes[0, 1].set_xlim(xmin_A - pad_A, 6)
    axes[1, 0].set_xlim(xmin_H, 1e2)
    axes[1, 1].set_xlim(xmin_H, 1e2)
    axes[1, 0].set_xscale("log")
    axes[1, 1].set_xscale("log")

    ymax_A = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(0, ymax_A)
    axes[0, 1].set_ylim(0, ymax_A)

    ymax_H = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(0, ymax_H)
    axes[1, 1].set_ylim(0, ymax_H)

    bordeaux_dark = "#6d0f2d"
    bordeaux_light = "#b34a6f"

    axes[0, 2].fill_between(c_values, dataset["kappa_A_true_q25"], dataset["kappa_A_true_q75"],
                            color=bordeaux_light, alpha=0.15)
    axes[0, 2].plot(c_values, dataset["kappa_A_true_med"], "--", linewidth=1.4, marker="o",
                    markersize=4, color=bordeaux_light, label="True")
    axes[0, 2].fill_between(c_values, dataset["kappa_A_est_q25"], dataset["kappa_A_est_q75"],
                            color=bordeaux_dark, alpha=0.15)
    axes[0, 2].plot(c_values, dataset["kappa_A_est_med"], "-", linewidth=1.4, marker="s",
                    markersize=4, color=bordeaux_dark, label="Estimated")

    axes[1, 2].fill_between(c_values, dataset["kappa_H_true_q25"], dataset["kappa_H_true_q75"],
                            color=bordeaux_light, alpha=0.15)
    axes[1, 2].plot(c_values, dataset["kappa_H_true_med"], "--", linewidth=1.4, marker="o",
                    markersize=4, color=bordeaux_light, label="True")
    axes[1, 2].fill_between(c_values, dataset["kappa_H_est_q25"], dataset["kappa_H_est_q75"],
                            color=bordeaux_dark, alpha=0.15)
    axes[1, 2].plot(c_values, dataset["kappa_H_est_med"], "-", linewidth=1.4, marker="s",
                    markersize=4, color=bordeaux_dark, label="Estimated")

    axes[0, 0].set_title(r"$A^*_{\mathrm{GOE}}$ spectrum")
    axes[0, 1].set_title(r"$\hat{A}_{\mathrm{GOE}}$ spectrum ($q=" + str(q) + r"$)")
    axes[0, 2].set_title(r"$\kappa(A_{\mathrm{GOE}})$ vs $\kappa(\hat{A}_{\mathrm{GOE}})$ ($q=" + str(q) + r"$)")
    axes[1, 0].set_title(r"$\mathcal{I}_{\mathrm{GOE}}$ spectrum")
    axes[1, 1].set_title(r"$\hat{\mathcal{I}}_{\mathrm{GOE}}$ spectrum ($q=" + str(q) + r"$)")
    axes[1, 2].set_title(r"$\kappa(\mathcal{I}_{\mathrm{GOE}})$ vs $\kappa(\hat{\mathcal{I}}_{\mathrm{GOE}})$ ($q=" + str(q) + r"$)")

    for ax in axes[:, :2].flatten():
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("density")

    axes[0, 2].set_xlabel(r"$c$")
    axes[0, 2].set_ylabel("condition number")
    axes[1, 2].set_xlabel(r"$c$")
    axes[1, 2].set_ylabel("condition number")

    for ax in axes.flatten():
        ax.legend(frameon=False, fontsize=8)

    axes[0, 2].grid(alpha=0.25)
    axes[1, 2].grid(alpha=0.25)

    fig.tight_layout()
    savefig(fig, "paper_plot_A_H_kappa")


######################################################################################################################
# FIGURE 11: projected soft-mode uncertainty (MC vs Wishart vs cW theory)
######################################################################################################################

def fig_soft_mode_uncertainty(dataset):

    print("\nSoft-mode uncertainty summary:")
    print(dataset["soft_mode_uncertainty_summary"].round(6).to_string(index=False))

    N = dataset["N"]
    q = dataset["q"]
    Delta_t = dataset["Delta_t"]

    r_plot = dataset["soft_mode_r_plot"]
    emp_plot = dataset["soft_mode_emp_plot"]
    emp_std_plot = dataset["soft_mode_emp_std_plot"]
    WN_plot = dataset["soft_mode_WN_plot"]
    cWN_plot = dataset["soft_mode_cWN_plot"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.errorbar(r_plot, emp_plot, yerr=emp_std_plot, fmt="o", markersize=6, capsize=3,
               elinewidth=1.2, color="black", label=r"Monte Carlo")

    ax.plot(r_plot, WN_plot, "s--", markersize=5, linewidth=2.0, color="firebrick",
           label=r"Wishart $\kappa_W=(M-1)/(M-N-2)$")

    ax.plot(r_plot, cWN_plot, "d-.", markersize=5, linewidth=2.0, color="steelblue",
           label=r"cW $\kappa_{\rm cW}=(M_{\rm eff}^N-1)/(M_{\rm eff}^N-N-2)$")

    ax.set_xlabel(r"$r$", fontsize=13)
    ax.set_ylabel(r"$\sigma_r/r$", fontsize=13)
    ax.set_title(rf"Relative soft-mode uncertainty $(N={N},\ q={q},\ \Delta t={Delta_t})$", fontsize=12)

    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11, direction="out", length=4, width=1)
    ax.legend(frameon=False, fontsize=9.5, loc="best")

    fig.tight_layout()
    savefig(fig, "projected_soft_mode_uncertainty")


######################################################################################################################
# MAIN
######################################################################################################################

FIGURES = {
    "lambda_min_vs_qeff": fig_lambda_min_vs_qeff,
    "spectra_unfiltered": fig_spectra_unfiltered,
    "spectra_filtered": fig_spectra_filtered,
    "frobenius_error_A": fig_frobenius_error_A_hist,
    "sigma_inf_error_vs_stability": fig_sigma_inf_error_vs_stability,
    "A_error_vs_stability": fig_A_error_vs_stability,
    "self_averaging_frobenius": fig_self_averaging_frobenius,
    "self_averaging_lambda_min": fig_self_averaging_lambda_min,
    "hessian_spectra": fig_hessian_spectra,
    "paper_plot": fig_paper_plot,
    "soft_mode_uncertainty": fig_soft_mode_uncertainty,
}


if __name__ == "__main__":

    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)

    print(f"Loaded processed dataset from: {args.dataset}")
    print(f"ENSEMBLE={dataset['ENSEMBLE']}  N={dataset['N']}  q={dataset['q']}  c_values={dataset['c_values']}")

    requested = list(FIGURES.keys()) if args.figures == ["all"] else args.figures

    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        raise ValueError(f"Unknown figure name(s): {unknown}. Available: {list(FIGURES.keys())}")

    for name in requested:
        print(f"\n--- Generating figure: {name} ---")
        FIGURES[name](dataset)

    print(f"\nAll requested figures saved to: {args.output_dir}")
