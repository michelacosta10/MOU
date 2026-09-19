import argparse
import os

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


######################################################################################################################
# ARGUMENTS
######################################################################################################################

parser = argparse.ArgumentParser(
    description="Theoretical resolvability / stability-inference phase diagrams for the GOE ensemble "
                 "(paper Fig. 3: 'reliable' = diagnosis boundary sigma_r/r=1, 'resolvability' = resolution "
                 "boundary sigma_r/r=0.1). Purely theoretical, no simulated/estimated data needed."
)
parser.add_argument("--N", type=int, default=100)
parser.add_argument("--delta-t", type=float, default=1.0, dest="Delta_t")
parser.add_argument("--q", type=int, default=10, help="fixed q=M/N used in the (c, Delta_t) panel")
parser.add_argument("--c-fixed", type=float, default=0.6, help="fixed c used in the (q, Delta_t) panel")
parser.add_argument("--output-dir", type=str, default="./figures")
parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument(
    "--figures", type=str, nargs="+", default=["all"],
    choices=["all", "reliable", "resolvability"],
    help="Which phase diagram variant(s) to generate."
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

c_crit = 1 / np.sqrt(2)


def savefig(fig, name):
    path = os.path.join(args.output_dir, f"{name}.{args.format}")
    fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {path}")


######################################################################################################################
# SHARED HELPERS
######################################################################################################################

def edges_from_centers(x, log=False):

    x = np.asarray(x)

    if log:
        lx = np.log10(x)
        le = np.empty(len(x) + 1)
        le[1:-1] = 0.5 * (lx[:-1] + lx[1:])
        le[0] = lx[0] - 0.5 * (lx[1] - lx[0])
        le[-1] = lx[-1] + 0.5 * (lx[-1] - lx[-2])
        return 10 ** le

    e = np.empty(len(x) + 1)
    e[1:-1] = 0.5 * (x[:-1] + x[1:])
    e[0] = x[0] - 0.5 * (x[1] - x[0])
    e[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return e


def draw_panel(ax, x_centers, y_centers, RE_col, RE_raw, xlabel, ylabel, title,
               xscale='linear', yscale='log', extra=None, contour_level=1.0):

    x_edges = edges_from_centers(x_centers, log=(xscale == 'log'))
    y_edges = edges_from_centers(y_centers, log=(yscale == 'log'))

    norm = Normalize(vmin=0, vmax=4.5)

    im = ax.pcolormesh(x_edges, y_edges, RE_col, cmap='RdBu_r', norm=norm, shading='flat', rasterized=True)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\sigma_r/r$')

    ax.contour(x_centers, y_centers, RE_raw, levels=[contour_level], colors='black', linewidths=2)

    ax.pcolormesh(
        x_edges, y_edges,
        np.where(RE_raw > contour_level, 1.0, np.nan),
        cmap='Reds', shading='flat', alpha=0.18, rasterized=True
    )

    if extra is not None:
        extra(ax)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(x_centers.min(), x_centers.max())
    ax.set_ylim(y_centers.min(), y_centers.max())
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=11)

    return im


######################################################################################################################
# Tracy-Widom corrected r(c) (paper Eq. rbar/TW_floor), tau-weighted effective sample size M_eff
# (paper Eq. Meff_alpha/Meff_tau, kappa_cW Eq. kcW_results).
# "reliable" uses contour/threshold = 1.0 (paper Fig. 3 top row), "resolvability" uses 0.1 (bottom row).
######################################################################################################################

def r_TW_corrected(c, N, mu_TW=-1.21):
    return (1 - np.sqrt(2) * c) - (np.sqrt(2) * c) ** (1 / 3) * mu_TW * N ** (-2 / 3)


def tau_avg_wigner_vec(c, dt_array):

    lam_min = max(1 - np.sqrt(2) * c, 1e-6)
    lam_max = 1 + np.sqrt(2) * c

    n_pts = 200
    lam_pts, weights = np.polynomial.legendre.leggauss(n_pts)
    lam_pts = 0.5 * (lam_max - lam_min) * lam_pts + 0.5 * (lam_max + lam_min)
    weights = weights * 0.5 * (lam_max - lam_min)

    rho = np.sqrt(np.maximum(2 * c ** 2 - (lam_pts - 1) ** 2, 0)) / (np.pi * c ** 2)

    dt_array = np.atleast_1d(dt_array)
    x = np.outer(lam_pts, dt_array)
    tau = np.where(
        x > 1e-6,
        (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x)),
        1.0 / (x + 1e-10)
    )

    numerator = np.einsum('i,i,ij->j', weights, rho, tau ** 2)
    denominator = np.einsum('i,i,ij->j', weights, rho, tau)

    return numerator / denominator


def compute_rel_err_tau(R, Q_or_q, DT, N, tau_grid=None):
    """kappa_cW = (Meff - 1) / (Meff - N - 2), Meff = M / <tau>_W (or plain M if tau_grid is None)."""

    M = Q_or_q * N
    T_eff = R * Q_or_q * N * DT

    Meff = M / np.maximum(tau_grid, 1e-6) if tau_grid is not None else M

    denom_kappa = Meff - N - 2
    kappa = np.where(denom_kappa > 0, (Meff - 1) / denom_kappa, np.inf)

    x = R * DT
    with np.errstate(over='ignore', invalid='ignore'):
        exp2x = np.exp(np.minimum(2 * x, 500))
    g = np.where(x > 1e-6, (exp2x - 1) / (2 * x), 1.0)

    with np.errstate(invalid='ignore', divide='ignore'):
        rel_err = np.where(
            (R > 0) & (T_eff > 0) & np.isfinite(kappa),
            np.sqrt(2 * kappa * g / T_eff),
            np.inf
        )

    rel_err_raw = np.where(np.isfinite(rel_err), rel_err, 999)
    rel_err_clipped = np.clip(rel_err_raw, 0, 4.5)

    return rel_err_clipped, rel_err_raw


def _build_tw_phase_diagram_grids(N, Delta_t, q, c_fixed):
    """Shared grid computation used by both the 'reliable' and 'resolvability' variants."""

    c_grid_1 = np.linspace(0.05, c_crit * 1.05, 300)
    q_grid_1 = np.logspace(0, 3, 300)
    C1, Q1 = np.meshgrid(c_grid_1, q_grid_1)
    R1 = r_TW_corrected(C1, N)

    tau_1d_c1 = np.array([tau_avg_wigner_vec(c, np.array([Delta_t]))[0] for c in c_grid_1])
    TAU1 = np.tile(tau_1d_c1, (len(q_grid_1), 1))
    q_star_1 = tau_1d_c1

    RE1_col, RE1_raw = compute_rel_err_tau(R1, Q1, Delta_t, N, tau_grid=TAU1)

    c_grid_2 = np.linspace(0.05, c_crit * 1.05, 300)
    dt_grid_2 = np.logspace(-2, 2, 300)
    C2, DT2 = np.meshgrid(c_grid_2, dt_grid_2)
    R2 = r_TW_corrected(C2, N)

    TAU2 = np.array([tau_avg_wigner_vec(c, dt_grid_2) for c in c_grid_2]).T

    RE2_col, RE2_raw = compute_rel_err_tau(R2, q * np.ones_like(C2), DT2, N, tau_grid=TAU2)

    r_line2 = r_TW_corrected(c_grid_2, N)
    dt_opt_2 = np.where(r_line2 > 0, 0.80 / r_line2, np.nan)

    r_fixed = r_TW_corrected(c_fixed, N)
    q_grid_3 = np.logspace(0, 3, 200)
    dt_grid_3 = np.logspace(-2, 2, 200)
    Q3, DT3 = np.meshgrid(q_grid_3, dt_grid_3)
    R3 = r_fixed * np.ones_like(Q3)

    tau_1d_dt3 = tau_avg_wigner_vec(c_fixed, dt_grid_3)
    TAU3 = np.tile(tau_1d_dt3[:, None], (1, len(q_grid_3)))

    RE3_col, RE3_raw = compute_rel_err_tau(R3, Q3, DT3, N, tau_grid=TAU3)

    dt_opt_3 = 0.80 / r_fixed

    return {
        "c_grid_1": c_grid_1, "q_grid_1": q_grid_1, "RE1_col": RE1_col, "RE1_raw": RE1_raw, "q_star_1": q_star_1,
        "c_grid_2": c_grid_2, "dt_grid_2": dt_grid_2, "RE2_col": RE2_col, "RE2_raw": RE2_raw, "dt_opt_2": dt_opt_2,
        "q_grid_3": q_grid_3, "dt_grid_3": dt_grid_3, "RE3_col": RE3_col, "RE3_raw": RE3_raw,
        "dt_opt_3": dt_opt_3, "r_fixed": r_fixed,
    }


def _plot_tw_phase_diagram(N, Delta_t, q, c_fixed, threshold, resolvable_label, unresolvable_label,
                           suptitle, filename, output_dir, fmt, dpi):

    g = _build_tw_phase_diagram_grids(N, Delta_t, q, c_fixed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def extra_1(ax):
        ax.plot(g["c_grid_1"], g["q_star_1"], color='gold', linestyle='-', lw=2.5,
               label=r'$q^*(c) = \langle\tau\rangle_W$  ($q_{\rm eff}=1$)')
        ax.legend(fontsize=9, loc='upper left')

    draw_panel(
        axes[0], g["c_grid_1"], g["q_grid_1"], g["RE1_col"], g["RE1_raw"],
        xlabel=r'$c$', ylabel=r'$q=M/N$',
        title=rf'$(c,q)$ — $\Delta t={Delta_t}$',
        xscale='linear', yscale='log', extra=extra_1, contour_level=threshold
    )
    axes[0].text(0.10, 50, resolvable_label, color='steelblue', fontsize=10, fontweight='bold')
    axes[0].text(0.45, 3, unresolvable_label, color='darkred', fontsize=10, fontweight='bold')

    def extra_2(ax):
        ax.plot(g["c_grid_2"], g["dt_opt_2"], color='green', linestyle='--', lw=2,
               label=r'$\Delta t^*=0.80/\bar{r}$')
        ax.legend(fontsize=9)

    draw_panel(
        axes[1], g["c_grid_2"], g["dt_grid_2"], g["RE2_col"], g["RE2_raw"],
        xlabel=r'$c$', ylabel=r'$\Delta t$',
        title=rf'$(c,\Delta t)$ — $q={q}$',
        xscale='linear', yscale='log', extra=extra_2, contour_level=threshold
    )
    axes[1].text(0.10, 10, resolvable_label, color='steelblue', fontsize=10, fontweight='bold')
    axes[1].text(0.45, 0.05, unresolvable_label, color='darkred', fontsize=10, fontweight='bold')

    def extra_3(ax):
        ax.axhline(g["dt_opt_3"], color='green', linestyle='--', lw=2,
                  label=rf'$\Delta t^*={g["dt_opt_3"]:.2f}$')
        ax.legend(fontsize=9)

    draw_panel(
        axes[2], g["q_grid_3"], g["dt_grid_3"], g["RE3_col"], g["RE3_raw"],
        xlabel=r'$q=M/N$', ylabel=r'$\Delta t$',
        title=rf'$(q,\Delta t)$ — $c={c_fixed}$, $r={g["r_fixed"]:.3f}$',
        xscale='log', yscale='log', extra=extra_3, contour_level=threshold
    )
    axes[2].text(100, 10, resolvable_label, color='steelblue', fontsize=10, fontweight='bold')
    axes[2].text(1.5, 0.05, unresolvable_label, color='darkred', fontsize=10, fontweight='bold')

    fig.suptitle(suptitle, fontsize=13, y=1.02)

    fig.tight_layout()
    savefig(fig, filename)


def plot_phase_diagram_reliable(N, Delta_t, q, c_fixed, output_dir, fmt, dpi):

    _plot_tw_phase_diagram(
        N, Delta_t, q, c_fixed,
        threshold=1.0,
        resolvable_label='RELIABLE\nSTABILITY ESTIMATE',
        unresolvable_label='UNRELIABLE\nSTABILITY ESTIMATE',
        suptitle=rf'Stability inference phase diagram — GOE ($N={N}$, '
                 rf'$\kappa_{{\rm cW}}$, Tracy--Widom correction)',
        filename="phase_diagram_reliable",
        output_dir=output_dir, fmt=fmt, dpi=dpi
    )


def plot_phase_diagram_resolvability(N, Delta_t, q, c_fixed, output_dir, fmt, dpi):

    _plot_tw_phase_diagram(
        N, Delta_t, q, c_fixed,
        threshold=0.1,
        resolvable_label='RESOLVABLE',
        unresolvable_label='UNRESOLVABLE',
        suptitle=rf'Stability inference phase diagram — GOE ($N={N}$, '
                 rf'$\kappa_{{\rm cW}}$, $\sigma_r/r = 0.1$ threshold)',
        filename="phase_diagram_resolvability",
        output_dir=output_dir, fmt=fmt, dpi=dpi
    )


def print_tw_diagnostics(N, Delta_t):

    print("\nTracy-Widom correction diagnostics (r_TW_corrected, mu_TW=-1.21):")

    c_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.70, 0.71, 0.72, 0.74])
    for c in c_test:
        tau = tau_avg_wigner_vec(c, np.array([Delta_t]))[0]
        r = r_TW_corrected(c, N)
        print(f"c={c:.2f}  r_TW={r:.4f}  q*(c)=<tau>={tau:.4f}")

    c_max = brentq(lambda c: r_TW_corrected(c, N), c_crit, 1.0)

    print(f"c_crit (large-N)     = {c_crit:.4f}")
    print(f"c_max  (TW, N={N})   = {c_max:.4f}")
    print(f"r_TW at c_crit       = {r_TW_corrected(c_crit, N):.4f}")
    print(f"r_TW at c_max        = {r_TW_corrected(c_max, N):.6f}")
    print(f"TW correction        = {c_max - c_crit:.4f}")


######################################################################################################################
# MAIN
######################################################################################################################

if __name__ == "__main__":

    requested = ["reliable", "resolvability"] if args.figures == ["all"] else args.figures

    if "reliable" in requested:
        print("\n--- Generating phase diagram: reliable (tau-weighted M_eff, threshold=1.0) ---")
        plot_phase_diagram_reliable(args.N, args.Delta_t, args.q, args.c_fixed, args.output_dir, args.format, args.dpi)

    if "resolvability" in requested:
        print("\n--- Generating phase diagram: resolvability (tau-weighted M_eff, threshold=0.1) ---")
        plot_phase_diagram_resolvability(args.N, args.Delta_t, args.q, args.c_fixed, args.output_dir, args.format, args.dpi)
        print_tw_diagnostics(args.N, args.Delta_t)

    print(f"\nAll requested phase diagrams saved to: {args.output_dir}")
