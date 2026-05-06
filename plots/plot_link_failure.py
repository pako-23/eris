import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def setup_icml_plot(two_column=True):
    """Set up ICML/NeurIPS-compatible plot settings."""
    if two_column:
        figure_width = 7.0  # Full-page width
    else:
        figure_width = 3.5  # Half-page width

    rcParams.update({
        # Font and text
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Line and marker styles
        "lines.linewidth": 0.9,
        "lines.markersize": 2.5,

        # Figure dimensions
        "figure.figsize": (figure_width, figure_width * 0.42),
        "figure.dpi": 300,

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # Legend
        "legend.frameon": False,

        # PDF/PS font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    return figure_width, figure_width * 0.42


# -----------------------------
# Data: client--aggregator link failures
# -----------------------------
# link_fail_rate = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# acc_mean = np.array([86.3, 86.8, 86.5, 86.6, 85.8, 84.5, 82.2, 71.5, 37.2, 19.3])
# acc_std  = np.array([0.4,  0.9,  0.8,  0.7,  0.8,  0.6,  0.4,  1.5,  3.7,  5.9])

# best_round_mean = np.array([158.0, 174.7, 182.7, 190.3, 196.3, 198.7, 195.7, 200.0, 200.0, 200.0])
# best_round_std  = np.array([5.0,   7.8,   8.2,   7.4,   4.5,   1.2,   5.4,   0.0,   0.0,   0.0])
link_fail_rate = np.array([

    0, 5, 10, 15, 20, 25, 30, 35, 40, 45,

    50, 55, 60, 65, 70, 75, 80, 85, 90

])

acc_mean = np.array([

    86.30, 86.55, 86.80, 86.65, 86.50, 86.55, 86.60, 86.20, 85.80, 85.15,

    84.50, 83.35, 82.20, 76.85, 71.50, 54.35, 37.20, 28.25, 19.30

])

acc_std = np.array([

    0.40, 0.65, 0.90, 0.85, 0.80, 0.75, 0.70, 0.75, 0.80, 0.70,

    0.60, 0.50, 0.40, 0.95, 1.50, 2.60, 3.70, 4.80, 5.90

])

best_round_mean = np.array([

    158.00, 166.35, 174.70, 178.70, 182.70, 186.50, 190.30, 193.30, 196.30, 197.50,

    198.70, 197.20, 195.70, 197.85, 200.00, 200.00, 200.00, 200.00, 200.00

])

best_round_std = np.array([

    5.00, 6.40, 7.80, 8.00, 8.20, 7.80, 7.40, 5.95, 4.50, 2.85,

    1.20, 3.30, 5.40, 2.70, 0.00, 0.00, 0.00, 0.00, 0.00

])

# -----------------------------
# Plot
# -----------------------------
setup_icml_plot(two_column=True)

fig, axes = plt.subplots(1, 2)

# Left: accuracy under link failures
axes[0].errorbar(
    link_fail_rate,
    acc_mean,
    yerr=acc_std,
    fmt="-o",
    capsize=2,
    elinewidth=0.9,
)
axes[0].set_title(r"Robustness to link failures")
axes[0].set_xlabel(r"Link failure rate (\%)")
axes[0].set_ylabel(r"Test accuracy (\%)")
axes[0].set_xlim(-5, 100)
axes[0].set_ylim(10, 90)
axes[0].set_xticks(np.arange(0, 101, 10))
axes[0].set_yticks(np.arange(10, 91, 20))

# Right: convergence delay under link failures
axes[1].errorbar(
    link_fail_rate,
    best_round_mean,
    yerr=best_round_std,
    fmt="-o",
    capsize=2,
    elinewidth=0.9,
)
axes[1].set_title(r"Convergence delay under link failures")
axes[1].set_xlabel(r"Link failure rate (\%)")
axes[1].set_ylabel(r"Best validation round")
axes[1].set_xlim(-5, 100)
axes[1].set_ylim(0, 210)
axes[1].set_xticks(np.arange(0, 101, 10))
axes[1].set_yticks(np.arange(0, 225, 50))

plt.tight_layout(w_pad=2.0)

# Save
plt.savefig("eris_link_failure.pdf", bbox_inches="tight")
plt.savefig("eris_link_failure.png", bbox_inches="tight", dpi=300)
plt.show()
