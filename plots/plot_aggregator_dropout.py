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
# Data: aggregator dropout
# -----------------------------
drop_rate = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
    50, 55, 60, 65, 70, 75, 80, 85, 90, 95
])

acc_mean = np.array([
    85.0, 85.2, 85.1, 85.0, 85.4, 85.2, 85.3, 85.3, 85.2, 85.4,
    85.1, 85.0, 85.1, 85.2, 85.2, 84.7, 83.3, 81.7, 77.5, 53.3
])
acc_std = np.array([
    0.9, 1.1, 0.7, 1.0, 0.9, 0.6, 0.7, 0.6, 0.7, 0.9,
    1.0, 0.8, 0.9, 0.7, 0.7, 0.3, 0.5, 0.7, 0.7, 6.9
])

best_round_mean = np.array([
    81.0, 89.3, 87.0, 89.7, 101.7, 100.3, 109.0, 111.7, 122.0, 133.0,
    137.0, 137.7, 167.3, 182.0, 192.3, 197.0, 200.0, 200.0, 200.0, 200.0
])
best_round_std = np.array([
    9.4, 15.9, 9.8, 13.1, 13.9, 8.2, 13.9, 11.0, 14.2, 20.1,
    22.0, 12.8, 20.2, 16.1, 9.5, 2.2, 0.0, 0.0, 0.0, 0.0
])


# -----------------------------
# Plot
# -----------------------------
setup_icml_plot(two_column=True)

fig, axes = plt.subplots(1, 2)

# Left: accuracy under aggregator dropout
axes[0].errorbar(
    drop_rate,
    acc_mean,
    yerr=acc_std,
    fmt="-o",
    capsize=2,
    elinewidth=0.9,
)
axes[0].set_title(r"Robustness to aggregator dropout")
axes[0].set_xlabel(r"Aggregator dropout rate (\%)")
axes[0].set_ylabel(r"Test accuracy (\%)")
axes[0].set_xlim(-5, 100)
axes[0].set_ylim(40, 90)
axes[0].set_xticks(np.arange(0, 101, 10))
axes[0].set_yticks(np.arange(40, 91, 10))

# Right: convergence delay under aggregator dropout
axes[1].errorbar(
    drop_rate,
    best_round_mean,
    yerr=best_round_std,
    fmt="-o",
    capsize=2,
    elinewidth=0.9,
)
axes[1].set_title(r"Convergence delay under dropout")
axes[1].set_xlabel(r"Aggregator dropout rate (\%)")
axes[1].set_ylabel(r"Best validation round")
axes[1].set_xlim(-5, 100)
axes[1].set_ylim(0, 210)
axes[1].set_xticks(np.arange(0, 101, 10))
axes[1].set_yticks(np.arange(0, 225, 50))

plt.tight_layout(w_pad=2.0)

# Save
plt.savefig("eris_aggregator_dropout.pdf", bbox_inches="tight")
plt.savefig("eris_aggregator_dropout.png", bbox_inches="tight", dpi=300)
plt.show()
