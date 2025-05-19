import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
 
# ICML plot settings
def setup_icml_plot(two_column=False):
    """Set up ICML-compatible plot settings."""
    if two_column:
        figure_width = 7  # Full-page width for two-column layout (in inches)
    else:
        figure_width = 3.5  # Half-page width for two-column layout (in inches)
 
    rcParams.update({
        # Font and text
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif fonts
        "font.serif": ["Times New Roman"],  # Set font to Times New Roman
        "axes.labelsize": 10,  # Font size for axis labels
        "axes.titlesize": 10,  # Font size for titles
        "legend.fontsize": 4,  # Font size for legends
        "xtick.labelsize": 10,  # Font size for x-axis ticks
        "ytick.labelsize": 10,  # Font size for y-axis ticks
 
        # Line and marker styles
        "lines.linewidth": 1.2,  # Line width
        "lines.markersize": 3,  # Marker size
 
        # Figure dimensions
        "figure.figsize": (figure_width, figure_width * 0.85),  # TODO change to better ratio
        "figure.dpi": 300,  # High resolution for publication
 
        # Grid
        "axes.grid": True,  # Enable grid
        "grid.alpha": 0.3,  # Grid transparency
        "grid.linestyle": "--",  # Dashed grid lines
 
        # Legend
        "legend.frameon": False,  # No border around legends
    })
    
    return (figure_width, figure_width * 0.85)
    
# Use a style suitable for publication quality figures
# plt.style.use('seaborn-whitegrid')

# Data from the table
# splits = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
# mia_accuracy_mean = np.array([0.65056, 0.588, 0.568587, 0.56428, 0.559872, 0.55488, 0.55216, 0.5501, 0.549315556, 0.548416])
# mia_accuracy_std = np.array([0.037664285, 0.032048963, 0.025793, 0.025114, 0.023035, 0.019464402, 0.017709, 0.017311384, 0.015499266, 0.016691])  / np.sqrt(5)
# Updated data
splits = np.array([10, 20, 30, 40, 50])
mia_accuracy_mean = np.array([0.76744, 0.75568, 0.742666667, 0.7196, 0.707504])
mia_accuracy_std = np.array([0.018489521, 0.016100484, 0.021330833, 0.026990072, 0.026983107]) / np.sqrt(5)
# Minimum leakage and its std
# min_leakage = 0.5456
# min_leakage_std = 0.013994  / np.sqrt(5)
min_leakage = 0.6544
min_leakage_std = 0.024344198 / np.sqrt(5)

# Set up the plot
fig_size = setup_icml_plot(two_column=True)

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(4, 4))

# Plot MIA Accuracy with error bars
ax.errorbar(splits, mia_accuracy_mean, yerr=mia_accuracy_std, fmt='o-', 
            capsize=5, label='_nolegend_', color='black', markersize=4, linewidth=1.2, alpha=0.4)
ax.plot(splits, mia_accuracy_mean, 'o-', color='black', markersize=4, label='ERIS', linewidth=1.2)

# Plot the dashed horizontal line for minimum leakage
ax.axhline(min_leakage, color='green', linestyle='--', linewidth=1.2, label='Min. Leakage')

# Plot a shaded band corresponding to the std of the minimum leakage
ax.axhspan(min_leakage - min_leakage_std, min_leakage + min_leakage_std, 
           facecolor='green', alpha=0.2, )#label='Minimum Leakage Std')

# Labeling and formatting
ax.set_xlabel(r'Number of Aggregators $(A)$', fontsize=14)
ax.set_ylabel('MIA Accuracy', fontsize=14)
ax.set_title('Model Partitioning', fontsize=16)
# set y-lim
ax.set_ylim(0.625, 0.78)
ax.legend(fontsize=12)
plt.tight_layout()

# Display the plot
# plt.show()
plt.savefig('plots-n_splits_cifar_same.pdf', bbox_inches='tight')
