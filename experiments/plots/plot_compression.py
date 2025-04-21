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
    
# Data
omega = np.array([85, 170, 340, 700, 1050])
mia_accuracy_mean = np.array([0.686704, 0.684752, 0.678336, 0.643312, 0.635056])
mia_accuracy_std = np.array([0.018281875, 0.022957887, 0.015947626, 0.010695686, 0.01948196]) / np.sqrt(5)

# Minimum leakage and its std (same as previous plot) # 0.6544	0.024344198
min_leakage = 0.6544
min_leakage_std = 0.024344198 / np.sqrt(5)

# Set up the plot
fig_size = setup_icml_plot(two_column=True)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(5, 4))

# Plot MIA Accuracy with error bars
ax.errorbar(omega, mia_accuracy_mean, yerr=mia_accuracy_std, fmt='o-', 
            capsize=5, label='_nolegend_', color='black', markersize=4, linewidth=1.2, alpha=0.4)
ax.plot(omega, mia_accuracy_mean, 'o-', color='black', markersize=4, label='ERIS', linewidth=1.2)

# Plot the dashed horizontal line for minimum leakage
ax.axhline(min_leakage, color='green', linestyle='--', linewidth=1.2, label='Min. Leakage')

# Plot a shaded band corresponding to the std of the minimum leakage
ax.axhspan(min_leakage - min_leakage_std, min_leakage + min_leakage_std, 
           facecolor='green', alpha=0.2, )#label='Minimum Leakage Std')

# Labeling and formatting
ax.set_xlabel(r'$\omega$', fontsize=14)
ax.set_ylabel('MIA Accuracy', fontsize=14)
ax.set_title('Privacy Leakage vs. Shifted Compression', fontsize=16)
# ax.set_ylim(0.625, 0.78)
ax.legend(fontsize=12)
plt.tight_layout()

# Save the plot as PDF
file_path = 'fig_compression.pdf'
plt.savefig(file_path, bbox_inches='tight')
