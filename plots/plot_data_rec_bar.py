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
        "xtick.labelsize": 8,  # Font size for x-axis ticks
        "ytick.labelsize": 8,  # Font size for y-axis ticks

        # Line and marker styles
        "lines.linewidth": 1.2,  # Line width
        "lines.markersize": 3,  # Marker size

        # Figure dimensions
        "figure.figsize": (figure_width, figure_width * 0.85),
        "figure.dpi": 300,  # High resolution for publication

        # Grid
        "axes.grid": True,  # Enable grid
        "grid.alpha": 0.3,  # Grid transparency
        "grid.linestyle": "--",  # Dashed grid lines

        # Legend
        "legend.frameon": False,  # No border around legends
    })
    
    return (figure_width, figure_width * 0.85)




# (Assuming setup_icml_plot is defined elsewhere)
fig_size = setup_icml_plot(two_column=True)

# Data arrays
splits = np.array([1, 1.001, 1.01, 1.1, 1.2, 1.4, 2, 4, 8, 16, 32])
x_positions = np.arange(len(splits))  # each group will span 1 unit

# --- MNIST ---
lpips_dlg_mean_mnist = np.array([0.109889, 0.156936, 0.232211, 0.518334, 
                                 0.551045, 0.552543, 0.562652, 0.573914, 0.573787, 0.574807, 0.580589])
lpips_dlg_std_mnist = np.array([0.225457, 0.228253, 0.177211, 0.111447, 
                                0.082754, 0.083656, 0.089072, 0.089199, 0.093084, 0.086936, 0.08634]) / np.sqrt(5)
lpips_idlg_mean_mnist = np.array([0.03717, 0.075882, 0.193202, 0.511458, 
                                  0.551025, 0.553632, 0.567528, 0.586007, 0.5895, 0.584929, 0.589685])
lpips_idlg_std_mnist = np.array([0.136978, 0.141556, 0.13649, 0.111773, 
                                 0.089535, 0.085258, 0.087831, 0.093293, 0.085115, 0.087164, 0.089007]) / np.sqrt(5)
lpips_random_mnist = 0.566315
lpips_random_std_mnist = 0.097331 / np.sqrt(5)

# --- CIFAR ---
lpips_dlg_mean_cifar = np.array([0.235061, 0.27145, 0.36258, 0.394891, 
                                 0.391388, 0.391977, 0.395981, 0.401567022, 0.40264, 0.410647691, 0.419258])
lpips_dlg_std_cifar = np.array([0.160128, 0.120143, 0.076478, 0.070736, 
                                0.063449, 0.071155, 0.073802, 0.076013595, 0.072612, 0.072433139, 0.06813]) / np.sqrt(5)
lpips_idlg_mean_cifar = np.array([0.171193, 0.238449, 0.34739, 0.393412, 
                                  0.39116, 0.393663, 0.394875, 0.406383881, 0.408849202, 0.41753449, 0.420213111])
lpips_idlg_std_cifar = np.array([0.137549, 0.107466, 0.075881, 0.074989, 
                                 0.07248, 0.071449, 0.072945, 0.074391271, 0.072059309, 0.073025471, 0.070792152]) / np.sqrt(5)
lpips_random_cifar = 0.415879
lpips_random_std_cifar = 0.072017 / np.sqrt(5)

# --- LFW ---
lpips_dlg_mean_lfw = np.array([0.306268, 0.339744, 
                               0.454117, 0.465857, 0.460154, 0.461122, 0.471396, 0.485633, 
                               0.490871, 0.489892, 0.495015])
lpips_dlg_std_lfw = np.array([0.171823, 0.134216, 
                              0.073346, 0.065453, 0.069977, 0.067561, 0.0699, 0.064886, 
                              0.069068, 0.073321, 0.069946]) / np.sqrt(5)
lpips_idlg_mean_lfw = np.array([0.183026, 0.257172, 
                                0.444235, 0.484002, 0.483562, 0.483421, 0.485871, 0.49435, 
                                0.495876, 0.497378, 0.499236])
lpips_idlg_std_lfw = np.array([0.144886, 0.118123, 
                               0.074746, 0.070512, 0.068119, 0.069509, 0.069284, 0.066782, 
                               0.065875, 0.06832, 0.069195]) / np.sqrt(5)
lpips_random_lfw = 0.472369
lpips_random_std_lfw = 0.063956 / np.sqrt(5)

# Set up the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Parameters for the bar plot
width = 0.5  # Each group spans 1 unit: left half for DLG, right half for iDLG
error_kw = {'capsize': 5, 'elinewidth': 1.2}

# --- MNIST Bar Plot ---
ax1.bar(x_positions, lpips_dlg_mean_mnist, width=width, 
        yerr=lpips_dlg_std_mnist, error_kw=error_kw, 
        color='black', alpha=0.3, label='DLG')
ax1.bar(x_positions + width, lpips_idlg_mean_mnist, width=width, 
        yerr=lpips_idlg_std_mnist, error_kw=error_kw, 
        color='blue', alpha=0.3, label='iDLG')
ax1.axhline(lpips_random_mnist, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax1.axhspan(lpips_random_mnist - lpips_random_std_mnist, lpips_random_mnist + lpips_random_std_mnist, 
            facecolor='green', alpha=0.2)
ax1.set_ylabel('Reconstruction Quality (LPIPS)', fontsize=14)
ax1.set_xlabel('Number of Model Splits', fontsize=14)
ax1.set_title('MNIST', fontsize=16)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
# Set xticks at the center of each group
ax1.set_xticks(x_positions + width/2)
ax1.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")
ax1.set_xlim(0, len(splits))

# --- CIFAR Bar Plot ---
ax2.bar(x_positions, lpips_dlg_mean_cifar, width=width, 
        yerr=lpips_dlg_std_cifar, error_kw=error_kw, 
        color='black', alpha=0.3, label='DLG')
ax2.bar(x_positions + width, lpips_idlg_mean_cifar, width=width, 
        yerr=lpips_idlg_std_cifar, error_kw=error_kw, 
        color='blue', alpha=0.3, label='iDLG')
ax2.axhline(lpips_random_cifar, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax2.axhspan(lpips_random_cifar - lpips_random_std_cifar, lpips_random_cifar + lpips_random_std_cifar, 
            facecolor='green', alpha=0.2)
ax2.set_xlabel('Number of Model Splits', fontsize=14)
ax2.set_title('CIFAR-10', fontsize=16)
ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
ax2.set_xticks(x_positions + width/2)
ax2.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")
ax2.set_xlim(0, len(splits))

# --- LFW Bar Plot ---
ax3.bar(x_positions, lpips_dlg_mean_lfw, width=width, 
        yerr=lpips_dlg_std_lfw, error_kw=error_kw, 
        color='black', alpha=0.3, label='DLG')
ax3.bar(x_positions + width, lpips_idlg_mean_lfw, width=width, 
        yerr=lpips_idlg_std_lfw, error_kw=error_kw, 
        color='blue', alpha=0.3, label='iDLG')
ax3.axhline(lpips_random_lfw, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax3.axhspan(lpips_random_lfw - lpips_random_std_lfw, lpips_random_lfw + lpips_random_std_lfw, 
            facecolor='green', alpha=0.2)
ax3.set_xlabel('Number of Model Splits', fontsize=14)
ax3.set_title('LFW', fontsize=16)
ax3.tick_params(axis='x', labelsize=10)
ax3.tick_params(axis='y', labelsize=10)
ax3.set_xticks(x_positions + width/2)
ax3.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")
ax3.set_xlim(0, len(splits))

# --- Add background highlighting for the ERIS region (from split value 2 onward) ---
eris_index = np.where(splits == 2)[0][0]  # index corresponding to split value 2
left_boundary = eris_index      # group where '2' is located
right_boundary = len(splits)    # last group edge (since each group spans 1 unit)

for ax in [ax1, ax2, ax3]:
    ax.axvspan(left_boundary, right_boundary, facecolor='lightgrey', alpha=0.3, 
               zorder=0, label='ERIS')
    ax.axvline(x=left_boundary, color='grey', linestyle='-', linewidth=1.2, zorder=1)

# Collect handles and labels from all axes and remove duplicates.
handles, labels = [], []
for ax in [ax1, ax2, ax3]:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

# Place a single combined legend below the entire figure.
fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.06), fontsize=10)

plt.tight_layout()
plt.savefig('plots-dlg-idlg_lpips_bar.pdf', bbox_inches='tight')
plt.close()
