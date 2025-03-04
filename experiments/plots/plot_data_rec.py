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




##############################################################################
# MNIST
##############################################################################


# Define the split values and use discrete positions for the x-axis
splits = np.array([1, 1.00001, 1.0001, 1.001, 1.01, 1.1, 1.2, 1.4, 2, 4, 8, 16, 32])
x_positions = np.arange(len(splits))  # equispaced positions

# Data for DLG (excluding the "Random" baseline)
ssim_dlg_mean = np.array([0.667902, 0.617078, 0.559451, 0.430053, 0.305153, 0.070389, 
                          0.034366, 0.041025, 0.044971, 0.030651, 0.019797, 0.018402, 0.012908])
ssim_dlg_std = np.array([0.331463, 0.341855, 0.292782, 0.254909, 0.163732, 0.066124, 
                         0.032914, 0.029098, 0.033563, 0.034273, 0.028049, 0.024514, 0.022268]) / np.sqrt(5)
lpips_dlg_mean = np.array([0.109889, 0.128435, 0.122332, 0.156936, 0.232211, 0.518334, 
                           0.551045, 0.552543, 0.562652, 0.573914, 0.573787, 0.574807, 0.580589])
lpips_dlg_std = np.array([0.225457, 0.234179, 0.23469, 0.228253, 0.177211, 0.111447, 
                          0.082754, 0.083656, 0.089072, 0.089199, 0.093084, 0.086936, 0.08634]) / np.sqrt(5)

# Data for iDLG (excluding the "Random" baseline)
ssim_idlg_mean = np.array([0.795236, 0.771033, 0.633857, 0.511222, 0.339384, 0.075204, 
                           0.034705, 0.038547, 0.037689, 0.02103, 0.011815, 0.00884, 0.004684])
ssim_idlg_std = np.array([0.239851, 0.232856, 0.246522, 0.186204, 0.138548, 0.059546, 
                          0.025311, 0.023247, 0.029731, 0.025968, 0.022925, 0.019476, 0.019316]) / np.sqrt(5)
lpips_idlg_mean = np.array([0.03717, 0.032397, 0.060986, 0.075882, 0.193202, 0.511458, 
                            0.551025, 0.553632, 0.567528, 0.586007, 0.5895, 0.584929, 0.589685])
lpips_idlg_std = np.array([0.136978, 0.121617, 0.168899, 0.141556, 0.13649, 0.111773, 
                           0.089535, 0.085258, 0.087831, 0.093293, 0.085115, 0.087164, 0.089007]) / np.sqrt(5)

# Random baseline (common to both methods)
ssim_random = 0.003431
ssim_random_std = 0.023822 / np.sqrt(5)
lpips_random = 0.566315
lpips_random_std = 0.097331 / np.sqrt(5)

# Set up the ICML-style plot
fig_size = setup_icml_plot(two_column=True)
# Create a figure with two subplots (one for SSIM, one for LPIPS)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

# --- SSIM Plot ---
ax1.errorbar(x_positions, ssim_dlg_mean, yerr=ssim_dlg_std, fmt='o-', 
             capsize=5, label='DLG', color='black', markersize=4, linewidth=1.2)
ax1.errorbar(x_positions, ssim_idlg_mean, yerr=ssim_idlg_std, fmt='s-', 
             capsize=5, label='iDLG', color='blue', markersize=4, linewidth=1.2)
# Plot random baseline as a horizontal dashed line with a shaded error band
ax1.axhline(ssim_random, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax1.axhspan(ssim_random - ssim_random_std, ssim_random + ssim_random_std, 
            facecolor='green', alpha=0.2)
ax1.set_ylabel('SSIM', fontsize=14)
ax2.set_xlabel('Number of Aggregators', fontsize=14)
ax1.set_title('Reconstruction Quality (SSIM)', fontsize=16)
ax1.legend(fontsize=8)
# Set x-axis ticks to display the original split values
ax1.set_xticks(x_positions)
ax1.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")

# --- LPIPS Plot ---
ax2.errorbar(x_positions, lpips_dlg_mean, yerr=lpips_dlg_std, fmt='o-', 
             capsize=5, label='DLG', color='black', markersize=4, linewidth=1.2)
ax2.errorbar(x_positions, lpips_idlg_mean, yerr=lpips_idlg_std, fmt='s-', 
             capsize=5, label='iDLG', color='blue', markersize=4, linewidth=1.2)
# Plot random baseline as a horizontal dashed line with a shaded error band
ax2.axhline(lpips_random, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax2.axhspan(lpips_random - lpips_random_std, lpips_random + lpips_random_std, 
            facecolor='green', alpha=0.2)
ax2.set_ylabel('LPIPS', fontsize=14)
ax2.set_xlabel('Number of Aggregators', fontsize=14)
ax2.set_title('Reconstruction Quality (LPIPS)', fontsize=16)
ax2.legend(fontsize=8)
# Set x-axis ticks to display the original split values
ax2.set_xticks(x_positions)
ax2.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")

plt.tight_layout()
plt.savefig('plots-dlg-idlg.pdf', bbox_inches='tight')
plt.close()
























##############################################################################
# CIFAR-10
##############################################################################
# Define the split values
splits = np.array([1, 1.00001, 1.0001, 1.001, 1.01, 1.1, 1.2, 1.4, 2, 4, 8, 16, 32])

# DLG Data
ssim_dlg_mean = np.array([0.361284, 0.363981, 0.353949, 0.258151, 0.132456, 0.075417, 
                          0.064962, 0.05358, 0.04482, 0.037064466, 0.03351, 0.032590536, 0.031743])
ssim_dlg_std = np.array([0.294573, 0.298955, 0.283444, 0.182986, 0.079072, 0.03207, 
                         0.026907, 0.026702, 0.024352, 0.019064869, 0.017917, 0.019347271, 0.018168]) / np.sqrt(5)
lpips_dlg_mean = np.array([0.235061, 0.231724, 0.23072, 0.27145, 0.36258, 0.394891, 
                           0.391388, 0.391977, 0.395981, 0.401567022, 0.40264, 0.410647691, 0.419258])
lpips_dlg_std = np.array([0.160128, 0.15721, 0.153744, 0.120143, 0.076478, 0.070736, 
                          0.063449, 0.071155, 0.073802, 0.076013595, 0.072612, 0.072433139, 0.06813]) / np.sqrt(5)

# iDLG Data
ssim_idlg_mean = np.array([0.469946, 0.470323, 0.452568, 0.314907, 0.159618, 0.085176, 
                           0.072816, 0.059518, 0.047268, 0.037255193, 0.032746463, 0.029286122, 0.032239616])
ssim_idlg_std = np.array([0.285075, 0.277823, 0.26357, 0.17356, 0.07116, 0.029656, 
                          0.027536, 0.022535, 0.02149, 0.019153657, 0.019291173, 0.014771053, 0.019485883]) / np.sqrt(5)
lpips_idlg_mean = np.array([0.171193, 0.171695, 0.174597, 0.238449, 0.34739, 0.393412, 
                            0.39116, 0.393663, 0.394875, 0.406383881, 0.408849202, 0.41753449, 0.420213111])
lpips_idlg_std = np.array([0.137549, 0.138385, 0.130402, 0.107466, 0.075881, 0.074989, 
                           0.07248, 0.071449, 0.072945, 0.074391271, 0.072059309, 0.073025471, 0.070792152]) / np.sqrt(5)

# Random baseline (common to both methods)
ssim_random = 0.008068
ssim_random_std = 0.011617 / np.sqrt(5)
lpips_random = 0.415879
lpips_random_std = 0.072017 / np.sqrt(5)

# Set up the ICML-style plot
fig_size = setup_icml_plot(two_column=True)
# Create a figure with two subplots (one for SSIM, one for LPIPS)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

# --- SSIM Plot ---
ax1.errorbar(x_positions, ssim_dlg_mean, yerr=ssim_dlg_std, fmt='o-', 
             capsize=5, label='DLG', color='black', markersize=4, linewidth=1.2)
ax1.errorbar(x_positions, ssim_idlg_mean, yerr=ssim_idlg_std, fmt='s-', 
             capsize=5, label='iDLG', color='blue', markersize=4, linewidth=1.2)
# Plot random baseline as a horizontal dashed line with a shaded error band
ax1.axhline(ssim_random, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax1.axhspan(ssim_random - ssim_random_std, ssim_random + ssim_random_std, 
            facecolor='green', alpha=0.2)
ax1.set_ylabel('SSIM', fontsize=14)
ax2.set_xlabel('Number of Aggregators', fontsize=14)
ax1.set_title('Reconstruction Quality (SSIM)', fontsize=16)
ax1.legend(fontsize=8)
# Set x-axis ticks to display the original split values
ax1.set_xticks(x_positions)
ax1.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")

# --- LPIPS Plot ---
ax2.errorbar(x_positions, lpips_dlg_mean, yerr=lpips_dlg_std, fmt='o-', 
             capsize=5, label='DLG', color='black', markersize=4, linewidth=1.2)
ax2.errorbar(x_positions, lpips_idlg_mean, yerr=lpips_idlg_std, fmt='s-', 
             capsize=5, label='iDLG', color='blue', markersize=4, linewidth=1.2)
# Plot random baseline as a horizontal dashed line with a shaded error band
ax2.axhline(lpips_random, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax2.axhspan(lpips_random - lpips_random_std, lpips_random + lpips_random_std, 
            facecolor='green', alpha=0.2)
ax2.set_ylabel('LPIPS', fontsize=14)
ax2.set_xlabel('Number of Aggregators', fontsize=14)
ax2.set_title('Reconstruction Quality (LPIPS)', fontsize=16)
ax2.legend(fontsize=8)
# Set x-axis ticks to display the original split values
ax2.set_xticks(x_positions)
ax2.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")

plt.tight_layout()
plt.savefig('plots-dlg-idlg_cifar.pdf', bbox_inches='tight')
plt.close()
























##############################################################################
# LFW
##############################################################################
# Define the split values
splits = np.array([1, 1.00001, 1.0001, 1.001, 1.01, 1.1, 1.2, 1.4, 2, 4, 8, 16, 32]) # 1.0000001, 1.000001,

# DLG Data
ssim_dlg_mean = np.array([0.292611, 0.298393, 0.299944, 0.232329, 
                          0.109009, 0.056725, 0.054344, 0.046807, 0.040194, 0.030838, 
                          0.028605, 0.02605, 0.026943]) # 0.289726, 0.300232,
ssim_dlg_std = np.array([0.26014, 0.26794, 0.250279, 0.183657, 
                         0.066035, 0.027808, 0.023326, 0.021682, 0.020612, 0.016466, 
                         0.017071, 0.01617, 0.015199]) / np.sqrt(5) # 0.257076, 0.251725,
lpips_dlg_mean = np.array([0.306268, 0.298462, 0.302072, 0.339744, 
                           0.454117, 0.465857, 0.460154, 0.461122, 0.471396, 0.485633, 
                           0.490871, 0.489892, 0.495015]) # 0.306133, 0.297219,
lpips_dlg_std = np.array([0.171823, 0.172603, 0.16744, 0.134216, 
                          0.073346, 0.065453, 0.069977, 0.067561, 0.0699, 0.064886, 
                          0.069068, 0.073321, 0.069946]) / np.sqrt(5) # 0.169103, 0.164938,

# iDLG Data
ssim_idlg_mean = np.array([0.489512, 0.486915, 0.484527, 0.355739, 
                           0.153597, 0.074756, 0.063178, 0.056601, 0.041232, 0.028128, 
                           0.024523, 0.022131, 0.020864]) # 0.48691, 0.482382,
ssim_idlg_std = np.array([0.222253, 0.220543, 0.220398, 0.146377, 
                          0.059472, 0.020254, 0.017982, 0.016738, 0.016242, 0.013373, 
                          0.014151, 0.014694, 0.013764]) / np.sqrt(5) # 0.221098, 0.218503,
lpips_idlg_mean = np.array([0.183026, 0.183764, 0.182633, 0.257172, 
                            0.444235, 0.484002, 0.483562, 0.483421, 0.485871, 0.49435, 
                            0.495876, 0.497378, 0.499236]) # 0.184236, 0.185861,
lpips_idlg_std = np.array([0.144886, 0.141139, 0.138153, 0.118123, 
                           0.074746, 0.070512, 0.068119, 0.069509, 0.069284, 0.066782, 
                           0.065875, 0.06832, 0.069195]) / np.sqrt(5) # 0.143395, 0.142521,

# Random baseline (common to both methods)
ssim_random = 0.006688
ssim_random_std = 0.011079 / np.sqrt(5)
lpips_random = 0.472369
lpips_random_std = 0.063956 / np.sqrt(5)


# Set up the ICML-style plot
fig_size = setup_icml_plot(two_column=True)
# Create a figure with two subplots (one for SSIM, one for LPIPS)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

# --- SSIM Plot ---
ax1.errorbar(x_positions, ssim_dlg_mean, yerr=ssim_dlg_std, fmt='o-', 
             capsize=5, label='DLG', color='black', markersize=4, linewidth=1.2)
ax1.errorbar(x_positions, ssim_idlg_mean, yerr=ssim_idlg_std, fmt='s-', 
             capsize=5, label='iDLG', color='blue', markersize=4, linewidth=1.2)
# Plot random baseline as a horizontal dashed line with a shaded error band
ax1.axhline(ssim_random, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax1.axhspan(ssim_random - ssim_random_std, ssim_random + ssim_random_std, 
            facecolor='green', alpha=0.2)
ax1.set_ylabel('SSIM', fontsize=14)
ax2.set_xlabel('Number of Aggregators', fontsize=14)
ax1.set_title('Reconstruction Quality (SSIM)', fontsize=16)
ax1.legend(fontsize=8)
# Set x-axis ticks to display the original split values
ax1.set_xticks(x_positions)
ax1.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")

# --- LPIPS Plot ---
ax2.errorbar(x_positions, lpips_dlg_mean, yerr=lpips_dlg_std, fmt='o-', 
             capsize=5, label='DLG', color='black', markersize=4, linewidth=1.2)
ax2.errorbar(x_positions, lpips_idlg_mean, yerr=lpips_idlg_std, fmt='s-', 
             capsize=5, label='iDLG', color='blue', markersize=4, linewidth=1.2)
# Plot random baseline as a horizontal dashed line with a shaded error band
ax2.axhline(lpips_random, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax2.axhspan(lpips_random - lpips_random_std, lpips_random + lpips_random_std, 
            facecolor='green', alpha=0.2)
ax2.set_ylabel('LPIPS', fontsize=14)
ax2.set_xlabel('Number of Aggregators', fontsize=14)
ax2.set_title('Reconstruction Quality (LPIPS)', fontsize=16)
ax2.legend(fontsize=8)
# Set x-axis ticks to display the original split values
ax2.set_xticks(x_positions)
ax2.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")

plt.tight_layout()
plt.savefig('plots-dlg-idlg_lfw.pdf', bbox_inches='tight')
plt.close()
















##############################################################################
# LPIPS unique plot across dataset 
##############################################################################
# Define the split values and use discrete positions for the x-axis
# splits = np.array([1, 1.00001, 1.0001, 1.001, 1.01, 1.1, 1.2, 1.4, 2, 4, 8, 16, 32])
# x_positions = np.arange(len(splits))  # equispaced positions

# # mnist 
# lpips_dlg_mean_mnist = np.array([0.109889, 0.128435, 0.122332, 0.156936, 0.232211, 0.518334, 
#                            0.551045, 0.552543, 0.562652, 0.573914, 0.573787, 0.574807, 0.580589])
# lpips_dlg_std_mnist = np.array([0.225457, 0.234179, 0.23469, 0.228253, 0.177211, 0.111447, 
#                           0.082754, 0.083656, 0.089072, 0.089199, 0.093084, 0.086936, 0.08634]) / np.sqrt(5)
# lpips_idlg_mean_mnist = np.array([0.03717, 0.032397, 0.060986, 0.075882, 0.193202, 0.511458, 
#                             0.551025, 0.553632, 0.567528, 0.586007, 0.5895, 0.584929, 0.589685])
# lpips_idlg_std_mnist = np.array([0.136978, 0.121617, 0.168899, 0.141556, 0.13649, 0.111773, 
#                            0.089535, 0.085258, 0.087831, 0.093293, 0.085115, 0.087164, 0.089007]) / np.sqrt(5)
# lpips_random_mnist = 0.566315
# lpips_random_std_mnist = 0.097331 / np.sqrt(5)

# # cifar
# lpips_dlg_mean_cifar = np.array([0.235061, 0.231724, 0.23072, 0.27145, 0.36258, 0.394891, 
#                            0.391388, 0.391977, 0.395981, 0.401567022, 0.40264, 0.410647691, 0.419258])
# lpips_dlg_std_cifar = np.array([0.160128, 0.15721, 0.153744, 0.120143, 0.076478, 0.070736, 
#                           0.063449, 0.071155, 0.073802, 0.076013595, 0.072612, 0.072433139, 0.06813]) / np.sqrt(5)
# lpips_idlg_mean_cifar = np.array([0.171193, 0.171695, 0.174597, 0.238449, 0.34739, 0.393412, 
#                             0.39116, 0.393663, 0.394875, 0.406383881, 0.408849202, 0.41753449, 0.420213111])
# lpips_idlg_std_cifar = np.array([0.137549, 0.138385, 0.130402, 0.107466, 0.075881, 0.074989, 
#                            0.07248, 0.071449, 0.072945, 0.074391271, 0.072059309, 0.073025471, 0.070792152]) / np.sqrt(5)
# lpips_random_cifar = 0.415879
# lpips_random_std_cifar = 0.072017 / np.sqrt(5)

# # lfw
# lpips_dlg_mean_lfw = np.array([0.306268, 0.298462, 0.302072, 0.339744, 
#                            0.454117, 0.465857, 0.460154, 0.461122, 0.471396, 0.485633, 
#                            0.490871, 0.489892, 0.495015]) # 0.306133, 0.297219,
# lpips_dlg_std_lfw = np.array([0.171823, 0.172603, 0.16744, 0.134216, 
#                           0.073346, 0.065453, 0.069977, 0.067561, 0.0699, 0.064886, 
#                           0.069068, 0.073321, 0.069946]) / np.sqrt(5) # 0.169103, 0.164938,
# lpips_idlg_mean_lfw = np.array([0.183026, 0.183764, 0.182633, 0.257172, 
#                             0.444235, 0.484002, 0.483562, 0.483421, 0.485871, 0.49435, 
#                             0.495876, 0.497378, 0.499236]) # 0.184236, 0.185861,
# lpips_idlg_std_lfw = np.array([0.144886, 0.141139, 0.138153, 0.118123, 
#                            0.074746, 0.070512, 0.068119, 0.069509, 0.069284, 0.066782, 
#                            0.065875, 0.06832, 0.069195]) / np.sqrt(5) # 0.143395, 0.142521,
# lpips_random_lfw = 0.472369
# lpips_random_std_lfw = 0.063956 / np.sqrt(5)

# splits = np.array([1, 1.001, 1.01, 1.1, 1.2, 1.4, 2, 4, 8, 16, 32])
splits = np.array([100, 99.9, 99.0, 90.0, 80.0, 70, 50, 25, 12.5, 6.25, 3.125])
x_positions = np.arange(len(splits))  # equispaced positions

# mnist 
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

# cifar
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

# lfw
lpips_dlg_mean_lfw = np.array([0.306268, 0.339744, 
                           0.454117, 0.465857, 0.460154, 0.461122, 0.471396, 0.485633, 
                           0.490871, 0.489892, 0.495015]) # 0.306133, 0.297219,
lpips_dlg_std_lfw = np.array([0.171823, 0.134216, 
                          0.073346, 0.065453, 0.069977, 0.067561, 0.0699, 0.064886, 
                          0.069068, 0.073321, 0.069946]) / np.sqrt(5) # 0.169103, 0.164938,
lpips_idlg_mean_lfw = np.array([0.183026, 0.257172, 
                            0.444235, 0.484002, 0.483562, 0.483421, 0.485871, 0.49435, 
                            0.495876, 0.497378, 0.499236]) # 0.184236, 0.185861,
lpips_idlg_std_lfw = np.array([0.144886, 0.118123, 
                           0.074746, 0.070512, 0.068119, 0.069509, 0.069284, 0.066782, 
                           0.065875, 0.06832, 0.069195]) / np.sqrt(5) # 0.143395, 0.142521,
lpips_random_lfw = 0.472369
lpips_random_std_lfw = 0.063956 / np.sqrt(5)



# Set up the ICML-style plot
fig_size = setup_icml_plot(two_column=True)
# Create a figure with two subplots (one for SSIM, one for LPIPS)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))


# --- MNIST Plot ---
ax1.errorbar(x_positions, lpips_dlg_mean_mnist, yerr=lpips_dlg_std_mnist, fmt='o-', 
             capsize=5, label='_nolegend_', color='black', markersize=4, linewidth=1.2, alpha=0.3)
ax1.errorbar(x_positions, lpips_idlg_mean_mnist, yerr=lpips_idlg_std_mnist, fmt='s-', 
             capsize=5, label='_nolegend_', color='blue', markersize=4, linewidth=1.2, alpha=0.3)
ax1.plot(x_positions, lpips_dlg_mean_mnist, 'o-', color='black', markersize=4, linewidth=1.2, label='DLG')
ax1.plot(x_positions, lpips_idlg_mean_mnist, 's-', color='blue', markersize=4, linewidth=1.2, label='iDLG')
# Plot random baseline as a horizontal dashed line with a shaded error band
ax1.axhline(lpips_random_mnist, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax1.axhspan(lpips_random_mnist - lpips_random_std_mnist, lpips_random_mnist + lpips_random_std_mnist, 
            facecolor='green', alpha=0.2)
ax1.set_ylabel('Reconstruction Quality (LPIPS)', fontsize=14)
# ax1.set_xlabel('Number of Model Splits', fontsize=14)
ax1.set_title('MNIST', fontsize=16)
# ax1.legend(fontsize=10)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
# Set x-axis ticks to display the original split values
ax1.set_xticks(x_positions)
ax1.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")
ax1.set_xlim(-0.5, len(splits) - 0.5)
# ax1.set_xscale('log')
# Optionally, set the ticks to your split values
# ax1.set_xticks(splits)
# ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())




# --- CIFAR Plot ---
ax2.errorbar(x_positions, lpips_dlg_mean_cifar, yerr=lpips_dlg_std_cifar, fmt='o-', 
             capsize=5, label='_nolegend_', color='black', markersize=4, linewidth=1.2, alpha=0.3)
ax2.errorbar(x_positions, lpips_idlg_mean_cifar, yerr=lpips_idlg_std_cifar, fmt='s-',
                capsize=5, label='_nolegend_', color='blue', markersize=4, linewidth=1.2, alpha=0.3)
ax2.plot(x_positions, lpips_dlg_mean_cifar, 'o-', color='black', markersize=4, linewidth=1.2, label='DLG')
ax2.plot(x_positions, lpips_idlg_mean_cifar, 's-', color='blue', markersize=4, linewidth=1.2, label='iDLG')
# Plot random baseline as a horizontal dashed line with a shaded error band
ax2.axhline(lpips_random_cifar, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax2.axhspan(lpips_random_cifar - lpips_random_std_cifar, lpips_random_cifar + lpips_random_std_cifar, 
            facecolor='green', alpha=0.2)
# ax2.set_ylabel('Reconstruction Quality (LPIPS)', fontsize=14)
ax2.set_xlabel(r'Shared Parameters (\%)', fontsize=14)
ax2.set_title('CIFAR-10', fontsize=16)
ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
# ax2.legend(fontsize=10)
# Set x-axis ticks to display the original split values
ax2.set_xticks(x_positions)
ax2.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")
ax2.set_xlim(-0.5, len(splits) - 0.5)
# ax2.set_xscale('log')
# Optionally, set the ticks to your split values
# ax2.set_xticks(splits)
# ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())




# --- LFW Plot ---
ax3.errorbar(x_positions, lpips_dlg_mean_lfw, yerr=lpips_dlg_std_lfw, fmt='o-', 
             capsize=5, label='_nolegend_', color='black', markersize=4, linewidth=1.2, alpha=0.3)
ax3.errorbar(x_positions, lpips_idlg_mean_lfw, yerr=lpips_idlg_std_lfw, fmt='s-',
                capsize=5, label='_nolegend_', color='blue', markersize=4, linewidth=1.2, alpha=0.3)
ax3.plot(x_positions, lpips_dlg_mean_lfw, 'o-', color='black', markersize=4, linewidth=1.2, label='DLG')
ax3.plot(x_positions, lpips_idlg_mean_lfw, 's-', color='blue', markersize=4, linewidth=1.2, label='iDLG')

# Plot random baseline as a horizontal dashed line with a shaded error band
ax3.axhline(lpips_random_lfw, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
ax3.axhspan(lpips_random_lfw - lpips_random_std_lfw, lpips_random_lfw + lpips_random_std_lfw, 
            facecolor='green', alpha=0.2)
# ax3.set_ylabel('Reconstruction Quality (LPIPS)', fontsize=14)
# ax3.set_xlabel('Number of Model Splits', fontsize=14)
ax3.set_title('LFW', fontsize=16)
# ax3.legend(fontsize=10)
ax3.tick_params(axis='x', labelsize=10)
ax3.tick_params(axis='y', labelsize=10)
# Set x-axis ticks to display the original split values
ax3.set_xticks(x_positions)
ax3.set_xticklabels([str(s) for s in splits], rotation=45, ha="right")
ax3.set_xlim(-0.5, len(splits) - 0.5)
# ax3.set_xscale('log')
# Optionally, set the ticks to your split values
# ax3.set_xticks(splits)
# ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())


# --- Add background highlighting for the ERIS region (from split value 2 onward) ---
# Find the index corresponding to split value 2.
eris_index = np.where(splits == 50)[0][0]  # should be 6
# Compute the left edge for shading (assuming each x_position is centered, subtract half a unit)
left_boundary = eris_index 
right_boundary = x_positions[-1] + 0.5  # extend a bit past the last marker

for ax in [ax1, ax2, ax3]:
    # Add translucent shaded background to indicate the ERIS region
    ax.axvspan(left_boundary, right_boundary, facecolor='lightgrey', alpha=0.3, zorder=0, label='ERIS')
    # Draw a vertical dashed line at the boundary (you can also use x=eris_index if you prefer)
    ax.axvline(x=left_boundary, color='grey', linestyle='-', linewidth=1.2, zorder=1)
    # ax.legend(fontsize=10)

# Collect handles and labels from each axis and remove duplicates.
handles, labels = [], []
for ax in [ax1, ax2, ax3]:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

# Place the legend below the entire figure.
fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.06), fontsize=10)


plt.tight_layout()
plt.savefig('plots-dlg-idlg_lpips.pdf', bbox_inches='tight')
plt.close()

# # --- MNIST Plot ---
# ax1.errorbar(splits, lpips_dlg_mean_mnist, yerr=lpips_dlg_std_mnist, fmt='o-', 
#              capsize=5, color='black', markersize=4, linewidth=1.2, alpha=0.3)
# ax1.errorbar(splits, lpips_idlg_mean_mnist, yerr=lpips_idlg_std_mnist, fmt='s-', 
#              capsize=5, color='blue', markersize=4, linewidth=1.2, alpha=0.3)
# ax1.plot(splits, lpips_dlg_mean_mnist, 'o-', color='black', markersize=4, linewidth=1.2, label='DLG')
# ax1.plot(splits, lpips_idlg_mean_mnist, 's-', color='blue', markersize=4, linewidth=1.2, label='iDLG')
# ax1.axhline(lpips_random_mnist, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
# ax1.axhspan(lpips_random_mnist - lpips_random_std_mnist, lpips_random_mnist + lpips_random_std_mnist, 
#             facecolor='green', alpha=0.2)
# ax1.set_ylabel('Reconstruction Quality (LPIPS)', fontsize=14)
# ax1.set_xlabel('Number of Model Splits', fontsize=14)
# ax1.set_title('MNIST', fontsize=16)
# ax1.legend(fontsize=10)
# ax1.tick_params(axis='x', labelsize=10)
# ax1.tick_params(axis='y', labelsize=10)
# ax1.set_xscale('log') 
# # ax1.set_xscale('symlog', linthresh=1.1, linscale=10.0)
# ax1.set_xticks(splits)
# # ax1.set_xlim(0.95, splits[-1])
# # ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
# # ax1.xaxis.set_major_locator(plt.LogLocator(base=1.01))


# # --- CIFAR Plot ---
# ax2.errorbar(splits, lpips_dlg_mean_cifar, yerr=lpips_dlg_std_cifar, fmt='o-', 
#              capsize=5, color='black', markersize=4, linewidth=1.2, alpha=0.3)
# ax2.errorbar(splits, lpips_idlg_mean_cifar, yerr=lpips_idlg_std_cifar, fmt='s-', 
#              capsize=5, color='blue', markersize=4, linewidth=1.2, alpha=0.3)
# ax2.plot(splits, lpips_dlg_mean_cifar, 'o-', color='black', markersize=4, linewidth=1.2, label='DLG')
# ax2.plot(splits, lpips_idlg_mean_cifar, 's-', color='blue', markersize=4, linewidth=1.2, label='iDLG')
# ax2.axhline(lpips_random_cifar, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
# ax2.axhspan(lpips_random_cifar - lpips_random_std_cifar, lpips_random_cifar + lpips_random_std_cifar, 
#             facecolor='green', alpha=0.2)
# ax2.set_xlabel('Number of Model Splits', fontsize=14)
# ax2.set_title('CIFAR-10', fontsize=16)
# ax2.legend(fontsize=10)
# ax2.tick_params(axis='x', labelsize=10)
# ax2.tick_params(axis='y', labelsize=10)
# ax2.set_xscale('log')
# # ax2.set_xscale('symlog', linthresh=1.1, linscale=10.0)
# ax2.set_xticks(splits)
# # ax2.set_xlim(0.95, splits[-1])
# # ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())


# # --- LFW Plot ---
# ax3.errorbar(splits, lpips_dlg_mean_lfw, yerr=lpips_dlg_std_lfw, fmt='o-', 
#              capsize=5, color='black', markersize=4, linewidth=1.2, alpha=0.3)
# ax3.errorbar(splits, lpips_idlg_mean_lfw, yerr=lpips_idlg_std_lfw, fmt='s-', 
#              capsize=5, color='blue', markersize=4, linewidth=1.2, alpha=0.3)
# ax3.plot(splits, lpips_dlg_mean_lfw, 'o-', color='black', markersize=4, linewidth=1.2, label='DLG')
# ax3.plot(splits, lpips_idlg_mean_lfw, 's-', color='blue', markersize=4, linewidth=1.2, label='iDLG')
# ax3.axhline(lpips_random_lfw, color='green', linestyle='--', linewidth=1.2, label='R. Noise')
# ax3.axhspan(lpips_random_lfw - lpips_random_std_lfw, lpips_random_lfw + lpips_random_std_lfw, 
#             facecolor='green', alpha=0.2)
# ax3.set_xlabel('Number of Model Splits', fontsize=14)
# ax3.set_title('LFW', fontsize=16)
# ax3.legend(fontsize=10)
# ax3.tick_params(axis='x', labelsize=10)
# ax3.tick_params(axis='y', labelsize=10)
# ax3.set_xscale('log')
# # ax3.set_xscale('symlog', linthresh=1.1, linscale=10.0)
# ax3.set_xticks(splits)
# # ax3.set_xlim(0.95, splits[-1])
# # ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# plt.tight_layout()
# plt.savefig('plots-dlg-idlg_lpips.pdf', bbox_inches='tight')
# plt.close()



