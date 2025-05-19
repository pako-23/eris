import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.colors as mcolors

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
        "legend.fontsize": 14,  # Font size for legends
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


# 2) OPTION WITH SAME SHAPE FOR EACH NUMBER OF SAMPLES - DIFFERENT SIZE

###############################################################################
# 1) DEFINE THE DATA FOR EACH DATASET
###############################################################################

samples = [4, 8, 16, 32, 64, 128] # 256

dataset_imdb = {
    'ERIS': {  # compression 2000 
        'priv_leak': [0.65216, 0.60512, 0.541178182, 0.534430476,0.527687442, 0.518000941],
        'priv_leak_std': [0.029503314, 0.024989521, 0.017824824, 0.012627739,0.016236438, 0.010716779] / np.sqrt(5)
    },
    'FedAvg': {
        'priv_leak': [0.829333333, 0.784, 0.669090909, 0.635809524,0.605395349, 0.568941176],
        'priv_leak_std': [0.063888792, 0.039517085, 0.018108945, 0.014724684,0.021084907, 0.008068907] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': { # epsilon 10
        'priv_leak': [0.528, 0.504, 0.498909091, 0.500190476,0.498418605, 0.503717647],
        'priv_leak_std': [0.058179798, 0.031189742, 0.017424216, 0.015219036,0.008634643, 0.009248466] / np.sqrt(5)
    },
    'SoteriaFL': {  # epsilon 10
        'priv_leak': [0.552, 0.5136, 0.501090909, 0.500952381,0.501209302, 0.506447059],
        'priv_leak_std': [0.056316566, 0.034834466, 0.015833848, 0.018349097,0.009291156, 0.009237924] / np.sqrt(5)
    },
    'Pruning (p=0.3)': {
        'priv_leak':     [0.704, 0.6576, 0.598181818, 0.564296788, 0.56627907, 0.542],
        'priv_leak_std': [0.032, 0.036274509, 0.028459047, 0.01548168, 0.017605435, 0.011789973] / np.sqrt(5)
    },
    'Min. Leakage': { # TODO TO CHANGE
        'priv_leak': [0.669333333, 0.6176, 0.549818182, 0.53752381,0.530790698, 0.516047059, 0.508210526],
        'priv_leak_std': [0.043328205, 0.027434285, 0.01757534, 0.009890099,0.016246878, 0.011731152, 0.006149155] / np.sqrt(5)
    },
}

dataset_mnist = {
    'ERIS': {   # compression 4
        'priv_leak': [0.684826667, 0.551424, 0.561105455, 0.527097143,0.517371163, 0.516486588],
        'priv_leak_std': [0.031100857, 0.027264554, 0.015841673, 0.008583277,0.006563869, 0.006434615] / np.sqrt(5)
    },    
    'FedAvg': {
        'priv_leak': [0.821333333, 0.72, 0.657818182, 0.599428571,0.566790698, 0.541364706],
        'priv_leak_std': [0.016546232, 0.03014631, 0.022368809, 0.01114896,0.019189866, 0.00650434] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': { # epsilon 10
        'priv_leak': [0.690666667, 0.5944, 0.576727273, 0.53047619,0.530028793, 0.512047059],
        'priv_leak_std': [0.016653328, 0.020015994, 0.016115287, 0.010875507,0.010021197, 0.009079305] / np.sqrt(5)
    },
    'SoteriaFL': { # epsilon 10
        'priv_leak': [0.714666667, 0.5768, 0.568727273, 0.532761905,0.521581395, 0.512894118],
        'priv_leak_std': [0.018086213, 0.018312837, 0.016876802, 0.005981074,0.008442085, 0.007725964] / np.sqrt(5)
    },
    'Pruning (p=0.01)': {
        'priv_leak':     [0.772, 0.6832, 0.635636364, 0.569904762, 0.545395349, 0.526541176],
        'priv_leak_std': [0.033306656, 0.042813082, 0.020939898, 0.011438091, 0.010331865, 0.00693603] / np.sqrt(5)
    },
    'Min. Leakage': {
        'priv_leak':     [0.666666667, 0.5432, 0.553818182, 0.520571429, 0.514139535, 0.514964706],
        'priv_leak_std': [0.026666667, 0.016666133, 0.015581298, 0.010443238, 0.008390677, 0.008479211] / np.sqrt(5)
    }
}

dataset_cifar = {
    'ERIS': {  # compression 24
        'priv_leak': [0.716346667, 0.684752, 0.595810909, 0.574937143,0.53979907, 0.517015529],
        'priv_leak_std': [0.042766433, 0.022957887, 0.022583375, 0.008547059,0.004078845, 0.005254299] / np.sqrt(5)
    },
    'FedAvg': {
        'priv_leak': [0.848, 0.7584, 0.701454545, 0.645714286,0.592930233, 0.561082353],
        'priv_leak_std': [0.045879068, 0.028464715, 0.014102341, 0.007177693,0.007944644, 0.007463832] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': {  # epsilon 10
        'priv_leak': [0.812, 0.724, 0.625454545, 0.571428571,0.539906977, 0.528141176],
        'priv_leak_std': [0.038964371, 0.026046113, 0.011894856, 0.013853788,0.008332724, 0.004918942] / np.sqrt(5)
    },
    'SoteriaFL': {# epsilon 10
        'priv_leak': [0.698666667, 0.6416, 0.582545455, 0.556761905,0.529395349, 0.520658824],
        'priv_leak_std': [0.018571184, 0.01254751, 0.021034407, 0.010935391,0.007356487, 0.005478666] / np.sqrt(5)
    },
    'Pruning (p=0.01)': {  # p 0.01
        'priv_leak': [0.748, 0.7536, 0.698181818, 0.637333333,0.570883721, 0.532188235],
        'priv_leak_std': [0.028720879, 0.036008888, 0.015933747, 0.009626117,0.006744026, 0.006964706] / np.sqrt(5)
    },
    'Min. Leakage': {
        'priv_leak':     [0.702666667, 0.6544, 0.568727273, 0.558095238, 0.530604651, 0.516658824],
        'priv_leak_std': [0.046875722, 0.024344198, 0.013380632, 0.008148286, 0.004719498, 0.004816121] / np.sqrt(5)
    }
}



###############################################################################
# 2) BASELINE COLORS
###############################################################################
baseline_colors = {
    'ERIS':        'tab:blue',
    'FedAvg':      'tab:orange',
    # 'fedavg+DP':   'tab:green',
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': 'tab:green',
    'SoteriaFL':     'tab:red',
    'Pruning (p=0.3)':  'tab:purple',
    'Pruning (p=0.01)': mcolors.to_rgba('tab:purple', alpha=0.5),
    'Min. Leakage': 'tab:gray',
}



###############################################################################
# 3) SETUP THE FIGURE WITH 3 SUBPLOTS
###############################################################################
# fig_size = setup_icml_plot(two_column=False)
# # fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
# fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

# fig.subplots_adjust(bottom=0.3)  # leave space at the bottom for legends

# datasets = [
#     ("IMDB",    dataset_imdb),
#     ("MNIST",   dataset_mnist),
#     ("CIFAR-10", dataset_cifar)
# ]

# # We'll collect handles for one legend (baselines) + one legend (samples)
# baseline_handles = {}
# sample_handles = {}



###############################################################################
# 4) PLOTTING FUNCTION
###############################################################################
###############################################################################
# 1) Compute ratio (Privacy leakage / "Min. Leakage" at the same index)
###############################################################################
baselines = list(dataset_cifar.keys())
min_leak_vals = dataset_cifar['Min. Leakage']['priv_leak']

# For convenience, define the number of samples (X-values) and y positions
x_vals = [4, 8, 16, 32, 64, 128]
y_positions = np.arange(1, len(x_vals)+1)  # 1 through 6

# Prepare a dictionary to the relative values
ratio_dict = {}
for baseline in baselines:
    if baseline == 'Min. Leakage':
        continue  # We'll skip storing ratio for "Min. Leakage" itself
    ratio_dict[baseline] = []
    for i, val in enumerate(dataset_cifar[baseline]['priv_leak']):
        ratio = val - min_leak_vals[i]
        ratio_dict[baseline].append(ratio)

###############################################################################
# 2) Plot the figure
###############################################################################
plt.figure(figsize=(6, 4))

# Draw 6 horizontal lines (one per each sample size).
for i, y in enumerate(y_positions):
    plt.axhline(y=y, color='lightgray', linestyle='--', linewidth=0.8)  # Horizontal line

# Plot each baseline’s ratio on the corresponding horizontal line
for baseline in ratio_dict:
    color = baseline_colors.get(baseline, 'black')
    # For each of the 6 points, we place a diamond marker at x=ratio, y=y_position
    for i, ratio_val in enumerate(ratio_dict[baseline]):
        y = y_positions[i]
        plt.plot(ratio_val, y, marker='D', color=color, label=baseline if i == 0 else "")

# Formatting: set y-ticks to show sample sizes
plt.yticks(y_positions, [str(x) for x in x_vals])
plt.xlabel("Relative Privacy Leakage (Baseline / Min. Leakage)")
plt.ylabel("Number of Training Samples")

# Build a legend (each baseline name will appear once because of the label trick above)
plt.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.show()
