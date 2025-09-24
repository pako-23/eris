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
        "axes.labelsize": 11,  # Font size for axis labels
        "axes.titlesize": 11,  # Font size for titles
        "legend.fontsize": 14,  # Font size for legends
        "xtick.labelsize": 11,  # Font size for x-axis ticks
        "ytick.labelsize": 11,  # Font size for y-axis ticks
 
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

samples = [4, 8, 16, 32, 64] # 256

dataset_imdb = {
    'Shatter': {
        'accuracy': [0.56596, 0.6022, 0.62536, 0.650675, 0.663775],
        'accuracy_std': [0.01306744, 0.048402841, 0.047075928, 0.046238683, 0.054469525] / np.sqrt(5),
        'priv_leak': [0.825, 0.765, 0.682272727, 0.59297619, 0.547848837],
        'priv_leak_std': [0.09537936, 0.040926764, 0.05045864, 0.037184615, 0.018470721] / np.sqrt(5)
    },
    'ERIS': {
        'accuracy': [0.5642, 0.6509, 0.69233, 0.72808, 0.73903],
        'accuracy_std': [0.0419, 0.074076811, 0.111552565, 0.084315621, 0.063460396] / np.sqrt(5),
        'priv_leak': [0.8075, 0.7585, 0.652272727, 0.586547619, 0.542651163],
        'priv_leak_std': [0.0932, 0.045855752, 0.046325742, 0.035142043, 0.015307379] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy': [0.57306, 0.65492, 0.69769, 0.73209, 0.74322],
        'accuracy_std': [0.050948134, 0.062841801, 0.113945915, 0.069884055, 0.054037206] / np.sqrt(5),
        'priv_leak': [0.916666667, 0.84, 0.682727273, 0.619047619, 0.55872093],
        'priv_leak_std': [0.05527708, 0.031622777, 0.039626354, 0.022587698, 0.017276346] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': {
        'accuracy': [0.53837, 0.53801, 0.53763, 0.53774, 0.53794],
        'accuracy_std': [0.000945463, 0.001535546, 0.003009369, 0.004764997, 0.004589858] / np.sqrt(5),
        'priv_leak': [0.525, 0.505, 0.534090909, 0.527380952, 0.50872093],
        'priv_leak_std': [0.014433757, 0.086458082, 0.055809223, 0.032186919, 0.009641351] / np.sqrt(5)
    },
    'SoteriaFL': { # epsilon 10
        'accuracy': [0.53233, 0.54533, 0.53464, 0.53848, 0.53671],
        'accuracy_std': [0.00441238, 0.005266906, 0.007741111, 0.01286468, 0.010062301] / np.sqrt(5),
        'priv_leak': [0.541666667, 0.52, 0.531818182, 0.533333333, 0.508139535],
        'priv_leak_std': [0.02763854, 0.073484692, 0.052025105, 0.03086067, 0.008778877] / np.sqrt(5)
    },
    'PriPrune (p=0.3)': {
        'accuracy': [0.5118, 0.5283, 0.55148, 0.55051, 0.55962],
        'accuracy_std': [0.014294782, 0.042210809, 0.035283707, 0.049161248, 0.043190429] / np.sqrt(5),
        'priv_leak': [0.741666667, 0.645, 0.634090909, 0.585714286, 0.545348837],
        'priv_leak_std': [0.08620067, 0.045552168, 0.043775819, 0.021821789, 0.014098088] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy': [0.718893333, ....],
    #     'accuracy_std': [0.019891609, ...] / np.sqrt(5),
    #     'priv_leak': [0.644444444, ...],
    #     'priv_leak_std': [0.022662309, ...] / np.sqrt(5)
    # }
}

dataset_mnist = {
    'Shatter': {
        'accuracy': [0.143044, 0.162549333, 0.170932, 0.197125333, 0.213074667],
        'accuracy_std': [0.027150507, 0.017947215, 0.077875436, 0.071296708, 0.083769997] / np.sqrt(5),
        'priv_leak': [0.670133333, 0.65728, 0.569890909, 0.557390476, 0.528623256],
        'priv_leak_std': [0.035564777, 0.046452384, 0.017632127, 0.025215904, 0.008883302] / np.sqrt(5)
    },
    'ERIS': { # compression 4
        'accuracy': [0.746025, 0.8161, 0.861075, 0.9051, 0.9269],
        'accuracy_std': [0.007301498, 0.01622421, 0.009649968, 0.005436911, 0.003436568] / np.sqrt(5),
        'priv_leak': [0.648173333, 0.63176, 0.559109091, 0.544533333, 0.524893023],
        'priv_leak_std': [0.033236125, 0.044157319, 0.014472682, 0.013369787, 0.006976248] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy': [0.7488, 0.80715, 0.8679, 0.89755, 0.91845],
        'accuracy_std': [0.05753508, 0.02301896, 0.00448497, 0.0022809, 0.00082006] / np.sqrt(5),
        'priv_leak': [0.687, 0.652, 0.581545455, 0.551428571, 0.53627907],
        'priv_leak_std': [0.013333333, 0.013856406, 0.018362736, 0.00680136, 0.011958102] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': { # epsilon 10
        'accuracy': [0.3198, 0.470075, 0.62255, 0.68065, 0.6914],
        'accuracy_std': [0.053222317, 0.044287435, 0.013956629, 0.018825315, 0.011125421] / np.sqrt(5),
        'priv_leak': [0.66, 0.602, 0.547272727, 0.533809524, 0.52627907],
        'priv_leak_std': [0.048534066, 0.018220867, 0.019666643, 0.019581795, 0.009540554] / np.sqrt(5)
    },
    'SoteriaFL': {
        'accuracy': [0.11825, 0.2942, 0.6654, 0.734075, 0.775425],
        'accuracy_std': [0.016500682, 0.045801255, 0.018561519, 0.036485982, 0.021186597] / np.sqrt(5),
        'priv_leak': [0.636666667, 0.62, 0.563636364, 0.54047619, 0.521162791],
        'priv_leak_std': [0.021858128, 0.025612497, 0.013606027, 0.015728709, 0.013210858] / np.sqrt(5)
    },
    'PriPrune (p=0.01)': {
        'accuracy': [0.4725, 0.760275, 0.8219, 0.837675, 0.847825],
        'accuracy_std': [0.105228418, 0.020879101, 0.012079528, 0.026426916, 0.024533281] / np.sqrt(5),
        'priv_leak': [0.643333333, 0.616, 0.568181818, 0.558571429, 0.538837209],
        'priv_leak_std': [0.025603819, 0.020396078, 0.012952552, 0.006226998, 0.013046076] / np.sqrt(5)
    },
    # 'Min. Leakage': { # DONE
    #     'accuracy': [0.75245, 0.81335, 0.8663, 0.897475, 0.917825],
    #     'accuracy_std': [0.052646106, 0.023496223, 0.005595087, 0.003979557, 0.000782224] / np.sqrt(5),
    #     'priv_leak': [0.604666667, 0.592, 0.547272727, 0.53952381, 0.520697674],
    #     'priv_leak_std': [0.028867513, 0.016, 0.010444659, 0.01897008, 0.009146958] / np.sqrt(5)
    # },
}

dataset_cifar = {
    'Shatter': {
        'accuracy': [0.1207616, 0.1143816, 0.116376, 0.1185656, 0.1111936],
        'accuracy_std': [0.017399252, 0.011062291, 0.011916171, 0.01028048, 0.014874011] / np.sqrt(5),
        'priv_leak': [0.736266667, 0.69032, 0.583454545, 0.568361905, 0.53095814],
        'priv_leak_std': [0.017275802, 0.047843478, 0.020951356, 0.005622985, 0.01748856] / np.sqrt(5)
    },
    'ERIS': {
        'accuracy': [0.19835, 0.279075, 0.31115, 0.3517, 0.367875],
        'accuracy_std': [0.01747348, 0.00859895, 0.01946413, 0.01185601, 0.01012407] / np.sqrt(5),
        'priv_leak': [0.722533333, 0.71488, 0.598090909, 0.570828571, 0.529492002],
        'priv_leak_std': [0.027714096, 0.024634285, 0.019421995, 0.01433114, 0.015892586] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy': [0.202875, 0.278675, 0.3102, 0.3439, 0.3556],
        'accuracy_std': [0.021333468, 0.004871537, 0.003100806, 0.006756848, 0.006433117] / np.sqrt(5),
        'priv_leak': [0.745, 0.71012, 0.605454545, 0.578095238, 0.5374135],
        'priv_leak_std': [0.022110832, 0.00663325, 0.01028519, 0.01771531, 0.007383745] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': { # epsilon 10
        'accuracy': [0.1006, 0.11905, 0.17675, 0.1814, 0.178475],
        'accuracy_std': [0.000927362, 0.019058397, 0.013825429, 0.023696941, 0.011151766] / np.sqrt(5),
        'priv_leak': [0.656666667, 0.634, 0.569090909, 0.548095238, 0.533715258],
        'priv_leak_std': [0.038151744, 0.014282857, 0.006555548, 0.020264688, 0.007064402] / np.sqrt(5)
    },
    'SoteriaFL': {
        'accuracy': [0.100025, 0.101375, 0.09805, 0.1924, 0.23095],
        'accuracy_std': [4.33013e-05, 0.00226757, 0.0031245, 0.008224962, 0.013991515] / np.sqrt(5),
        'priv_leak': [0.623333333, 0.624, 0.597272727, 0.562380952, 0.546318775],
        'priv_leak_std': [0.014529663, 0.025922963, 0.016539459, 0.015900769, 0.008987335] / np.sqrt(5)
    },
    'PriPrune (p=0.01)': {
        'accuracy': [0.11315, 0.20715, 0.2399, 0.24945, 0.255275],
        'accuracy_std': [0.015678887, 0.05898006, 0.01164839, 0.020142306, 0.011634297] / np.sqrt(5),
        'priv_leak': [0.68, 0.62, 0.585454545, 0.566666667, 0.534191719],
        'priv_leak_std': [0.00942809, 0.016492423, 0.015212, 0.030371845, 0.004629711] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy': [0.202225, 0.279125, 0.30915, 0.3439, 0.355775],
    #     'accuracy_std': [0.021336749, 0.005684354, 0.002613905, 0.006756848, 0.006430931] / np.sqrt(5),
    #     'priv_leak': [0.706666667, 0.67, 0.603636364, 0.558095238, 0.557680091],
    #     'priv_leak_std': [0.024944383, 0.073184698, 0.026906634, 0.01771531, 0.01166393] / np.sqrt(5)
    # },
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
    'PriPrune (p=0.3)':  'tab:purple',
    'PriPrune (p=0.01)': mcolors.to_rgba('tab:purple', alpha=0.5),
    'Min. Leakage': 'tab:gray',
    'Shatter':     'tab:brown',
}

# Instead of different markers, we define a single marker ('D') but
# scale the size with the number of samples. For instance:
# size_scale = {
#     4:   40,
#     8:   70,
#     16:  100,
#     32:  130,
#     64:  160,
#     128: 190
# }

size_scale = {
    4:   30,
    8:   60,
    16:  90,
    32:  120,
    64:  150,
    128: 180
}

# These values are the "area" in points^2 for scatter(..., s=...).

###############################################################################
# 3) SETUP THE FIGURE WITH 3 SUBPLOTS
###############################################################################
fig_size = setup_icml_plot(two_column=False)
# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

fig.subplots_adjust(bottom=0.3)  # leave space at the bottom for legends

datasets = [
    ("IMDB",    dataset_imdb),
    ("CIFAR-10", dataset_cifar),
    ("MNIST",   dataset_mnist),
]

# We'll collect handles for one legend (baselines) + one legend (samples)
baseline_handles = {}
sample_handles = {}

###############################################################################
# 4) PLOTTING FUNCTION
###############################################################################
def plot_dataset(ax, dataset_name, data_dict):
    ax.set_title(dataset_name, fontsize=16)
    for baseline_name, vals in data_dict.items():
        color = baseline_colors.get(baseline_name, 'grey')
        accuracies = vals['accuracy']
        leaks      = vals['priv_leak']

        if baseline_name == 'Min. Leakage':
            marker_style = '*'
            # Possibly bump up the marker size or outline thickness:
            edgecolors = 'k'
            alpha_val = 1.0
        else:
            marker_style = 'D'  # diamond for the normal baselines
            edgecolors = 'k'
            alpha_val = 0.5 if 'p=0.01' in baseline_name else 0.8
        
        accuracies = vals['accuracy']
        leaks      = vals['priv_leak']
        
        for i, n_sample in enumerate(samples):
            x_val = 1.0 - leaks[i]
            y_val = accuracies[i]
            # The area of the marker is scaled by n_sample
            s_val = size_scale[n_sample] + 25 if baseline_name=='Min. Leakage' else size_scale[n_sample]  # area in points^2

            ax.scatter(
                x_val,
                y_val,
                color=color,
                marker=marker_style,     # diamond for all
                edgecolors=edgecolors,
                s=s_val,        # scale area by sample size
                alpha=alpha_val,
            )
            
            # Draw 2D error bars if available
            if ('accuracy_std' in vals) and ('priv_leak_std' in vals):
                x_err = vals['priv_leak_std'][i]
                y_err = vals['accuracy_std'][i]
                ax.errorbar(
                    x_val, y_val,
                    xerr=x_err,
                    yerr=y_err,
                    fmt='none',
                    ecolor=color,
                    elinewidth=1,
                    capsize=3,
                    alpha=0.4
                )
            
            # Collect a dummy handle for the baseline color legend
            if baseline_name not in baseline_handles:
                baseline_handles[baseline_name] = plt.Line2D(
                    [0], [0],
                    marker=marker_style, color=color, label=baseline_name,
                    markerfacecolor=color, markersize=8, linewidth=0
                )
            
            # Collect a dummy handle for the sample size legend
            # For legend, we must convert area in pts^2 -> marker size in pts
            if n_sample not in sample_handles:
                marker_size_pts = np.sqrt(s_val)  # convert area -> diameter in points
                sample_handles[n_sample] = plt.Line2D(
                    [0], [0],
                    marker='D',
                    color='black',
                    label=f"{n_sample} samples",
                    markerfacecolor='white',
                    markersize=marker_size_pts,
                    linewidth=0
                )
    
    # Optionally add a random guess line:
    if dataset_name == 'IMDB':
        ax.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess = 50%')
        ax.set_ylabel('Accuracy', fontsize=15)
    elif dataset_name == 'MNIST':
        ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
    elif dataset_name == 'CIFAR-10':
        ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')

    ax.set_xlabel('1 - MIA Accuracy', fontsize=15)
    # ax.set_ylabel('Accuracy', fontsize=14)

###############################################################################
# 5) PLOT EACH DATASET
###############################################################################
for i, (name, data_dict) in enumerate(datasets):
    plot_dataset(axes[i], name, data_dict)

###############################################################################
# 6) CREATE LEGENDS
###############################################################################
baseline_labels = list(baseline_handles.keys())

# Reorder the baseline handles so Min. Leakage appears last
baseline_labels = list(baseline_handles.keys())
if 'Min. Leakage' in baseline_labels:
    baseline_labels.remove('Min. Leakage')
    # baseline_labels.append('Min. Leakage')
    baseline_labels.insert(2, 'Min. Leakage')
    
baseline_hlist  = [baseline_handles[lab] for lab in baseline_labels]

sample_labels = sorted(sample_handles.keys())
sample_hlist  = [sample_handles[s] for s in sample_labels]

# Baselines legend
fig.legend(
    handles=baseline_hlist,
    labels=baseline_labels,
    loc='upper center',
    bbox_to_anchor=(0.28, 0.015),
    title=r"$\mathbf{Baselines}$",
    ncol=3, #if 'Min. Leakage' in baseline_labels else 3,
    fontsize=12,
    title_fontsize=12,
    labelspacing=0.85
)

# Sample-sizes legend
fig.legend(
    handles=sample_hlist,
    labels=[f"{s} samples" for s in sample_labels],
    loc='upper center',
    bbox_to_anchor=(0.75, 0.015),
    title=r"$\mathbf{Samples\: Per\: Client}$",
    ncol=3,
    fontsize=12,
    title_fontsize=12,
    labelspacing=0.85
)

plt.tight_layout()
# plt.show()
if 'Min. Leakage' in baseline_labels:
    plt.savefig('figure_nonIID_same_symbol_different_dimension_min_leak_unbiased.pdf', bbox_inches='tight')
else:
    plt.savefig('figure_nonIID_same_symbol_different_dimension_unbiased.pdf', bbox_inches='tight')
