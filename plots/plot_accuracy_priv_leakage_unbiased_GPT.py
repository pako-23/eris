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

samples = [16, 32, 64, 128] # 256

dataset_cnn = {
    'Shatter': {
        'accuracy': [0.300456612, 0.303825555, 0.330394341, 0.343499757],
        'accuracy_std': [0.012166854, 0.005886243, 0.006464076, 0.00375722] / np.sqrt(5),
        'priv_leak': [0.785, 0.691166667, 0.66175, 0.681568627],
        'priv_leak_std': [0.075, 0.026020825, 0.051158455, 0.008985443] / np.sqrt(5)
    },
    'ERIS': {
        'accuracy': [0.30044508, 0.316027844, 0.341442621, 0.356178438],
        'accuracy_std': [0.009485325, 0.009482723, 0.007574087, 0.004805357] / np.sqrt(5),
        'priv_leak': [0.777333333, 0.682666667, 0.64375, 0.678431373],
        'priv_leak_std': [0.062915287, 0.036084392, 0.051158455, 0.007401798] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy': [0.303741239, 0.32205069, 0.342682099, 0.360441006],
        'accuracy_std': [0.012514085, 0.014644316, 0.006536082, 0.005705798] / np.sqrt(5),
        'priv_leak': [1.0, 0.9875, 0.964583, 0.965686],
        'priv_leak_std': [0.0, 0.0125, 0.003608, 0.008985] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': {
        'accuracy': [0.256585942, 0.262612104, 0.262977151, 0.25779961],
        'accuracy_std': [0.00858558, 0.000689898, 0.000991788, 0.001004381] / np.sqrt(5),
        'priv_leak': [0.541666667, 0.5, 0.493333333, 0.544117647],
        'priv_leak_std': [0.05204165, 0.033071891, 0.025259074, 0.014705882] / np.sqrt(5)
    },
    'SoteriaFL': {
        'accuracy': [0.251796931, 0.259023373, 0.247522325, 0.257808327],
        'accuracy_std': [0.003565785, 0.007463313, 0.007904873, 0.009071697] / np.sqrt(5),
        'priv_leak': [0.541666667, 0.508333333, 0.495416667, 0.540196078],
        'priv_leak_std': [0.05204165, 0.026020825, 0.026020825, 0.014803597] / np.sqrt(5)
    },
    'PriPrune (p=0.3)': {
        'accuracy': [0.204067796, 0.217950317, 0.289982084, 0.294852497],
        'accuracy_std': [0.101103156, 0.006656693, 0.011039067, 0.016663041] / np.sqrt(5),
        'priv_leak': [0.741667, 0.7, 0.708333333, 0.703921569],
        'priv_leak_std': [0.038188, 0.033071891, 0.034422316, 0.007401798] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy': [0.303741239, 0.32405069, 0.342682099, 0.360441006],
    #     'accuracy_std': [0.012514085, 0.014644316, 0.006536082, 0.005705798] / np.sqrt(5),
    #     'priv_leak': [0.678333333, 0.605833333, 0.540833333, 0.596078431],
    #     'priv_leak_std': [0.101036297, 0.019094065, 0.047735163, 0.025357877] / np.sqrt(5)
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
fig, axes = plt.subplots(1, 1, figsize=(4, 4), sharey=False)

fig.subplots_adjust(bottom=0.3)  # leave space at the bottom for legends

datasets = [
    ("CNN/DailyMail",    dataset_cnn),
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
    elif dataset_name == 'CNN/DailyMail':
        ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
        ax.set_ylabel('ROUGE-1', fontsize=15)


    ax.set_xlabel('1 - MIA Accuracy', fontsize=15)
    # ax.set_ylabel('Accuracy', fontsize=14)

###############################################################################
# 5) PLOT EACH DATASET
###############################################################################
for i, (name, data_dict) in enumerate(datasets):
    plot_dataset(axes, name, data_dict)

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
    bbox_to_anchor=(0.55, 0.015),
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
    bbox_to_anchor=(0.55, -0.215),
    title=r"$\mathbf{Samples\: Per\: Client}$",
    ncol=3,
    fontsize=12,
    title_fontsize=12,
    labelspacing=0.85
)

plt.tight_layout()
# plt.show()
if 'Min. Leakage' in baseline_labels:
    plt.savefig('figure_same_symbol_different_dimension_min_leak_unbiased_CNN.pdf', bbox_inches='tight')
else:
    plt.savefig('figure_same_symbol_different_dimension_unbiased_CNN.pdf', bbox_inches='tight')
