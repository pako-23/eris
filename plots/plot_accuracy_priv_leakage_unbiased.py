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

samples = [4, 8, 16, 32, 64, 128] # 256

dataset_imdb = {
    'Shatter': {
        'accuracy': [0.685176, 0.748424, 0.779128, 0.793368, 0.801864, 0.808424],
        'accuracy_std': [0.04662688152, 0.01875747531, 0.00554259145, 0.00639521071, 0.00483790037, 0.00194022679] / np.sqrt(5),
        'priv_leak': [0.676666667, 0.62448, 0.5472, 0.539790476, 0.535386047, 0.520847059],
        'priv_leak_std': [0.028867513, 0.037861896, 0.018989758, 0.014811395, 0.009304874, 0.010741995] / np.sqrt(5)
    },
    'ERIS': {  # compression 2000 
        'accuracy': [0.712808, 0.792848, 0.801064, 0.809856, 0.811584, 0.815912],
        'accuracy_std': [0.047428757, 0.007079182, 0.004625951, 0.002154749,0.001894936, 0.002143104] / np.sqrt(5),
        'priv_leak': [0.65216, 0.60512, 0.541178182, 0.534430476,0.527687442, 0.518000941],
        'priv_leak_std': [0.029503314, 0.024989521, 0.017824824, 0.012627739,0.016236438, 0.010716779] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy': [0.717344, 0.795608, 0.805216, 0.816208, 0.817048, 0.824456],
        'accuracy_std': [0.049347438, 0.006140093, 0.002999691, 0.001105448,0.000499856, 0.00176356] / np.sqrt(5),
        'priv_leak': [0.829333333, 0.784, 0.669090909, 0.635809524,0.605395349, 0.568941176],
        'priv_leak_std': [0.063888792, 0.039517085, 0.018108945, 0.014724684,0.021084907, 0.008068907] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': { # epsilon 10
        'accuracy': [0.538048, 0.538104, 0.538296, 0.539736, 0.54116, 0.542968],
        'accuracy_std': [0.00032239, 0.00023269, 0.00059146, 0.00101338,0.00120877, 0.00187385] / np.sqrt(5),
        'priv_leak': [0.528, 0.504, 0.498909091, 0.500190476,0.498418605, 0.503717647],
        'priv_leak_std': [0.058179798, 0.031189742, 0.017424216, 0.015219036,0.008634643, 0.009248466] / np.sqrt(5)
    },
    'SoteriaFL': {  # epsilon 10
        'accuracy': [0.533552, 0.540112, 0.536664, 0.54348, 0.547896, 0.5528],
        'accuracy_std': [0.002922509, 0.002449411, 0.002852603, 0.003920163,0.0034512, 0.005730089] / np.sqrt(5),
        'priv_leak': [0.552, 0.5136, 0.501090909, 0.500952381,0.501209302, 0.506447059],
        'priv_leak_std': [0.056316566, 0.034834466, 0.015833848, 0.018349097,0.009291156, 0.009237924] / np.sqrt(5)
    },
    'PriPrune (p=0.3)': {
        'accuracy':      [0.535192, 0.559248, 0.60318, 0.61198, 0.62008, 0.68945],
        'accuracy_std':  [0.027913414, 0.034909982, 0.057080648, 0.070242491, 0.059051008, 0.019244789] / np.sqrt(5),
        'priv_leak':     [0.704, 0.6576, 0.598181818, 0.564296788, 0.56627907, 0.542],
        'priv_leak_std': [0.032, 0.036274509, 0.028459047, 0.01548168, 0.017605435, 0.011789973] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy': [0.718893333, 0.79328, 0.80676, 0.81584, 0.816733333, 0.824413333],
    #     'accuracy_std': [0.019891609, 0.00748139, 0.00091971, 0.001087137, 0.000361232, 0.000915836] / np.sqrt(5),
    #     'priv_leak': [0.644444444, 0.586667, 0.532121212, 0.525714286, 0.530542636, 0.515294118],
    #     'priv_leak_std': [0.022662309, 0.020997, 0.025251717, 0.012730119, 0.013206536, 0.011221992] / np.sqrt(5)
    # }
}

dataset_mnist = {
    'Shatter': {
        'accuracy': [0.1195536667, 0.1232192, 0.1455024, 0.1651232, 0.1850225, 0.212887],
        'accuracy_std': [0.02332434071, 0.02921841866, 0.04236111163, 0.06159952491, 0.06061265952, 0.0722007673] / np.sqrt(5),
        'priv_leak': [0.70392, 0.563536, 0.556043636, 0.520838095, 0.514472558, 0.515301647],
        'priv_leak_std': [0.021896825, 0.028905868, 0.012681735, 0.010380802, 0.008539584, 0.007897464] / np.sqrt(5)
    },
    'ERIS': {   # compression 4
        'accuracy': [0.78716, 0.84838, 0.9026, 0.92564, 0.93576, 0.94024],
        'accuracy_std': [0.01194196, 0.005779412, 0.00108074, 0.003013702,0.002317412, 0.001902209] / np.sqrt(5),
        'priv_leak': [0.684826667, 0.551424, 0.561105455, 0.527097143,0.517371163, 0.516486588],
        'priv_leak_std': [0.031100857, 0.027264554, 0.015841673, 0.008583277,0.006563869, 0.006434615] / np.sqrt(5)
    },    
    'FedAvg': {
        'accuracy': [0.80692, 0.8642, 0.89226, 0.91484, 0.92546, 0.93114],
        'accuracy_std': [0.017090278, 0.008792042, 0.007437903, 0.003692479,0.000618385, 0.001611955] / np.sqrt(5),
        'priv_leak': [0.821333333, 0.72, 0.657818182, 0.599428571,0.566790698, 0.541364706],
        'priv_leak_std': [0.016546232, 0.03014631, 0.022368809, 0.01114896,0.019189866, 0.00650434] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': { # epsilon 10
        'accuracy': [0.39654, 0.5084, 0.64396, 0.7035, 0.7043, 0.70476],
        'accuracy_std': [0.031428751, 0.04749101, 0.015298444, 0.01311564,0.01267249, 0.003895947] / np.sqrt(5),
        'priv_leak': [0.690666667, 0.5944, 0.576727273, 0.53047619,0.530028793, 0.512047059],
        'priv_leak_std': [0.016653328, 0.020015994, 0.016115287, 0.010875507,0.010021197, 0.009079305] / np.sqrt(5)
    },
    'SoteriaFL': { # epsilon 10
        'accuracy': [0.08828, 0.32146, 0.67006, 0.77952, 0.78154, 0.79498],
        'accuracy_std': [0.026506859, 0.020026443, 0.013089935, 0.023811207,0.02656265, 0.014538968] / np.sqrt(5),
        'priv_leak': [0.714666667, 0.5768, 0.568727273, 0.532761905,0.521581395, 0.512894118],
        'priv_leak_std': [0.018086213, 0.018312837, 0.016876802, 0.005981074,0.008442085, 0.007725964] / np.sqrt(5)
    },
    'PriPrune (p=0.01)': {
        'accuracy':      [0.47886, 0.70604, 0.84808, 0.87772, 0.87012, 0.86378],
        'accuracy_std':  [0.083292199, 0.036687905, 0.00306294, 0.001506519, 0.001718604, 0.001875527] / np.sqrt(5),
        'priv_leak':     [0.772, 0.6832, 0.635636364, 0.569904762, 0.545395349, 0.526541176],
        'priv_leak_std': [0.033306656, 0.042813082, 0.020939898, 0.011438091, 0.010331865, 0.00693603] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy':      [0.80678, 0.86296, 0.89264, 0.91466, 0.9256, 0.9314],
    #     'accuracy_std':  [0.019521414, 0.010587653, 0.007363586, 0.003940355, 0.000525357, 0.001667333] / np.sqrt(5),
    #     'priv_leak':     [0.666666667, 0.5432, 0.553818182, 0.520571429, 0.514139535, 0.514964706],
    #     'priv_leak_std': [0.026666667, 0.016666133, 0.015581298, 0.010443238, 0.008390677, 0.008479211] / np.sqrt(5)
    # }
}

dataset_cifar = {
    'Shatter': {
        'accuracy': [0.1146736, 0.1157064, 0.1242276, 0.1231576, 0.1295816, 0.1363896],
        'accuracy_std': [0.01749687694, 0.01964500432, 0.0164506518, 0.02030418038, 0.02162403947, 0.01552329201] / np.sqrt(5),
        'priv_leak': [0.77904, 0.707504, 0.642050909, 0.58631619, 0.546029767, 0.520413176],
        'priv_leak_std': [0.055474192, 0.026983107, 0.019545441, 0.00979706, 0.004858406, 0.004148291] / np.sqrt(5)
    },
    'ERIS': {  # compression 24
        'accuracy': [0.2631, 0.33278, 0.34622, 0.37398, 0.38164, 0.38296],
        'accuracy_std': [0.011622908, 0.010554127, 0.014213008, 0.013610496,0.010101208, 0.008779203] / np.sqrt(5),
        'priv_leak': [0.716346667, 0.684752, 0.595810909, 0.574937143,0.53979907, 0.517015529],
        'priv_leak_std': [0.042766433, 0.022957887, 0.022583375, 0.008547059,0.004078845, 0.005254299] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy': [0.27118, 0.32984, 0.3443, 0.37236, 0.38504, 0.38884],
        'accuracy_std': [0.012046477, 0.00608296, 0.010382871, 0.00411028,0.004391628, 0.00321596] / np.sqrt(5),
        'priv_leak': [0.848, 0.7584, 0.701454545, 0.645714286,0.592930233, 0.561082353],
        'priv_leak_std': [0.045879068, 0.028464715, 0.014102341, 0.007177693,0.007944644, 0.007463832] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': {  # epsilon 10
        'accuracy': [0.10326, 0.14926, 0.18916, 0.22312, 0.23364, 0.2413],
        'accuracy_std': [0.005274694, 0.020109858, 0.013146041, 0.011190603,0.008483065, 0.003201874] / np.sqrt(5),
        'priv_leak': [0.812, 0.724, 0.625454545, 0.571428571,0.539906977, 0.528141176],
        'priv_leak_std': [0.038964371, 0.026046113, 0.011894856, 0.013853788,0.008332724, 0.004918942] / np.sqrt(5)
    },
    'SoteriaFL': {# epsilon 10
        'accuracy': [0.10002, 0.10062, 0.1085, 0.1968, 0.2604, 0.26458],
        'accuracy_std': [4E-05, 0.00119063, 0.010611315, 0.007781774,0.005216129, 0.002470142] / np.sqrt(5),
        'priv_leak': [0.698666667, 0.6416, 0.582545455, 0.556761905,0.529395349, 0.520658824],
        'priv_leak_std': [0.018571184, 0.01254751, 0.021034407, 0.010935391,0.007356487, 0.005478666] / np.sqrt(5)
    },
    'PriPrune (p=0.01)': {  # p 0.01
        'accuracy': [0.1374, 0.28418, 0.29568, 0.29386, 0.28696, 0.27988],
        'accuracy_std': [0.020517017, 0.003927035, 0.00699068, 0.005035315,0.005140661, 0.003202124] / np.sqrt(5),
        'priv_leak': [0.748, 0.7536, 0.698181818, 0.637333333,0.570883721, 0.532188235],
        'priv_leak_std': [0.028720879, 0.036008888, 0.015933747, 0.009626117,0.006744026, 0.006964706] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy':      [0.27106, 0.33106, 0.34622, 0.37252, 0.38574, 0.38878],
    #     'accuracy_std':  [0.011716416, 0.006242948, 0.009108326, 0.003781746, 0.003713004, 0.003581843] / np.sqrt(5),
    #     'priv_leak':     [0.702666667, 0.6544, 0.568727273, 0.558095238, 0.530604651, 0.516658824],
    #     'priv_leak_std': [0.046875722, 0.024344198, 0.013380632, 0.008148286, 0.004719498, 0.004816121] / np.sqrt(5)
    # }
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
    plt.savefig('figure_same_symbol_different_dimension_min_leak_unbiased.pdf', bbox_inches='tight')
else:
    plt.savefig('figure_same_symbol_different_dimension_unbiased.pdf', bbox_inches='tight')
