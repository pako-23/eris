
# 1) OPTION WITH DIFFERENT SHAPE FOR EACH NUMBER OF SAMPLES

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


    
###############################################################################
# 1) DEFINE THE DATA FOR EACH DATASET
###############################################################################

samples = [4, 8, 16, 32, 64, 128, 256]
samples = [4, 8, 16, 32, 64, 128]

# dataset_imdb = {
#     'eris': {  # compression 2000 
#         'accuracy':      [0.678312, 0.769136, 0.793752, 0.799168, 0.807344, 0.80752, 0.816536],
#         'accuracy_std':  [0.049142987, 0.010037326, 0.012337115, 0.003314685, 0.001627287, 0.006345367, 0.002932368]/ np.sqrt(5),
#         'priv_leak':     [0.682453333, 0.618688, 0.545978182, 0.54023619, 0.540301395, 0.518825412, 0.514069708],
#         'priv_leak_std': [0.050962337, 0.025282511, 0.024087659, 0.006144676, 0.016307247, 0.011586023, 0.004873402]/ np.sqrt(5)
#     },
#     'fedavg': {
#         'accuracy':      [0.669096, 0.779912, 0.795064, 0.809704, 0.815152, 0.817472, 0.82764],
#         'accuracy_std':  [0.055515611, 0.012725267, 0.008257287, 0.002052058, 0.002867462, 0.004081992, 0.006320354]/ np.sqrt(5),
#         'priv_leak':     [0.922666667, 0.8944, 0.837090909, 0.779047619, 0.707162791, 0.6672, 0.643368421],
#         'priv_leak_std': [0.028472209, 0.03010382, 0.021500721, 0.017079247, 0.017267281, 0.012341261, 0.010445014]/ np.sqrt(5)
#     },
#     'fedavg+DP': {  # epsilon 100
#         'accuracy':      [0.53128, 0.535248, 0.543944, 0.547848, 0.570376, 0.571592, 0.591304],
#         'accuracy_std':  [0.010936061, 0.009601013, 0.015365553, 0.007382115, 0.00648689, 0.016375814, 0.008964021]/ np.sqrt(5),
#         'priv_leak':     [0.562666667, 0.5216, 0.519272727, 0.518095238, 0.522976744, 0.5168, 0.510877193],
#         'priv_leak_std': [0.055553778, 0.032946016, 0.031145619, 0.018190227, 0.020578124, 0.005842879, 0.009410373]/ np.sqrt(5)
#     },
#     'SoteriaFL': {  # epsilon 100
#         'accuracy':      [0.530064, 0.543288, 0.56336, 0.55848, 0.55216, 0.576904, 0.578952],
#         'accuracy_std':  [0.015529473, 0.020847843, 0.018765551, 0.020596341, 0.014085408, 0.023198256, 0.027074161]/ np.sqrt(5),
#         'priv_leak':     [0.565333333, 0.5392, 0.523636364, 0.528, 0.529116279, 0.521411765, 0.512093567],
#         'priv_leak_std': [0.051708585, 0.041291161, 0.022178837, 0.016225173, 0.014783383, 0.01100412, 0.00798809]/ np.sqrt(5)
#     },
#     'pruning': {  # p 0.3
#         'accuracy':      [0.540992, 0.585384, 0.637088, 0.6694, 0.738696, 0.777472, 0.776264],
#         'accuracy_std':  [0.029956463, 0.024206718, 0.024124204, 0.017667199, 0.015074357, 0.002999883, 0.001394756]/ np.sqrt(5),
#         'priv_leak':     [0.706666667, 0.6864, 0.587636364, 0.580190476, 0.568744186, 0.543529412, 0.530573099],
#         'priv_leak_std': [0.02921187, 0.037318092, 0.021402093, 0.010404953, 0.014151769, 0.01336338, 0.010315126]/ np.sqrt(5)
#     }
# }

# dataset_mnist = {
#     'eris': {  # compression 3
#         'accuracy':      [0.81446, 0.86516, 0.89256, 0.9221, 0.94726, 0.96822, 0.97444],
#         'accuracy_std':  [0.01685771, 0.01000192, 0.006573462, 0.003414674, 0.0021077, 0.002003397, 0.002006589] / np.sqrt(5),
#         'priv_leak':     [0.670506667, 0.540624, 0.558887273, 0.523417143, 0.517631628, 0.515913412, 0.509974269],
#         'priv_leak_std': [0.034832465, 0.028523131, 0.014508929, 0.009384773, 0.006637327, 0.006862925, 0.002745959] / np.sqrt(5)
#     },
#     'fedavg': {
#         'accuracy':      [0.79632, 0.86928, 0.89542, 0.9214, 0.94648, 0.96808, 0.97472],
#         'accuracy_std':  [0.027197897, 0.006587078, 0.006048934, 0.003147698, 0.002419421, 0.000549181, 0.00130292] / np.sqrt(5),
#         'priv_leak':     [0.890666667, 0.7952, 0.690545455, 0.617904762, 0.577767442, 0.546964706, 0.529824561],
#         'priv_leak_std': [0.02407396, 0.023812602, 0.018413075, 0.009757142, 0.01345272, 0.00808481, 0.006602544] / np.sqrt(5)
#     },
#     'fedavg+DP10': {  # epsilon 10
#         'accuracy':      [0.37752, 0.6099, 0.78134, 0.76118, 0.78698, 0.80924, 0.79678],
#         'accuracy_std':  [0.042004638, 0.02455785, 0.015328222, 0.010937349, 0.005714333, 0.026628151, 0.020091232] / np.sqrt(5),
#         'priv_leak':     [0.770666667, 0.644, 0.597454545, 0.54152381, 0.538046512, 0.528094118, 0.520093567],
#         'priv_leak_std': [0.015549205, 0.027247018, 0.017001701, 0.00784429, 0.009637676, 0.007713342, 0.004468632] / np.sqrt(5)
#     },
#     'SoteriaFL': {  # epsilon 10
#         'accuracy':      [0.31242, 0.51972, 0.7691, 0.76906, 0.77012, 0.766, 0.758],
#         'accuracy_std':  [0.035545655, 0.040101491, 0.00872399, 0.011459948, 0.012460241, 0.004310452, 0.00267058] / np.sqrt(5),
#         'priv_leak':     [0.694666667, 0.5816, 0.568727273, 0.528952381, 0.530325581, 0.525411765, 0.516444444],
#         'priv_leak_std': [0.023626727, 0.006499231, 0.015741717, 0.011342533, 0.008435932, 0.004164108, 0.004333866] / np.sqrt(5)
#     },
#     'pruning': {  # p 0.3
#         'accuracy':      [0.10142, 0.1016, 0.10356, 0.1032, 0.10834, 0.24724, 0.5921],
#         'accuracy_std':  [0.000116619, 0.000109545, 0.000402989, 0.000394968, 0.001284679, 0.003127043, 0.005283559] / np.sqrt(5),
#         'priv_leak':     [0.56, 0.4864, 0.526545455, 0.498285714, 0.507348837, 0.507247059, 0.505052632],
#         'priv_leak_std': [0.037475918, 0.027434285, 0.012256218, 0.012868073, 0.011203028, 0.007933844, 0.006651424] / np.sqrt(5)
#     }
# }

# dataset_cifar = {
#     'eris': {  # compression
#         'accuracy':      [0.27628, 0.32726, 0.35812, 0.42564, 0.46086, 0.49882, 0.56512],
#         'accuracy_std':  [0.005898949, 0.009133148, 0.006745191, 0.009270297, 0.004608948, 0.007149378, 0.003091537] / np.sqrt(5),
#         'priv_leak':     [0.870702585, 0.829456, 0.814472727, 0.792586267, 0.751847442, 0.734665412, 0.709490058],
#         'priv_leak_std': [0.042460687, 0.022192606, 0.0104308, 0.009920529, 0.01177371, 0.007824994, 0.007726473] / np.sqrt(5)
#     },
#     'fedavg': {
#         'accuracy':      [0.27806, 0.32598, 0.36064, 0.42558, 0.4638, 0.49952, 0.56606],
#         'accuracy_std':  [0.010603886, 0.010926555, 0.007801179, 0.006534646, 0.00348425, 0.006447139, 0.004964111] / np.sqrt(5),
#         'priv_leak':     [0.98, 0.9424, 0.941090909, 0.941333333, 0.938697674, 0.898541176, 0.858479532],
#         'priv_leak_std': [0.012649111, 0.006974238, 0.006956773, 0.005132264, 0.006857272, 0.002574939, 0.0025496] / np.sqrt(5)
#     },
#     'fedavg+DP10': {  # epsilon 10
#         'accuracy':      [0.10198, 0.12512, 0.17968, 0.25878, 0.27208, 0.29932, 0.33204],
#         'accuracy_std':  [0.004722033, 0.014304601, 0.003890707, 0.00476168, 0.004298558, 0.001963059, 0.002991053] / np.sqrt(5),
#         'priv_leak':     [0.84, 0.7568, 0.695636364, 0.603047619, 0.581023256, 0.563764706, 0.544982456],
#         'priv_leak_std': [0.037475918, 0.020143485, 0.012202154, 0.011307293, 0.00220918, 0.004489126, 0.005044503] / np.sqrt(5)
#     },
#     'SoteriaFL': {  # epsilon 10
#         'accuracy':      [0.1, 0.1, 0.1, 0.13264, 0.1193, 0.15018, 0.10652],
#         'accuracy_std':  [0.0, 0.0, 0.0, 0.017755067, 0.023638612, 0.003499371, 0.007727199] / np.sqrt(5),
#         'priv_leak':     [0.7, 0.6464, 0.605454545, 0.585905, 0.555906977, 0.547670588, 0.52594152],
#         'priv_leak_std': [0.005962848, 0.01689497, 0.01096953, 0.006772, 0.00692008, 0.006118733, 0.003803166] / np.sqrt(5)
#     },
#     'pruning': {  # p 0.1
#         'accuracy':      [0.1015, 0.1022, 0.10004, 0.11002, 0.15862, 0.26248, 0.34758],
#         'accuracy_std':  [0.004531666, 0.003777301, 4.89898E-05, 0.015387319, 0.019657304, 0.006284075, 0.003105737] / np.sqrt(5),
#         'priv_leak':     [0.669333333, 0.608, 0.536727273, 0.527238095, 0.523069767, 0.516705882, 0.512818713],
#         'priv_leak_std': [0.028158283, 0.026532998, 0.01873345, 0.018066142, 0.004329331, 0.004305304, 0.006800785] / np.sqrt(5)
#     }
# }



# ###############################################################################
# # 2) BASELINE COLORS + MARKER STYLES
# ###############################################################################
# baseline_colors = {
#     'eris':    'tab:blue',
#     'fedavg':       'tab:orange',
#     # 'fedavg+DP100': 'tab:green',
#     # 'fedavg+DP10':  'tab:green',
#     # 'SoteriaFL-100':  'tab:red',
#     # 'SoteriaFL-10':   'tab:red',
#     'fedavg+DP':    'tab:green',
#     'SoteriaFL':      'tab:red',
#     # 'pruning-0.3':  'tab:purple'
#     'pruning':      'tab:purple'
# }

# marker_styles = {
#     4:   'o',   # circle
#     8:   '^',   # triangle_up
#     16:  's',   # square
#     32:  'P',   # plus (filled)
#     64:  'D',   # diamond
#     128: 'X',   # X (filled)
#     256: '*',   # star
# }

# ###############################################################################
# # 3) SETUP THE FIGURE WITH 3 SUBPLOTS
# ###############################################################################
# fig_size = setup_icml_plot(two_column=False)
# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
# fig.subplots_adjust(bottom=0.3)  # leave space at bottom for legends

# # Put the three datasets in a list of (name, data_dict)
# datasets = [
#     ("IMDB",       dataset_imdb),
#     ("MNIST",       dataset_mnist),
#     ("CIFAR-10",    dataset_cifar)
# ]

# # We will collect handles for the legends (baselines + samples) only once
# baseline_handles = {}
# sample_handles = {}

# ###############################################################################
# # 4) PLOTTING FUNCTION
# ###############################################################################
# def plot_dataset(ax, dataset_name, data_dict):
#     """
#     ax          : the subplot axis
#     dataset_name: string for subplot title
#     data_dict   : dictionary containing per-baseline accuracy & priv_leak arrays
#     """
#     ax.set_title(dataset_name, fontsize=13)
#     for baseline_name, vals in data_dict.items():
#         # If we have no color for this baseline in baseline_colors, pick a default
#         color = baseline_colors.get(baseline_name, 'grey')
#         accuracies = vals['accuracy']
#         leaks      = vals['priv_leak']
        
#         for i, n_sample in enumerate(samples):
#             x_val = 1.0 - leaks[i]   # invert the privacy leakage
#             y_val = accuracies[i]
            
#             # Plot each point
#             sc = ax.scatter(
#                 x_val, y_val,
#                 color=color,
#                 marker=marker_styles[n_sample],
#                 edgecolors='k',
#                 s=70,
#                 alpha=0.8
#             )

#             # If std arrays exist, draw error bars in both x and y directions
#             if ('accuracy_std' in vals) and ('priv_leak_std' in vals):
#                 x_err = vals['priv_leak_std'][i]
#                 y_err = vals['accuracy_std'][i]
                
#                 ax.errorbar(
#                     x_val, y_val,
#                     xerr=x_err,
#                     yerr=y_err,
#                     fmt='none',      # no extra marker (the scatter is separate)
#                     ecolor=color,    # same color as the baseline
#                     elinewidth=1,
#                     capsize=3,
#                     alpha=0.8
#                 )
            
#             # Collect a "dummy" handle for the baseline (color) legend, if not already
#             if baseline_name not in baseline_handles:
#                 baseline_handles[baseline_name] = plt.Line2D(
#                     [0], [0],
#                     marker='o', color=color, label=baseline_name,
#                     markerfacecolor=color, markersize=8, linewidth=0
#                 )
            
#             # Collect a "dummy" handle for sample size (marker) legend, if not already
#             if n_sample not in sample_handles:
#                 sample_handles[n_sample] = plt.Line2D(
#                     [0], [0],
#                     marker=marker_styles[n_sample], color='black',
#                     label=f'{n_sample} samples',
#                     markerfacecolor='white', markersize=8, linewidth=0
#                 )
    
#     # After plotting, add a horizontal line for random guessing:
#     if dataset_name == 'IMDB':
#         ax.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess = 50%')
#     elif dataset_name == 'MNIST':
#         ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
#     elif dataset_name == 'CIFAR-10':
#         ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
    
#     ax.set_xlabel('1 - Privacy Leakage', fontsize=11)
#     ax.set_ylabel('Accuracy', fontsize=11)

# ###############################################################################
# # 5) PLOT EACH DATASET IN ITS SUBPLOT
# ###############################################################################
# for i, (name, data_dict) in enumerate(datasets):
#     plot_dataset(axes[i], name, data_dict)

# ###############################################################################
# # 6) CREATE LEGENDS (OUTSIDE / BELOW THE FIGURE)
# ###############################################################################

# # Convert the dicts to sorted lists of handles/labels, if desired
# # Baseline legend
# baseline_labels = list(baseline_handles.keys())
# baseline_hlist  = [baseline_handles[lab] for lab in baseline_labels]

# # Sample legend
# sample_labels = sorted(sample_handles.keys())
# sample_hlist  = [sample_handles[s] for s in sample_labels]

# # We can place them below the figure using fig.legend(...):
# #   Use bbox_to_anchor and loc='upper center' or so to shift them below.
# fig.legend(
#     handles=baseline_hlist,
#     labels=baseline_labels,
#     loc='upper center',
#     bbox_to_anchor=(0.25, -0.0),
#     title=r"$\mathbf{Baselines:}$",
#     ncol=7,  # you can adjust number of columns,
#     # fontsize=12
# )

# fig.legend(
#     handles=sample_hlist,
#     labels=[f"{s} samples" for s in sample_labels],
#     loc='upper center',
#     bbox_to_anchor=(0.73, -0.0),
#     title=r"$\mathbf{Sample\:Per\:Client:}$",
#     ncol=7 , # you can adjust number of columns
#     # fontsize=12
# )

# plt.tight_layout()
# # plt.show()

# plt.savefig('figure_different_symbol.pdf', bbox_inches='tight')























# 2) OPTION WITH SAME SHAPE FOR EACH NUMBER OF SAMPLES - DIFFERENT SIZE

###############################################################################
# 1) DEFINE THE DATA FOR EACH DATASET
###############################################################################

samples = [4, 8, 16, 32, 64, 128] # 256

dataset_imdb = {
    'ERIS': {  # compression 2000 
        'accuracy':      [0.678312, 0.769136, 0.793752, 0.799168, 0.807344, 0.80752, 0.816536],
        'accuracy_std':  [0.049142987, 0.010037326, 0.012337115, 0.003314685, 0.001627287, 0.006345367, 0.002932368]/ np.sqrt(5),
        'priv_leak':     [0.682453333, 0.618688, 0.545978182, 0.54023619, 0.540301395, 0.518825412, 0.514069708],
        'priv_leak_std': [0.050962337, 0.025282511, 0.024087659, 0.006144676, 0.016307247, 0.011586023, 0.004873402]/ np.sqrt(5)
    },
    'FedAvg': {
        'accuracy':      [0.669096, 0.779912, 0.795064, 0.809704, 0.815152, 0.817472, 0.82764],
        'accuracy_std':  [0.055515611, 0.012725267, 0.008257287, 0.002052058, 0.002867462, 0.004081992, 0.006320354]/ np.sqrt(5),
        'priv_leak':     [0.922666667, 0.8944, 0.837090909, 0.779047619, 0.707162791, 0.6672, 0.643368421],
        'priv_leak_std': [0.028472209, 0.03010382, 0.021500721, 0.017079247, 0.017267281, 0.012341261, 0.010445014]/ np.sqrt(5)
    },
    # 'FedAvg (ε, δ)-LDP': {  # epsilon 100
    #     'accuracy':      [0.53128, 0.535248, 0.543944, 0.547848, 0.570376, 0.571592, 0.591304],
    #     'accuracy_std':  [0.010936061, 0.009601013, 0.015365553, 0.007382115, 0.00648689, 0.016375814, 0.008964021]/ np.sqrt(5),
    #     'priv_leak':     [0.562666667, 0.5216, 0.519272727, 0.518095238, 0.522976744, 0.5168, 0.510877193],
    #     'priv_leak_std': [0.055553778, 0.032946016, 0.031145619, 0.018190227, 0.020578124, 0.005842879, 0.009410373]/ np.sqrt(5)
    # },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': {  # epsilon 10
        'accuracy':      [0.534392, 0.535856, 0.544856, 0.542528, 0.552392, 0.553656, 0.561336],
        'accuracy_std':  [0.004830198, 0.003181909, 0.006134263, 0.006008602, 0.00561047, 0.008635021, 0.004081067]/ np.sqrt(5),
        'priv_leak':     [0.557333333, 0.5104, 0.510545455, 0.508190476, 0.507348837, 0.510964706, 0.50554386],
        'priv_leak_std': [0.055553778, 0.034465055, 0.024727273, 0.02193042, 0.016479563, 0.008362182, 0.010848759]/ np.sqrt(5)
    },    
    # 'SoteriaFL': {  # epsilon 100
    #     'accuracy':      [0.530064, 0.543288, 0.56336, 0.55848, 0.55216, 0.576904, 0.578952],
    #     'accuracy_std':  [0.015529473, 0.020847843, 0.018765551, 0.020596341, 0.014085408, 0.023198256, 0.027074161]/ np.sqrt(5),
    #     'priv_leak':     [0.565333333, 0.5392, 0.523636364, 0.528, 0.529116279, 0.521411765, 0.512093567],
    #     'priv_leak_std': [0.051708585, 0.041291161, 0.022178837, 0.016225173, 0.014783383, 0.01100412, 0.00798809]/ np.sqrt(5)
    # },
    'SoteriaFL': {  # epsilon 10
        'accuracy': [0.531424, 0.538416, 0.561016, 0.560216, 0.537376, 0.565472, 0.555816],    
        'accuracy_std': [0.008307244, 0.011377243, 0.004205414, 0.005974913, 0.003948517, 0.010864295, 0.008219181]/ np.sqrt(5),
        'priv_leak': [0.56, 0.5232, 0.520727273, 0.521142857, 0.519813953, 0.514635294, 0.511251462],
        'priv_leak_std': [0.053995885, 0.031839598, 0.028298293, 0.019222058, 0.018876158, 0.011366825, 0.007668234]/ np.sqrt(5)
    },
    'Pruning (p=0.3)': {  # p 0.3
        'accuracy':      [0.540992, 0.585384, 0.637088, 0.6694, 0.738696, 0.777472, 0.776264],
        'accuracy_std':  [0.029956463, 0.024206718, 0.024124204, 0.017667199, 0.015074357, 0.002999883, 0.001394756]/ np.sqrt(5),
        'priv_leak':     [0.706666667, 0.6864, 0.587636364, 0.580190476, 0.568744186, 0.543529412, 0.530573099],
        'priv_leak_std': [0.02921187, 0.037318092, 0.021402093, 0.010404953, 0.014151769, 0.01336338, 0.010315126]/ np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy': [0.669072, 0.779936, 0.795064, 0.809664,0.815168, 0.817472, 0.827632],
    #     'accuracy_std': [0.055480507, 0.012732386, 0.008234662, 0.002019838,0.00286947, 0.004081992, 0.006314398] / np.sqrt(5),
    #     'priv_leak': [0.669333333, 0.6176, 0.549818182, 0.53752381,0.530790698, 0.516047059, 0.508210526],
    #     'priv_leak_std': [0.043328205, 0.027434285, 0.01757534, 0.009890099,0.016246878, 0.011731152, 0.006149155] / np.sqrt(5)
    # },
}

dataset_mnist = {
    'ERIS': {  # compression 3
        'accuracy':      [0.81446, 0.86516, 0.89256, 0.9221, 0.94726, 0.96822, 0.97444],
        'accuracy_std':  [0.01685771, 0.01000192, 0.006573462, 0.003414674, 0.0021077, 0.002003397, 0.002006589] / np.sqrt(5),
        'priv_leak':     [0.670506667, 0.540624, 0.558887273, 0.523417143, 0.517631628, 0.515913412, 0.509974269],
        'priv_leak_std': [0.034832465, 0.028523131, 0.014508929, 0.009384773, 0.006637327, 0.006862925, 0.002745959] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy':      [0.79632, 0.86928, 0.89542, 0.9214, 0.94648, 0.96808, 0.97472],
        'accuracy_std':  [0.027197897, 0.006587078, 0.006048934, 0.003147698, 0.002419421, 0.000549181, 0.00130292] / np.sqrt(5),
        'priv_leak':     [0.890666667, 0.7952, 0.690545455, 0.617904762, 0.577767442, 0.546964706, 0.529824561],
        'priv_leak_std': [0.02407396, 0.023812602, 0.018413075, 0.009757142, 0.01345272, 0.00808481, 0.006602544] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': {  # epsilon 10
        'accuracy':      [0.37752, 0.6099, 0.78134, 0.76118, 0.78698, 0.80924, 0.79678],
        'accuracy_std':  [0.042004638, 0.02455785, 0.015328222, 0.010937349, 0.005714333, 0.026628151, 0.020091232] / np.sqrt(5),
        'priv_leak':     [0.770666667, 0.644, 0.597454545, 0.54152381, 0.538046512, 0.528094118, 0.520093567],
        'priv_leak_std': [0.015549205, 0.027247018, 0.017001701, 0.00784429, 0.009637676, 0.007713342, 0.004468632] / np.sqrt(5)
    },
    'SoteriaFL': {  # epsilon 10
        'accuracy':      [0.31242, 0.51972, 0.7691, 0.76906, 0.77012, 0.766, 0.758],
        'accuracy_std':  [0.035545655, 0.040101491, 0.00872399, 0.011459948, 0.012460241, 0.004310452, 0.00267058] / np.sqrt(5),
        'priv_leak':     [0.694666667, 0.5816, 0.568727273, 0.528952381, 0.530325581, 0.525411765, 0.516444444],
        'priv_leak_std': [0.023626727, 0.006499231, 0.015741717, 0.011342533, 0.008435932, 0.004164108, 0.004333866] / np.sqrt(5)
    },
    # 'Pruning (p=0.3)': {  # p 0.3
    #     'accuracy':      [0.10142, 0.1016, 0.10356, 0.1032, 0.10834, 0.24724, 0.5921],
    #     'accuracy_std':  [0.000116619, 0.000109545, 0.000402989, 0.000394968, 0.001284679, 0.003127043, 0.005283559] / np.sqrt(5),
    #     'priv_leak':     [0.56, 0.4864, 0.526545455, 0.498285714, 0.507348837, 0.507247059, 0.505052632],
    #     'priv_leak_std': [0.037475918, 0.027434285, 0.012256218, 0.012868073, 0.011203028, 0.007933844, 0.006651424] / np.sqrt(5)
    # },
    'Pruning (p=0.01)': {# p 0.01
        'accuracy': [0.75632, 0.82522, 0.88532, 0.88498, 0.9317, 0.96556, 0.97398],
        'accuracy_std': [0.018342235, 0.005874487, 0.003000933, 0.00310316,0.000513809, 0.000677052, 0.000661513] / np.sqrt(5),
        'priv_leak': [0.858666667, 0.7344, 0.576207792, 0.564571429,0.545953488, 0.528376471, 0.520468],
        'priv_leak_std': [0.031944396, 0.030421045, 0.032374859, 0.011145706,0.009040023, 0.007960595, 0.006932] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy': [0.80198, 0.86858, 0.89584, 0.92218, 0.9465, 0.96864, 0.97492],
    #     'accuracy_std': [0.022394, 0.007315846, 0.006297492, 0.003301757,0.002461707, 0.000668132, 0.00124964] / np.sqrt(5),
    #     'priv_leak': [0.665333333, 0.544, 0.550909091, 0.52247619,0.514976744, 0.514305882, 0.506269006],
    #     'priv_leak_std': [0.039417988, 0.012132601, 0.016663912, 0.008445638,0.005922886, 0.007373984, 0.005274115] / np.sqrt(5)
    # },
}

dataset_cifar = {
    # 'ERIS': {  # No compression
    #     'accuracy':      [0.27628, 0.32726, 0.35812, 0.42564, 0.46086, 0.49882, 0.56512],
    #     'accuracy_std':  [0.005898949, 0.009133148, 0.006745191, 0.009270297, 0.004608948, 0.007149378, 0.003091537] / np.sqrt(5),
    #     'priv_leak':     [0.870702585, 0.829456, 0.814472727, 0.792586267, 0.751847442, 0.734665412, 0.709490058],
    #     'priv_leak_std': [0.042460687, 0.022192606, 0.0104308, 0.009920529, 0.01177371, 0.007824994, 0.007726473] / np.sqrt(5)
    # },
    'ERIS': {  # compression 3
        'accuracy': [0.27766, 0.32252, 0.35554, 0.42862, 0.46208, 0.49746, 0.55986],
        'accuracy_std': [0.009568615, 0.014202169, 0.011529024, 0.009214641,0.009331538, 0.006924623, 0.007029822] / np.sqrt(5),
        'priv_leak': [0.80704, 0.787712, 0.759236364, 0.725355238,0.711008534, 0.698407529, 0.675476491],
        'priv_leak_std': [0.033945776, 0.033673162, 0.029245544, 0.007194985,0.008199878, 0.007071347, 0.007072233] / np.sqrt(5)
    },
    'FedAvg': {
        'accuracy':      [0.27806, 0.32598, 0.36064, 0.42558, 0.4638, 0.49952, 0.56606],
        'accuracy_std':  [0.010603886, 0.010926555, 0.007801179, 0.006534646, 0.00348425, 0.006447139, 0.004964111] / np.sqrt(5),
        'priv_leak':     [0.98, 0.9424, 0.941090909, 0.941333333, 0.938697674, 0.898541176, 0.858479532],
        'priv_leak_std': [0.012649111, 0.006974238, 0.006956773, 0.005132264, 0.006857272, 0.002574939, 0.0025496] / np.sqrt(5)
    },
    'FedAvg ($\\varepsilon$, $\\delta$)-LDP': {  # epsilon 10
        'accuracy':      [0.10198, 0.12512, 0.17968, 0.25878, 0.27208, 0.29932, 0.33204],
        'accuracy_std':  [0.004722033, 0.014304601, 0.003890707, 0.00476168, 0.004298558, 0.001963059, 0.002991053] / np.sqrt(5),
        'priv_leak':     [0.84, 0.7568, 0.695636364, 0.603047619, 0.581023256, 0.563764706, 0.544982456],
        'priv_leak_std': [0.037475918, 0.020143485, 0.012202154, 0.011307293, 0.00220918, 0.004489126, 0.005044503] / np.sqrt(5)
    },
    'SoteriaFL': {  # epsilon 10
        'accuracy':      [0.1, 0.1, 0.1, 0.13264, 0.1193, 0.15018, 0.10652],
        'accuracy_std':  [0.0, 0.0, 0.0, 0.017755067, 0.023638612, 0.003499371, 0.007727199] / np.sqrt(5),
        'priv_leak':     [0.7, 0.6464, 0.605454545, 0.585905, 0.555906977, 0.547670588, 0.52594152],
        'priv_leak_std': [0.005962848, 0.01689497, 0.01096953, 0.006772, 0.00692008, 0.006118733, 0.003803166] / np.sqrt(5)
    },
    # 'Pruning (p=0.1)': {  # p 0.1 
    #     'accuracy':      [0.1015, 0.1022, 0.10004, 0.11002, 0.15862, 0.26248, 0.34758],
    #     'accuracy_std':  [0.004531666, 0.003777301, 4.89898E-05, 0.015387319, 0.019657304, 0.006284075, 0.003105737] / np.sqrt(5),
    #     'priv_leak':     [0.669333333, 0.608, 0.536727273, 0.527238095, 0.523069767, 0.516705882, 0.512818713],
    #     'priv_leak_std': [0.028158283, 0.026532998, 0.01873345, 0.018066142, 0.004329331, 0.004305304, 0.006800785] / np.sqrt(5)
    # },
    'Pruning (p=0.01)': {
        'accuracy': [0.2239, 0.3074, 0.34575, 0.410375, 0.44955, 0.4904],
        'accuracy_std': [0.02277301, 0.007760477, 0.004552197, 0.00187133, 0.002634862, 0.005429088] / np.sqrt(5),
        'priv_leak': [0.903333333, 0.876, 0.842727273, 0.788333333, 0.743604651, 0.741235294],
        'priv_leak_std': [0.061191866, 0.019390719, 0.017272727, 0.013725111, 0.008011451, 0.010455872] / np.sqrt(5)
    },
    # 'Min. Leakage': {
    #     'accuracy': [0.27528, 0.32478, 0.35756, 0.43294, 0.46412, 0.49924, 0.56964],
    #     'accuracy_std': [0.016946669, 0.01058327, 0.008656466, 0.006213405,0.002805281, 0.007181532, 0.002386294] / np.sqrt(5),
    #     'priv_leak': [0.804, 0.792, 0.780363636, 0.75752381,0.726325581, 0.712894118, 0.699415205],
    #     'priv_leak_std': [0.034149996, 0.035145412, 0.01197242, 0.016369874,0.009197549, 0.007761711, 0.004856267] / np.sqrt(5)
    # }, 
}



# ###############################################################################
# # 2) BASELINE COLORS
# ###############################################################################
# baseline_colors = {
#     'ERIS':        'tab:blue',
#     'fedavg':      'tab:orange',
#     # 'fedavg+DP':   'tab:green',
#     'fedavg ($\\varepsilon$, $\\delta$)-LDP': 'tab:green',
#     'SoteriaFL':     'tab:red',
#     'Pruning (p=0.3)':  'tab:purple',
#     'Pruning (p=0.1)': mcolors.to_rgba('tab:purple', alpha=0.5),
#     'Min. Leakage': 'tab:gray',
# }

# # Instead of different markers, we define a single marker ('D') but
# # scale the size with the number of samples. For instance:
# # size_scale = {
# #     4:   40,
# #     8:   70,
# #     16:  100,
# #     32:  130,
# #     64:  160,
# #     128: 190
# # }

# size_scale = {
#     4:   30,
#     8:   60,
#     16:  90,
#     32:  120,
#     64:  150,
#     128: 180
# }

# # These values are the "area" in points^2 for scatter(..., s=...).

# ###############################################################################
# # 3) SETUP THE FIGURE WITH 3 SUBPLOTS
# ###############################################################################
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

# ###############################################################################
# # 4) PLOTTING FUNCTION
# ###############################################################################
# def plot_dataset(ax, dataset_name, data_dict):
#     ax.set_title(dataset_name, fontsize=16)
#     for baseline_name, vals in data_dict.items():
#         color = baseline_colors.get(baseline_name, 'grey')
#         accuracies = vals['accuracy']
#         leaks      = vals['priv_leak']

#         if baseline_name == 'Min. Leakage':
#             marker_style = '*'
#             # Possibly bump up the marker size or outline thickness:
#             edgecolors = 'k'
#             alpha_val = 1.0
#         else:
#             marker_style = 'D'  # diamond for the normal baselines
#             edgecolors = 'k'
#             alpha_val = 0.5 if 'p=0.1' in baseline_name else 0.8
        
#         accuracies = vals['accuracy']
#         leaks      = vals['priv_leak']
        
#         for i, n_sample in enumerate(samples):
#             x_val = 1.0 - leaks[i]
#             y_val = accuracies[i]
#             # The area of the marker is scaled by n_sample
#             s_val = size_scale[n_sample]  # area in points^2

#             ax.scatter(
#                 x_val,
#                 y_val,
#                 color=color,
#                 marker=marker_style,     # diamond for all
#                 edgecolors=edgecolors,
#                 s=s_val,        # scale area by sample size
#                 alpha=alpha_val,
#             )
            
#             # Draw 2D error bars if available
#             if ('accuracy_std' in vals) and ('priv_leak_std' in vals):
#                 x_err = vals['priv_leak_std'][i]
#                 y_err = vals['accuracy_std'][i]
#                 ax.errorbar(
#                     x_val, y_val,
#                     xerr=x_err,
#                     yerr=y_err,
#                     fmt='none',
#                     ecolor=color,
#                     elinewidth=1,
#                     capsize=3,
#                     alpha=0.4
#                 )
            
#             # Collect a dummy handle for the baseline color legend
#             if baseline_name not in baseline_handles:
#                 baseline_handles[baseline_name] = plt.Line2D(
#                     [0], [0],
#                     marker=marker_style, color=color, label=baseline_name,
#                     markerfacecolor=color, markersize=8, linewidth=0
#                 )
            
#             # Collect a dummy handle for the sample size legend
#             # For legend, we must convert area in pts^2 -> marker size in pts
#             if n_sample not in sample_handles:
#                 marker_size_pts = np.sqrt(s_val)  # convert area -> diameter in points
#                 sample_handles[n_sample] = plt.Line2D(
#                     [0], [0],
#                     marker='D',
#                     color='black',
#                     label=f"{n_sample} samples",
#                     markerfacecolor='white',
#                     markersize=marker_size_pts,
#                     linewidth=0
#                 )
    
#     # Optionally add a random guess line:
#     if dataset_name == 'IMDB':
#         ax.axhline(y=0.5, color='gray', linestyle='--', label='Random Guess = 50%')
#         ax.set_ylabel('Accuracy', fontsize=14)
#     elif dataset_name == 'MNIST':
#         ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
#     elif dataset_name == 'CIFAR-10':
#         ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
    
#     ax.set_xlabel('1 - Privacy Leakage', fontsize=14)
#     # ax.set_ylabel('Accuracy', fontsize=14)

# ###############################################################################
# # 5) PLOT EACH DATASET
# ###############################################################################
# for i, (name, data_dict) in enumerate(datasets):
#     plot_dataset(axes[i], name, data_dict)

# ###############################################################################
# # 6) CREATE LEGENDS
# ###############################################################################
# baseline_labels = list(baseline_handles.keys())

# # Reorder the baseline handles so Min. Leakage appears last
# baseline_labels = list(baseline_handles.keys())
# if 'Min. Leakage' in baseline_labels:
#     baseline_labels.remove('Min. Leakage')
#     baseline_labels.append('Min. Leakage')
    
# baseline_hlist  = [baseline_handles[lab] for lab in baseline_labels]

# sample_labels = sorted(sample_handles.keys())
# sample_hlist  = [sample_handles[s] for s in sample_labels]

# # Baselines legend
# fig.legend(
#     handles=baseline_hlist,
#     labels=baseline_labels,
#     loc='upper center',
#     bbox_to_anchor=(0.25, 0.015),
#     title=r"$\mathbf{Baselines}$",
#     ncol=4 if 'Min. Leakage' in baseline_labels else 3,
#     fontsize=11,
#     title_fontsize=11,
#     labelspacing=0.85
# )

# # Sample-sizes legend
# fig.legend(
#     handles=sample_hlist,
#     labels=[f"{s} samples" for s in sample_labels],
#     loc='upper center',
#     bbox_to_anchor=(0.73, 0.015),
#     title=r"$\mathbf{Samples\: Per\: Client}$",
#     ncol=3,
#     fontsize=11,
#     title_fontsize=11,
#     labelspacing=0.85
# )

# plt.tight_layout()
# # plt.show()
# if 'Min. Leakage' in baseline_labels:
#     plt.savefig('figure_same_symbol_different_dimension_min_leak.pdf', bbox_inches='tight')
# else:
#     plt.savefig('figure_same_symbol_different_dimension.pdf', bbox_inches='tight')



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
        ax.set_ylabel('Accuracy', fontsize=14)
    elif dataset_name == 'MNIST':
        ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
    elif dataset_name == 'CIFAR-10':
        ax.axhline(y=0.1, color='gray', linestyle='--', label='Random Guess = 10%')
    
    ax.set_xlabel('1 - Privacy Leakage', fontsize=14)
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
    fontsize=11,
    title_fontsize=11,
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
    fontsize=11,
    title_fontsize=11,
    labelspacing=0.85
)

plt.tight_layout()
# plt.show()
if 'Min. Leakage' in baseline_labels:
    plt.savefig('figure_same_symbol_different_dimension_min_leak_biased.pdf', bbox_inches='tight')
else:
    plt.savefig('figure_same_symbol_different_dimension_biased.pdf', bbox_inches='tight')
