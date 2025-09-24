import matplotlib.pyplot as plt
import numpy as np
import numpy as np
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
        "axes.labelsize": 10.6,  # Font size for axis labels
        "axes.titlesize": 10.6,  # Font size for titles
        "legend.fontsize": 14,  # Font size for legends
        "xtick.labelsize": 10.6,  # Font size for x-axis ticks
        "ytick.labelsize": 10.6,  # Font size for y-axis ticks

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

results_4samples =  {
    'FedAvg': {  
        'accuracy': [0.27118, 0.23014, 0.22202, 0.18196, 0.12042, 0.10516, 0.10294],
        'accuracy_std': [0.015046477, 0.020022947, 0.00714462, 0.015758629, 0.01983879, 0.005542057, 0.0039717] / np.sqrt(5),
        'priv_leak': [0.152, 0.153333333, 0.166666667, 0.169333333, 0.186666667, 0.202666667, 0.252],
        'priv_leak_std': [0.045879068, 0.042373996,0.041739936, 0.027840817, 0.049351348, 0.03592585, 0.030811253] / np.sqrt(5)
    },
    'ERIS': { 
        'accuracy': [0.2714, 0.241, 0.22292, 0.17236, 0.12034, 0.103796, 0.09972],
        'accuracy_std': [0.009530792, 0.019810603, 0.008626332, 0.021533193, 0.017688256, 0.00160574, 0.000386782] / np.sqrt(5),
        'priv_leak': [0.273253333, 0.306533333, 0.313333333, 0.329573333,0.341333333, 0.333948369, 0.345626667],
        'priv_leak_std': [0.049876976, 0.029545054, 0.033730962, 0.033970173,0.008399788, 0.011153863, 0.012688234] / np.sqrt(5)
    },
    'PriPrune': {  
        'accuracy': [0.27118, 0.24448, 0.21568, 0.18596, 0.1374, 0.1105, 0.10088],
        'accuracy_std': [0.012046477, 0.010036214, 0.035119875, 0.021529756,0.020517017, 0.012233397, 0.003614084] / np.sqrt(5),
        'priv_leak': [0.152, 0.146666667, 0.157333333, 0.222666667,0.252, 0.296, 0.326666667],
        'priv_leak_std': [0.045879068, 0.033993463, 0.041225666, 0.038781439,0.028720879, 0.029992592, 0.026331224] / np.sqrt(5)
    },
    'SoteriaFL': {  
        'accuracy': [
            0.28114, 0.27078, 0.23792, 0.10156, 0.10088, 0.1, 0.1],
        'accuracy_std': [0.006134036, 0.025994261, 0.036205657, 0.002648471,0.00176, 0.0, 0.0] / np.sqrt(5),
        'priv_leak': [0.126666667, 0.170666667, 0.197333333, 0.214666667,0.278666667, 0.312, 0.332],
        'priv_leak_std': [0.032386554, 0.033888707, 0.031155524, 0.014236104,0.027455014, 0.011469767, 0.016546232] / np.sqrt(5)
    },
}

results_8samples = {
    'FedAvg': { 
        'accuracy': [0.32784, 0.26134, 0.25914, 0.23596, 0.1795, 0.12502, 0.10812],
        'accuracy_std': [0.00908296, 0.022984221, 0.022308259, 0.007649732,0.020822776, 0.01419259, 0.008755661] / np.sqrt(5),
        'priv_leak': [0.2416, 0.2456, 0.2592, 0.2784, 0.3088, 0.2712, 0.2872],
        'priv_leak_std': [0.028464715, 0.019032604, 0.025848791, 0.040603448,0.040111844, 0.031025151, 0.023650793] / np.sqrt(5)
    },
    'ERIS': { 
        'accuracy': [0.32414, 0.26868, 0.27078, 0.2409, 0.18718, 0.12868, 0.10912],
        'accuracy_std': [0.013158815, 0.023678378, 0.033285336, 0.003260675,0.02026459, 0.017252409, 0.017983148] / np.sqrt(5),
        'priv_leak': [0.333824, 0.366982809, 0.3867, 0.383981618,0.387584, 0.377904, 0.379608],
        'priv_leak_std': [0.022962169, 0.021002613, 0.034668222, 0.030906831,0.029202231, 0.020603962, 0.028817444] / np.sqrt(5)
    },
    'PriPrune': { 
        'accuracy': [0.32784, 0.30058, 0.28418, 0.19828, 0.15196, 0.14, 0.12766],
        'accuracy_std': [0.00608296, 0.023030884, 0.003927035, 0.01894966,0.026683223, 0.031823828, 0.025228603] / np.sqrt(5),
        'priv_leak': [0.2416, 0.2384, 0.2464, 0.3256, 0.364, 0.3736, 0.3832],
        'priv_leak_std': [0.028464715, 0.028011426, 0.036008888, 0.040366323,0.037094474, 0.029997333, 0.0304] / np.sqrt(5)
    },
    'SoteriaFL': { 
        'accuracy': [0.33756, 0.28562, 0.2776, 0.20864, 0.10518, 0.10042, 0.10002],
        'accuracy_std': [0.013811676, 0.020450663, 0.016669853, 0.015560026,0.007595367, 0.00084, 4E-05] / np.sqrt(5),
        'priv_leak': [0.2216, 0.284, 0.3104, 0.3088, 0.3408, 0.3552, 0.3584],
        'priv_leak_std': [0.029241067, 0.029826163, 0.033903392, 0.018137254,0.017232527, 0.0144, 0.012026637] / np.sqrt(5)
    },
}


results_16samples = {
    'FedAvg': {  
        'accuracy': [0.3443, 0.30876, 0.2848, 0.2497, 0.1914, 0.19218, 0.1332],
        'accuracy_std': [0.010382871, 0.013042024, 0.007222465, 0.004768228,0.011181592, 0.011201678, 0.012442508] / np.sqrt(5),
        'priv_leak': [0.298545455, 0.304727273, 0.330545455, 0.358181818,0.388727273, 0.360727273, 0.348727273],
        'priv_leak_std': [0.014102341, 0.013141317, 0.012877041, 0.020538213,0.009851794, 0.010450988, 0.010116675] / np.sqrt(5)
    },   
    'ERIS': {
        'accuracy': [0.35054, 0.3095, 0.2855, 0.24956, 0.1876, 0.18432, 0.12154],
        'accuracy_std': [0.017494411, 0.013213781, 0.007107461, 0.006315885,0.012321688, 0.009295031, 0.016929335] / np.sqrt(5),
        'priv_leak': [0.411076364, 0.445476364, 0.454138182, 0.453563636,0.455076364, 0.458150056, 0.460094545],
        'priv_leak_std': [0.024482387, 0.017728451, 0.007673103, 0.007308602,0.011015376, 0.014533019, 0.011097896] / np.sqrt(5)
    },
    'PriPrune': {  
        'accuracy': [0.3443, 0.3267, 0.29568, 0.20984, 0.15076, 0.12792, 0.11552],
        'accuracy_std': [0.010382871, 0.0079307, 0.00699068, 0.017651697,0.024869869, 0.021711693, 0.018046761] / np.sqrt(5),
        'priv_leak': [0.298545455, 0.296363636, 0.301818182, 0.401454545,0.442545455, 0.448363636, 0.455636364],
        'priv_leak_std': [0.014102341, 0.010220705, 0.015933747, 0.019897257, 0.01612349, 0.016286688, 0.015019546] / np.sqrt(5)
    },
    'SoteriaFL': {  
        'accuracy': [0.35964, 0.33484, 0.30838, 0.25592, 0.16098, 0.1026, 0.1],
        'accuracy_std': [0.012819766, 0.023732476, 0.019266904, 0.017066622,0.007775963, 0.0052, 0.0] / np.sqrt(5),
        'priv_leak': [0.273454545, 0.356, 0.371636364, 0.389818182,0.416, 0.413818182, 0.432363636],
        'priv_leak_std': [0.017233448, 0.018159987, 0.017001701, 0.017836725,0.019083115, 0.021957133, 0.013673888] / np.sqrt(5)
    },
    'Shatter': {  
        'accuracy': [0.1242276, 0.12462, 0.1148394, 0.1130692, 0.1052931, 0.1062184, 0.100031],
        'accuracy_std': [0.0164506518, 0.0150292, 0.0169302, 0.0138293, 0.012392, 0.0129942, 0.0139281] / np.sqrt(5),
        'priv_leak': [0.642050909, 0.614523636, 0.582042, 0.562932, 0.552821, 0.548849944, 0.546905455],
        'priv_leak_std': [0.025454407, 0.027728451, 0.011673103, 0.010308602, 0.012015376, 0.015533019, 0.014097896] / np.sqrt(5)
    },
}

results_16samples['Shatter']['priv_leak'] = [1 - val for val in results_16samples['Shatter']['priv_leak']]


# (Optional) color map to stay consistent with your previous style:
baseline_colors = {
    'FedAvg': 'tab:orange',
    'ERIS':   'tab:blue',
    'PriPrune': 'tab:purple',
    'SoteriaFL': 'tab:red',
    'Shatter': 'tab:brown',
}

fig_size = setup_icml_plot(two_column=False)


# 4 samples plot
fig, ax = plt.subplots(figsize=(5, 4))

# Plot markers with error bars for each method (without connecting lines)
for method_name, vals in results_4samples.items():
    # Convert lists to numpy arrays for calculations
    x = np.array(vals['priv_leak'])
    y = np.array(vals['accuracy'])
    xerr = np.array(vals['priv_leak_std'])
    yerr = np.array(vals['accuracy_std'])
    
    color = baseline_colors.get(method_name, 'gray')
    
    ax.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt='D', markersize=6,
        markeredgecolor='k',
        markerfacecolor=color,
        linestyle='None',  # No line connecting individual points
        ecolor=color,
        elinewidth=1,
        capsize=3,
        alpha=0.8,
        label=method_name
    )

# Combine all points across methods for Pareto front computation
all_points = []
for vals in results_4samples.values():
    x = np.array(vals['priv_leak'])
    y = np.array(vals['accuracy'])
    for xi, yi in zip(x, y):
        all_points.append((xi, yi))
all_points = np.array(all_points)  # shape (N, 2)

# Function to compute the Pareto front (for maximization: higher is better)
def compute_pareto_front(points):
    num_points = points.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        for j in range(num_points):
            if i != j and is_pareto[i]:
                # A point j dominates point i if it is at least as good in both objectives
                # and strictly better in at least one.
                if (points[j, 0] >= points[i, 0] and 
                    points[j, 1] >= points[i, 1] and 
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break
    return is_pareto

pareto_mask = compute_pareto_front(all_points)
pareto_points = all_points[pareto_mask]

# Sort Pareto front points by x-value (i.e. 1 - MIA Accuracy) for a smooth connected line
pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

# Plot the Pareto front connecting line (black solid line)
ax.plot(pareto_points[:, 0], pareto_points[:, 1], color='k', linestyle='-', linewidth=1.5, label='Pareto Front')

# Axis labels and legend (formatted for a scientific paper)
ax.set_xlabel('1 - MIA Accuracy', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(title='Method', fontsize=10.3, title_fontsize=11, frameon=True, framealpha=1)
# legend.get_frame().set_facecolor('grey')

# title
ax.set_title('4 Samples', fontsize=16)

plt.tight_layout()
plt.savefig('pareto_front_same_style_4samples.pdf', bbox_inches='tight')
# plt.show()



# 8 samples plot
fig, ax = plt.subplots(figsize=(5, 4))

# Plot markers with error bars for each method (without connecting lines)
for method_name, vals in results_8samples.items():
    # Convert lists to numpy arrays for calculations
    x = np.array(vals['priv_leak'])
    y = np.array(vals['accuracy'])
    xerr = np.array(vals['priv_leak_std'])
    yerr = np.array(vals['accuracy_std'])
    
    color = baseline_colors.get(method_name, 'gray')
    
    ax.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt='D', markersize=6,
        markeredgecolor='k',
        markerfacecolor=color,
        linestyle='None',  # No line connecting individual points
        ecolor=color,
        elinewidth=1,
        capsize=3,
        alpha=0.8,
        label=method_name
    )

# Combine all points across methods for Pareto front computation
all_points = []
for vals in results_8samples.values():
    x = np.array(vals['priv_leak'])
    y = np.array(vals['accuracy'])
    for xi, yi in zip(x, y):
        all_points.append((xi, yi))
all_points = np.array(all_points)  # shape (N, 2)

# Function to compute the Pareto front (for maximization: higher is better)
def compute_pareto_front(points):
    num_points = points.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        for j in range(num_points):
            if i != j and is_pareto[i]:
                # A point j dominates point i if it is at least as good in both objectives
                # and strictly better in at least one.
                if (points[j, 0] >= points[i, 0] and 
                    points[j, 1] >= points[i, 1] and 
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break
    return is_pareto

pareto_mask = compute_pareto_front(all_points)
pareto_points = all_points[pareto_mask]

# Sort Pareto front points by x-value (i.e. 1 - MIA Accuracy) for a smooth connected line
pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

# Plot the Pareto front connecting line (black solid line)
ax.plot(pareto_points[:, 0], pareto_points[:, 1], color='k', linestyle='-', linewidth=1.5, label='Pareto Front')

# Axis labels and legend (formatted for a scientific paper)
ax.set_xlabel('1 - MIA Accuracy', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(title='Method', fontsize=10.3, title_fontsize=11, frameon=True, framealpha=1)
# legend.get_frame().set_facecolor('white')

# title
ax.set_title('8 Samples', fontsize=16)

plt.tight_layout()
plt.savefig('pareto_front_same_style_8samples.pdf', bbox_inches='tight')
# plt.show()



# 16 samples plot
fig, ax = plt.subplots(figsize=(5, 4))

# Plot markers with error bars for each method (without connecting lines)
for method_name, vals in results_16samples.items():
    # Convert lists to numpy arrays for calculations
    x = np.array(vals['priv_leak'])
    y = np.array(vals['accuracy'])
    xerr = np.array(vals['priv_leak_std'])
    yerr = np.array(vals['accuracy_std'])
    
    color = baseline_colors.get(method_name, 'gray')
    
    ax.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt='D', markersize=6,
        markeredgecolor='k',
        markerfacecolor=color,
        linestyle='None',  # No line connecting individual points
        ecolor=color,
        elinewidth=1,
        capsize=3,
        alpha=0.8,
        label=method_name
    )

# Combine all points across methods for Pareto front computation
all_points = []
for vals in results_16samples.values():
    x = np.array(vals['priv_leak'])
    y = np.array(vals['accuracy'])
    for xi, yi in zip(x, y):
        all_points.append((xi, yi))
all_points = np.array(all_points)  # shape (N, 2)

# Function to compute the Pareto front (for maximization: higher is better)
def compute_pareto_front(points):
    num_points = points.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        for j in range(num_points):
            if i != j and is_pareto[i]:
                # A point j dominates point i if it is at least as good in both objectives
                # and strictly better in at least one.
                if (points[j, 0] >= points[i, 0] and 
                    points[j, 1] >= points[i, 1] and 
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break
    return is_pareto

pareto_mask = compute_pareto_front(all_points)
pareto_points = all_points[pareto_mask]

# Sort Pareto front points by x-value (i.e. 1 - MIA Accuracy) for a smooth connected line
pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

# Plot the Pareto front connecting line (black solid line)
ax.plot(pareto_points[:, 0], pareto_points[:, 1], color='k', linestyle='-', linewidth=1.5, label='Pareto Front')

# Axis labels and legend (formatted for a scientific paper)
ax.set_xlabel('1 - MIA Accuracy', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.legend(title='Method', fontsize=10.3, title_fontsize=11, frameon=True, framealpha=1)
# legend.get_frame().set_facecolor('white')

# title
ax.set_title('16 Samples', fontsize=16)

plt.tight_layout()
plt.savefig('pareto_front_same_style_16samples.pdf', bbox_inches='tight')
# plt.show()












# combined plot
# Define function to compute the Pareto front (for maximization: higher is better)
def compute_pareto_front(points):
    num_points = points.shape[0]
    is_pareto = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        for j in range(num_points):
            if i != j and is_pareto[i]:
                if (points[j, 0] >= points[i, 0] and 
                    points[j, 1] >= points[i, 1] and 
                    (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break
    return is_pareto

# Prepare a list of datasets with corresponding titles
datasets = [(results_4samples, '4 samples'),
            (results_8samples, '8 samples'),
            (results_16samples, '16 samples')]

# Create one figure with 3 subplots (side by side)
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

# Loop over each dataset and its corresponding axis
for ax, (results, title) in zip(axes, datasets):
    # Plot markers with error bars for each method (without connecting lines)
    for method_name, vals in results.items():
        x = np.array(vals['priv_leak'])
        y = np.array(vals['accuracy'])
        xerr = np.array(vals['priv_leak_std'])
        yerr = np.array(vals['accuracy_std'])
        color = baseline_colors.get(method_name, 'gray')
    
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt='D', markersize=6,
            markeredgecolor='k',
            markerfacecolor=color,
            linestyle='None',
            ecolor=color,
            elinewidth=1,
            capsize=3,
            alpha=0.8,
            label=method_name
        )
    
    # Combine all points across methods for Pareto front computation
    all_points = []
    for vals in results.values():
        x = np.array(vals['priv_leak'])
        y = np.array(vals['accuracy'])
        for xi, yi in zip(x, y):
            all_points.append((xi, yi))
    all_points = np.array(all_points)
    
    # Compute and sort Pareto front points by x-value
    pareto_mask = compute_pareto_front(all_points)
    pareto_points = all_points[pareto_mask]
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
    
    # Plot the Pareto front connecting line
    ax.plot(pareto_points[:, 0], pareto_points[:, 1],
            color='k', linestyle='-', linewidth=1.5,
            label='Pareto Front')
    
    # Set the x-axis label and subplot title
    ax.set_xlabel(r'1 - MIA Accuracy', fontsize=14)
    ax.set_title(title, fontsize=16)

# Label y-axis only for the leftmost subplot
axes[0].set_ylabel('Accuracy', fontsize=14)

# # Add legends for each subplot
# axes[0].legend(
#     title='Method',
#     fontsize=10,
#     title_fontsize=11,
#     frameon=True,
#     framealpha=1,
#     loc='upper left',         # Position the upper left corner of the legend at the specified bbox
#     bbox_to_anchor=(0.63, 1.019)  # x and y offset values; adjust these numbers to move the legend
# )

for i, ax in enumerate(axes):
    if i == 1:
        ax.legend(fontsize=9.8, frameon=True, framealpha=1, loc="lower left")
    else:
        ax.legend(fontsize=9.8, frameon=True, framealpha=1)

plt.tight_layout()
plt.savefig('combined_pareto_front_plots.pdf', bbox_inches='tight')
# plt.show()



