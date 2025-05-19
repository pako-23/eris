import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


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




# # DistilBERT
# rounds = list(range(2, 22, 2))
# num_bins = 100
# bin_range = (-1, 1)
# all_hist = []

# for r in rounds:
#     # Load the precomputed weights
#     flat_params = np.load(f"./weight_distributions_distilbert/weights_{r}.npy")
    
#     hist, bin_edges = np.histogram(flat_params, bins=num_bins, range=bin_range)
#     all_hist.append(hist)

# all_hist = np.array(all_hist)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Define the bar dimensions
# dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(bin_centers)  # bin width for each bar
# dy = 1.0  # a constant width for the training round dimension

# # Loop over training rounds and plot the bars for each histogram bin.
# for i, r in enumerate(rounds):
#     xs = bin_centers  # x positions (bin centers)
#     ys = np.full_like(bin_centers, r)  # y positions (training round r)
#     zs = np.zeros_like(bin_centers)  # start at z=0
#     dz = all_hist[i]  # height is given by the histogram frequency
#     ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.6)

# ax.set_xlabel("Weight Value")
# ax.set_ylabel("Training Round")
# ax.set_zlabel("Frequency")
# ax.set_title("3D Histogram of Weight Distributions Across Training Rounds - DistilBERT - IMDB")
# plt.show()





# # ResNet9
# rounds = list(range(20, 220, 20))
# num_bins = 100
# bin_range = (-1, 1)
# all_hist = []

# for r in rounds:
#     # Load the precomputed weights
#     flat_params = np.load(f"./weight distributions/weights_{r}.npy")
    
#     hist, bin_edges = np.histogram(flat_params, bins=num_bins, range=bin_range)
#     all_hist.append(hist)

# all_hist = np.array(all_hist)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Define the bar dimensions
# dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(bin_centers)  # bin width for each bar
# dy = 1.0  # a constant width for the training round dimension

# # Loop over training rounds and plot the bars for each histogram bin.
# for i, r in enumerate(rounds):
#     xs = bin_centers  # x positions (bin centers)
#     ys = np.full_like(bin_centers, r)  # y positions (training round r)
#     zs = np.zeros_like(bin_centers)  # start at z=0
#     dz = all_hist[i]  # height is given by the histogram frequency
#     ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.6)

# ax.set_xlabel("Weight Value")
# ax.set_ylabel("Training Round")
# ax.set_zlabel("Frequency")
# ax.set_title("3D Histogram of Weight Distributions Across Training Rounds - ResNet9 - CIFAR10")
# plt.show()




# # Lenet5
# rounds = list(range(30, 330, 30))
# num_bins = 100
# bin_range = (-1, 1)
# all_hist = []

# for r in rounds:
#     # Load the precomputed weights
#     flat_params = np.load(f"./weight_distributions_lenet/weights_{r}.npy")
    
#     hist, bin_edges = np.histogram(flat_params, bins=num_bins, range=bin_range)
#     all_hist.append(hist)

# all_hist = np.array(all_hist)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# fig = plt.figure(figsize=(14, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Define the bar dimensions
# dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(bin_centers)  # bin width for each bar
# dy = 1.0  # a constant width for the training round dimension

# # Loop over training rounds and plot the bars for each histogram bin.
# for i, r in enumerate(rounds):
#     xs = bin_centers  # x positions (bin centers)
#     ys = np.full_like(bin_centers, r)  # y positions (training round r)
#     zs = np.zeros_like(bin_centers)  # start at z=0
#     dz = all_hist[i]  # height is given by the histogram frequency
#     ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.6)

# ax.set_xlabel("Weight Value")
# ax.set_ylabel("Training Round")
# ax.set_zlabel("Frequency")
# ax.set_title("3D Histogram of Weight Distributions Across Training Rounds - LeNet5 - MNIST")
# plt.show()


# Apply ICML style settings.
# Since we want three subplots in one figure, we'll override the overall figure size.
setup_icml_plot(two_column=True)

# Create a figure with 3 subplots (1 row, 3 columns) with 3D projection
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

def plot_3d_histogram(ax, rounds, weight_path_template, model_title):
    """
    Plot a 3D histogram on the provided axis.
    
    Parameters:
        ax: The matplotlib 3D subplot axis.
        rounds: List of training rounds.
        weight_path_template: A string template for np.load.
                             Use .format(r=<round_number>) to load the file.
        model_title: Title for the subplot.
    """
    num_bins = 100
    bin_range = (-1, 1)
    all_hist = []
    
    # Compute histograms for each training round.
    for r in rounds:
        # Load precomputed flattened weights from a .npy file.
        flat_params = np.load(weight_path_template.format(r=r))
        # Compute histogram counts for a fixed bin range.
        hist, bin_edges = np.histogram(flat_params, bins=num_bins, range=bin_range)
        all_hist.append(hist)
    all_hist = np.array(all_hist)
    
    # Compute bin centers from the bin edges.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Define bar dimensions
    dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(bin_centers)  # width of each bar (in x-dimension)
    dy = 1.0  # constant depth for each training round
    
    # Plot bars for each round.
    for i, r in enumerate(rounds):
        xs = bin_centers                     # x positions: centers of weight bins.
        ys = np.full_like(bin_centers, r)      # y positions: current training round.
        zs = np.zeros_like(bin_centers)        # z positions: starting at 0.
        dz = all_hist[i]                       # bar heights.
        ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.6)
    
    # Set axis labels and title.
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Training Round")
    ax.set_zlabel("Frequency")
    ax.set_title(model_title)

# --- DistilBERT ---
rounds_distil = list(range(2, 22, 2))
# Adjust the file path template as needed. The placeholder {r} will be replaced by the round number.
weight_path_distil = "./weight_distributions/distilbert/weights_{r}.npy"
plot_3d_histogram(ax1, rounds_distil, weight_path_distil, "DistilBERT - IMDB")

# --- ResNet9 ---
rounds_resnet = list(range(20, 220, 20))
# Note: If the folder name contains a space, ensure it is correctly specified.
weight_path_resnet = "./weight distributions/resnet/weights_{r}.npy"
plot_3d_histogram(ax2, rounds_resnet, weight_path_resnet, "ResNet9 - CIFAR10")

# --- LeNet5 ---
rounds_lenet = list(range(30, 330, 30))
weight_path_lenet = "./weight_distributions/lenet/weights_{r}.npy"
plot_3d_histogram(ax3, rounds_lenet, weight_path_lenet, "LeNet5 - MNIST")

plt.tight_layout()
plt.show()
