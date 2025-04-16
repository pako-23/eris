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






# # Apply ICML style settings.
# # Since we want three subplots in one figure, we'll override the overall figure size.
# setup_icml_plot(two_column=True)

# # Create a figure with 3 subplots (1 row, 3 columns) with 3D projection
# fig = plt.figure(figsize=(14, 5))
# ax1 = fig.add_subplot(131, projection='3d')
# ax2 = fig.add_subplot(132, projection='3d')
# ax3 = fig.add_subplot(133, projection='3d')

# def plot_3d_histogram(ax, rounds, weight_path_template, model_title):
#     """
#     Plot a 3D histogram on the provided axis.
    
#     Parameters:
#         ax: The matplotlib 3D subplot axis.
#         rounds: List of training rounds.
#         weight_path_template: A string template for np.load.
#                              Use .format(r=<round_number>) to load the file.
#         model_title: Title for the subplot.
#     """
#     num_bins = 100
#     bin_range = (-1, 1)
#     all_hist = []
    
#     # Compute histograms for each training round.
#     for r in rounds:
#         # Load precomputed flattened weights from a .npy file.
#         flat_params = np.load(weight_path_template.format(r=r))
#         # Compute histogram counts for a fixed bin range.
#         hist, bin_edges = np.histogram(flat_params, bins=num_bins, range=bin_range)
#         all_hist.append(hist)
#     all_hist = np.array(all_hist)
    
#     # Compute bin centers from the bin edges.
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     # Define bar dimensions
#     dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(bin_centers)  # width of each bar (in x-dimension)
#     dy = 1.0  # constant depth for each training round
    
#     # Plot bars for each round.
#     for i, r in enumerate(rounds):
#         xs = bin_centers                     # x positions: centers of weight bins.
#         ys = np.full_like(bin_centers, r)      # y positions: current training round.
#         zs = np.zeros_like(bin_centers)        # z positions: starting at 0.
#         dz = all_hist[i]                       # bar heights.
#         ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.6)
    
#     # Set axis labels and title.
#     ax.set_xlabel("Weight Value")
#     ax.set_ylabel("Training Round")
#     ax.set_zlabel("Frequency")
#     ax.set_title(model_title)

# # --- DistilBERT ---
# rounds_distil = list(range(2, 22, 2))
# # Adjust the file path template as needed. The placeholder {r} will be replaced by the round number.
# weight_path_distil = "./weight_distributions/distilbert/weights_{r}.npy"
# plot_3d_histogram(ax1, rounds_distil, weight_path_distil, "DistilBERT - IMDB")

# # --- ResNet9 ---
# rounds_resnet = list(range(20, 220, 20))
# # Note: If the folder name contains a space, ensure it is correctly specified.
# weight_path_resnet = "./weight_distributions/resnet/weights_{r}.npy"
# plot_3d_histogram(ax2, rounds_resnet, weight_path_resnet, "ResNet9 - CIFAR10")

# # --- LeNet5 ---
# rounds_lenet = list(range(30, 330, 30))
# weight_path_lenet = "./weight_distributions/lenet/weights_{r}.npy"
# plot_3d_histogram(ax3, rounds_lenet, weight_path_lenet, "LeNet5 - MNIST")

# plt.tight_layout()
# plt.savefig("weight_distributions_3d.png", dpi=300, bbox_inches='tight')
# plt.show()



from matplotlib.ticker import FuncFormatter

# # Apply ICML style settings for two-column plot.
# setup_icml_plot(two_column=True)

# # Formatter that abbreviates z-axis ticks (e.g., 7000 -> 7)
# def short_formatter(x, pos):
#     if x >= 1000:
#         return f"{int(x/1000)}"  # e.g., 7000 -> "7"
#     else:
#         return f"{int(x)}"

# def plot_3d_histogram(ax, rounds, weight_path_template, model_title, show_zlabel=True):
#     """
#     Plot a 3D histogram on the provided axis.
    
#     Parameters:
#         ax: The matplotlib 3D subplot axis.
#         rounds: List of training rounds.
#         weight_path_template: A string template for np.load.
#                              Use .format(r=<round_number>) to load the file.
#         model_title: Title for the subplot.
#         show_zlabel: Whether to display the z-axis label.
#     """
#     num_bins = 100
#     bin_range = (-0.5, 0.5)
#     all_hist = []
    
#     # Compute histograms for each training round.
#     for r in rounds:
#         # Load precomputed flattened weights from a .npy file.
#         flat_params = np.load(weight_path_template.format(r=r))
#         # 
#         flat_params = flat_params[(flat_params > -0.5) & (flat_params < 0.5)]
#         # Compute histogram counts for a fixed bin range.
#         hist, bin_edges = np.histogram(flat_params, bins=num_bins, range=bin_range)
#         all_hist.append(hist)
#     all_hist = np.array(all_hist)
    
#     # Compute bin centers from the bin edges.
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     # Define bar dimensions.
#     dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(bin_centers)  # width of each bar (x-dimension)
#     dy = 1.0  # constant depth for each training round
    
    
    
#     # Plot bars for each round.
#     for i, r in enumerate(rounds):
#         xs = bin_centers                     # x positions: centers of weight bins.
#         ys = np.full_like(bin_centers, r)      # y positions: current training round.
#         zs = np.zeros_like(bin_centers)        # z positions: starting at 0.
#         dz = all_hist[i]                       # bar heights.
#         ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.6)
    
#     # Set axis labels and title.
#     ax.set_xlabel("Weight Value")
#     ax.set_ylabel("Training Round")
#     if show_zlabel:
#         # For rightmost subplot: show z-axis label and apply custom tick formatting.
#         ax.set_zlabel("Frequency (k)")
#         ax.zaxis.set_major_formatter(FuncFormatter(short_formatter))
#     else:
#         # For other subplots: remove z-axis label.
#         ax.set_zlabel("")
    
#     ax.set_title(model_title)

# # Create figure with 3 subplots (1 row, 3 columns) with 3D projection.
# fig = plt.figure(figsize=(14, 5))
# ax1 = fig.add_subplot(131, projection='3d')
# ax2 = fig.add_subplot(132, projection='3d')
# ax3 = fig.add_subplot(133, projection='3d')

# # --- DistilBERT ---
# rounds_distil = list(range(2, 22, 2))
# weight_path_distil = "./weight_distributions/distilbert/weights_{r}.npy"
# plot_3d_histogram(ax1, rounds_distil, weight_path_distil, "DistilBERT - IMDB", show_zlabel=False)

# # --- ResNet9 ---
# rounds_resnet = list(range(20, 220, 20))
# weight_path_resnet = "./weight_distributions/resnet/weights_{r}.npy"
# plot_3d_histogram(ax2, rounds_resnet, weight_path_resnet, "ResNet9 - CIFAR10", show_zlabel=False)

# # --- LeNet5 ---
# rounds_lenet = list(range(30, 330, 30))
# weight_path_lenet = "./weight_distributions/lenet/weights_{r}.npy"
# plot_3d_histogram(ax3, rounds_lenet, weight_path_lenet, "LeNet5 - MNIST", show_zlabel=True)

# # Increase spacing between subplots to prevent overlapping y-axis tick labels.
# plt.subplots_adjust(wspace=0.2)
# plt.savefig("weight_distributions_3d.png", dpi=300, bbox_inches='tight')
# plt.show()





# Apply ICML style settings.
setup_icml_plot(two_column=True)

# Formatter for z-axis ticks (e.g., 7000 -> 7)
def short_formatter(x, pos):
    if x >= 1000:
        return f"{int(x/1000)}"  # For example, 7000 becomes "7"
    else:
        return f"{int(x)}"

def plot_3d_histogram(ax, rounds, weight_path_template, model_title):
    """
    Plot a 3D histogram on the provided axis.
    
    Parameters:
        ax: The matplotlib 3D axis.
        rounds: List of training rounds.
        weight_path_template: A template string for np.load; use .format(r=<round>) to load the file.
        model_title: Title for the figure.
    """
    num_bins = 100
    bin_range = (-0.5, 0.5)
    all_hist = []
    
    # Compute histogram for each training round with weights limited to the range.
    for r in rounds:
        flat_params = np.load(weight_path_template.format(r=r))
        # Limit the weight values between -0.5 and 0.5.
        flat_params = flat_params[(flat_params > -0.5) & (flat_params < 0.5)]
        hist, bin_edges = np.histogram(flat_params, bins=num_bins, range=bin_range)
        all_hist.append(hist)
    all_hist = np.array(all_hist)
    
    # Calculate bin centers.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(bin_centers)  # Bar width
    dy = 1.0  # Constant depth per training round
    
    # Plot bars for each training round.
    for i, r in enumerate(rounds):
        xs = bin_centers                      # X positions: centers of weight bins.
        ys = np.full_like(bin_centers, r)       # Y positions: current training round.
        zs = np.zeros_like(bin_centers)         # Z positions: starting at 0.
        dz = all_hist[i]                        # Bar heights.
        ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.6)
    
    # Set axis labels and title.
    ax.set_xlabel("Weight Value", fontsize=13)
    ax.set_ylabel("Training Round", fontsize=13)
    ax.set_zlabel(r"$\mathrm{Frequency}\,(10^3)$", fontsize=13)
    ax.zaxis.set_major_formatter(FuncFormatter(short_formatter))
    ax.set_title(model_title)

# --- Figure for DistilBERT ---
rounds_distil = list(range(2, 22, 2))
weight_path_distil = "./weight_distributions/distilbert/weights_{r}.npy"
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
plot_3d_histogram(ax1, rounds_distil, weight_path_distil, "DistilBERT - IMDB")
fig1.savefig("weight_distributions_3d_distilbert.png", dpi=300, bbox_inches='tight')
# plt.show()

# --- Figure for ResNet9 ---
rounds_resnet = list(range(20, 220, 20))
weight_path_resnet = "./weight_distributions/resnet/weights_{r}.npy"
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
plot_3d_histogram(ax2, rounds_resnet, weight_path_resnet, "ResNet9 - CIFAR10")
fig2.savefig("weight_distributions_3d_resnet.png", dpi=300, bbox_inches='tight')
# plt.show()

# --- Figure for LeNet5 ---
rounds_lenet = list(range(30, 330, 30))
weight_path_lenet = "./weight_distributions/lenet/weights_{r}.npy"
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
plot_3d_histogram(ax3, rounds_lenet, weight_path_lenet, "LeNet5 - MNIST")
fig3.savefig("weight_distributions_3d_lenet.png", dpi=300, bbox_inches='tight')
# plt.show()
