import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

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


# (Optional) color map to stay consistent with your previous style:
baseline_colors = {
    'FedAvg': 'tab:orange',
    'ERIS':   'tab:blue',
    'Pruning': 'tab:purple',
    'SoteriaFL': 'tab:red',
    'Ako': 'tab:green',
    'Shatter': 'tab:olive',
}


fig_size = setup_icml_plot(two_column=False)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

clients = 50
rate = 100_000_000
multiplier = 1_000_000
compression_rate = 0.05

x = np.linspace(1, 1000, 40)

y_fedavg = 2 * (clients * x * multiplier * 8 * 4) / rate
y_eris2 = 2 * (x * multiplier * compression_rate * 8 * 4 * (clients - 1)) / (2 * rate)
y_eris25 = 2 * (x * multiplier * compression_rate * 8 * 4 * (clients - 1)) / (25 * rate)
y_eris50 = 2 * (x * multiplier * compression_rate * 8 * 4 * (clients - 1)) / (50 * rate)
y_priprune = 2 * (clients * x * multiplier * (1 - 0.3) * 8 * 4) / rate
y_soteria = 2 * (clients * x * multiplier * compression_rate * 8 * 4) / rate
y_ako = (x * multiplier * 8 * 4) / rate
y_shatter = np.maximum((4*x*multiplier*8*4)/(rate*clients), (4*x*multiplier*8*4)/rate)
print(f"fedavg/eris2: {y_fedavg[0] / y_eris2[0]}")
print(f"fedavg/eris25: {y_fedavg[0] / y_eris25[0]}")
print(f"fedavg/eris50: {y_fedavg[0] / y_eris50[0]}")
print(f"soteria/eris2: {y_soteria[0] / y_eris2[0]}")
print(f"soteria/eris25: {y_soteria[0] / y_eris25[0]}")
print(f"soteria/eris50: {y_soteria[0] / y_eris50[0]}")
ax2.plot(x, y_fedavg, label='FedAvg', color=baseline_colors['FedAvg'], marker='o', markersize=4, linewidth=1)
ax2.plot(x, y_eris2, label='ERIS (A=2)', color=baseline_colors['ERIS'], marker='^', markersize=4, linewidth=1)
ax2.plot(x, y_eris25, label='ERIS (A=25)', color=baseline_colors['ERIS'], marker='x', markersize=4, linewidth=1)
ax2.plot(x, y_eris50, label='ERIS (A=50)', color=baseline_colors['ERIS'], marker='s', markersize=4, linewidth=1)
ax2.plot(x, y_priprune, label='PriPrune', color=baseline_colors['Pruning'], marker='o', markersize=4, linewidth=1)
ax2.plot(x, y_soteria, label='SoteriaFL', color=baseline_colors['SoteriaFL'], marker='o', markersize=4, linewidth=1)
ax2.plot(x, y_ako, label='Ako', color=baseline_colors['Ako'], marker='o', markersize=4, linewidth=1)
ax2.plot(x, y_shatter, label='Shatter', color=baseline_colors['Shatter'], marker='o', markersize=4, linewidth=1)

ax2.set_yscale('log')
ax2.set_xlabel('Parameters', fontsize=14)
ax2.set_ylabel('Minimum Distribution Time (seconds)', fontsize=14)
ax2.set_title('Effect of Model Size', fontsize=16)
ax2.set_xlim(1, 1000)
ax2.set_xticks(np.arange(200, 1001, 200))



param = 10000000
model_size = param*8*4
priprune_size = param*(1-0.3)*8*4
compr_size = param*compression_rate*8*4

x = np.linspace(10, 300, 50)

y_fedavg = 2 * (x * model_size) / rate
y_eris2 = 2 * (compr_size * (x - 1)) / (2 * rate)
y_eris25 = 2 * (compr_size * (x - 1)) / (25 * rate)
y_eris50 = 2 * (compr_size * (x - 1)) / (50 * rate)
y_priprune = 2 * (x * priprune_size) / rate
y_soteria = 2 * (x * compr_size) / rate
y_ako = (model_size / rate) * np.ones(x.shape)
y_shatter = np.maximum((4*model_size)/(rate*x), (4*model_size)/rate)
print(f"fedavg/eris2: {y_fedavg[0] / y_eris2[0]}")
print(f"fedavg/eris25: {y_fedavg[0] / y_eris25[0]}")
print(f"fedavg/eris50: {y_fedavg[0] / y_eris50[0]}")
print(f"soteria/eris2: {y_soteria[0] / y_eris2[0]}")
print(f"soteria/eris25: {y_soteria[0] / y_eris25[0]}")
print(f"soteria/eris50: {y_soteria[0] / y_eris50[0]}")

ax1.plot(x, y_fedavg, label='FedAvg', color=baseline_colors['FedAvg'], marker='o', markersize=4, linewidth=1)
ax1.plot(x, y_eris2, label='ERIS (A=2)', color=baseline_colors['ERIS'], marker='^', markersize=4, linewidth=1)
ax1.plot(x, y_eris25, label='ERIS (A=25)', color=baseline_colors['ERIS'], marker='x', markersize=4, linewidth=1)
ax1.plot(x, y_eris50, label='ERIS (A=50)', color=baseline_colors['ERIS'], marker='s', markersize=4, linewidth=1)
ax1.plot(x, y_priprune, label='PriPrune', color=baseline_colors['Pruning'], marker='o', markersize=4, linewidth=1)
ax1.plot(x, y_soteria, label='SoteriaFL', color=baseline_colors['SoteriaFL'], marker='o', markersize=4, linewidth=1)
ax1.plot(x, y_ako, label='Ako', color=baseline_colors['Ako'], marker='o', markersize=4, linewidth=1)
ax1.plot(x, y_shatter, label='Shatter', color=baseline_colors['Shatter'], marker='o', markersize=4, linewidth=1)

ax1.set_yscale('log')
ax1.set_xlabel('Clients', fontsize=14)
ax1.set_ylabel('Minimum Distribution Time (seconds)', fontsize=14)
ax1.set_title('Effect of Number of Clients', fontsize=16)
ax1.set_xlim(10, 300)
ax1.set_xticks(np.arange(50, 301, 50))



handles, labels = ax1.get_legend_handles_labels()


fig.legend(
    handles=handles,
    labels=labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.05),
    title=r"$\mathbf{Methods}$",
    ncol=4,
    fontsize=11,
    title_fontsize=11,
    labelspacing=0.85)


plt.tight_layout()
plt.savefig('scalability.pdf', bbox_inches='tight')
