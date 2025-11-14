import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def setup_icml_plot(two_column=False):
    """Set up ICML-compatible plot settings."""
    if two_column:
        figure_width = 7  # Full-page width for two-column layout (in inches)
    else:
        figure_width = 3.5  # Half-page width for two-column layout (in inches)

    rcParams.update(
        {
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
            "figure.figsize": (
                figure_width,
                figure_width * 0.85,
            ),  # TODO change to better ratio
            "figure.dpi": 300,  # High resolution for publication
            # Grid
            "axes.grid": True,  # Enable grid
            "grid.alpha": 0.3,  # Grid transparency
            "grid.linestyle": "--",  # Dashed grid lines
            # Legend
            "legend.frameon": False,  # No border around legends
        }
    )

    return (figure_width, figure_width * 0.85)


# (Optional) color map to stay consistent with your previous style:
baseline_colors = {
    "FedAvg": "tab:orange",
    "ERIS": "tab:blue",
    "Pruning": "tab:purple",
    "SoteriaFL": "tab:red",
    "Ako": "tab:green",
    "Shatter": "tab:olive",
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
y_shatter = np.maximum(
    (4 * x * multiplier * 8 * 4) / (rate * clients), (4 * x * multiplier * 8 * 4) / rate
)
y_eris50_uncomp = 2 * (x * multiplier * 8 * 4 * (clients - 1)) / (50 * rate)
print(f"fedavg/eris2: {y_fedavg[0] / y_eris2[0]}")
print(f"fedavg/eris25: {y_fedavg[0] / y_eris25[0]}")
print(f"fedavg/eris50: {y_fedavg[0] / y_eris50[0]}")
print(f"soteria/eris2: {y_soteria[0] / y_eris2[0]}")
print(f"soteria/eris25: {y_soteria[0] / y_eris25[0]}")
print(f"soteria/eris50: {y_soteria[0] / y_eris50[0]}")
ax2.plot(
    x,
    y_fedavg,
    label="FedAvg",
    color=baseline_colors["FedAvg"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_eris2,
    label="ERIS (A=2)",
    color=baseline_colors["ERIS"],
    marker="^",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_eris25,
    label="ERIS (A=25)",
    color=baseline_colors["ERIS"],
    marker="x",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_eris50,
    label="ERIS (A=50)",
    color=baseline_colors["ERIS"],
    marker="s",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_eris50_uncomp,
    label="ERIS (A=50) w/o compression",
    color=baseline_colors["ERIS"],
    marker="P",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_priprune,
    label="PriPrune",
    color=baseline_colors["Pruning"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_soteria,
    label="SoteriaFL",
    color=baseline_colors["SoteriaFL"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_ako,
    label="Ako",
    color=baseline_colors["Ako"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax2.plot(
    x,
    y_shatter,
    label="Shatter",
    color=baseline_colors["Shatter"],
    marker="o",
    markersize=4,
    linewidth=1,
)

ax2.set_yscale("log")
ax2.set_xlabel("Parameters", fontsize=14)
ax2.set_ylabel("Minimum Distribution Time (seconds)", fontsize=14)
ax2.set_title("Effect of Model Size", fontsize=16)
ax2.set_xlim(1, 1000)
ax2.set_xticks(np.arange(200, 1001, 200))


param = 10000000
model_size = param * 8 * 4
priprune_size = param * (1 - 0.3) * 8 * 4
compr_size = param * compression_rate * 8 * 4

x = np.linspace(10, 300, 50)

y_fedavg = 2 * (x * model_size) / rate
y_eris2 = 2 * (compr_size * (x - 1)) / (2 * rate)
y_eris25 = 2 * (compr_size * (x - 1)) / (25 * rate)
y_eris50 = 2 * (compr_size * (x - 1)) / (50 * rate)
y_priprune = 2 * (x * priprune_size) / rate
y_soteria = 2 * (x * compr_size) / rate
y_ako = (model_size / rate) * np.ones(x.shape)
y_shatter = np.maximum((4 * model_size) / (rate * x), (4 * model_size) / rate)
y_eris50_uncomp = 2 * (model_size * (x - 1)) / (50 * rate)
print(f"fedavg/eris2: {y_fedavg[0] / y_eris2[0]}")
print(f"fedavg/eris25: {y_fedavg[0] / y_eris25[0]}")
print(f"fedavg/eris50: {y_fedavg[0] / y_eris50[0]}")
print(f"soteria/eris2: {y_soteria[0] / y_eris2[0]}")
print(f"soteria/eris25: {y_soteria[0] / y_eris25[0]}")
print(f"soteria/eris50: {y_soteria[0] / y_eris50[0]}")

ax1.plot(
    x,
    y_fedavg,
    label="FedAvg",
    color=baseline_colors["FedAvg"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_eris2,
    label="ERIS (A=2)",
    color=baseline_colors["ERIS"],
    marker="^",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_eris25,
    label="ERIS (A=25)",
    color=baseline_colors["ERIS"],
    marker="x",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_eris50,
    label="ERIS (A=50)",
    color=baseline_colors["ERIS"],
    marker="s",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_eris50_uncomp,
    label="ERIS (A=50) w/o compression",
    color=baseline_colors["ERIS"],
    marker="P",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_priprune,
    label="PriPrune",
    color=baseline_colors["Pruning"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_soteria,
    label="SoteriaFL",
    color=baseline_colors["SoteriaFL"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_ako,
    label="Ako",
    color=baseline_colors["Ako"],
    marker="o",
    markersize=4,
    linewidth=1,
)
ax1.plot(
    x,
    y_shatter,
    label="Shatter",
    color=baseline_colors["Shatter"],
    marker="o",
    markersize=4,
    linewidth=1,
)

ax1.set_yscale("log")
ax1.set_xlabel("Clients", fontsize=14)
ax1.set_ylabel("Minimum Distribution Time (seconds)", fontsize=14)
ax1.set_title("Effect of Number of Clients", fontsize=16)
ax1.set_xlim(10, 300)
ax1.set_xticks(np.arange(50, 301, 50))


handles, labels = ax1.get_legend_handles_labels()


fig.legend(
    handles=handles,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),
    title=r"$\mathbf{Methods}$",
    ncol=3,
    fontsize=11,
    title_fontsize=11,
    labelspacing=0.85,
)


plt.tight_layout()
plt.savefig("scalability.pdf", bbox_inches="tight")


def format_num(number):
    if number == 0:
        return f"{round(number, 2)}"

    i = 2
    formatted = f"{round(number, i)}"
    while len(list(filter(lambda x: x == "0", list(formatted)))) == len(formatted) - 1:
        i += 1
        formatted = f"{round(number, i)}"

    return formatted


def format_bytes(size):
    units = ["B", "KB", "MB", "GB"]
    multiplier = 1000
    size = float(size)
    i = 0

    while size >= multiplier and i < len(units) - 1:
        size /= multiplier
        i += 1

    return f"{round(size, 2)} {units[i]}"


def communication_table(**argv):
    dataset = argv["dataset"]
    clients = argv["clients"]
    params = argv["params"]
    shatter = argv["shatter"]
    compressions = argv["compressions"]
    table = []
    rate = 20000000
    header = [
        "Method",
        "Upload per-client",
        "Download per-client",
        "Compression Rate",
        "Dist. Time",
    ]

    model_size = params * 4
    compressed = model_size * compressions[0] / 100
    table.append(
        [
            "FedAvg",
            format_bytes(model_size),
            format_bytes(model_size),
            f"{model_size/compressed}",
            f"{2*((clients*model_size)/rate)} s",
        ]
    )

    compressed = model_size * compressions[1] / 100
    table.append(
        [
            "Shatter",
            format_bytes(model_size + 2 * model_size * shatter),
            format_bytes(model_size + 2 * model_size * shatter),
            format_num(model_size / compressed),
            f"{format_num(max(model_size/rate, (shatter*model_size)/rate, (shatter*model_size)/(clients*rate)))} s",
        ]
    )

    compressed = model_size * compressions[2] / 100
    table.append(
        [
            "PriPrune (0.01)",
            format_bytes(compressed),
            format_bytes(compressed),
            format_num(model_size / compressed),
            f"{format_num(2*((clients*compressed)/rate))} s",
        ]
    )

    compressed = model_size * compressions[3] / 100
    table.append(
        [
            "PriPrune (0.05)",
            format_bytes(compressed),
            format_bytes(compressed),
            format_num(model_size / compressed),
            f"{format_num(2*((clients*compressed)/rate))} s",
        ]
    )

    compressed = model_size * compressions[4] / 100
    table.append(
        [
            "PriPrune (0.1)",
            format_bytes(compressed),
            format_bytes(compressed),
            format_num(model_size / compressed),
            f"{format_num(2*((clients*compressed)/rate))} s",
        ]
    )

    compressed = model_size * compressions[5] / 100
    table.append(
        [
            "SoteriaFL",
            format_bytes(compressed),
            format_bytes(compressed),
            format_num(model_size / compressed),
            f"{format_num(2*((clients*compressed)/rate))} s",
        ]
    )

    compressed = model_size * compressions[6] / 100
    distribution = 2 * max(
        ((clients - 1) * compressed) / (clients * rate), compressed / rate
    )
    table.append(
        [
            "ERIS",
            format_bytes((clients - 1) * 2 * (compressed / clients)),
            format_bytes((clients - 1) * 2 * (compressed / clients)),
            format_num(model_size / compressed),
            f"{format_num(distribution)} s",
        ]
    )

    cellsizes = [len(i) for i in header]
    for row in table:
        for i in range(len(row)):
            cellsizes[i] = max(cellsizes[i], len(row[i]))

    print()
    header = "|".join(
        [
            ("{:^" + str(cellsizes[i] + 2) + "}").format(header[i])
            for i in range(len(header))
        ]
    )
    print(("{:^" + str(len(header)) + "}").format(dataset))
    print(("{:^" + str(len(header)) + "}").format("-" * (len(dataset) + 2)))
    print(header)
    print("+".join(["-" * (i + 2) for i in cellsizes]))
    for row in table:
        print(
            "|".join(
                [
                    ("{:^" + str(cellsizes[i] + 2) + "}").format(row[i])
                    for i in range(len(row))
                ]
            )
        )
    print()


communication_table(
    dataset="MNIST",
    clients=50,
    params=62000,
    shatter=4,
    compressions=[100, 100, 99, 95, 90, 5, 3.3],
)
communication_table(
    dataset="CIFAR10",
    clients=50,
    params=1650000,
    shatter=4,
    compressions=[100, 100, 99, 95, 90, 5, 0.6],
)
communication_table(
    dataset="IMDB",
    clients=25,
    params=67000000,
    shatter=4,
    compressions=[100, 100, 90, 80, 70, 5, 0.012],
)
communication_table(
    dataset="CNN/Daily Mail",
    clients=10,
    params=1300000000,
    shatter=3,
    compressions=[100, 100, 90, 80, 70, 5, 1],
)
