import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re, shutil

# ------------------------- Style helper --------------------------------- #
def setup_icml_plot(two_column=False):
    """ICML‑inspired plot settings (Times, thin lines, small markers)."""
    if two_column:
        figure_width = 7  # inches
    else:
        figure_width = 3.5

    # Use LaTeX only if it is available to avoid runtime errors
    use_tex = shutil.which("latex") is not None
    rcParams.update({
        # Font & text
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 4,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Lines & markers
        "lines.linewidth": 1.2,
        "lines.markersize": 3,

        # Figure size & dpi
        "figure.figsize": (figure_width, figure_width * 0.85),
        "figure.dpi": 300,

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # Legend
        "legend.frameon": False,
    })
    return (figure_width, figure_width * 0.85)


# ------------------------- Raw data ------------------------------------- #
raw_data = """
	Accuracy	
No compr	mean	std
4	0.27134	0.011895646
8	0.329	0.00398447
16	0.3432	0.009097032
32	0.37124	0.00553122
64	0.38594	0.00504801
128	0.38954	0.003232089
		
	Accuracy	
omega 13	mean	std
4	0.27106	0.012559076
8	0.32238	0.018903587
16	0.34958	0.016579433
32	0.37216	0.013265685
64	0.37728	0.011321731
128	0.38246	0.003504055
		
	Accuracy	
omega 20	mean	std
4	0.2684	0.006793821
8	0.32482	0.01429495
16	0.34076	0.010543358
32	0.37358	0.015949972
64	0.38434	0.014508701
128	0.38412	0.005128509
		
	Accuracy	
omega 27	mean	std
4	0.26908	0.00969503
8	0.31448	0.00621881
16	0.3341	0.008877612
32	0.38292	0.015574903
64	0.37782	0.005395702
128	0.37914	0.010857366
		
	Accuracy	
omega 40	mean	std
4	0.26934	0.014108239
8	0.33558	0.016099863
16	0.35374	0.012844547
32	0.37236	0.011405718
64	0.3846	0.015855977
128	0.38788	0.014034728
		
	Accuracy	
omega 85	mean	std
4	0.266	0.009795305
8	0.3311	0.005792754
16	0.35104	0.00641704
32	0.37726	0.006415793
64	0.37812	0.003862331
128	0.38712	0.007919444
		
	Accuracy	
omega 170	mean	std
4	0.2631	0.011622908
8	0.33278	0.010554127
16	0.34622	0.014213008
32	0.37398	0.013610496
64	0.38164	0.010101208
128	0.38296	0.008779203
		
	Accuracy	
omega 340	mean	std
4	0.25876	0.013417094
8	0.34018	0.015552672
16	0.33872	0.017129086
32	0.37002	0.008803272
64	0.37896	0.007197388
128	0.38078	0.012444983
		
	Accuracy	
omega 700	mean	std
4	0.21592	0.014125495
8	0.28472	0.010698486
16	0.28862	0.014671251
32	0.32966	0.014846629
64	0.32972	0.026698569
128	0.36046	0.01477709
		
	Accuracy	
omega 1050	mean	std
4	0.2127	0.014815532
8	0.24152	0.01011917
16	0.23942	0.010782838
32	0.2562	0.019234552
64	0.2435	0.012693936
128	0.25116	0.014248172
"""

# ------------------------- Parse data ----------------------------------- #
datasets = {}
current_key = None

for line in raw_data.strip().splitlines():
    line = line.strip()
    if not line or line.lower().startswith("accuracy"):
        continue

    # New dataset
    if line.startswith("No compr"):
        current_key = "No compression"
        datasets[current_key] = {"x": [], "mean": [], "std": []}
        continue
    m = re.match(r"omega\s+(\d+)", line)
    if m:
        current_key = rf"$\omega={m.group(1)}$"
        datasets[current_key] = {"x": [], "mean": [], "std": []}
        continue

    cols = line.split()
    if cols and cols[0].isdigit():
        n_samples = int(cols[0])
        mean_val = float(cols[1])
        std_val = float(cols[2]) / np.sqrt(5)  # Adjust std for 5 runs
        datasets[current_key]["x"].append(n_samples)
        datasets[current_key]["mean"].append(mean_val)
        datasets[current_key]["std"].append(std_val)

# ------------------------- Plot ----------------------------------------- #
_ = setup_icml_plot(two_column=True)  # apply style
fig, ax = plt.subplots(figsize=(5, 4))             # figsize taken from rcParams

for label, data in datasets.items():
    ax.errorbar(
        data["x"],
        data["mean"],
        yerr=data["std"],
        fmt="o-",
        label=label,
        capsize=2,
    )

ax.set_xlabel(r"Number of samples", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title(r"Effect of Shifted Compression ($\omega$)", fontsize=14)
ax.set_xticks([4, 8, 16, 32, 64, 128])
ax.set_ylim(0.2, 0.4)
ax.legend(fontsize=8, ncol=2)  # override tiny default legend font
plt.tight_layout()

# Save to PDF
pdf_path = "accuracy_vs_samples.pdf"
plt.savefig(pdf_path, bbox_inches="tight")

# plt.show()
