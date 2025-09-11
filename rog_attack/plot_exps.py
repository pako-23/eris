#!/usr/bin/env python3
"""
Summarize experiment metrics from experiments/*/fedavg.json

- Methods (folder names) become rows.
- Metrics become columns with mean and std (from std_<metric>).
- Outputs:
  - experiments_summary_wide.csv  -> separate mean/std columns
  - experiments_summary_pm.csv    -> "mean ± std" columns
  - experiments_summary.md        -> optional Markdown (if pandas has to_markdown)

Usage: just run in the repo root where the 'experiments/' folder lives.
"""

from pathlib import Path
import json
import pandas as pd

def load_metrics_dict(d: dict) -> dict:
    """
    Pair keys like 'snr' with 'std_snr'. Return a flat dict with MultiIndex-like keys:
    { (metric, 'mean'): val, (metric, 'std'): val, ... }
    """
    paired = {}
    # First, record stds
    std_map = {k[4:]: v for k, v in d.items() if isinstance(k, str) and k.startswith("std_")}
    for k, v in d.items():
        if not isinstance(k, str) or k.startswith("std_"):
            continue
        metric = k
        mean_val = v
        std_val = std_map.get(metric, None)
        # keep numeric only
        if isinstance(mean_val, (int, float)):
            paired[(metric, "mean")] = float(mean_val)
        if isinstance(std_val, (int, float)):
            paired[(metric, "std")] = float(std_val)
    # also include stray std_* without mean if they exist
    for metric, std_val in std_map.items():
        if (metric, "std") not in paired and isinstance(std_val, (int, float)):
            paired[(metric, "std")] = float(std_val)
    return paired

def summarize_experiments(root="experiments", metrics_file="fedavg.json", decimals=4):
    root = Path(root)
    rows = {}
    for exp_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        mf = exp_dir / metrics_file
        if not mf.exists():
            continue
        try:
            with open(mf, "r") as f:
                data = json.load(f)
            rows[exp_dir.name] = load_metrics_dict(data)
        except Exception as e:
            print(f"[WARN] Skipping {mf}: {e}")

    if not rows:
        raise SystemExit("No experiments found with metrics. Check paths and filenames.")

    # Build DataFrame with MultiIndex-like columns (tuples)
    df = pd.DataFrame.from_dict(rows, orient="index")

    # Sort columns by metric then mean/std order
    if isinstance(df.columns, pd.MultiIndex):
        # unlikely here because we used tuple keys; still keep sorting logic consistent
        pass
    ordered_cols = sorted(df.columns, key=lambda x: (x[0], 0 if x[1] == "mean" else 1))
    df = df.reindex(columns=ordered_cols)

    # Round numeric columns
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(decimals)
    df.index.name = "method"

    # Save wide CSV
    df_wide = df.copy()
    df_wide.columns = pd.MultiIndex.from_tuples(df_wide.columns, names=["metric", "stat"])
    df_wide.to_csv("experiments_summary_wide.csv")

    # Build "mean ± std" compact table
    compact = pd.DataFrame(index=df.index)
    metrics = sorted(set(m for (m, _) in df_wide.columns))
    for m in metrics:
        mean_col = (m, "mean")
        std_col = (m, "std")
        mean_series = df_wide[mean_col] if mean_col in df_wide.columns else None
        std_series = df_wide[std_col] if std_col in df_wide.columns else None

        if mean_series is not None and std_series is not None:
            compact[m] = mean_series.map(lambda x: f"{x:g}") + " ± " + std_series.map(lambda x: f"{x:g}")
        elif mean_series is not None:
            compact[m] = mean_series.map(lambda x: f"{x:g}")
        elif std_series is not None:
            compact[m] = std_series.map(lambda x: f"± {x:g}")
        else:
            compact[m] = ""

    compact.index.name = "method"
    compact.to_csv("experiments_summary_pm.csv")

    # Optional Markdown
    try:
        md = compact.reset_index().to_markdown(index=False)
        with open("experiments_summary.md", "w") as f:
            f.write(md)
    except Exception:
        pass

    # Preview to console
    print("\n=== Wide table (first 20 rows) ===")
    print(df_wide.head(20))
    print("\nSaved:")
    print(" - experiments_summary_wide.csv")
    print(" - experiments_summary_pm.csv")
    print(" - experiments_summary.md (if supported)")

if __name__ == "__main__":
    summarize_experiments()
