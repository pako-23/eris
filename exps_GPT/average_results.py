#!/usr/bin/env python3
import os, glob, argparse, math
import numpy as np
import pandas as pd


def _read_xlsx(path: str):
    """Read all sheets; return dict[str, DataFrame]. Missing sheets -> empty DF."""
    if not os.path.isfile(path):
        return {}
    try:
        return pd.read_excel(path, sheet_name=None)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {}


def _concat_with_fold(dfs, fold_idx, expected_cols=None):
    """Attach a 'fold' column to each DF and align columns if expected_cols given."""
    out = {}
    for name, df in dfs.items():
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["fold"] = fold_idx
        if expected_cols is not None:
            # align numeric columns union-wise later; here just keep all
            missing = [c for c in expected_cols.get(name, []) if c not in df.columns]
            for m in missing:
                df[m] = np.nan
            # Also ensure we keep any extra columns dynamically
        out[name] = df
    return out


def _agg_summary(dfs_summary: list) -> pd.DataFrame:
    """Stack 1-row summaries (one per fold) and compute mean/std per numeric metric."""
    if not dfs_summary:
        return pd.DataFrame()
    df = pd.concat(dfs_summary, ignore_index=True, sort=False)

    # Coerce booleans to ints to compute averages (interpreted as proportions)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)

    num = df.select_dtypes(include=[np.number])
    means = num.mean(axis=0, skipna=True)
    stds  = num.std(axis=0, ddof=1, skipna=True)

    out = pd.DataFrame({
        "metric": means.index,
        "mean": means.values,
        "std": stds.values,
        "n_folds": [num.shape[0]] * len(means),
    })
    return out


def _agg_rounds(dfs_rounds: list) -> pd.DataFrame:
    """Aggregate per-round metrics (weighted_val_loss, mean_mia_acc)."""
    if not dfs_rounds:
        return pd.DataFrame()
    df = pd.concat(dfs_rounds, ignore_index=True, sort=False)
    # keep only numeric columns + 'round'
    keep = ["round"]
    num_cols = [c for c in df.columns if c != "round" and pd.api.types.is_numeric_dtype(df[c])]
    keep += num_cols
    df = df[keep]
    # Mean/std across folds (ignoring NaNs)
    grouped = df.groupby("round", dropna=False).agg(["mean", "std", "count"])
    # flatten columns
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index().sort_values("round")


def _agg_mia_by_round_client(dfs_mia: list) -> pd.DataFrame:
    """Aggregate MIA accuracy by (round, client) across folds."""
    if not dfs_mia:
        return pd.DataFrame()
    df = pd.concat(dfs_mia, ignore_index=True, sort=False)
    if not {"round", "client"}.issubset(df.columns):
        return pd.DataFrame()
    # numeric-only aggregation (accuracy, thr_in/out may be present)
    numeric_cols = [c for c in df.columns if c not in {"round", "client", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame()
    grouped = df.groupby(["round", "client"], dropna=False)[numeric_cols].agg(["mean", "std", "count"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index().sort_values(["round", "client"])


def _agg_mia_by_round(dfs_mia: list) -> pd.DataFrame:
    """Aggregate MIA accuracy by round (across clients and folds)."""
    if not dfs_mia:
        return pd.DataFrame()
    df = pd.concat(dfs_mia, ignore_index=True, sort=False)
    if "round" not in df.columns:
        return pd.DataFrame()
    numeric_cols = [c for c in df.columns if c not in {"round", "client", "fold"} and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return pd.DataFrame()
    grouped = df.groupby("round", dropna=False)[numeric_cols].agg(["mean", "std", "count"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index().sort_values("round")


def _agg_rouge(dfs_rouge: list) -> pd.DataFrame:
    """Aggregate single-row rouge sheets across folds."""
    if not dfs_rouge:
        return pd.DataFrame()
    df = pd.concat(dfs_rouge, ignore_index=True, sort=False)
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    means = num.mean(axis=0, skipna=True)
    stds  = num.std(axis=0, ddof=1, skipna=True)
    out = pd.DataFrame({
        "metric": means.index,
        "mean": means.values,
        "std": stds.values,
        "n_folds": [num.shape[0]] * len(means),
    })
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate k-fold XLSX results from local_fl_training.py")
    p.add_argument("--results_dir", type=str, required=True, help="Directory containing results_summary_F{fold}.xlsx files")
    p.add_argument("--n_folds", type=int, required=True)
    p.add_argument("--pattern", type=str, default="results_summary_F{fold}.xlsx",
                   help="Filename pattern; {fold} will be replaced by fold index")
    p.add_argument("--method", type=str, default="FedAvg", choices=["FedAvg", "ERIS", "SoteriaFL", "PriPrune", "FedAvg+DP", "Shatter"],)
    p.add_argument("--n_samples", type=int, default=None, help="(optional) total number of samples (for info only)")
    args = p.parse_args()

    # Collect files
    fold_paths = []
    for f in range(args.n_folds):
        fname = args.pattern.format(fold=f)
        path = os.path.join(args.results_dir, fname)
        if os.path.isfile(path):
            fold_paths.append((f, path))
        else:
            print(f"[WARN] Missing fold file: {path}")

    if not fold_paths:
        raise SystemExit("[ERROR] No XLSX files found.")

    dfs_summary, dfs_rounds, dfs_mia, dfs_rouge = [], [], [], []

    for fold_idx, path in fold_paths:
        sheets = _read_xlsx(path)
        if not sheets:
            continue

        if "summary" in sheets and not sheets["summary"].empty:
            df = sheets["summary"].copy()
            df["fold"] = fold_idx
            dfs_summary.append(df)

        if "rounds" in sheets and not sheets["rounds"].empty:
            df = sheets["rounds"].copy()
            df["fold"] = fold_idx
            dfs_rounds.append(df)

        if "mia" in sheets and not sheets["mia"].empty:
            df = sheets["mia"].copy()
            df["fold"] = fold_idx
            dfs_mia.append(df)

        if "rouge" in sheets and not sheets["rouge"].empty:
            df = sheets["rouge"].copy()
            df["fold"] = fold_idx
            dfs_rouge.append(df)

    # Aggregate
    agg_summary = _agg_summary(dfs_summary)
    agg_rounds  = _agg_rounds(dfs_rounds)
    agg_mia_rc  = _agg_mia_by_round_client(dfs_mia)
    agg_mia_r   = _agg_mia_by_round(dfs_mia)
    agg_rouge   = _agg_rouge(dfs_rouge)

    # Write output Excel
    out_xlsx = f"summaries/results_summary_{args.method}_F{len(fold_paths)}_N{args.n_samples}.xlsx"
    out_path = os.path.join(args.results_dir, out_xlsx)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        writer = pd.ExcelWriter(out_path, engine="xlsxwriter")
    except Exception:
        writer = pd.ExcelWriter(out_path)  # fallback

    with writer:
        if not agg_summary.empty:
            agg_summary.to_excel(writer, sheet_name="summary_mean_std", index=False)
        if not agg_rounds.empty:
            agg_rounds.to_excel(writer, sheet_name="rounds_mean_std", index=False)
        if not agg_mia_rc.empty:
            agg_mia_rc.to_excel(writer, sheet_name="mia_mean_std_by_round_client", index=False)
        if not agg_mia_r.empty:
            agg_mia_r.to_excel(writer, sheet_name="mia_mean_std_by_round", index=False)
        if not agg_rouge.empty:
            agg_rouge.to_excel(writer, sheet_name="rouge_mean_std", index=False)

    print(f"[OK] Wrote aggregated stats to {out_path}")
    print(f"  Folds aggregated: {[f for f, _ in fold_paths]}")
    print(f"  Sheets: "
          f"summary={not agg_summary.empty}, "
          f"rounds={not agg_rounds.empty}, "
          f"mia_rc={not agg_mia_rc.empty}, "
          f"mia_r={not agg_mia_r.empty}, "
          f"rouge={not agg_rouge.empty}")
    
    # delete previous xlsx files
    # for _, path in fold_paths:
    #     try:
    #         os.remove(path)
    #         print(f"[OK] Deleted fold file: {path}")
    #     except Exception as e:
    #         print(f"[WARN] Failed to delete fold file {path}: {e}")


if __name__ == "__main__":
    main()
