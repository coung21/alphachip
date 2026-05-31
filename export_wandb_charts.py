#!/usr/bin/env python3
"""Export and save metric charts from Weights & Biases runs.

Usage examples:
  python export_wandb_charts.py --entity myuser --project myproj --metrics loss reward --outdir wandb_charts

The script fetches runs from the specified `entity/project`, downloads history for requested metrics
and saves PNG charts (one per metric per run) into the output directory.
"""
import argparse
import os
import math
from pathlib import Path
import wandb
import matplotlib.pyplot as plt


def plot_metric(df, metric, outpath, run_name):
    if metric not in df.columns:
        return False
    x = list(range(len(df[metric].values)))
    y = df[metric].values
    if len(y) == 0:
        return False
    plt.figure(figsize=(8,4))
    plt.plot(x, y, label=f"{metric}")
    plt.title(f"{metric} — {run_name}")
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Export WandB charts for runs")
    parser.add_argument("--entity", required=True, help="WandB entity/user name")
    parser.add_argument("--project", required=True, help="WandB project name")
    parser.add_argument("--metrics", nargs="+", help="Metrics to export (space separated)")
    parser.add_argument("--outdir", default="wandb_charts", help="Output directory for charts")
    parser.add_argument("--limit", type=int, default=10, help="Max runs to process (default 10)")
    parser.add_argument("--filter", help="Optional W&B run filter (e.g. 'config.seed=42')")
    args = parser.parse_args()

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    print(f"Fetching runs for {project_path} (limit={args.limit})...")
    runs = api.runs(project_path, filters=args.filter) if args.filter else api.runs(project_path)

    outdir = Path(args.outdir)
    processed = 0
    for run in runs:
        if processed >= args.limit:
            break
        print(f"Processing run: {run.id}  name={run.name}")
        try:
            # Download history as pandas DataFrame (requires pandas)
            df = run.history(pandas=True)
        except Exception:
            # fallback: try without pandas (history -> list of dicts)
            hist = list(run.scan_history())
            if len(hist) == 0:
                print(f"  No history for run {run.id}")
                processed += 1
                continue
            import pandas as _pd
            df = _pd.DataFrame(hist)

        # determine metrics to save
        metrics = args.metrics
        if not metrics:
            # pick top numeric columns excluding internal keys
            metrics = [c for c in df.columns if not str(c).startswith("_")][:6]

        run_dir = outdir / f"{run.id}_{run.name or 'noname'}"
        for metric in metrics:
            safe_metric = str(metric).replace("/", "_")
            outpath = run_dir / f"{safe_metric}.png"
            ok = plot_metric(df, metric, outpath, run.name or run.id)
            if ok:
                print(f"  Saved {outpath}")
            else:
                print(f"  Metric '{metric}' not found or empty in run {run.id}")

        processed += 1

    print("Done.")


if __name__ == "__main__":
    main()
