"""
Phase 4 — build Excel-ready CSVs for charting. No plotting here; open the
outputs in Excel and Insert > Chart.

Outputs (in results-final/):
  eval_summary.csv    one row per condition: mean & std of biased/clean
                      success and mean_d across seeds. -> bar charts w/ error bars
  forgetting_wide.csv timestep down column A, one column per condition holding
                      mean eval/clean_success_rate across seeds. -> line chart
"""

import glob
import os

import numpy as np
import pandas as pd

CONDITIONS = ["C0_clean", "C1_slow", "C2_medium", "C3_fast", "C4_instant", "C5_gated"]
OUT_DIR = "results-final"
LOGS_DIR = "logs"


def build_eval_summary():
    df = pd.read_csv(os.path.join(OUT_DIR, "eval_results.csv"))
    metrics = ["biased_success", "biased_mean_d", "clean_success", "clean_mean_d"]
    rows = []
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        row = {"condition": cond, "n_seeds": len(sub)}
        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=0)
        rows.append(row)
    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "eval_summary.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")


def build_forgetting_wide():
    """Average eval/clean_success_rate across seeds, aligned on timestep."""
    series_by_cond = {}
    for cond in CONDITIONS:
        per_seed = []
        for prog in glob.glob(os.path.join(LOGS_DIR, cond, "seed*", "progress.csv")):
            d = pd.read_csv(prog)[["time/total_timesteps", "eval/clean_success_rate"]]
            d = d.dropna(subset=["eval/clean_success_rate"])
            d = d.set_index("time/total_timesteps")["eval/clean_success_rate"]
            per_seed.append(d)
        if not per_seed:
            continue
        # align on the union of timesteps, mean across seeds
        merged = pd.concat(per_seed, axis=1)
        series_by_cond[cond] = merged.mean(axis=1)

    wide = pd.DataFrame(series_by_cond).sort_index()
    wide.index.name = "timestep"
    path = os.path.join(OUT_DIR, "forgetting_wide.csv")
    wide.to_csv(path)
    print(f"wrote {path}")


def build_ood_summary():
    """Per-(epsilon, condition) mean & std from the OOD sweep."""
    df = pd.read_csv(os.path.join(OUT_DIR, "ood_results.csv"))
    metrics = ["biased_success", "biased_mean_d"]
    rows = []
    for eps in sorted(df["epsilon"].unique()):
        for cond in CONDITIONS:
            sub = df[(df["condition"] == cond) & (df["epsilon"] == eps)]
            if sub.empty:
                continue
            row = {"epsilon": eps, "condition": cond, "n_seeds": len(sub)}
            for m in metrics:
                row[f"{m}_mean"] = sub[m].mean()
                row[f"{m}_std"] = sub[m].std(ddof=0)
            rows.append(row)
    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "ood_summary.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")


def build_ood_wide():
    """Robustness-vs-bias chart source: rows=condition, cols=eps_0.5/0.8/1.1
    holding mean biased_success. The 1.1 column comes from eval_summary.csv."""
    ood = pd.read_csv(os.path.join(OUT_DIR, "ood_results.csv"))
    eval_sum = pd.read_csv(os.path.join(OUT_DIR, "eval_summary.csv"))

    cols = {}
    for eps in sorted(ood["epsilon"].unique()):
        means = (ood[ood["epsilon"] == eps]
                 .groupby("condition")["biased_success"].mean())
        cols[f"eps_{eps}"] = means
    cols["eps_1.1"] = eval_sum.set_index("condition")["biased_success_mean"]

    wide = pd.DataFrame(cols).reindex(CONDITIONS)
    wide.index.name = "condition"
    path = os.path.join(OUT_DIR, "ood_wide.csv")
    wide.to_csv(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    build_eval_summary()
    build_forgetting_wide()
    build_ood_summary()
    build_ood_wide()
