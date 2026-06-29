"""
Phase 4 (secondary) — OOD bias sweep.

Re-evaluates every trained model at bias magnitudes OTHER than the training
ε_max=1.1, where the task is genuinely solvable, to see if a curriculum
advantage emerges below saturation. Reuses the biased eval protocol from
evaluate_all.evaluate_model (per-episode resampled vector bias, deterministic).

Writes results-final/ood_results.csv (one row per condition/seed/eps).
"""

import csv
import os

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC  # noqa: F401  (ensures SB3 import path is warm)

from evaluation.evaluate_all import evaluate_model, CONDITIONS, SEEDS, RESULTS_DIR, OUT_DIR

gym.register_envs(gymnasium_robotics)

EPS_LIST = [0.5, 0.8]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ood_results.csv")
    rows = []

    for eps in EPS_LIST:
        for cond in CONDITIONS:
            for seed in SEEDS:
                model_path = os.path.join(RESULTS_DIR, cond, f"seed{seed}.zip")
                if not os.path.exists(model_path):
                    print(f"  MISSING {model_path} — skipping")
                    continue
                succ, mean_d = evaluate_model(model_path, biased=True, seed=seed, eps=eps)
                print(f"eps={eps} {cond} seed{seed}: succ={succ:.3f} mean_d={mean_d:.4f}")
                rows.append({
                    "epsilon": eps, "condition": cond, "seed": seed,
                    "biased_success": succ, "biased_mean_d": mean_d,
                })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "epsilon", "condition", "seed", "biased_success", "biased_mean_d",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    print("\nSummary (mean +/- std across seeds):")
    print(f"{'eps':<6}{'condition':<12}{'biased_succ':<18}")
    for eps in EPS_LIST:
        for cond in CONDITIONS:
            cr = [r for r in rows if r["condition"] == cond and r["epsilon"] == eps]
            if not cr:
                continue
            bs = np.array([r["biased_success"] for r in cr])
            print(f"{eps:<6}{cond:<12}{bs.mean():.3f} +/- {bs.std():.3f}")


if __name__ == "__main__":
    main()
