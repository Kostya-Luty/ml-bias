"""
Phase 4 — deterministic evaluation of every trained model.

Sweeps results/<condition>/seed{N}.zip and evaluates each model under:
  - biased env: mode="fixed", _fixed_bias resampled per episode (vector,
    magnitude eps=1.1) — the same protocol that produced the Phase 3.1 baseline
  - clean env:  no bias (eps=0) — forgetting cross-check

Metrics per model: success rate (is_success) and mean_d (final
||achieved_goal - desired_goal||) over N=100 deterministic episodes.

Writes results-final/eval_results.csv.
"""

import csv
import os

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

from envs.systematic_bias_wrapper import SystematicBiasWrapper

gym.register_envs(gymnasium_robotics)

EPS = 1.1
N_EPISODES = 100
CONDITIONS = ["C0_clean", "C1_slow", "C2_medium", "C3_fast", "C4_instant", "C5_gated"]
SEEDS = [0, 1, 2]
RESULTS_DIR = "results"
OUT_DIR = "results-final"


def evaluate_model(model_path, biased, seed, n_episodes=N_EPISODES, eps=EPS):
    """Return (success_rate, mean_d) over n_episodes, deterministic.

    biased=True: per-episode resampled vector bias of magnitude eps.
    biased=False: clean env (no bias).
    """
    base = gym.make("FetchReach-v4")
    env = SystematicBiasWrapper(base, epsilon_max=eps, alpha=0.0, mode="fixed")
    model = SAC.load(model_path, env=env)

    rng = np.random.default_rng(seed)
    successes = []
    dists = []

    for ep in range(n_episodes):
        if biased:
            env._fixed_bias = rng.uniform(-eps, eps, size=env.action_space.shape)
        else:
            env._fixed_bias = np.zeros(env.action_space.shape)

        obs, _ = env.reset(seed=seed + ep)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes.append(float(info["is_success"]))
                # desired/achieved goal are in obs (Dict), reliable source
                d = float(np.linalg.norm(
                    np.asarray(obs["achieved_goal"]) - np.asarray(obs["desired_goal"])
                ))
                dists.append(d)
                break

    env.close()
    return sum(successes) / len(successes), sum(dists) / len(dists)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "eval_results.csv")
    rows = []

    for cond in CONDITIONS:
        for seed in SEEDS:
            model_path = os.path.join(RESULTS_DIR, cond, f"seed{seed}.zip")
            if not os.path.exists(model_path):
                print(f"  MISSING {model_path} — skipping")
                continue

            b_succ, b_d = evaluate_model(model_path, biased=True, seed=seed)
            c_succ, c_d = evaluate_model(model_path, biased=False, seed=seed)
            print(f"{cond} seed{seed}: "
                  f"biased succ={b_succ:.3f} mean_d={b_d:.4f} | "
                  f"clean succ={c_succ:.3f} mean_d={c_d:.4f}")
            rows.append({
                "condition": cond, "seed": seed,
                "biased_success": b_succ, "biased_mean_d": b_d,
                "clean_success": c_succ, "clean_mean_d": c_d,
            })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "condition", "seed",
            "biased_success", "biased_mean_d",
            "clean_success", "clean_mean_d",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")

    # quick aggregate summary
    print("\nCondition summary (mean +/- std across seeds):")
    print(f"{'condition':<12} {'biased_succ':<18} {'clean_succ':<18}")
    for cond in CONDITIONS:
        cr = [r for r in rows if r["condition"] == cond]
        if not cr:
            continue
        bs = np.array([r["biased_success"] for r in cr])
        cs = np.array([r["clean_success"] for r in cr])
        print(f"{cond:<12} {bs.mean():.3f} +/- {bs.std():<10.3f} "
              f"{cs.mean():.3f} +/- {cs.std():<10.3f}")


if __name__ == "__main__":
    main()
