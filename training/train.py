# train.py
#
# Trains a single SAC+HER agent on FetchReach-v4 wrapped with SystematicBiasWrapper
# in curriculum mode. Supports batching multiple seeds in one call.
#
# Single run (seed 0):
#   python -m training.train --log-dir logs/curriculum --seed 0
#
# Batch run (seeds 0-4, sequential):
#   python -m training.train --log-dir logs/curriculum --seed 0 --runs 5
#
# Each run writes to:
#   logs/<log-dir>/seed{N}/        — TensorBoard events, progress.csv, config.json
#   results/seed{N}.zip            — saved model
#
# View all runs in TensorBoard:
#   tensorboard --logdir logs/curriculum

import argparse
import json
import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.logger import configure
from envs.systematic_bias_wrapper import SystematicBiasWrapper
from training.curriculum_callback import CurriculumStepCallback

# --- CLI args ---
# --log-dir : base log directory; each run gets its own subdir named seed{N}
# --seed    : starting seed; batch runs use seed, seed+1, seed+2, ...
# --save-dir: output directory for model .zips, one file per run named seed{N}.zip
# --runs    : how many sequential runs to launch (default 1)
parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", default="logs/curriculum_test", help="Base directory for TensorBoard/CSV logs")
parser.add_argument("--seed", type=int, default=0, help="Starting random seed (batch runs increment from here)")
parser.add_argument("--save-dir", default="results", help="Directory to save trained models")
parser.add_argument("--runs", type=int, default=1, help="Number of sequential runs with incrementing seeds")
args = parser.parse_args()

# --- Curriculum schedule hyperparameters ---
# These are shared across all runs in a batch so conditions are identical.
#
# epsilon_max : maximum actuation bias magnitude added to each action dimension
# alpha       : ramp rate; epsilon(t) = min(epsilon_max, alpha * t)
#               with alpha = epsilon_max / 15000, the bias reaches its peak at
#               step 15000 — the halfway point of the 30k-step budget
# total_timesteps : training budget; keep fixed across all experimental conditions
EPSILON_MAX = 0.2
ALPHA = 0.2 / 15000  # ramps to epsilon_max at step 15000 (50% of training)
TOTAL_TIMESTEPS = 30_000

# Register gymnasium-robotics envs (FetchReach, FetchPush, etc.) once before any
# gym.make calls — safe to call even if already registered.
gym.register_envs(gymnasium_robotics)


def train_one(seed: int, log_dir: str, save_path: str) -> None:
    """Run one full training session and save the model.

    Args:
        seed: random seed passed to SAC (covers policy init, env, and replay sampling)
        log_dir: directory for TensorBoard events, progress.csv, and config.json
        save_path: path (without .zip) where the final model is saved
    """
    # Write config.json first so the run is self-documenting even if training crashes.
    os.makedirs(log_dir, exist_ok=True)
    config = {
        "env": "FetchReach-v4",
        "algorithm": "SAC+HER",
        "mode": "curriculum",
        "epsilon_max": EPSILON_MAX,
        "alpha": ALPHA,
        "total_timesteps": TOTAL_TIMESTEPS,
        "seed": seed,
        "save_path": save_path,
        "n_sampled_goal": 4,
        "goal_selection_strategy": "future",
    }
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Environment ---
    # SystematicBiasWrapper adds a per-episode constant bias to every action.
    # In curriculum mode the bias magnitude grows linearly with global step count;
    # CurriculumStepCallback (below) is responsible for ticking that counter.
    base = gym.make("FetchReach-v4")
    env = SystematicBiasWrapper(base, epsilon_max=EPSILON_MAX, alpha=ALPHA, mode="curriculum")

    # --- Model ---
    # MultiInputPolicy is required for dict observation spaces (FetchReach returns
    # separate "observation", "achieved_goal", and "desired_goal" keys).
    # HER replays failed episodes with substituted goals — critical for sparse rewards.
    # n_sampled_goal=4 and strategy="future" are the SB3 Zoo defaults for Fetch tasks.
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
        seed=seed,
        verbose=1,
    )
    # Configure logger to emit both human-readable stdout and machine-readable CSV.
    model.set_logger(configure(log_dir, ["stdout", "csv"]))

    # --- Training ---
    # CurriculumStepCallback increments the wrapper's internal step counter after
    # each env step so epsilon(t) tracks training progress, not wall-clock time.
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CurriculumStepCallback())

    model.save(save_path)
    env.close()


# --- Batch loop ---
# Runs are sequential (not parallel) to avoid GPU/CPU contention.
# Seeds are contiguous starting from --seed so results are reproducible and comparable.
for i in range(args.runs):
    seed = args.seed + i
    log_dir = os.path.join(args.log_dir, f"seed{seed}")
    save_path = os.path.join(args.save_dir, f"seed{seed}")
    print(f"\n=== Run {i + 1}/{args.runs} | seed={seed} | log={log_dir} ===\n")
    train_one(seed, log_dir, save_path)
