# train.py
#
# Trains a single SAC+HER agent on FetchReach-v4 wrapped with SystematicBiasWrapper.
# The experimental condition (mode, epsilon_max, alpha, gate) is fully described by a
# JSON config in configs/, so all conditions share identical code and differ only in
# their config file. Supports batching multiple seeds in one call.
#
# Single run (seed 0):
#   python -m training.train --config configs/c2_medium.json --seed 0
#
# Batch run (seeds 0-2, sequential):
#   python -m training.train --config configs/c2_medium.json --seed 0 --runs 3
#
# Each run writes to:
#   logs/<condition>/seed{N}/   — TensorBoard events, progress.csv, config.json
#   results/<condition>/seed{N}.zip   — saved model
#
# View all runs in TensorBoard:
#   tensorboard --logdir logs/

import argparse
import json
import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList
from envs.systematic_bias_wrapper import SystematicBiasWrapper
from training.curriculum_callback import CurriculumStepCallback
from training.forgetting_probe import ForgettingProbeCallback

# --- CLI args ---
# --config  : path to the condition's JSON config (defines mode/epsilon_max/alpha/gate)
# --seed    : starting seed; batch runs use seed, seed+1, seed+2, ...
# --runs    : how many sequential runs to launch (default 3 — RL is high-variance)
# --log-dir : base log dir; defaults to logs/<condition>. Each run gets a seed{N} subdir.
# --save-dir: base model dir; defaults to results/<condition>.
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to condition JSON config")
parser.add_argument("--seed", type=int, default=0, help="Starting seed (batch runs increment)")
parser.add_argument("--runs", type=int, default=3, help="Number of sequential runs with incrementing seeds")
parser.add_argument("--log-dir", default=None, help="Base log dir (default: logs/<condition>)")
parser.add_argument("--save-dir", default=None, help="Base model dir (default: results/<condition>)")
parser.add_argument("--eval-freq", type=int, default=5000, help="Forgetting-probe cadence (steps)")
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

condition = cfg["condition"]
base_log_dir = args.log_dir or os.path.join("logs", condition)
base_save_dir = args.save_dir or os.path.join("results", condition)

# Register gymnasium-robotics envs once before any gym.make call — idempotent.
gym.register_envs(gymnasium_robotics)


def train_one(seed: int, log_dir: str, save_path: str) -> None:
    """Run one full training session and save the model.

    Args:
        seed: random seed passed to SAC (covers policy init, env, replay sampling)
        log_dir: directory for TensorBoard events, progress.csv, config.json
        save_path: path (without .zip) where the final model is saved
    """
    # Write config.json (with this run's seed) first, so the run is self-documenting
    # even if training crashes partway.
    os.makedirs(log_dir, exist_ok=True)
    run_cfg = dict(cfg, seed=seed, save_path=save_path, eval_freq=args.eval_freq)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)

    # --- Environment ---
    # SystematicBiasWrapper adds a per-episode constant bias to every action. The
    # mode/epsilon_max/alpha/gate come straight from the condition config, so every
    # condition runs identical code.
    base = gym.make(cfg["env"])
    env = SystematicBiasWrapper(
        base,
        epsilon_max=cfg["epsilon_max"],
        alpha=cfg["alpha"],
        mode=cfg["mode"],
        gate_success_rate=cfg.get("gate_success_rate"),
        gate_patience=cfg.get("gate_patience", 0),
    )

    # --- Model ---
    # MultiInputPolicy is required for dict observation spaces (FetchReach returns
    # separate "observation", "achieved_goal", and "desired_goal" keys).
    # HER replays failed episodes with substituted goals — critical for sparse rewards.
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=cfg["n_sampled_goal"],
            goal_selection_strategy=cfg["goal_selection_strategy"],
        ),
        seed=seed,
        verbose=1,
    )
    # Emit both human-readable stdout and machine-readable CSV (progress.csv).
    model.set_logger(configure(log_dir, ["stdout", "csv"]))

    # --- Callbacks ---
    # CurriculumStepCallback ticks the wrapper's step counter so eps(t) tracks
    # training progress. ForgettingProbeCallback periodically evaluates on a clean
    # env to catch catastrophic forgetting. Both log to progress.csv.
    callbacks = CallbackList([
        CurriculumStepCallback(),
        ForgettingProbeCallback(
            env_id=cfg["env"],
            eval_freq=args.eval_freq,
            n_eval_episodes=20,
        ),
    ])

    model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    env.close()


# --- Batch loop ---
# Runs are sequential (not parallel) to keep CPU headroom for other conditions.
# Seeds are contiguous from --seed so results are reproducible and comparable.
for i in range(args.runs):
    seed = args.seed + i
    log_dir = os.path.join(base_log_dir, f"seed{seed}")
    save_path = os.path.join(base_save_dir, f"seed{seed}")
    print(f"\n=== {condition} | run {i + 1}/{args.runs} | seed={seed} | log={log_dir} ===\n")
    train_one(seed, log_dir, save_path)
