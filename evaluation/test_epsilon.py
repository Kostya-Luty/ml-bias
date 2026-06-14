# test_epsilon.py
#
# Evaluates a trained model under a fixed actuation bias of magnitude epsilon.
# The bias is held constant within each episode (same as the wrapper's eval mode)
# but does not ramp — this is a direct OOD stress test, not a curriculum.
#
# Usage:
#   python -m evaluation.test_epsilon --epsilon 0.1 --seed 0
#   python -m evaluation.test_epsilon --epsilon 0.3 --seed 42 --model results/seed2
#
# Results are appended to evaluation/epsilon.csv so repeated calls accumulate
# a table you can plot directly:
#   epsilon, seed, n_episodes, success_rate

import argparse
import csv
import os
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from envs.systematic_bias_wrapper import SystematicBiasWrapper

CSV_PATH = os.path.join(os.path.dirname(__file__), "epsilon.csv")
CSV_FIELDS = ["epsilon", "seed", "n_episodes", "success_rate", "model", "_fixed_bias"]

parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", type=float, required=True, help="Fixed bias magnitude to apply during evaluation")
parser.add_argument("--seed", type=int, default=0, help="Random seed for episode resets")
parser.add_argument("--model", default="results/test1", help="Path to saved model .zip")
parser.add_argument("--n-episodes", type=int, default=100, help="Number of episodes to evaluate")
args = parser.parse_args()

gym.register_envs(gymnasium_robotics)

# Wrap the env in fixed mode: the same bias vector is applied for every episode.
# We sample it once here (seeded for reproducibility) from Uniform(-eps, +eps)
# per action dimension, then assign it to _fixed_bias before the first reset.
base = gym.make("FetchReach-v4")
env = SystematicBiasWrapper(base, epsilon_max=args.epsilon, alpha=0, mode="fixed")
rng = np.random.default_rng(args.seed)
model = SAC.load(args.model, env=env)

successes = []
obs, _ = env.reset(seed=args.seed)

for _ in range(args.n_episodes):
    env._fixed_bias = rng.uniform(-args.epsilon, args.epsilon, size=env.action_space.shape)
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            successes.append(float(info["is_success"]))
            obs, _ = env.reset()
            break

env.close()

success_rate = sum(successes) / len(successes)
print(f"epsilon={args.epsilon}  seed={args.seed}  success_rate={success_rate:.1%}")

# Append one row to epsilon.csv, writing the header only if the file is new.
write_header = not os.path.exists(CSV_PATH)
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
    writer.writerow({
        "epsilon": args.epsilon,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "success_rate": success_rate,
        "model": args.model,
        "_fixed_bias": env._fixed_bias
    })
