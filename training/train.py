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
# --log-dir: where TensorBoard/CSV logs and config.json are written
# --seed: controls all randomness for reproducibility across conditions
# --save-path: where the final model .zip is saved
parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", default="logs/curriculum_test", help="Directory for TensorBoard/CSV logs")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--save-path", default="results/test1", help="Path to save the trained model")
args = parser.parse_args()

# --- Curriculum schedule hyperparameters ---
# epsilon_max: peak actuation bias magnitude (fraction of action range)
# alpha: ramp rate — epsilon reaches epsilon_max at step epsilon_max/alpha (here: step 15000, i.e. 50% of training)
# total_timesteps: all conditions use identical budgets so comparisons are fair
EPSILON_MAX = 0.2
ALPHA = 0.2 / 15000  # reaches epsilon_max at the halfway point of training
TOTAL_TIMESTEPS = 30_000

# --- Save config alongside logs so every run is self-documenting ---
os.makedirs(args.log_dir, exist_ok=True)
config = {
    "env": "FetchReach-v4",
    "algorithm": "SAC+HER",
    "mode": "curriculum",
    "epsilon_max": EPSILON_MAX,
    "alpha": ALPHA,
    "total_timesteps": TOTAL_TIMESTEPS,
    "seed": args.seed,
    "save_path": args.save_path,
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
}
with open(os.path.join(args.log_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# --- Build environment ---
gym.register_envs(gymnasium_robotics)
base = gym.make("FetchReach-v4")
# Curriculum mode: bias ramps from 0 to epsilon_max over the first half of training,
# then stays constant. A new bias vector is sampled each episode and held fixed within it.
env = SystematicBiasWrapper(base, epsilon_max=EPSILON_MAX, alpha=ALPHA, mode="curriculum")

# --- Build model ---
# MultiInputPolicy handles the dict observation space (obs + achieved_goal + desired_goal).
# HER with "future" strategy is standard for sparse-reward Fetch tasks.
model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"),
            seed=args.seed, verbose=1)
model.set_logger(configure(args.log_dir, ["stdout", "csv"]))

# --- Train ---
# CurriculumStepCallback advances the wrapper's global step counter each env step
# so the epsilon schedule tracks actual training progress, not wall time.
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CurriculumStepCallback())

model.save(args.save_path)
