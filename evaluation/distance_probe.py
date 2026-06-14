"""
Diagnostic: continuous distance metric vs. bias magnitude.

Instead of only the binary success flag, this logs the final
end-effector-to-goal distance for every episode. The continuous signal
reveals what the binary success rate hides:
  - whether the bias actually moves the arm (saturation check)
  - how much the per-episode error SPREADS across random goals

For each epsilon we report, over N episodes:
  success_rate     - the binary metric (what we had before)
  mean_distance    - average final distance to goal
  std_distance     - spread across episodes (the re-smoothing signal)
  min/max_distance - range

Run from the project root:  python evaluation/distance_probe.py
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from envs.systematic_bias_wrapper import SystematicBiasWrapper

gym.register_envs(gymnasium_robotics)

# ---- config ----
MODEL_PATH = "results/test1"   # adjust if your good model is elsewhere
N_EPISODES = 100
EPSILONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3]
SEED_BASE = 1000   # same goal set across all epsilons -> fair comparison
# ----------------




def final_distance(info, obs):
    """
    End-effector-to-goal distance at episode end.
    FetchReach's observation dict carries achieved_goal and desired_goal,
    both 3D positions; the distance is the norm of their difference.
    """
    return float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))


print(f"{'eps':>5} | {'success':>7} | {'mean_d':>7} | {'std_d':>6} | {'min_d':>6} | {'max_d':>6}")
print("-" * 52)

for eps in EPSILONS:
    base = gym.make("FetchReach-v4")
    env = SystematicBiasWrapper(base, epsilon_max=eps, alpha=0, mode="fixed")

    model = SAC.load(MODEL_PATH, env=env)

    successes = 0
    distances = []

    for ep in range(N_EPISODES):
        # fixed bias for this episode, at magnitude eps, random direction
        env._fixed_bias = np.random.uniform(-eps, eps, size=env.action_space.shape)
        obs, info = env.reset(seed=SEED_BASE + ep)

        done = False
        last_obs = obs
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            last_obs = obs
            done = term or trunc

        successes += int(info.get("is_success", 0.0))
        distances.append(final_distance(info, last_obs))

    distances = np.array(distances)
    print(f"{eps:5.2f} | {successes / N_EPISODES:7.2f} | "
          f"{distances.mean():7.3f} | {distances.std():6.3f} | "
          f"{distances.min():6.3f} | {distances.max():6.3f}")