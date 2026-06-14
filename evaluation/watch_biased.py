import argparse
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from envs.systematic_bias_wrapper import SystematicBiasWrapper

# Same as watch.py but wraps the env with a fixed epsilon=0.5 bias so you can
# visually compare behaviour against the unperturbed viewer.
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="results/test1", help="Path to saved model .zip (e.g. results/seed0)")
parser.add_argument("--epsilon", type=float, default=0.9, help="Fixed bias magnitude applied each episode")
args = parser.parse_args()

gym.register_envs(gymnasium_robotics)

# render_mode="human" opens a live MuJoCo viewer window
base = gym.make("FetchReach-v4", render_mode="human")
# alpha=0 keeps the bias at epsilon_max from the first step — no ramp
env = SystematicBiasWrapper(base, epsilon_max=args.epsilon, alpha=0, mode="fixed")

try:

    model = SAC.load(args.model, env=env)

    env._fixed_bias = np.full(env.action_space.shape, args.epsilon)
    obs, info = env.reset(seed=0)
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)  # deterministic = no exploration noise
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

finally:
    env.close()
