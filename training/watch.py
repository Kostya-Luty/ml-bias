import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)

# render_mode="human" opens a live MuJoCo viewer window
env = gym.make("FetchReach-v4", render_mode="human")

model = SAC.load("results/test1", env=env)

obs, info = env.reset(seed=0)
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)   # deterministic = no exploration noise
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()