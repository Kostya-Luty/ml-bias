import argparse
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
import numpy as np
from envs.systematic_bias_wrapper import SystematicBiasWrapper

gym.register_envs(gymnasium_robotics)


base = gym.make("FetchReach-v4")
env = SystematicBiasWrapper(base, epsilon_max=1, alpha=0, mode="fixed")
model = SAC.load("results/test1", env=env)
env._fixed_bias = np.full(env.action_space.shape, 0)   # deliberate, known bias
obs, info = env.reset(seed=0)
print("bias after reset:", env._bias)        # <-- should print 0.3,0.3,0.3,0.3
for _ in range(5):
    act, _ = model.predict(obs, deterministic=True)
    biased = env.action(act)
    print("raw:", np.round(act,2), " biased:", np.round(biased,2))
    obs, *_ = env.step(act)