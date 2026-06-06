import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.logger import configure

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4")

log_path = "logs/test1"
new_logger = configure(log_path, ["stdout", "csv"])

model = SAC(
    policy="MultiInputPolicy",      # (a) dict observations -> MUST be MultiInput, not Mlp
    env=env,
    replay_buffer_class=HerReplayBuffer,   # (b) this is the "logbook" HER will relabel
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                  # (c) make 4 relabeled copies per real transition
        goal_selection_strategy="future",  # (d) relabel using goals achieved later in the same episode
    ),
    verbose=1,                      # print training progress to the console
    seed=0,                         # (e) fixed seed for this first reproducible run
)

model.set_logger(new_logger)

model.learn(total_timesteps=20000)

model.save("results/test1")