# forgetting_probe.py
#
# Periodically evaluates the in-training policy on a CLEAN (bias-free) FetchReach
# env and logs success rate + mean final distance. A mid-training dip in the clean
# success rate signals catastrophic forgetting: the curriculum's growing bias has
# pushed the policy off the clean task it had already solved. This can only be
# observed live during training, so it must be wired in before the real runs.

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.callbacks import BaseCallback

gym.register_envs(gymnasium_robotics)


class ForgettingProbeCallback(BaseCallback):
    """Evaluates the current policy on a clean (unbiased) env every `eval_freq` steps.

    Logs:
      eval/clean_success_rate — mean info['is_success'] over n_eval_episodes
      eval/clean_mean_d       — mean final achieved->desired goal distance (continuous;
                                doubles result resolution vs. the binary success metric)
    """

    def __init__(self, env_id="FetchReach-v4", eval_freq=5000,
                 n_eval_episodes=20, seed=10_000, verbose=0):
        super().__init__(verbose)
        self.env_id = env_id
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self._clean_env = None

    def _init_callback(self) -> None:
        # Build a dedicated clean env once. No bias wrapper: this is the reference
        # task. A fixed seed offset keeps eval goals reproducible across runs and
        # disjoint from training seeds.
        self._clean_env = gym.make(self.env_id)

    def _evaluate(self):
        successes, dists = [], []
        for i in range(self.n_eval_episodes):
            obs, _ = self._clean_env.reset(seed=self.seed + i)
            success, last_d = 0.0, np.nan
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self._clean_env.step(action)
                last_d = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
                success = max(success, float(info.get("is_success", 0.0)))
                done = terminated or truncated
            successes.append(success)
            dists.append(last_d)
        return float(np.mean(successes)), float(np.mean(dists))

    def _on_step(self) -> bool:
        # num_timesteps is SB3's global step count; probe on a fixed cadence.
        if self.num_timesteps % self.eval_freq == 0:
            sr, mean_d = self._evaluate()
            self.logger.record("eval/clean_success_rate", sr)
            self.logger.record("eval/clean_mean_d", mean_d)
            if self.verbose:
                print(f"[forgetting-probe] step={self.num_timesteps} "
                      f"clean_success={sr:.3f} clean_mean_d={mean_d:.4f}")
        return True

    def _on_training_end(self) -> None:
        if self._clean_env is not None:
            self._clean_env.close()
