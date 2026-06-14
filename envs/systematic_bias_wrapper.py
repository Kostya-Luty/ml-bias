import numpy as np
import gymnasium as gym


class SystematicBiasWrapper(gym.ActionWrapper):
    """
    Wraps a FetchReach-style env and injects a per-episode systematic bias
    into actions, simulating a consistently miscalibrated robot.

    The bias is sampled once per episode from Uniform(-eps(t), +eps(t)),
    held constant for the whole episode, where eps(t) grows linearly with
    the global training step:  eps(t) = min(eps_max, alpha * t).
    """

    def __init__(self, env, epsilon_max, alpha, mode="curriculum",
                 gate_success_rate=None, gate_patience=0):
        super().__init__(env)
        self.epsilon_max = epsilon_max
        self.alpha = alpha
        self.mode = mode
        self.gate_success_rate = gate_success_rate   # e.g. 1.0; None = no gate (pure time schedule)
        self.gate_patience = gate_patience           # how many consecutive successes required

        self._global_step = 0
        self._bias = None
        self._fixed_bias = None
        self._gate_open_step = None                  # step at which the gate opened; None until then
        self._success_streak = 0

    def reset(self, **kwargs):
        # how wide the bias range is right now
        epsilon = self._current_epsilon()

        if self.mode == "fixed":
            # to manually fix bias for entire simulation
            self._bias = self._fixed_bias
        else:
            # curriculum / instant: sample a bias for this episode
            self._bias = np.random.uniform(
                low=-epsilon,
                high=+epsilon,
                size=self.action_space.shape,   # one random value per dimension
            )

        return self.env.reset(**kwargs)
    
    def action(self, act):
        # add this episode's systematic error
        bias = self._bias if self._bias is not None else 0.0
        biased = act + bias

        # keep the result within the environment's legal action range
        biased = np.clip(
            biased,
            self.action_space.low,
            self.action_space.high,
        )
        return biased
    
    def _current_epsilon(self):
        if self.mode == "instant":
            return self.epsilon_max
        if self.mode == "fixed":
            return 0.0

        # curriculum mode:
        if self.gate_success_rate is not None and self._gate_open_step is None:
            # gate exists and hasn't opened yet -> no bias at all
            return 0.0

        # ramp counts from the gate-open step (or from 0 if no gate)
        start = self._gate_open_step if self._gate_open_step is not None else 0
        elapsed = self._global_step - start
        return min(self.epsilon_max, self.alpha * elapsed)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))

        # gate logic: only relevant if a gate is set and not yet open
        if self.gate_success_rate is not None and self._gate_open_step is None:
            if info.get("is_success", 0.0) >= 1.0:
                self._success_streak += 1
            else:
                self._success_streak = 0
            if self._success_streak >= self.gate_patience:
                self._gate_open_step = self._global_step   # this becomes ramp "time zero"

        return obs, reward, terminated, truncated, info
    
    def set_global_step(self, step):
        self._global_step = step

    def get_current_epsilon(self):
        return self._current_epsilon()