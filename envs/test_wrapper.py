import numpy as np
import gymnasium as gym
import gymnasium_robotics
from systematic_bias_wrapper import SystematicBiasWrapper

gym.register_envs(gymnasium_robotics)


def make(mode, epsilon_max=0.2, alpha=1e-5):
    base = gym.make("FetchReach-v4")
    return SystematicBiasWrapper(base, epsilon_max=epsilon_max, alpha=alpha, mode=mode)


# --- Test 1: bias is CONSTANT within an episode (systematic, not noise) ---
env = make("instant")          # instant -> full epsilon, easy to see
env.reset(seed=0)
bias_at_start = env._bias.copy()
for _ in range(10):
    env.step(env.action_space.sample())
assert np.array_equal(env._bias, bias_at_start), "Bias changed mid-episode!"
print("Test 1 passed: bias is frozen within an episode")


# --- Test 2: bias CHANGES across episodes ---
env.reset(seed=0)
b1 = env._bias.copy()
env.reset(seed=1)
b2 = env._bias.copy()
assert not np.array_equal(b1, b2), "Bias did not change across episodes!"
print("Test 2 passed: bias is resampled each episode")


# --- Test 3: epsilon(t) grows linearly and caps at epsilon_max ---
env = make("curriculum", epsilon_max=0.2, alpha=1e-5)
env.set_global_step(0)
assert env._current_epsilon() == 0.0
env.set_global_step(5000)
assert abs(env._current_epsilon() - 0.05) < 1e-9      # 1e-5 * 5000
env.set_global_step(10_000_000)
assert env._current_epsilon() == 0.2                  # capped
print("Test 3 passed: epsilon grows linearly and caps")


# --- Test 4: actions stay within bounds after biasing ---
env = make("instant", epsilon_max=0.5)
env.reset(seed=2)
extreme = env.action_space.high                       # command max action
out = env.action(extreme)
assert np.all(out <= env.action_space.high + 1e-9)
assert np.all(out >= env.action_space.low - 1e-9)
print("Test 4 passed: biased actions stay in bounds")


# --- Test 5 (THE important one): epsilon_max=0 is a perfect no-op ---
env = make("curriculum", epsilon_max=0.0)
env.reset(seed=3)
a = env.action_space.sample()
assert np.array_equal(env.action(a), np.clip(a, env.action_space.low, env.action_space.high)), \
    "Zero-bias wrapper altered the action!"
print("Test 5 passed: epsilon_max=0 is a no-op")

# --- Test 6: with a gate set, epsilon stays 0 until the gate opens ---
# Gate: require 3 consecutive successful episodes before the ramp starts.
env = make("curriculum", epsilon_max=0.2, alpha=1e-5)
env.gate_success_rate = 1.0
env.gate_patience = 3

env.set_global_step(50_000)          # plenty of steps elapsed...
assert env._current_epsilon() == 0.0, "Gate should keep epsilon at 0 before opening"
print("Test 6 passed: epsilon stays 0 while gate is closed, regardless of step")


# --- Test 7: the gate opens only after `patience` consecutive successes ---
# We drive step() manually, feeding controlled is_success values via a stub env.
class _StubEnv(gym.Env):
    """Minimal env that lets us script the is_success flag each step."""
    def __init__(self, action_space, success_schedule):
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self._schedule = list(success_schedule)
        self._i = 0
    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}
    def step(self, action):
        success = self._schedule[self._i]
        self._i += 1
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {"is_success": float(success)}

# Real action space so clipping/sampling behave; success pattern: 1,1,0,1,1,1
real_action_space = gym.make("FetchReach-v4").action_space
stub = _StubEnv(real_action_space, success_schedule=[1, 1, 0, 1, 1, 1])
genv = SystematicBiasWrapper(stub, epsilon_max=0.2, alpha=1e-5,
                             mode="curriculum")
genv.gate_success_rate = 1.0
genv.gate_patience = 3
genv.set_global_step(0)
genv.reset(seed=0)

a = real_action_space.sample()
genv.step(a)   # success 1 -> streak 1
genv.step(a)   # success 1 -> streak 2
assert genv._gate_open_step is None, "Gate opened too early (streak only 2)"
genv.step(a)   # success 0 -> streak RESETS to 0
assert genv._gate_open_step is None, "Gate should not open; streak was broken"
assert genv._success_streak == 0, "Streak should reset on a failure"
genv.step(a)   # success 1 -> streak 1
genv.step(a)   # success 1 -> streak 2
genv.set_global_step(100)
genv.step(a)   # success 1 -> streak 3 -> GATE OPENS at this step
assert genv._gate_open_step == 100, "Gate should open exactly when streak hits patience"
print("Test 7 passed: gate opens only after `patience` consecutive successes, resets on failure")


# --- Test 8: after the gate opens, the ramp counts from the gate-open step ---
# Gate opened at step 100 above. Advance 5000 steps past it.
genv.set_global_step(100 + 5000)
# elapsed since gate = 5000, so epsilon = alpha * 5000 = 1e-5 * 5000 = 0.05
assert abs(genv._current_epsilon() - 0.05) < 1e-9, "Ramp should count from gate-open step, not from 0"
print("Test 8 passed: ramp is measured from the gate-open step")


# --- Test 9: no gate (gate_success_rate=None) preserves original time-based behavior ---
env = make("curriculum", epsilon_max=0.2, alpha=1e-5)   # gate defaults to None
env.set_global_step(5000)
assert abs(env._current_epsilon() - 0.05) < 1e-9, "Ungated ramp should count from step 0"
print("Test 9 passed: ungated schedule unchanged (ramps from step 0)")

print("\nAll tests passed.")