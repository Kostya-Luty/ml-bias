# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Goal:** Investigate whether gradually introducing systematic actuation error during RL training (curriculum domain randomization) produces more robust policies than clean training or standard instant domain randomization.

**Hypothesis:** An optimal randomization rate `α` exists — slow enough not to disrupt early learning, fast enough that the policy adapts before training ends.

**Error model:** Per-episode systematic bias sampled from `Uniform(-ε(t), +ε(t))` per action dimension, held constant within an episode. Schedule: `ε(t) = min(ε_max, α·t)`.

## Environment & Stack

- **RL environment:** FetchReach-v3 (gymnasium-robotics) — sparse reward, goal-conditioned, 4D action space
- **Algorithm:** SAC or TD3 + HER (HindsightExperienceReplay from stable-baselines3)
- **Python:** venv at `venv/` — activate with `.\venv\Scripts\activate` (Windows)

## Commands

```bash
# Activate environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train a single agent (once training/train.py exists)
python training/train.py --config configs/<condition>.yaml

# Run unit tests (once tests exist)
python -m pytest envs/tests/

# Launch TensorBoard
tensorboard --logdir logs/
```

## Planned Directory Structure

```
envs/                  # Gymnasium wrappers
  systematic_bias_wrapper.py   # ActionWrapper with curriculum schedule
training/
  train.py             # Entry point: loads config, trains one agent, saves model
evaluation/            # Eval loops and robustness curve scripts
analysis/              # Aggregation scripts, plot generation
configs/               # YAML configs for each training condition
results/               # Saved models, keyed by condition+seed
logs/                  # TensorBoard event files
```

## Architecture

**Wrapper-based design:** `SystematicBiasWrapper(gymnasium.ActionWrapper)` in `envs/systematic_bias_wrapper.py` is the core abstraction. It must remain task-agnostic so it works on FetchPush/PickAndPlace without modification.

Key wrapper behaviors:
- `reset()` samples a new bias vector and caches it for the episode
- `action(act)` returns `clip(act + bias, action_bounds)`
- A global step counter drives `ε(t)`; an SB3 callback must advance it
- **Instant mode:** `ε(t) = ε_max` always (C4 baseline)
- **Fixed-bias eval mode:** bias set externally, never resampled (for the OOD test set in Phase 4)
- **No-op invariant:** `ε_max=0` must reproduce the clean baseline exactly

**Training conditions:**
- C0: Clean baseline (`ε_max=0`)
- C1–C3: Slow/medium/fast curriculum (α s.t. `ε_max` reached at ~25%/50%/75% of training)
- C4: Instant DR (`ε(t) = ε_max` from step 0)

All conditions share identical hyperparameters, architecture, total steps, and algorithm. Run each with ≥3 seeds.

## Key Implementation Notes

- Use `stable_baselines3.HerReplayBuffer` — it's the standard for sparse-reward Fetch tasks
- SB3 Zoo has tuned hyperparameters for FetchReach; start from those, don't tune
- `info['is_success']` is the success signal from the env
- Evaluation function: N=100 episodes, report mean success rate — reused across all phases
- Always record config (hyperparams, α, ε_max, seed) alongside each run's results
- Report mean ± std across seeds; never report a single seed as a result
