# Sim-to-Real via Curriculum Domain Randomization — Implementation Plan

**Project goal:** Investigate whether *gradually* introducing systematic actuation error during RL training (curriculum domain randomization) produces policies more robust to consistent calibration biases than either clean training or standard (instant) domain randomization.

**Core hypothesis:** There exists an optimal randomization rate `α` — slow enough to avoid disrupting early learning, fast enough that the policy adapts to bias before training ends. Curriculum policies will outperform both the clean baseline and instant-randomization baseline when evaluated on out-of-distribution biases.

**Independent variable (Phase 1):** randomization rate `α`
**Error model:** per-episode systematic bias sampled from `Uniform(-ε(t), +ε(t))`, held constant within an episode, where `ε(t) = min(ε_max, α·t)`.
**Dependent variables:** success rate under perturbation, sample efficiency, clean-sim performance over time (forgetting probe).

---

## Phase 0 — Environment Setup

### 0.1 — Create an isolated Python environment
- [x] Install latest Python version
- [x] Create a virtual environment: `python -m venv venv` then activate it.
- [x] Upgrade pip: `pip install --upgrade pip`.

### 0.2 — Install the stack
- [x] `pip install mujoco`
- [x] `pip install gymnasium gymnasium-robotics`
- [x] `pip install stable-baselines3`
- [x] `pip install numpy matplotlib pandas tensorboard`
- [x] Pin versions once everything works: `pip freeze > requirements.txt`.

### 0.3 — Verify the install
- [x] Run a one-line sanity check that imports `gymnasium`, makes `FetchReach-v4`, and prints the observation/action space.
- [x] Confirm MuJoCo renders (or at least runs headless) without errors.
- [x] If on a headless machine, set `MUJOCO_GL=egl` or `osmesa` and confirm rendering still initializes.

### 0.4 — Set up the project skeleton
- [x] Create directories: `envs/`, `training/`, `evaluation/`, `analysis/`, `configs/`, `results/`, `logs/`.
- [x] Initialize a git repo. Add a `.gitignore` for `venv/`, `results/`, `logs/`, `__pycache__/`.
- [x] Create a `README.md` stub describing the project (you'll expand it later — it's the portfolio front door).

---

## Phase 1 — Baseline: Get FetchReach Running Unmodified

The goal here is to confirm a standard agent trains and reaches high success on the clean task before touching perturbations.

### 1.1 — Understand the environment
- [x] Print and document the observation space (FetchReach returns a dict: `observation`, `achieved_goal`, `desired_goal`).
- [x] Print and document the action space (4D: 3 end-effector deltas + 1 gripper, gripper unused in Reach).
- [x] Note that FetchReach uses **sparse** rewards by default and is a **goal-conditioned** env — this matters for algorithm choice.

### 1.2 — Choose and configure the algorithm
- [x] Use **SAC** or **TD3** with **HER (Hindsight Experience Replay)** — the standard combo for sparse-reward goal-conditioned Fetch tasks. `stable-baselines3` supports `HerReplayBuffer`.
- [x] Start from known-good hyperparameters (SB3 Zoo has tuned configs for Fetch). Don't tune yet.
- [x] Set a fixed random seed for this first run so results are reproducible.

### 1.3 — Train the clean baseline
- [x] Write `training/train.py` that takes a config and trains one agent.
- [ ] Train until success rate plateaus (FetchReach is easy; expect near-100% success well before 1M steps).
- [ ] Log to TensorBoard. Confirm the success-rate curve climbs and plateaus.
- [x] Save the trained model to `results/test1` (smoke-test run; full baseline TBD).

### 1.4 — Define and verify the success metric
- [x] Confirm what "success" means: end-effector within a distance threshold of the goal (the env reports this via `info['is_success']`).
- [x] Write a quick evaluation loop that runs N=100 episodes and reports mean success rate (`evaluation/evaluate.py`). This is the function you'll reuse everywhere.

**Phase 1 exit criterion:** a saved clean policy that achieves >95% success on unmodified FetchReach, and a reusable evaluation function.

---

## Phase 2 — Build the Perturbation Wrapper

This is the technical heart of the project. The bias injection lives in a Gymnasium wrapper so it's cleanly separable from the training code.

### 2.1 — Design the wrapper interface
- [ ] Create `envs/systematic_bias_wrapper.py` subclassing `gymnasium.ActionWrapper`.
- [ ] Constructor parameters: `epsilon_max`, `alpha` (rate), `schedule_type` (default linear), and a flag to enable/disable a fixed-bias evaluation mode.
- [ ] The wrapper must track a global training-step counter so `ε(t)` can grow over time.

### 2.2 — Implement per-episode bias sampling
- [ ] On each `reset()`, sample a new bias vector: `bias ~ Uniform(-ε(t), +ε(t))` per action dimension (or a single scalar magnitude with random direction — decide and document which; per-dimension is simpler to start).
- [ ] Store the bias for the episode. Do **not** resample mid-episode (this is what makes it *systematic*, not noise).

### 2.3 — Implement the action transformation
- [ ] In `action(act)`, return `act + bias` (clipped to the valid action range).
- [ ] Decide whether to apply bias to the gripper dimension (for Reach, irrelevant — document the choice).

### 2.4 — Implement the schedule
- [ ] `ε(t) = min(ε_max, α·t)` where `t` is the current global step.
- [ ] Expose a method to set/advance the step counter from the training loop (SB3 callbacks can do this).
- [ ] Add an "instant" mode (`ε(t) = ε_max` always) for the standard-DR baseline.
- [ ] Add a "fixed bias" evaluation mode where the bias is set externally and never resampled — needed for the controlled test set in Phase 4.

### 2.5 — Unit-test the wrapper in isolation
- [ ] Verify bias is constant within an episode and changes across episodes.
- [ ] Verify `ε(t)` grows linearly and caps at `ε_max`.
- [ ] Verify actions stay within bounds after biasing.
- [ ] Verify instant mode and fixed-bias mode behave correctly.
- [ ] Sanity check: with `ε_max = 0`, the wrapper is a no-op and reproduces the clean baseline exactly.

### 2.6 — Wire the step counter to training
- [ ] Write an SB3 callback that pushes the current timestep into the wrapper every step (or every rollout) so the schedule advances.
- [ ] Confirm via logging that `ε(t)` actually increases during a short training run.

**Phase 2 exit criterion:** a tested wrapper that injects per-episode systematic bias on a linear schedule, with verified instant and fixed-bias modes, and an `ε_max=0` no-op that matches the clean baseline.

---

## Phase 3 — Run the Training Conditions

### 3.1 — Choose ε_max
- [ ] Pick `ε_max` by experiment: find the bias magnitude that drops the clean policy's success rate to roughly 40–60% (enough headroom to show improvement, not so much the task is impossible).
- [ ] Do this by evaluating the Phase 1 clean policy under several fixed biases and plotting success vs. bias magnitude. Document the chosen `ε_max`.

### 3.2 — Define the condition matrix
- [ ] **C0 — Clean baseline:** `ε_max = 0` (already have from Phase 1).
- [ ] **C1 — Slow curriculum:** small `α`.
- [ ] **C2 — Medium curriculum:** moderate `α`.
- [ ] **C3 — Fast curriculum:** large `α` (reaches `ε_max` early in training).
- [ ] **C4 — Instant DR baseline:** `ε(t) = ε_max` from step 0.
- [ ] Choose `α` values so that C1–C3 reach `ε_max` at roughly 25%, 50%, 75% of total training steps (clean, interpretable spacing).

### 3.3 — Control for confounds
- [ ] All conditions use identical hyperparameters, network architecture, total training steps, and algorithm.
- [ ] The **only** differences are `ε_max` (C0 vs rest) and `α` (C1–C4).
- [ ] Run each condition with **multiple seeds** (start with 3, ideally 5) — single-seed RL results are not trustworthy.

### 3.4 — Probe for catastrophic forgetting during training
- [ ] Add a periodic evaluation callback that, every K steps, evaluates the *current* policy on the **clean** env (bias off).
- [ ] Log this clean-success curve alongside training. A mid-training dip is the forgetting signal — capture it whether or not it appears.

### 3.5 — Execute and log everything
- [ ] Run all conditions × all seeds. Save each model and its full log.
- [ ] Record config (all hyperparameters, `α`, `ε_max`, seed) alongside each run so results are reproducible.

**Phase 3 exit criterion:** trained policies for every condition × seed, with clean-eval-during-training curves logged.

---

## Phase 4 — Evaluation Under Out-of-Distribution Bias

### 4.1 — Build a fixed evaluation test set
- [ ] Generate a fixed, seeded set of bias vectors at evaluation magnitude (e.g., sampled at `ε_max`, and optionally *beyond* `ε_max` to test extrapolation).
- [ ] This set is identical across all conditions — same biases tested against every policy. This is what makes the comparison fair.

### 4.2 — Evaluate every policy on the test set
- [ ] For each policy (condition × seed), run the fixed-bias evaluation mode across the full test set, N episodes per bias.
- [ ] Record success rate per bias and aggregate.

### 4.3 — Measure sample efficiency
- [ ] From training logs, extract steps-to-threshold (e.g., steps to first reach 80% success on the perturbed objective) for each condition.

### 4.4 — Aggregate across seeds
- [ ] Report mean ± standard deviation (or confidence intervals) across seeds for every metric. Never report a single seed.

**Phase 4 exit criterion:** a results table of success rate (mean ± std over seeds) for each condition on the OOD bias test set, plus sample-efficiency numbers.

---

## Phase 5 — Analysis & Writeup

### 5.1 — Core plots
- [ ] **Success rate vs. α** — the headline plot. Does a sweet spot exist? Where does C4 (instant) fall?
- [ ] **Training curves per condition** — success over steps, overlaid, with seed variance shaded.
- [ ] **Forgetting plot** — clean-env success during training, looking for mid-training dips at high `α`.
- [ ] **Robustness curve** — success rate vs. bias magnitude for each condition (including extrapolation beyond `ε_max`).

### 5.2 — Interpret the result honestly
- [ ] State which hypothesis outcome occurred (optimal α exists / curriculum beats instant / null result).
- [ ] A null result is still a finding — report it plainly.

### 5.3 — Document limitations
- [ ] Note that uniform additive action bias is a simplified model of real systematic error (no joint-specific nonlinearity, backlash, or drift).
- [ ] Note that this is sim-to-sim transfer, not validated on hardware.
- [ ] Frame system identification and hardware validation as natural next steps.

### 5.4 — Portfolio packaging
- [ ] Clean README: problem, hypothesis, method, key plot, how to reproduce.
- [ ] Reproducibility: `requirements.txt`, seeds, config files, one command to rerun.
- [ ] Short technical writeup (blog post or PDF) walking through the design decisions and the result.
- [ ] Make the headline plot the visual hook at the top of the README.

---

## Open Decisions To Revisit
- Per-dimension bias vs. single magnitude with random direction (currently: per-dimension, simpler).
- Whether to extend to FetchPush later (harder task, systematic bias matters more for contact).
- Number of seeds (3 minimum, 5 preferred — depends on compute/time budget).
- Whether to add random *noise* as a secondary condition once systematic error is characterized.

---

## Notes
- FetchReach is deliberately easy — it's the right starting task because it isolates the *transfer* question from the *can-it-learn-the-task* question. If results are clean here, escalate to a harder task.
- Keep the wrapper task-agnostic so it drops onto FetchPush/PickAndPlace without rewrites.
- The implementation is intentionally the small part. The value is the controlled experiment and the analysis.