# ml-bias — Curriculum Domain Randomization for Systematic Actuation Error

This project experiments with gradually introducing systematic bias into a 
reinforcement learning (RL) training algorithm, rather than introducing it
in its full extent immediately. 

**Hypothesis:** an optimal ramp rate `α` exists; slow enough not to disrupt early
learning, fast enough that the policy adapts to the bias before training ends.

**Short answer:** The hypothesis is partially supported. A slow curriculum is the
best policy at every bias level tested, not because it uniquely learns to
overcome bias, but rather by not destroying the original unbiased policy like 
instant and fast rates do. See [Results](#results).

---

## Method

- **Environment:** `FetchReach-v4` (gymnasium-robotics) — sparse reward,
  goal-conditioned, 4D action space (`Box(-1, 1)`; 3 end effector deltas + 1 inert gripper),
  50-step episodes, 5 cm success threshold.
- **Algorithm:** SAC + HER (`HerReplayBuffer`, `MultiInputPolicy`) from
  stable-baselines3.
- **Bias model:** a per-episode additive (e.g. biased output = unbiased + bias) offset modeling a miscalibrated actuator,
  sampled once per episode and held constant within it:

  ```
  bias ~ Uniform(-ε(t), +ε(t))   per action dimension
  ε(t) = min(ε_max, α · t)
  ```

- **Controls:** all conditions share the same seed resultin in identical 
  hyperparameters, architecture, total steps (100k), and algorithm. The only 
  differences are `ε_max` (C0 vs. rest) and `α`. Each condition is run with 
  3 seeds, results are reported mean ± std.
- **ε_max = 1.1**, chosen as the magnitude that drops the clean, unbiased
  policy to ~57% success (mid-degradation band).

### Conditions

| Condition | mode | alpha | ε_max reached at |
|---|---|---|---|
| C0_clean | curriculum | 0 (no-op) | never (ε≡0) |
| C1_slow | curriculum | 1.467e-5 | 75k (75%) |
| C2_medium | curriculum | 2.2e-5 | 50k (50%) |
| C3_fast | curriculum | 4.4e-5 | 25k (25%) |
| C4_instant | instant | — | step 0 |
| C5_gated | curriculum | 2.2e-5 | 50k timesteps after model reaches 100% success rate |

`alpha = 1.1 / (f · 100000)`, where `f` is the fraction of training spent ramping.
C5_gated holds ε=0 until 20 consecutive successes, then starts the medium ramp from
that step.

---

## Results

Deterministic evaluation, N=100 episodes per seed, mean ± std over 3 seeds. Each model 
is subject to the same envionment to ensure accurate comparison. The biased protocol 
resamples a fresh vector bias within the given magnitude every episode.
(`mode="fixed"`). Chart-ready CSVs are in `results-final/`.

### Robustness vs. bias magnitude — biased success rate

`results-final/ood_wide.csv` (and `ood_summary.csv` for error bars)

| Condition | ε=0.5 | ε=0.8 | ε=1.1 |
|---|---|---|---|
| C0_clean | 0.937 | 0.733 | 0.383 |
| **C1_slow** | **0.967** | **0.760** | 0.420 |
| C2_medium | 0.870 | 0.743 | 0.403 |
| C3_fast | 0.823 | 0.743 | 0.460 |
| C4_instant | 0.667 | 0.603 | 0.410 |
| C5_gated | 0.840 | 0.697 | 0.357 |

### Clean competence (forgetting cross-check) at ε=1.1

How each model fares against the original policy, without bias. Note that 
C4_instant (maximum bias introduced immediately) performs significantly worse.
`results-final/eval_summary.csv`; training-time trend in `forgetting_wide.csv`

| Condition | Clean success |
|---|---|
| C0_clean | 1.000 ± 0.000 |
| C1_slow | 0.997 ± 0.005 |
| C2_medium | 0.933 ± 0.042 |
| C3_fast | 0.843 ± 0.054 |
| C4_instant | 0.660 ± 0.120 |
| C5_gated | 0.913 ± 0.116 |

---

## Interpretation

- At the saturating bias (ε=1.1) the conditions are indistinguishable (all
  0.38–0.46, overlapping std). When a bias component exceeds the `±1.0` action
  bound it clips that dimension for the whole episode, making a fraction of goals
  in the test set physically unreachable by any policy, resulting in these values. 
- Below saturation, an order is clear. At ε=0.5, C1_slow (0.97) ≈ clean
  (0.94) > C2 > C5 > C3 > C4_instant (0.67). The harder bias was forced during
  training, the worse the policy does at moderate test bias.
- The blocker in training is forgetting. Instant bias causes the model to drop clean
  competence to 0.66; meanwhile, the performance of C3/C4 in the clean policy
  dipped mid-training while C0/C1 stayed at 1.0. C1 (slow) wins by preserving the base
  skill while adapting just enough to edge out clean training.
- Verdict: the hypothesis holds in part; the curriculum learning beat introducing error
  instantly. However, there is no interior optimal `α`; in this setup, slower is
  monotonically better, and standard instant randomization is the worst option.

---

## Limitations

- Uniform additive action bias is a simplified model of real systematic error.
  A more accurate representation would include backlash, drift, etc.
- This is sim-to-sim transfer, not validated on hardware. While this is intended 
  to aid in beating the sim-to-real gap, the extent of this experiment only covers
  software.
- FetchReach is a deliberately easy model, chosen to demonstrate the theory at low 
  cost. Results may not carry to contact-rich tasks such as FetchPush and PickAndPlace,
  where systematic bias matters more.
- 3 seeds is the minimum for trustworthy RL comparison; more would tighten the
  std bands, several of which currently overlap.

---

## Reproduce

```powershell
# Setup (Windows)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Unit tests for the bias wrapper
python -m pytest envs/test_wrapper.py

# Train one condition, 3 seeds (training is already done; re-run only to verify)
python -m training.train --config configs/c1_slow.json --seed 0 --runs 3

# Evaluation (writes results-final/*.csv)
python -m evaluation.evaluate_all   # ε=1.1 biased + clean cross-check
python -m evaluation.evaluate_ood   # test values of ε=0.5, 0.8

# Build Excel-ready summary/wide tables
python -m analysis.make_tables

# Training curves
tensorboard --logdir logs/
```

Charts are produced in Excel from the `results-final/*.csv` files (no plotting code
in the repo): `ood_wide.csv` → robustness-vs-bias line chart, `eval_summary.csv` →
success/forgetting bar charts, `forgetting_wide.csv` → clean-success-during-training
line chart.

---

## Project structure

```
envs/
  systematic_bias_wrapper.py   # ActionWrapper: curriculum / instant / fixed modes + mastery gate
  test_wrapper.py              # 9 unit tests
training/
  train.py                     # Config-driven SAC+HER trainer (--config, --seed, --runs)
  curriculum_callback.py       # Pushes timestep into the wrapper, logs curriculum/epsilon
  forgetting_probe.py          # Periodic clean-env eval during training (forgetting signal)
evaluation/
  evaluate.py                  # evaluate_policy(model_path, n_episodes, seed) -> float
  evaluate_all.py              # All models @ ε=1.1 biased + clean -> eval_results.csv
  evaluate_ood.py              # All models @ ε=0.5, 0.8 -> ood_results.csv
analysis/
  make_tables.py               # Builds Excel-ready summary + wide CSVs
configs/                       # One JSON per condition (c0_clean.json … c5_gated.json)
results/<condition>/seed{N}.zip  # Trained models (18: 6 conditions × 3 seeds)
logs/<condition>/seed{N}/      # progress.csv, config.json, TensorBoard events
results-final/                 # Evaluation CSVs (chart sources)
```
