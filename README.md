# ml-bias

Investigates whether gradually introducing systematic actuation error during RL training (**curriculum domain randomization**) produces more robust policies than clean training or standard instant domain randomization.

**Hypothesis:** an optimal ramp rate `α` exists — slow enough not to disrupt early learning, fast enough that the policy adapts before training ends.

---

## Setup

```powershell
# Create and activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training

### Single run

```powershell
python -m training.train --log-dir logs/curriculum --seed 0
```

### Batch runs (multiple seeds, sequential)

```powershell
python -m training.train --log-dir logs/curriculum --seed 0 --runs 5
```

Each run writes to:
- `logs/<log-dir>/seed{N}/` — TensorBoard events, `progress.csv`, `config.json`
- `results/seed{N}.zip` — saved model

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--log-dir` | `logs/curriculum_test` | Base directory for logs; each run gets a `seed{N}` subdir |
| `--seed` | `0` | Starting seed; batch runs use `seed`, `seed+1`, ... |
| `--save-dir` | `results` | Directory for saved model `.zip` files |
| `--runs` | `1` | Number of sequential runs |

---

## Monitoring

```powershell
tensorboard --logdir logs/curriculum
```

Open `http://localhost:6006` to view training curves across all seeds.

---

## Experimental Conditions

All conditions use identical hyperparameters, architecture, total steps (30k), and algorithm (SAC + HER).

| Condition | Mode | Description |
|---|---|---|
| C0 | Clean | `ε_max = 0` — no bias, clean baseline |
| C1 | Curriculum (slow) | `ε_max` reached at ~75% of training |
| C2 | Curriculum (medium) | `ε_max` reached at ~50% of training (current default) |
| C3 | Curriculum (fast) | `ε_max` reached at ~25% of training |
| C4 | Instant DR | `ε(t) = ε_max` from step 0 |

Run each condition with ≥ 3 seeds and report mean ± std success rate.

---

## Error Model

Per-episode systematic bias sampled from `Uniform(-ε(t), +ε(t))` per action dimension, held constant within an episode.

```
ε(t) = min(ε_max, α · t)
```

With the current default (`α = ε_max / 15000`, `ε_max = 0.2`), the bias ramps from 0 to 0.2 over the first 15k steps (50% of the training budget), then stays constant.

---

## Project Structure

```
envs/
  systematic_bias_wrapper.py   # ActionWrapper implementing the curriculum bias schedule
training/
  train.py                     # Main training script (supports batching)
  curriculum_callback.py       # SB3 callback that ticks the wrapper's step counter
evaluation/
  evaluate.py                  # evaluate_policy(model_path, n_episodes, seed) -> float
results/                       # Saved model .zip files
logs/                          # TensorBoard event files and per-run config.json
```
