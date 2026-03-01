## Noise Implementation in New Verl

This document focuses on how parameter noise and adaptive noise are implemented in the new `verl` framework and how they are wired into training.

### 1) Parameter Noise: Where and How It Is Applied

Implementation location:
- `xinyue/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`

Key mechanics:
- Noise is applied during rollout weight synchronization only, so training weights remain clean.
- Noise is generated on CPU in `fp32` and then cast back to the original parameter dtype.
- A deterministic per-parameter seed is derived from a shared step seed and parameter name hash.
- Optional filtering by module name substring allows noise to target only specific submodules (e.g. `['.mlp.']`).

Core methods:
- `set_noise_sigma(new_sigma)`: update rollout noise at runtime.
- `_apply_param_noise(name, weight, step_seed)`: returns a noisy copy of the tensor.
- `_normalize_noise_target_modules(...)`: handles list or string config values.

### 2) Rollout Configuration Fields

Rollout config additions:
- `actor_rollout_ref.rollout.use_param_noise`
- `actor_rollout_ref.rollout.noise_sigma`
- `actor_rollout_ref.rollout.noise_target_modules`
- `actor_rollout_ref.rollout.param_noise_base_seed`

Definition files:
- `xinyue/verl/verl/workers/config/rollout.py`
- `xinyue/verl/verl/trainer/config/rollout/rollout.yaml`

### 3) Adaptive Noise Controller

Adaptive noise is handled in the trainer:
- `xinyue/verl/verl/trainer/ppo/ray_trainer.py`

Behavior:
- After `old_log_probs` are recomputed, compare to `rollout_log_probs`.
- Compute “noisy KL” as `rollout_log_probs - old_log_probs`.
- If adaptive noise is enabled, update sigma using a target KL band and a multiplicative coefficient.
- Push updated sigma to rollout workers via `set_noise_sigma()`.

Adaptive noise config fields:
- `actor_rollout_ref.actor.adaptive_noise`
- `actor_rollout_ref.actor.adaptive_noise_target_kl`
- `actor_rollout_ref.actor.adaptive_noise_coeff`
- `actor_rollout_ref.actor.adaptive_noise_min_sigma`
- `actor_rollout_ref.actor.adaptive_noise_max_sigma`
- `actor_rollout_ref.actor.adaptive_noise_initial_sigma`

Definition files:
- `xinyue/verl/verl/workers/config/actor.py`
- `xinyue/verl/verl/trainer/config/actor/actor.yaml`

### 4) Worker Plumbing for Sigma Updates

To allow runtime updates to rollout noise, `set_noise_sigma()` is implemented in:
- `xinyue/verl/verl/workers/fsdp_workers.py`
- `xinyue/verl/verl/workers/engine_workers.py`
- `xinyue/verl/verl/workers/megatron_workers.py`

These forward the sigma to the rollout adapter (`vllm` server adapter).

### 5) Rollout Logprobs and TIS (Noise-Dependent)

Noise metrics and adaptive updates require rollout logprobs:
- `actor_rollout_ref.rollout.calculate_log_probs=True`

TIS/rollout correction uses the same logprobs to compute truncated IS weights:
- `algorithm.rollout_correction.rollout_is=token`
- `algorithm.rollout_correction.rollout_is_threshold=<tis_imp_ratio_cap>`

### 6) Script-Level Wiring

The new launcher exposes the key noise flags:
- `--noise_sigma`, `--param_noise_base_seed`, `--adaptive_noise`, `--noise_target_kl`

These are mapped to new config fields in:
- `xinyue/verl/train_grpo_math_tune_ray.sh`

### 7) Files Touched for Noise Support

- `xinyue/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- `xinyue/verl/verl/trainer/ppo/ray_trainer.py`
- `xinyue/verl/verl/workers/fsdp_workers.py`
- `xinyue/verl/verl/workers/engine_workers.py`
- `xinyue/verl/verl/workers/megatron_workers.py`
- `xinyue/verl/verl/workers/config/rollout.py`
- `xinyue/verl/verl/trainer/config/rollout/rollout.yaml`
- `xinyue/verl/verl/workers/config/actor.py`
- `xinyue/verl/verl/trainer/config/actor/actor.yaml`
