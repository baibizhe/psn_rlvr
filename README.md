# PSN-RLVR on VERL

Implementation and training entrypoint for:

**Learning to Explore with Parameter-Space Noise: A Deep Dive into Parameter-Space Noise for Reinforcement Learning with Verifiable Rewards**

- Paper (arXiv PDF): https://arxiv.org/pdf/2602.02555
- Core code directory: `psn_rlvr/`
- Implementation notes: `psn_rlvr/VERL_IMPLEMENTATION_DETAILS.md`

## What This Repository Adds

This workspace adapts VERL-based GRPO training to support **parameter-space exploration** for RLVR:

- Parameter-space noise during rollout (clean training weights are preserved).
- Optional **module-targeted noise injection** (e.g. MLP-only).
- **Adaptive noise scheduling** based on rollout/old-policy KL proxy.
- **Truncated Importance Sampling (TIS)** rollout correction support.
- Script-level flags for reproducible fixed-noise and adaptive-noise runs.

## Repository Layout

- `psn_rlvr/train_grpo_math_tune_ray.sh`  
  Main launcher with PSN, adaptive noise, and TIS wiring.
- `psn_rlvr/VERL_IMPLEMENTATION_DETAILS.md`  
  Detailed mapping of where each feature is implemented.
- `psn_rlvr/verl/workers/rollout/vllm_rollout/vllm_rollout.py`  
  Rollout-time parameter noise injection.
- `psn_rlvr/verl/trainer/ppo/ray_trainer.py`  
  Adaptive sigma updates and rollout-vs-old logprob comparison.
- `psn_rlvr/verl/trainer/config/actor/actor.yaml`  
  Adaptive noise config defaults.
- `psn_rlvr/verl/trainer/config/rollout/rollout.yaml`  
  Rollout noise config defaults.

## PSN Implementation Summary

### 1) Parameter Noise in Rollout

Noise is applied when syncing rollout weights, not to the optimizer-updated training parameters:

- CPU-side `fp32` noise sampling, cast back to parameter dtype.
- Deterministic per-parameter seeding from global step seed + parameter hash.
- Runtime sigma update support via `set_noise_sigma(...)`.
- Optional filtering via `noise_target_modules` (e.g. `['.mlp.']`).

Key rollout config fields:

- `actor_rollout_ref.rollout.use_param_noise`
- `actor_rollout_ref.rollout.noise_sigma`
- `actor_rollout_ref.rollout.noise_target_modules`
- `actor_rollout_ref.rollout.param_noise_base_seed`

### 2) Adaptive Noise Controller

Implemented in trainer logic after `old_log_probs` recomputation:

- Computes noisy KL proxy from `rollout_log_probs - old_log_probs`.
- Updates sigma with multiplicative coefficient around a target KL.
- Pushes the new sigma to rollout workers through `set_noise_sigma(...)`.

Key actor config fields:

- `actor_rollout_ref.actor.adaptive_noise`
- `actor_rollout_ref.actor.adaptive_noise_target_kl`
- `actor_rollout_ref.actor.adaptive_noise_coeff`
- `actor_rollout_ref.actor.adaptive_noise_min_sigma`
- `actor_rollout_ref.actor.adaptive_noise_max_sigma`
- `actor_rollout_ref.actor.adaptive_noise_initial_sigma`

### 3) Rollout Logprobs and TIS

Adaptive control and TIS both depend on rollout logprobs:

- Enable rollout logprobs via `actor_rollout_ref.rollout.calculate_log_probs=True`.
- Enable TIS correction via:
  - `algorithm.rollout_correction.rollout_is=token`
  - `algorithm.rollout_correction.rollout_is_threshold=<cap>`

## Quick Start

From repository root:

```bash
cd psn_rlvr
bash train_grpo_math_tune_ray.sh
```

### Fixed-Noise PSN Example

```bash
cd psn_rlvr
bash train_grpo_math_tune_ray.sh \
  --noise_sigma 0.004 \
  --param_noise_base_seed 42 \
  --tis_imp_ratio_cap 10 \
  --adaptive_noise False
```

### Adaptive-Noise PSN Example

```bash
cd psn_rlvr
bash train_grpo_math_tune_ray.sh \
  --adaptive_noise True \
  --adaptive_noise_initial_sigma 0.004 \
  --noise_target_kl 0.003 \
  --adaptive_noise_coeff 1.01 \
  --adaptive_noise_min_sigma 0.000001 \
  --adaptive_noise_max_sigma 0.5 \
  --tis_imp_ratio_cap 10 \
  --param_noise_base_seed 42
```

## Important Runtime Notes

- The launcher currently assumes local/offline-style paths for data/models/logs/checkpoints (see env vars in `train_grpo_math_tune_ray.sh`).
- Required extra packages installed by the launcher: `math_verify`, `tensordict`.
- Rollout logprobs are automatically enabled when adaptive noise is on or when TIS cap is non-zero.

## TODO

- [x] Support fixed parameter noise (`noise_sigma`).
- [x] Support adaptive noise Variant I (target-KL style sigma controller).
- [ ] Support adaptive noise Variant II (real-time scheduler using semantic diversity + self-certainty probes).
- [ ] Add a dedicated launcher/config preset for Variant II once implemented.

## Citation

If this implementation helps your research, please cite:

```bibtex
@article{bai2026psnrlvr,
  title   = {Learning to Explore with Parameter-Space Noise: A Deep Dive into Parameter-Space Noise for Reinforcement Learning with Verifiable Rewards},
  author  = {Bai, Bizhe and Wang, Xinyue and Ye, Peng and Chen, Tao},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.02555},
  url     = {https://arxiv.org/pdf/2602.02555}
}
```
