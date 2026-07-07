# Vibe Coding Log — HDMI Residual Policy

> **Date**: 2026-07-06
> **Branch**: main
> **Base commit**: `b2ed39d` (Align dependencies with mjlab 1.4.0)

## Summary

Implemented HDMI-style residual joint-position policy as described in
[`docs/hdmi_residual_policy_implementation.md`](../docs/hdmi_residual_policy_implementation.md).

The policy predicts a **residual** (`delta_action`) around the motion reference
joint position, rather than directly outputting absolute joint targets.

```
q_target = q_ref + action_scale * delta_action
```

Registered as **separate tasks** alongside the original non-residual tasks so
both can be trained and compared side-by-side.

---

## Files changed

### New files

| File | Purpose |
|------|---------|
| `src/instinct_mj/rl/residual_policy.py` | `ResidualActorCritic` + `ResidualEncoderActorCritic` — HDMI residual composition at distribution-mean level |
| `src/instinct_mj/envs/mdp/observations/residual_action.py` | `ref_action_as_state` — computes `ref_action = (q_ref - q_default) / action_scale` |
| `src/instinct_mj/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_residual_cfg.py` | 4 env config classes inheriting from base Part2Link, adding `ref_action` observation |

### Modified files

| File | Change |
|------|--------|
| `src/instinct_mj/rl/config.py` | Added `residual_action_component: str \| None` field to `InstinctRlActorCriticCfg` |
| `src/instinct_mj/rl/vecenv_wrapper.py` | Added `_log_residual_debug` for HDMI diagnostic logging |
| `src/instinct_mj/envs/mdp/observations/__init__.py` | Added `from .residual_action import *` |
| `src/instinct_mj/tasks/interaction/config/g1/rl_cfgs.py` | Added 2 residual RL config functions; original 2 unchanged |
| `src/instinct_mj/tasks/interaction/config/g1/__init__.py` | Registered 4 new residual tasks; original 4 unchanged |

---

## New tasks registered

### Original (unchanged, no residual)

| Task ID | Policy |
|---------|--------|
| `Instinct-Interaction-Sitting-Part2Link-G1-v0` | `EncoderActorCritic` |
| `Instinct-Interaction-Sitting-Part2Link-G1-Play-v0` | `EncoderActorCritic` |
| `Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0` | `EncoderActorCritic` |
| `Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Play-v0` | `EncoderActorCritic` |

### Residual (new)

| Task ID | Policy |
|---------|--------|
| `Instinct-Interaction-Sitting-Part2Link-G1-Residual-v0` | `ResidualEncoderActorCritic` |
| `Instinct-Interaction-Sitting-Part2Link-G1-Residual-Play-v0` | `ResidualEncoderActorCritic` |
| `Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Residual-v0` | `ResidualEncoderActorCritic` |
| `Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Residual-Play-v0` | `ResidualEncoderActorCritic` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Observation Pipeline                  │
│                                                         │
│  MotionReference ──→ q_ref                              │
│  Robot.default_joint_pos ──→ q_default                  │
│  ActionManager.scale ──→ action_scale                   │
│                                                         │
│  ref_action = (q_ref - q_default) / action_scale        │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────────────┐           │
│  │        Policy Observations               │           │
│  │  [joint_pos_ref, joint_vel_ref, ...,     │           │
│  │   depth_image, ..., ref_action]          │           │
│  └──────────────────────────────────────────┘           │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────────────┐           │
│  │     Encoder (ParallelLayer)              │           │
│  │  - Transformer/Conv2d → encoded features │           │
│  │  - ref_action → PASSTHROUGH (unchanged)  │           │
│  └──────────────────────────────────────────┘           │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────────────┐           │
│  │    ResidualActorCritic                   │           │
│  │                                          │           │
│  │  ref_action = extract(obs)               │           │
│  │  delta_mean = MLP(obs)     ← residual    │           │
│  │  final_mean = ref_action + delta_mean    │           │
│  │  dist = Normal(final_mean, std)          │           │
│  │  action = dist.sample()                  │           │
│  └──────────────────────────────────────────┘           │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────────────────┐           │
│  │    JointPositionAction                   │           │
│  │  q_target = q_default + scale * action   │           │
│  │  ≡ q_ref + scale * delta_action          │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Key design decisions

### 1. Residual at distribution-mean level

The residual is added to the Gaussian **mean before sampling**, not after.
This guarantees `log_prob` is always computed under the correct distribution.

```python
# ✓ Correct (this implementation)
final_mean = ref_action + delta_mean
dist = Normal(final_mean, std)
action = dist.sample()
log_prob = dist.log_prob(action)  # consistent

# ✗ Wrong (would break PPO)
action = dist.sample()
action = ref_action + action       # log_prob no longer matches
```

### 2. `ref_action` in observations, passthrough encoder

`ref_action` is an observation term in the `"policy"` group. It is **not**
listed in any encoder's `component_names`, so `ParallelLayer` forwards it
unchanged. The MLP sees it as informative context alongside other observations.

### 3. Separate task registration

Residual variants are registered as **separate tasks** (`*-Residual-v0`)
rather than replacing the originals. This enables A/B comparison.

### 4. Name-mangling workaround

`EncoderActorCriticMixin` redefines `obs_segments` with its own name-mangled
`__obs_segments` (pointing to raw input segments). `ResidualActorCritic`
stores a direct reference to `ActorCritic.obs_segments` (post-encoder layout)
via `_mlp_obs_segments` to ensure slicing matches the actual tensor layout.

---

## Debug logging

The `VecEnvWrapper` logs three diagnostic signals per step:

| Key | Meaning |
|-----|---------|
| `Step/mean_abs_final_action` | \|final_action\| (normalized) |
| `Step/mean_abs_ref_action` | \|ref_action\| from observations |
| `Step/q_target_minus_q_ref_abs` | \|q_target - q_ref\| (rad) — the learned residual magnitude |

---

## How to train

```bash
# Original (no residual)
uv run python -m instinct_mj.scripts.instinct_rl.train \
    Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0

# Residual
uv run python -m instinct_mj.scripts.instinct_rl.train \
    Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Residual-v0
```

---

## Related files (existing, unchanged)

- `docs/hdmi_residual_policy_implementation.md` — reference design doc
- `instinct_rl/modules/act_residual.py` — **non-functional stub** (`exit(-1)` in `__init__`); this implementation supersedes it
- `instinct_rl/modules/actor_critic.py` — base `ActorCritic` class
- `instinct_rl/modules/encoder_actor_critic.py` — `EncoderActorCriticMixin`
- `src/instinct_mj/assets/unitree_g1.py` — `beyondmimic_action_scale`

---

## 2026-07-06 #2 — Consolidate assets & data into `data/`

> **Branch**: contact

Moved robot assets and motion-reference datasets into a unified `data/`
directory at the project root.

### Directory structure

```
data/
├── assets/
│   └── unitree_g1/         # moved from src/instinct_mj/assets/resources/unitree_g1/
│       ├── meshes/         # 42MB STL meshes
│       ├── xml/            # MJCF XML
│       └── urdf/           # URDF variants
└── datasets/
    └── interaction/        # symlink → /home/yangke/KY/InstinctLab_interact/datasets/interaction/
        └── output_npz_29dof_with_object/  # 1.3GB motion data
```

### Files changed

| File | Change |
|------|--------|
| `src/instinct_mj/assets/resources/unitree_g1/` | **moved** → `data/assets/unitree_g1/` |
| `data/datasets/interaction` | **new symlink** → external dataset |
| `src/instinct_mj/assets/unitree_g1.py` | `G1_MJCF_PATH`, `G1_MESHES_DIR` → project-root-relative via `_DATA_DIR` |
| `src/instinct_mj/tasks/interaction/config/g1/g1_interaction_sitting_part2link_shadowing_cfg.py` | `PART2LINK_DATASET_ROOT` → `data/datasets/interaction/...` |
| `src/instinct_mj/scripts/prepare_part2link_assets.py` | `DEFAULT_DATASET_ROOT` → `data/datasets/interaction/...` |
| `src/instinct_mj/tasks/parkour/mjcf/g1_29dof_torsoBase_popsicle_with_shoe.xml` | `meshdir` → `../../../../data/assets/unitree_g1/meshes` |

### Path resolution pattern

All paths now resolve relative to the project root (computed from `__file__`):

```python
_PROJECT_ROOT = os.path.dirname(...)  # walk up from __file__
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", ...)
```

No hardcoded `/home/yangke/` paths remain in source.
