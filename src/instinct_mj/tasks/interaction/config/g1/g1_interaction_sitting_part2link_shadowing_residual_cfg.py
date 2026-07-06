"""HDMI residual-policy variants of the G1 Part2Link interaction task.

Adds ``ref_action`` to the policy observation group so that
``ResidualEncoderActorCritic`` can compose the HDMI-style residual
action (see ``docs/hdmi_residual_policy_implementation.md``).

Usage
-----
These configs are registered as separate tasks (``*-Residual-v0`` /
``*-Residual-Play-v0``) alongside the original non-residual tasks, so
both can be trained and compared side-by-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.managers import ObservationTermCfg as ObsTermCfg
from mjlab.managers import SceneEntityCfg

import instinct_mj.envs.mdp as instinct_mdp
from .g1_interaction_sitting_part2link_shadowing_cfg import (
    G1InteractionSittingPart2LinkShadowingEnvCfg,
    G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY,
)

# ---------------------------------------------------------------------------
# Shared helper: inject the ref_action observation term
# ---------------------------------------------------------------------------


def _add_ref_action_observation(env_cfg: G1InteractionSittingPart2LinkShadowingEnvCfg) -> None:
    """Register the HDMI residual observation term in a post-init hook.

    .. note::
        Do **not** add ``"ref_action"`` to any encoder ``component_names``
        — it must pass through raw (``ParallelLayer`` forwards unknown
        components as-is).
    """
    env_cfg.observations["policy"].terms["ref_action"] = ObsTermCfg(
        func=instinct_mdp.ref_action_as_state,
        params={
            "motion_asset_cfg": SceneEntityCfg("motion_reference"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


# ---------------------------------------------------------------------------
# Residual env configs
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkShadowingResidualEnvCfg(
    G1InteractionSittingPart2LinkShadowingEnvCfg
):
    """Depth-only Part2Link env config with HDMI residual action."""

    def __post_init__(self):
        super().__post_init__()
        _add_ref_action_observation(self)
        self.run_name += "_HDMIResidual"


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkShadowingResidualEnvCfg_PLAY(
    G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY
):
    """Play variant of the residual depth-only config."""

    def __post_init__(self):
        super().__post_init__()
        _add_ref_action_observation(self)
        self.run_name += "_HDMIResidual"


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkTransformerResidualEnvCfg(
    G1InteractionSittingPart2LinkShadowingEnvCfg
):
    """Transformer Part2Link env config with HDMI residual action."""

    def __post_init__(self):
        super().__post_init__()
        _add_ref_action_observation(self)
        self.run_name += "_TransformerPolicy_HDMIResidual"


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkTransformerResidualEnvCfg_PLAY(
    G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY
):
    """Play variant of the residual transformer config."""

    def __post_init__(self):
        super().__post_init__()
        _add_ref_action_observation(self)
        self.run_name += "_TransformerPolicy_HDMIResidual"
