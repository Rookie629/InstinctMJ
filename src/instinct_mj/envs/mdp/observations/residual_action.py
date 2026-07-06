"""Observation term that computes the normalised reference action for HDMI-style
residual policy.

Core formula
------------
  ref_action = (q_ref - q_default) / action_scale

where:
  q_ref        : reference joint position from motion dataset (rad)
  q_default    : default robot joint position (rad)
  action_scale : per-joint action scale (rad)
  ref_action   : normalised reference action (unitless)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv as ManagerBasedEnv

    from instinct_mj.motion_reference.motion_reference_manager import MotionReferenceManager


class ref_action_as_state(ManagerTermBase):
    """Compute normalised reference action for HDMI residual policy.

    Observation term that converts the reference joint position ``q_ref`` into
    the normalised action space:

        ref_action = (q_ref - q_default) / action_scale

    The result is added to the policy observation group so that the
    ``ResidualActorCritic`` can extract it and compose the residual action.

    Config parameters
    -----------------
    motion_asset_cfg : SceneEntityCfg
        Entity configuration for the motion reference sensor.
        Default: ``SceneEntityCfg("motion_reference")``.
    robot_cfg : SceneEntityCfg
        Entity configuration for the robot.
        Default: ``SceneEntityCfg("robot")``.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(env)

        motion_asset_cfg: SceneEntityCfg = cfg.params.get(
            "motion_asset_cfg", SceneEntityCfg("motion_reference")
        )
        robot_cfg: SceneEntityCfg = cfg.params.get(
            "robot_cfg", SceneEntityCfg("robot")
        )

        self._motion_reference: MotionReferenceManager = env.scene[motion_asset_cfg.name]
        self._robot = env.scene[robot_cfg.name]

        # Resolve the joint-position action term to obtain per-joint scale and
        # the mapping from action-dimension indices to MuJoCo joint indices.
        joint_pos_action = env.action_manager.get_term("joint_pos")
        self._action_scale: torch.Tensor = joint_pos_action.scale  # (num_envs, action_dim)
        self._target_ids: torch.Tensor = joint_pos_action.target_ids  # (action_dim,)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
    ) -> torch.Tensor:
        """Return ``ref_action``, shape ``(num_envs, action_dim)``."""
        # q_ref: reference joint positions for the actuated joints (rad).
        # motion_reference.reference_frame.joint_pos: (num_envs, 1, num_joints_mjcf)
        q_ref = self._motion_reference.reference_frame.joint_pos[
            :, 0, self._target_ids
        ]  # (num_envs, action_dim)

        # Apply joint-position mask: zero out invalid reference joints.
        joint_mask = self._motion_reference.reference_frame.joint_pos_mask[
            :, 0, self._target_ids
        ]  # (num_envs, action_dim)
        q_ref = q_ref * joint_mask

        # q_default: default joint positions for actuated joints (rad).
        q_default = self._robot.data.default_joint_pos[
            :, self._target_ids
        ]  # (num_envs, action_dim)

        # action_scale: per-joint action scale (rad).
        # self._action_scale has shape (num_envs, action_dim).
        action_scale = self._action_scale

        # HDMI formula: ref_action = (q_ref - q_default) / action_scale
        ref_action = (q_ref - q_default) / action_scale.clamp_min(1e-8)

        return ref_action  # (num_envs, action_dim)
