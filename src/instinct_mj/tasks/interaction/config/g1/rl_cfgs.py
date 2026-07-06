"""Instinct-RL configs for G1 sitting Part2Link interaction tasks."""

from __future__ import annotations

import os

from instinct_mj.rl import (
    InstinctRlActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)
from instinct_mj.tasks.config.rl_utils import default_policy_critic_normalizers


def _depth_encoder_cfg() -> dict[str, dict]:
    return {
        "depth_image": {
            "class_name": "Conv2dHeadModel",
            "component_names": ["depth_image"],
            "output_size": 32,
            "takeout_input_components": True,
            "channels": [32, 32],
            "kernel_sizes": [3, 3],
            "strides": [1, 1],
            "hidden_sizes": [32],
            "paddings": [1, 1],
            "nonlinearity": "ReLU",
            "use_maxpool": False,
        }
    }


def _motion_ref_transformer_cfg() -> dict[str, dict]:
    return {
        "motion_ref": {
            "class_name": "TransformerHeadModel",
            "component_names": [
                "joint_pos_ref",
                "joint_vel_ref",
                "position_ref",
                "rotation_ref",
            ],
            "output_size": 128,
            "takeout_input_components": True,
            "num_heads": 4,
            "num_layers": 2,
            "d_model": 128,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "activation": "gelu",
            "nonlinearity": "CELU",
            "output_selection": "maxpool",
        },
        **_depth_encoder_cfg(),
    }


def _critic_motion_ref_transformer_cfg() -> dict[str, dict]:
    return {
        "motion_ref": {
            "class_name": "TransformerHeadModel",
            "component_names": [
                "joint_pos_ref",
                "joint_vel_ref",
                "position_ref",
            ],
            "output_size": 128,
            "takeout_input_components": True,
            "num_heads": 4,
            "num_layers": 2,
            "d_model": 128,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "activation": "gelu",
            "nonlinearity": "CELU",
            "output_selection": "maxpool",
        },
    }


def g1_interaction_sitting_part2link_depth_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    run_name = "".join(
        [
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
        ]
    )
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="EncoderActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_configs=_depth_encoder_cfg(),
            critic_encoder_configs=None,
        ),
        algorithm=InstinctRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=500,
        log_interval=10,
        experiment_name="g1_interaction_part2link",
        run_name=run_name,
        resume=False,
        load_run=None,
        policy_observation_group="policy",
        critic_observation_group="critic",
    )


def g1_interaction_sitting_part2link_transformer_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    run_name = "".join(
        [
            "_DepthMotionRefTransformerPolicy",
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
        ]
    )
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="EncoderActorCritic",
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_configs=_motion_ref_transformer_cfg(),
            critic_encoder_configs=_critic_motion_ref_transformer_cfg(),
        ),
        algorithm=InstinctRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=500,
        log_interval=10,
        experiment_name="g1_interaction_part2link_transformer",
        run_name=run_name,
        resume=False,
        load_run=None,
        policy_observation_group="policy",
        critic_observation_group="critic",
    )


# ---------------------------------------------------------------------------
# HDMI residual policy variants
# ---------------------------------------------------------------------------


def g1_interaction_sitting_part2link_depth_residual_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Depth-only encoder with HDMI residual action composition."""
    run_name = "".join(
        [
            "_HDMIResidual",
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
        ]
    )
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="instinct_mj.rl.residual_policy:ResidualEncoderActorCritic",
            residual_action_component="ref_action",
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_configs=_depth_encoder_cfg(),
            critic_encoder_configs=None,
        ),
        algorithm=InstinctRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=500,
        log_interval=10,
        experiment_name="g1_interaction_part2link_residual",
        run_name=run_name,
        resume=False,
        load_run=None,
        policy_observation_group="policy",
        critic_observation_group="critic",
    )


def g1_interaction_sitting_part2link_transformer_residual_instinct_rl_cfg() -> InstinctRlOnPolicyRunnerCfg:
    """Transformer encoder with HDMI residual action composition."""
    run_name = "".join(
        [
            "_DepthMotionRefTransformerPolicy_HDMIResidual",
            f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
        ]
    )
    return InstinctRlOnPolicyRunnerCfg(
        policy=InstinctRlActorCriticCfg(
            class_name="instinct_mj.rl.residual_policy:ResidualEncoderActorCritic",
            residual_action_component="ref_action",
            init_noise_std=1.0,
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_configs=_motion_ref_transformer_cfg(),
            critic_encoder_configs=_critic_motion_ref_transformer_cfg(),
        ),
        algorithm=InstinctRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        normalizers=default_policy_critic_normalizers(),
        num_steps_per_env=24,
        max_iterations=50_000,
        save_interval=500,
        log_interval=10,
        experiment_name="g1_interaction_part2link_transformer_residual",
        run_name=run_name,
        resume=False,
        load_run=None,
        policy_observation_group="policy",
        critic_observation_group="critic",
    )

