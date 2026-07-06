"""Register G1 interaction sitting Part2Link tasks."""

from instinct_mj.tasks.registry import register_instinct_task

from .rl_cfgs import (
    g1_interaction_sitting_part2link_depth_instinct_rl_cfg,
    g1_interaction_sitting_part2link_depth_residual_instinct_rl_cfg,
    g1_interaction_sitting_part2link_transformer_instinct_rl_cfg,
    g1_interaction_sitting_part2link_transformer_residual_instinct_rl_cfg,
)


def _part2link_env_cfg():
    from .g1_interaction_sitting_part2link_shadowing_cfg import G1InteractionSittingPart2LinkShadowingEnvCfg

    return G1InteractionSittingPart2LinkShadowingEnvCfg()


def _part2link_play_env_cfg():
    from .g1_interaction_sitting_part2link_shadowing_cfg import G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY

    return G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY()


def _part2link_transformer_env_cfg():
    from .g1_interaction_sitting_part2link_transformer_shadowing_cfg import (
        G1InteractionSittingPart2LinkTransformerEnvCfg,
    )

    return G1InteractionSittingPart2LinkTransformerEnvCfg()


def _part2link_transformer_play_env_cfg():
    from .g1_interaction_sitting_part2link_transformer_shadowing_cfg import (
        G1InteractionSittingPart2LinkTransformerEnvCfg_PLAY,
    )

    return G1InteractionSittingPart2LinkTransformerEnvCfg_PLAY()


# ---------------------------------------------------------------------------
# Residual env-cfg factories
# ---------------------------------------------------------------------------


def _part2link_residual_env_cfg():
    from .g1_interaction_sitting_part2link_shadowing_residual_cfg import (
        G1InteractionSittingPart2LinkShadowingResidualEnvCfg,
    )

    return G1InteractionSittingPart2LinkShadowingResidualEnvCfg()


def _part2link_residual_play_env_cfg():
    from .g1_interaction_sitting_part2link_shadowing_residual_cfg import (
        G1InteractionSittingPart2LinkShadowingResidualEnvCfg_PLAY,
    )

    return G1InteractionSittingPart2LinkShadowingResidualEnvCfg_PLAY()


def _part2link_transformer_residual_env_cfg():
    from .g1_interaction_sitting_part2link_shadowing_residual_cfg import (
        G1InteractionSittingPart2LinkTransformerResidualEnvCfg,
    )

    return G1InteractionSittingPart2LinkTransformerResidualEnvCfg()


def _part2link_transformer_residual_play_env_cfg():
    from .g1_interaction_sitting_part2link_shadowing_residual_cfg import (
        G1InteractionSittingPart2LinkTransformerResidualEnvCfg_PLAY,
    )

    return G1InteractionSittingPart2LinkTransformerResidualEnvCfg_PLAY()


# ===================================================================
# Original tasks (no residual)
# ===================================================================

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-G1-v0",
    env_cfg_factory=_part2link_env_cfg,
    play_env_cfg_factory=_part2link_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_depth_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-G1-Play-v0",
    env_cfg_factory=_part2link_play_env_cfg,
    play_env_cfg_factory=_part2link_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_depth_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-Transformer-G1-v0",
    env_cfg_factory=_part2link_transformer_env_cfg,
    play_env_cfg_factory=_part2link_transformer_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_transformer_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Play-v0",
    env_cfg_factory=_part2link_transformer_play_env_cfg,
    play_env_cfg_factory=_part2link_transformer_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_transformer_instinct_rl_cfg,
)

# ===================================================================
# HDMI residual tasks
# ===================================================================

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-G1-Residual-v0",
    env_cfg_factory=_part2link_residual_env_cfg,
    play_env_cfg_factory=_part2link_residual_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_depth_residual_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-G1-Residual-Play-v0",
    env_cfg_factory=_part2link_residual_play_env_cfg,
    play_env_cfg_factory=_part2link_residual_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_depth_residual_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Residual-v0",
    env_cfg_factory=_part2link_transformer_residual_env_cfg,
    play_env_cfg_factory=_part2link_transformer_residual_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_transformer_residual_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Interaction-Sitting-Part2Link-Transformer-G1-Residual-Play-v0",
    env_cfg_factory=_part2link_transformer_residual_play_env_cfg,
    play_env_cfg_factory=_part2link_transformer_residual_play_env_cfg,
    instinct_rl_cfg_factory=g1_interaction_sitting_part2link_transformer_residual_instinct_rl_cfg,
)

