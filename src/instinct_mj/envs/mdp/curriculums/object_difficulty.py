"""Object-difficulty curriculum terms for domain randomization.

These curriculum classes anneal object variant / scale / precision parameters
over the course of training, gradually increasing the difficulty.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import CurriculumTermCfg


class ObjectAlphaCurriculum:
    """Linearly anneal object alpha from *initial_alpha* to *final_alpha*.

    Alpha = 1.0 is the canonical / easiest reference shape; alpha = 0.0
    is the hardest morph.  The current value is stored on
    ``env._object_variant_current_alpha`` and consumed by the variant
    selector at the next reset.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self._env = env
        self.initial_alpha = float(cfg.params.get("initial_alpha", 1.0))
        self.final_alpha = float(cfg.params.get("final_alpha", 0.0))

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: Sequence[int],
        initial_alpha: float = 1.0,
        final_alpha: float = 0.0,
        start_step: int = 0,
        end_step: int = 3_000_000,
    ) -> dict[str, float]:
        del env_ids
        initial_alpha = self.initial_alpha if initial_alpha == 1.0 else float(initial_alpha)
        final_alpha = self.final_alpha if final_alpha == 0.0 else float(final_alpha)
        if end_step <= start_step:
            progress = 1.0 if env.common_step_counter >= start_step else 0.0
        else:
            progress = (float(env.common_step_counter) - float(start_step)) / float(
                end_step - start_step
            )
            progress = min(max(progress, 0.0), 1.0)
        alpha = initial_alpha + progress * (final_alpha - initial_alpha)
        env._object_variant_current_alpha = float(alpha)
        # Also set on Part2Link-compatible attribute
        env._part2link_object_current_alpha = float(alpha)
        return {"object_alpha_progress": progress, "object_alpha": float(alpha)}


class ObjectScaleCurriculum:
    """Gradually widen the active-scale sampling range.

    Early training: scale close to 1.0 (easier).  Later training: wider
    range (harder).  The updated bounds are stored on
    ``env._object_variant_scale_range`` and should be read by the reset
    event to override ``scale_distribution_params``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self._env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: Sequence[int],
        initial_scale_range: tuple[float, float] = (0.95, 1.05),
        final_scale_range: tuple[float, float] = (0.8, 1.2),
        start_step: int = 0,
        end_step: int = 1_000_000,
    ) -> dict[str, float]:
        del env_ids
        if end_step <= start_step:
            progress = 1.0 if env.common_step_counter >= start_step else 0.0
        else:
            progress = (float(env.common_step_counter) - float(start_step)) / float(
                end_step - start_step
            )
            progress = min(max(progress, 0.0), 1.0)
        lo = initial_scale_range[0] + progress * (final_scale_range[0] - initial_scale_range[0])
        hi = initial_scale_range[1] + progress * (final_scale_range[1] - initial_scale_range[1])
        env._object_variant_scale_range = (float(lo), float(hi))
        return {"object_scale_lo": float(lo), "object_scale_hi": float(hi)}


class ObjectPrecisionScaleCurriculum:
    """Gradually tighten the precision-scale sampling range.

    Early training: generous tolerance.  Later training: strict.
    The updated bounds are stored on ``env._object_variant_precision_range``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self._env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: Sequence[int],
        initial_precision_range: tuple[float, float] = (1.0, 1.6),
        final_precision_range: tuple[float, float] = (0.8, 1.4),
        start_step: int = 0,
        end_step: int = 1_000_000,
    ) -> dict[str, float]:
        del env_ids
        if end_step <= start_step:
            progress = 1.0 if env.common_step_counter >= start_step else 0.0
        else:
            progress = (float(env.common_step_counter) - float(start_step)) / float(
                end_step - start_step
            )
            progress = min(max(progress, 0.0), 1.0)
        lo = initial_precision_range[0] + progress * (
            final_precision_range[0] - initial_precision_range[0]
        )
        hi = initial_precision_range[1] + progress * (
            final_precision_range[1] - initial_precision_range[1]
        )
        env._object_variant_precision_range = (float(lo), float(hi))
        return {"object_precision_lo": float(lo), "object_precision_hi": float(hi)}


class ObjectVariantCountCurriculum:
    """Gradually introduce more variants during training.

    Early: only the canonical variant (index 0 per type) is available.
    Late: all variants are available for random selection.
    The current count is stored on ``env._object_variant_active_count``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self._env = env

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: Sequence[int],
        initial_count: int = 1,
        final_count: int = 5,
        start_step: int = 0,
        end_step: int = 2_000_000,
    ) -> dict[str, float]:
        del env_ids
        if end_step <= start_step:
            progress = 1.0 if env.common_step_counter >= start_step else 0.0
        else:
            progress = (float(env.common_step_counter) - float(start_step)) / float(
                end_step - start_step
            )
            progress = min(max(progress, 0.0), 1.0)
        count = int(initial_count + progress * (final_count - initial_count))
        count = max(count, 1)
        env._object_variant_active_count = count
        return {"object_variant_count": count}
