"""HDMI-style residual joint-position policy.

Implements the residual action composition described in
docs/hdmi_residual_policy_implementation.md.

Core formula
------------
  q_target = q_ref + action_scale * delta_action

where:
  q_ref        : reference joint position from motion dataset (rad)
  q_target     : final PD joint-position target (rad)
  action_scale : per-joint action scaling (rad)
  delta_action : residual action predicted by the policy (normalized)

In normalized action space:
  ref_action   = (q_ref - q_default) / action_scale
  delta_action = policy(obs)
  final_action = ref_action + delta_action          ← residual composition
  q_target     = q_default + action_scale * final_action

The residual is added to the **distribution mean before sampling**, which
guarantees correct log-probability computation under the Gaussian policy.
"""

from __future__ import annotations

import torch
from torch.distributions import Normal

from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.modules.encoder_actor_critic import EncoderActorCriticMixin
from instinct_rl.utils.utils import get_obs_slice


class ResidualActorCritic(ActorCritic):
    """Actor-critic with HDMI-style residual action composition.

    The policy (MLP) predicts ``delta_action`` — a *residual* around the
    normalised reference action ``ref_action``.  The residual is added to the
    Gaussian mean / loc **before** sampling so that ``log_prob`` stays
    consistent with the distribution that was actually sampled.

    Parameters
    ----------
    residual_action_component : str
        Name of the observation component that carries the pre-computed
        normalised reference action ``ref_action``.  Defaults to ``"ref_action"``.
    """

    def __init__(self, *args, residual_action_component: str = "ref_action", **kwargs):
        self._residual_action_component = residual_action_component
        super().__init__(*args, **kwargs)

        # Snapshot the observation segments that the MLP *actually* sees.
        # When wrapped by EncoderActorCriticMixin (Transformer / Conv2d),
        # ``self.obs_segments`` may resolve to the *pre-encoder* layout via
        # MRO (EncoderActorCriticMixin redefines the property with its own
        # name-mangled ``__obs_segments``).  We need the post-encoder layout
        # stored by ActorCritic, so we grab it directly from the descriptor.
        self._mlp_obs_segments = ActorCritic.obs_segments.fget(self)

    # ------------------------------------------------------------------
    # Core override: inject the residual into the distribution mean
    # ------------------------------------------------------------------

    def update_distribution(self, observations: torch.Tensor):
        """Compute delta_mean, add ref_action, and create Normal distribution.

        This is the single method that guarantees the residual is added at the
        **mean** level before the distribution is constructed, so that
        ``log_prob`` is always computed under the residualized distribution.

        The MLP sees the **full** observation vector (including ``ref_action``)
        as context; its output is interpreted as the residual correction
        ``delta_action``.  The final action mean is::

            final_mean = ref_action + delta_mean
        """
        ref_action = self._get_ref_action(observations)

        # MLP predicts the residual correction in normalised action space.
        # It sees ref_action as informative context alongside other obs.
        delta_mean = self.actor(observations)

        # HDMI residual composition: final_mean = ref_action + delta_action.
        mean = ref_action + delta_mean
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    # ------------------------------------------------------------------
    # Inference (deterministic) – same residual logic
    # ------------------------------------------------------------------

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action for deployment / evaluation rollouts."""
        ref_action = self._get_ref_action(observations)
        delta_mean = self.actor(observations)
        return ref_action + delta_mean

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ref_action(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract ``ref_action`` from the observation vector.

        Uses the MLP-level observation segments (post-encoder when encoders
        are active) so that slicing matches the actual tensor layout.

        Returns
        -------
        ref_action : Tensor  shape (..., action_dim)
            The normalised reference action sliced from the full observation.
        """
        obs_slice, _shape = get_obs_slice(
            self._mlp_obs_segments, self._residual_action_component
        )
        return observations[..., obs_slice]


# ------------------------------------------------------------------
# Encoder-enabled variant (Transformer, Conv2d, …)
# ------------------------------------------------------------------


class ResidualEncoderActorCritic(EncoderActorCriticMixin, ResidualActorCritic):
    """Encoder-aware residual actor-critic.

    Combines ``EncoderActorCriticMixin`` (Transformer / Conv2d encoders) with
    ``ResidualActorCritic`` (HDMI-style residual composition).

    ``ref_action`` passes through the encoder stack unchanged because it is
    **not** listed in any encoder's ``component_names`` — ``ParallelLayer``
    forwards unknown components as-is.

    Usage in RL config
    ------------------
    .. code-block:: python

        InstinctRlActorCriticCfg(
            class_name="instinct_mj.rl.residual_policy:ResidualEncoderActorCritic",
            residual_action_component="ref_action",
            ...
        )
    """

    pass
