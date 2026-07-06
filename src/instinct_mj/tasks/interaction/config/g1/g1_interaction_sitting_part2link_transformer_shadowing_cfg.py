from __future__ import annotations

from dataclasses import dataclass

from .g1_interaction_sitting_part2link_shadowing_cfg import (
    G1InteractionSittingPart2LinkShadowingEnvCfg,
    G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY,
)


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkTransformerEnvCfg(G1InteractionSittingPart2LinkShadowingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.run_name += "_TransformerPolicy"


@dataclass(kw_only=True)
class G1InteractionSittingPart2LinkTransformerEnvCfg_PLAY(G1InteractionSittingPart2LinkShadowingEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.run_name += "_TransformerPolicy"

