from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .amass_motion_cfg import AmassMotionCfg
from .part2link_motion import Part2LinkMotion


@dataclass(kw_only=True)
class Part2LinkMotionCfg(AmassMotionCfg):
    """Configuration for G1 sitting Part2Link object-interaction motion data."""

    class_type: type = Part2LinkMotion

    supported_file_endings: Sequence[str] = field(default_factory=lambda: ["retargeted.npz", "retargetted.npz"])
    metadata_yaml: str | None = None
    object_key: str = "box"
    object_velocity_estimation_method: str | None = "frontbackward"
