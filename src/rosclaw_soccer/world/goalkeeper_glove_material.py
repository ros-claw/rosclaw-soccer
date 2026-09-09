"""Explicit, shared SIM_ONLY glove-contact contract; never a policy parameter.

The historical three-player runner used an explicit contact material and
wrist-limit margin. This contract makes that assumption available to both
teams in the shared pitch without changing glove dimensions or ball state.
It is not a measured real-world latex material or evidence of policy learning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoalkeeperGloveMaterial:
    time_constant_sec: float = 0.02
    damping_ratio: float = 1.0
    joint_guard_margin_rad: float = 0.08

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.time_constant_sec)
            and 0.004 <= self.time_constant_sec <= 0.04
            and math.isfinite(self.damping_ratio)
            and 0.10 <= self.damping_ratio <= 2.0
            and math.isfinite(self.joint_guard_margin_rad)
            and 0.04 <= self.joint_guard_margin_rad <= 0.12
        ):
            raise ValueError("glove material is outside its explicit simulation envelope")

    def apply(self, model: Any, *, prefix: str) -> None:
        """Configure a model before simulation; resolve all targets before writes."""
        if not re.fullmatch(r"[a-zA-Z0-9_]*", prefix):
            raise ValueError("invalid goalkeeper model prefix")
        geoms = [model.geom(prefix + side + "_goalkeeper_glove").id for side in ("left", "right")]
        joints = [
            model.joint(prefix + side + "_wrist_pitch_joint").id for side in ("left", "right")
        ]
        for geom, joint in zip(geoms, joints, strict=True):
            model.geom_solref[geom] = (self.time_constant_sec, self.damping_ratio)
            model.geom_priority[geom] = 1
            if self.time_constant_sec != 0.02 or self.damping_ratio != 1.0:
                model.jnt_margin[joint] = max(0.04, self.joint_guard_margin_rad)
                model.jnt_solref[joint] = (0.003, 1.0)
