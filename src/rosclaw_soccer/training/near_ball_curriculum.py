"""Deterministic, bilateral role courses for private near-ball learning.

Course selection changes only initial conditions and high-level skill bindings;
the same contact physics and frozen locomotion foundation execute every role.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.training.active_team_probe import run_probe

ROLES = ("goalkeeper", "defender", "playmaker", "finisher")
TRAIN_OFFSETS = (-0.12, 0.0, 0.12, 0.04, -0.04)
EXAM_OFFSETS = (-0.06, 0.06)


@dataclass(frozen=True)
class RoleCourse:
    role: str
    blue: bool
    offset: float

    def __post_init__(self) -> None:
        if (
            self.role not in ROLES
            or type(self.blue) is not bool
            or not math.isfinite(self.offset)
            or abs(self.offset) > 0.2
        ):
            raise ValueError("bounded role course required")

    @property
    def key(self) -> str:
        return f"{'blue' if self.blue else 'red'}-{self.role}-{self.offset:+.2f}"


def training_courses(iteration: int) -> tuple[RoleCourse, ...]:
    if type(iteration) is not int or iteration < 0:
        raise ValueError("nonnegative curriculum iteration required")
    return tuple(
        RoleCourse(role, blue, TRAIN_OFFSETS[iteration % len(TRAIN_OFFSETS)])
        for role in ROLES
        for blue in (False, True)
    )


def examination_courses(*, strict_handoff: bool = False) -> tuple[RoleCourse, ...]:
    if type(strict_handoff) is not bool:
        raise ValueError("explicit handoff examination contract required")
    return tuple(
        RoleCourse(role, blue, offset)
        for role in ROLES
        for blue in (False, True)
        for offset in ((-0.10, 0.10) if strict_handoff else EXAM_OFFSETS)
    )


@dataclass(frozen=True)
class RoleRolloutJob:
    assets: Path
    destination: Path
    checkpoint: Path
    duration: float
    course: RoleCourse
    seed: int
    explore: bool
    strict_handoff: bool = False


def collect_role_course(job: RoleRolloutJob) -> str:
    run_probe(
        asset_root=job.assets,
        output=job.destination,
        active=True,
        four_vs_four=True,
        duration=job.duration,
        blue_kickoff=job.course.blue,
        kickoff_offset_m=job.course.offset,
        basic_ball_play=True,
        kickoff_role=job.course.role,
        anticipatory_contact=True,
        forward_receiver_lane=job.course.role == "playmaker",
        contact_policy=OwnedBallContactPolicy(),
        all_role_clearance=True,
        near_ball_policy=NearBallResidualPolicy.load(job.checkpoint),
        near_ball_seed=job.seed,
        near_ball_explore=job.explore,
        strict_receive_handoff=job.strict_handoff,
    )
    return str(job.destination / "probe.json")
