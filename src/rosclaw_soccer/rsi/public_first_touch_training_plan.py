"""Declared balanced public training draws, never private exam authority.

This binds public generator inputs, actual behavior identity and exploration
draws before collection. A declaration is not evidence: callers still need
complete physical replay, policy-density reconstruction and dataset admission.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.rsi.balanced_first_touch_curriculum import balanced_public_first_touch_jobs
from rosclaw_soccer.sim.contracts import hash_json


def _hash(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(c in "0123456789abcdef" for c in value[7:])
    )


@dataclass(frozen=True, slots=True)
class PublicTrainingDraw:
    group: int
    context_id: int
    sample_index: int
    seed: int
    lane: int
    course: tuple[float, float, float]
    sampling_seed: int


@dataclass(frozen=True, slots=True)
class PublicFirstTouchTrainingPlan:
    seeds: tuple[int, ...]
    behavior_model_hash: str
    stream: int

    def __post_init__(self) -> None:
        # V1 deliberately fixes four independent draws per context. No caller
        # can shrink the declared bank after seeing failures.
        balanced_public_first_touch_jobs(self.seeds, repeats=4)
        if (
            not _hash(self.behavior_model_hash)
            or type(self.stream) is not int
            or not 0 <= self.stream < 32
        ):
            raise ValueError("exact behavior hash and bounded public exploration stream required")

    def draws(self) -> tuple[PublicTrainingDraw, ...]:
        jobs = balanced_public_first_touch_jobs(self.seeds, repeats=4)
        draws = []
        for job in jobs:
            # A dedicated domain avoids silently inheriting consumed-bank
            # streams. 32-bit collision is checked for the complete declaration.
            payload = (
                f"soccer.rsi.public_training.v1:{self.stream}:{job.seed}:{job.lane}:{job.repeat}"
            )
            sample_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
            draws.append(
                PublicTrainingDraw(
                    job.group,
                    job.group // 4,
                    job.repeat,
                    job.seed,
                    job.lane,
                    job.course,
                    sample_seed,
                )
            )
        if len({d.sampling_seed for d in draws}) != len(draws):
            raise ValueError("complete public draw declaration contains an exploration collision")
        return tuple(draws)

    def contract(self) -> dict[str, Any]:
        draws = self.draws()
        result = dict(
            schema="soccer.rsi.public_first_touch_training_plan.v1",
            partition="DECLARED_PUBLIC_TRAINING",
            seeds=list(self.seeds),
            stream=self.stream,
            behavior_model_hash=self.behavior_model_hash,
            jobs=[asdict(d) for d in draws],
            episodes=len(draws),
            contexts=len(draws) // 4,
            draws_per_context=4,
            four_x_direction_strata_counts=[len(draws) // 4] * 4,
            approach_mode="legacy",
            swing_mode="legacy",
            all_success_and_failure_trajectories_required=True,
            private_pool_or_labels_read=False,
            global_ancestry_separation_proven=False,
            physics_or_behavior_density_validated=False,
            training_data_admitted=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        return dict(result, plan_hash=hash_json(result))

    def require_draw(
        self, *, group: int | None, seed: int, lane: int, course: tuple[float, float, float]
    ) -> PublicTrainingDraw:
        if type(group) is not int or not 0 <= group < len(self.seeds) * 16:
            raise ValueError("bounded declared public training group required")
        draw = self.draws()[group]
        if (
            type(seed) is not int
            or type(lane) is not int
            or type(course) is not tuple
            or any(type(v) is not float for v in course)
            or (seed, lane, course) != (draw.seed, draw.lane, draw.course)
        ):
            raise ValueError("exact predeclared public course and identity required")
        return draw

    def require_sampling_view(self, draw: PublicTrainingDraw, view: Any) -> None:
        if (
            type(draw) is not PublicTrainingDraw
            or draw not in self.draws()
            or type(view) is not dict
            or view.get("schema") != "soccer.rsi.recurrent_motor_sampling.v1"
            or type(view.get("seed")) is not int
            or view["seed"] != draw.sampling_seed
            or type(view.get("mean_model")) is not dict
            or view["mean_model"].get("model_hash") != self.behavior_model_hash
        ):
            raise ValueError("exact declared current mean and independent public draw required")
