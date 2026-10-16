"""Public planning declarations and choice arithmetic, never a physics verifier.

The executor must authenticate all native receipts and persist selection before
evaluation. Typed role labels prevent accidental mixing, not hostile forgery.
No actor, private exam, simulation callback, transport or authority is exposed.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.rsi.initial_context_control_portfolio import ControlOutcome
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"sha256:[0-9a-f]{64}")
PERTURBATIONS = ((0.01, 0.9, 0.95), (0.02, 0.8, 1.05))


def _hash(value: str) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError("content-bound planning identity required")


@dataclass(frozen=True, slots=True)
class PublicPlanningContext:
    context_index: int
    source_group: int
    source_context_id: int
    source_seed: int
    source_lane: int
    sampling_seed: int
    course: tuple[float, float, float]
    perturbation_index: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    self.context_index,
                    self.source_group,
                    self.source_context_id,
                    self.source_seed,
                    self.source_lane,
                    self.sampling_seed,
                    self.perturbation_index,
                )
            )
            or not 0 <= self.context_index < 8
            or self.perturbation_index not in (0, 1)
            or self.source_seed >= 2**32
            or self.sampling_seed >= 2**32
            or self.source_lane >= 16
            or type(self.course) is not tuple
            or len(self.course) != 3
            or any(type(value) is not float or not math.isfinite(value) for value in self.course)
        ):
            raise ValueError("bounded explicit public planning context required")
        x, y, vx = self.course
        if not ((2.2 <= x <= 2.4 or 2.6 <= x <= 2.8) and abs(y) <= 0.16 and 0.25 <= abs(vx) <= 0.7):
            raise ValueError("perturbation leaves original public train domain; no clipping")

    @property
    def context_hash(self) -> str:
        return hash_json(
            dict(schema="soccer.rsi.public_physics_planning_context.v1", **asdict(self))
        )


def planning_contexts(
    plan: PublicFirstTouchTrainingPlan,
    *,
    base_groups: tuple[int, ...] = (64, 68, 72, 76),
) -> tuple[PublicPlanningContext, ...]:
    """Two fixed perturbations per stratum; fail the whole plan before allocation."""
    if type(plan) is not PublicFirstTouchTrainingPlan:
        raise ValueError("exact public source plan required")
    plan.__post_init__()
    draws = plan.draws()
    if (
        type(base_groups) is not tuple
        or len(base_groups) != 4
        or any(
            type(group) is not int or group < 0 or group % 4 or group + 1 >= len(draws)
            for group in base_groups
        )
        or len(set(base_groups)) != 4
    ):
        raise ValueError("four distinct context-major public base groups required")
    contexts: list[PublicPlanningContext] = []
    strata = []
    for group in base_groups:
        for index, (dx, sy, sv) in enumerate(PERTURBATIONS):
            draw = draws[group + index]
            x, y, vx = draw.course
            context = PublicPlanningContext(
                len(contexts),
                draw.group,
                draw.context_id,
                draw.seed,
                draw.lane,
                draw.sampling_seed,
                (x + dx, y * sy, vx * sv),
                index,
            )
            contexts.append(context)
            strata.append((context.course[0] >= 2.6, context.course[2] > 0))
    if (
        any(
            strata.count(stratum) != 2
            for stratum in ((False, False), (False, True), (True, False), (True, True))
        )
        or len({c.course for c in contexts}) != 8
        or len({c.sampling_seed for c in contexts}) != 8
    ):
        raise ValueError("eight unique contexts/draws and complete four-stratum balance required")
    return tuple(contexts)


def validate_context_source(
    context: PublicPlanningContext, plan: PublicFirstTouchTrainingPlan, *, group: int
) -> None:
    """Bind a single derived course to its exact public source draw, without IO."""
    if type(context) is not PublicPlanningContext or type(plan) is not PublicFirstTouchTrainingPlan:
        raise ValueError("exact public planning context and source plan required")
    context.__post_init__()
    plan.__post_init__()
    draws = plan.draws()
    if type(group) is not int or group != context.source_group or not 0 <= group < len(draws):
        raise ValueError("planning source group mismatch")
    draw = draws[group]
    if draw.sample_index != context.perturbation_index:
        raise ValueError("fixed perturbation must bind its independent source draw")
    dx, sy, sv = PERTURBATIONS[context.perturbation_index]
    expected = PublicPlanningContext(
        context.context_index,
        draw.group,
        draw.context_id,
        draw.seed,
        draw.lane,
        draw.sampling_seed,
        (draw.course[0] + dx, draw.course[1] * sy, draw.course[2] * sv),
        context.perturbation_index,
    )
    if expected != context:
        raise ValueError("derived public context differs from the frozen perturbation")


@dataclass(frozen=True, slots=True)
class PlanningTrial:
    context_hash: str
    option_index: int
    option_hash: str
    attempt_hash: str
    receipt_hash: str
    binding_hash: str
    outcome: ControlOutcome
    role: str = "PLANNING_SIMULATION"

    def __post_init__(self) -> None:
        for value in (
            self.context_hash,
            self.option_hash,
            self.attempt_hash,
            self.receipt_hash,
            self.binding_hash,
        ):
            _hash(value)
        if (
            type(self.option_index) is not int
            or not 0 <= self.option_index < 16
            or type(self.outcome) is not ControlOutcome
            or type(self.role) is not str
            or self.role != "PLANNING_SIMULATION"
        ):
            raise ValueError("planning-only typed trial; evaluation outcomes are not inputs")
        self.outcome.__post_init__()


def choose_planned_option(
    trials: tuple[PlanningTrial, ...],
    *,
    context_hash: str,
    option_hashes: tuple[str, ...],
    binding_hash: str,
) -> dict[str, Any]:
    """Complete finite search arithmetic; caller MUST authenticate physical receipts."""
    _hash(context_hash)
    _hash(binding_hash)
    if type(option_hashes) is not tuple or not 2 <= len(option_hashes) <= 16:
        raise ValueError("bounded distinct predeclared option catalog required")
    for option in option_hashes:
        _hash(option)
    if len(set(option_hashes)) != len(option_hashes):
        raise ValueError("distinct planning options required")
    if type(trials) is not tuple or len(trials) != len(option_hashes):
        raise ValueError("every planning option, including failures, required")
    for index, trial in enumerate(trials):
        if type(trial) is not PlanningTrial:
            raise ValueError("exact planning trial required")
        trial.__post_init__()
        if (
            trial.option_index != index
            or trial.option_hash != option_hashes[index]
            or trial.context_hash != context_hash
            or trial.binding_hash != binding_hash
        ):
            raise ValueError("ordered complete context, option and execution binding required")
    if len({trial.attempt_hash for trial in trials}) != len(trials):
        raise ValueError("an attempt cannot count as multiple planning executions")
    feasible = [trial for trial in trials if trial.outcome.task_pass]
    chosen = (
        max(feasible, key=lambda trial: (trial.outcome.training_reward, -trial.option_index))
        if feasible
        else None
    )
    result = dict(
        schema="soccer.rsi.physics_planning_option_choice.v1",
        context_hash=context_hash,
        binding_hash=binding_hash,
        option_hashes=list(option_hashes),
        planning_trials=[asdict(trial) for trial in trials],
        planning_trial_count=len(trials),
        status="PLANNING_OPTION_SELECTED" if chosen else "PLANNING_NO_FEASIBLE_OPTION",
        option_index=chosen.option_index if chosen else None,
        option_hash=chosen.option_hash if chosen else None,
        planning_receipt_hash=chosen.receipt_hash if chosen else None,
        evaluated_real_task_outcomes_are_inputs=False,
        physical_receipts_authenticated_here=False,
        selection_requires_persistence_before_evaluation=True,
        teacher_qualified=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        activation_ceiling="SIM_ONLY",
    )
    return dict(result, selection_hash=hash_json(result))
