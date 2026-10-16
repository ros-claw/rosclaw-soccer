"""Non-neural, initial-state conditional SIM teacher research prototype.

Fits a fixed nearest-neighbour choice from complete measured control matrices.
Training outcomes are labels, never query inputs. Training fit is not physical
teacher qualification, calibrated success probability or unseen validation.
Callers must authenticate receipts and bind body, parent, world, initial robot
state and outcome contract into execution_binding_hash before using this model.
"""

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Self

from rosclaw_soccer.sim.contracts import hash_json

BallContext = tuple[float, float, float]
CONTEXT_SCALES = (0.6, 0.32, 1.4)
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def _hash(value: str) -> None:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError("content-bound execution/options/evidence required")


def _context(value: BallContext) -> None:
    if (
        type(value) is not tuple
        or len(value) != 3
        or any(type(v) is not float or not math.isfinite(v) for v in value)
        or not 0.0 <= value[0] <= 10.0
        or abs(value[1]) > 2.0
        or abs(value[2]) > 3.0
    ):
        raise ValueError("finite initial measured world-relative ball x/y and world vx required")


@dataclass(frozen=True, slots=True)
class ControlOutcome:
    safe: bool
    task_pass: bool
    training_reward: float

    def __post_init__(self) -> None:
        if (
            type(self.safe) is not bool
            or type(self.task_pass) is not bool
            or (self.task_pass and not self.safe)
            or type(self.training_reward) is not float
            or not math.isfinite(self.training_reward)
            or abs(self.training_reward) > 100.0
        ):
            raise ValueError("finite primitive measured control outcomes required")


@dataclass(frozen=True, slots=True)
class ControlCase:
    context: BallContext
    outcomes: tuple[ControlOutcome, ...]
    evidence_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        _context(self.context)
        if (
            type(self.outcomes) is not tuple
            or type(self.evidence_hashes) is not tuple
            or not 2 <= len(self.outcomes) <= 16
            or len(self.outcomes) != len(self.evidence_hashes)
        ):
            raise ValueError("complete bounded per-context option matrix required")
        for outcome in self.outcomes:
            if type(outcome) is not ControlOutcome:
                raise ValueError("typed physical outcomes required")
            outcome.__post_init__()
        for value in self.evidence_hashes:
            _hash(value)


@dataclass(frozen=True, slots=True)
class InitialContextControlPortfolio:
    cases: tuple[ControlCase, ...]
    option_hashes: tuple[str, ...]
    execution_binding_hash: str
    partition: str = "consumed_public_research"

    def __post_init__(self) -> None:
        _hash(self.execution_binding_hash)
        if (
            type(self.partition) is not str
            or self.partition != "consumed_public_research"
            or type(self.cases) is not tuple
            or not 4 <= len(self.cases) <= 160
            or type(self.option_hashes) is not tuple
            or not 2 <= len(self.option_hashes) <= 16
        ):
            raise ValueError("bounded consumed-public multistate fit only; no Fresh admission")
        for option in self.option_hashes:
            _hash(option)
        if len(set(self.option_hashes)) != len(self.option_hashes):
            raise ValueError("distinct content-bound control options required")
        evidence: list[str] = []
        for case in self.cases:
            if type(case) is not ControlCase:
                raise ValueError("typed complete control matrix required")
            case.__post_init__()
            if len(case.outcomes) != len(self.option_hashes):
                raise ValueError("every declared option required at every training context")
            evidence.extend(case.evidence_hashes)
        if len({case.context for case in self.cases}) != len(self.cases):
            raise ValueError("repeated initial contexts are not independent training cases")
        if len(set(evidence)) != len(evidence):
            raise ValueError("replayed evidence cannot inflate the training matrix")

    @classmethod
    def fit(
        cls,
        *,
        cases: tuple[ControlCase, ...],
        option_hashes: tuple[str, ...],
        execution_binding_hash: str,
        partition: str = "consumed_public_research",
    ) -> Self:
        """Fixed non-neural fit, without hyperparameter or checkpoint selection."""
        return cls(cases, option_hashes, execution_binding_hash, partition)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        """Reload the complete fixed JSON contract, not pickle or partial weights."""
        if (
            type(raw) is not dict
            or type(raw.get("cases")) is not list
            or not 4 <= len(raw["cases"]) <= 160
            or type(raw.get("option_hashes")) is not list
            or not 2 <= len(raw["option_hashes"]) <= 16
        ):
            raise ValueError("complete bounded JSON portfolio required")
        for case in raw["cases"]:
            if (
                type(case) is not dict
                or set(case) != {"context", "outcomes", "evidence_hashes"}
                or type(case["context"]) is not list
                or len(case["context"]) != 3
                or type(case["outcomes"]) is not list
                or len(case["outcomes"]) != len(raw["option_hashes"])
                or type(case["evidence_hashes"]) is not list
                or len(case["evidence_hashes"]) != len(raw["option_hashes"])
            ):
                raise ValueError("bounded complete JSON case rows required before fit allocation")
        try:
            cases = tuple(
                ControlCase(
                    tuple(case["context"]),
                    tuple(ControlOutcome(**outcome) for outcome in case["outcomes"]),
                    tuple(case["evidence_hashes"]),
                )
                for case in raw["cases"]
            )
            result = cls.fit(
                cases=cases,
                option_hashes=tuple(raw["option_hashes"]),
                execution_binding_hash=raw["execution_binding_hash"],
                partition=raw["partition"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError("complete typed JSON portfolio required") from error
        if hash_json(result.to_dict()) != hash_json(raw):
            raise ValueError("fixed portfolio algorithm/claims/fit differ from complete contract")
        return result

    @property
    def selected_options(self) -> tuple[int | None, ...]:
        selected = []
        for case in self.cases:
            safe = [i for i, outcome in enumerate(case.outcomes) if outcome.safe]
            selected.append(
                max(
                    safe,
                    key=lambda i: (
                        case.outcomes[i].task_pass,
                        case.outcomes[i].training_reward,
                        -i,
                    ),
                )
                if safe
                else None
            )
        return tuple(selected)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            schema="soccer.rsi.initial_context_control_portfolio.v1",
            **asdict(self),
            selected_options=self.selected_options,
            algorithm="fixed_one_nearest_measured_context_with_max_coordinate_support",
            context_fields=("ball_minus_root_x_m", "ball_minus_root_y_m", "ball_world_vx_m_s"),
            context_scales=CONTEXT_SCALES,
            support_max_coordinate_radius=1.0,
            outcomes_are_training_labels_only=True,
            input_evidence_authenticated_here=False,
            non_neural_supervised_fit=True,
            neural_optimizer_updates=0,
            teacher_qualified=False,
            unseen_generalization_proven=False,
            calibrated_probability_model=False,
            activation_ceiling="SIM_ONLY",
            promotion_authorized=False,
            hardware_authorized=False,
        )

    @property
    def model_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def query(self, context: BallContext, *, execution_binding_hash: str) -> dict[str, Any]:
        _context(context)
        _hash(execution_binding_hash)
        result: dict[str, Any] = dict(
            model_hash=self.model_hash,
            status="UNKNOWN",
            option_index=None,
            option_hash=None,
            training_context_match=False,
            calibrated_probability_model=False,
            unseen_generalization_proven=False,
            teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        if execution_binding_hash != self.execution_binding_hash:
            return dict(result, reason="execution_binding_mismatch")
        neighbours = []
        for index, case in enumerate(self.cases):
            delta = tuple(
                (a - b) / scale
                for a, b, scale in zip(context, case.context, CONTEXT_SCALES, strict=True)
            )
            if max(abs(d) for d in delta) <= 1.0:
                neighbours.append((sum(d * d for d in delta), case.context, index))
        if not neighbours:
            return dict(result, reason="outside_measured_context_support")
        distance, _, index = min(neighbours)
        chosen = self.selected_options[index]
        if chosen is None:
            return dict(result, reason="nearest_training_context_has_no_safe_option")
        return dict(
            result,
            status="SIM_RESEARCH_OPTION",
            reason="unqualified_neighbour_choice_not_success_probability",
            option_index=chosen,
            option_hash=self.option_hashes[chosen],
            training_context_match=context == self.cases[index].context,
            normalized_squared_distance=distance,
            source_training_evidence_hash=self.cases[index].evidence_hashes[chosen],
        )
