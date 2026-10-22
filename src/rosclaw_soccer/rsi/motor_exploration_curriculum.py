"""Fixed TRAIN-only on-policy curricula, not arbitrary unreviewed case lists."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MotorExplorationCurriculum:
    name: str
    protocol: str
    context_ids: tuple[int, ...]
    seeds_per_context: int
    audit_status: str

    @property
    def worlds(self) -> int:
        return len(self.context_ids) * self.seeds_per_context

    def seeds(self, generation: int) -> list[int]:
        if type(generation) is not int or not 0 <= generation < 100:
            raise ValueError("bounded preregistered exploration generation required")
        return [710220 + 1000 * generation + i for i in range(self.seeds_per_context)]


def motor_curriculum(name: str) -> MotorExplorationCurriculum:
    if name == "run2":
        return MotorExplorationCurriculum(
            name,
            "dual256.actual_AR1_calibration.TRAIN2x4.v1",
            (20, 21),
            4,
            "DUAL_AR1_ALL8_OWN_NATIVE_AND_SAMPLING_LAW_AUDIT_PASS_NOT_TRAINED",
        )
    if name == "retention-run64":
        return MotorExplorationCurriculum(
            name,
            "dual256.actual_AR1_curriculum.TRAIN4x16.v2",
            (4, 5, 20, 21),
            16,
            "DUAL_AR1_ALL64_OWN_NATIVE_AND_SAMPLING_LAW_AUDIT_PASS_NOT_TRAINED",
        )
    raise ValueError("explicit reviewed motor exploration curriculum required")


def declared_motor_curriculum(d: dict[str, Any]) -> MotorExplorationCurriculum:
    c = motor_curriculum(d.get("curriculum", "run2"))
    if (
        d["protocol"] != c.protocol
        or d.get("sampling_law", "gaussian-ar1") not in ("gaussian-ar1", "box-truncated-ar1")
        or d["seeds"] != c.seeds(d.get("generation_index", 0))
        or d.get("source_context_ids", list(c.context_ids)) != list(c.context_ids)
        or len(d["contexts"]) != len(c.context_ids)
        or d["maximum_worlds"] != c.worlds
        or d["maximum_native_steps"] != 4000 * c.worlds
        or d["maximum_neural_decisions"] != 400 * c.worlds
    ):
        raise ValueError("exact fixed case/seed product and work budget required")
    return c
