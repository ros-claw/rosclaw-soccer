"""Predeclared full-horizon ball-aware continuation, not a promotion gate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BallAwareLearningBudget:
    course: bool = False

    def __post_init__(self) -> None:
        if type(self.course) is not bool:
            raise ValueError("explicit smoke/course protocol required")

    @property
    def worlds(self) -> int:
        return 128 if self.course else 32

    @property
    def iterations(self) -> int:
        return 200 if self.course else 2

    @property
    def seed(self) -> int:
        return 2701 if self.course else 2601

    @property
    def controls(self) -> int:
        return self.iterations * 24

    @property
    def samples(self) -> int:
        return self.controls * self.worlds

    @property
    def optimizer_updates(self) -> int:
        return self.iterations * 21

    @property
    def gpu_world_steps(self) -> int:
        return self.samples * 10
