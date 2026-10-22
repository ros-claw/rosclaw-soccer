"""Bounded teacher-only predictive clearance hypotheses; not motor authority."""


def shin_clearance_profile(name: str) -> tuple[float, float]:
    if name == "standard-v1":
        return 0.04, 1.0
    if name == "anticipatory-priority-v1":
        return 0.08, 4.0
    raise ValueError("explicit reviewed teacher prediction/constraint profile required")
