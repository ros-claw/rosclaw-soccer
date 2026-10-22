"""Preregistered motor exploration interventions, never action authority changes."""

from typing import Any


def exploration_sigma(profile: str, law: str) -> float:
    if profile == "micro-v1" and law in ("gaussian-ar1", "box-truncated-ar1"):
        return 0.005
    if profile == "contact-v1" and law == "box-truncated-ar1":
        return 0.02
    raise ValueError("reviewed exploration profile and matching probability law required")


def declared_exploration_sigma(declaration: dict[str, Any]) -> float:
    sigma = exploration_sigma(
        declaration.get("exploration_profile", "micro-v1"),
        declaration.get("sampling_law", "gaussian-ar1"),
    )
    if declaration["sigma_policy_units"] != sigma:
        raise ValueError("sampling amplitude must match the preregistered profile")
    return sigma
