from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
)
from rosclaw_soccer.training.intended_foot_alignment_discovery import (
    default_intended_foot_alignment_probes,
    strict_intended_contact_quality,
)


def _context(index: int) -> CausalTransitionContext:
    return CausalTransitionContext(
        f"test.s130.foot.{index}",
        (5.1, -0.164, 0.0),
        -0.11 + 0.005 * index,
        1.34 - 0.005 * index,
        (1.19 + 0.002 * index, -0.15),
        0.8,
        0.09 + 0.002 * index,
    )


def test_alignment_curriculum_varies_lateral_stance_for_every_context() -> None:
    contexts = tuple(_context(index) for index in range(6))
    probes = default_intended_foot_alignment_probes(contexts)
    assert len(probes) == 18
    for context in contexts:
        assert {probe.action.stance_offset_y_m for probe in probes if probe.context == context} == {
            -0.12,
            0.0,
            0.12,
        }


def test_strict_quality_rejects_support_foot_contact(monkeypatch) -> None:
    monkeypatch.setattr(
        "rosclaw_soccer.training.intended_foot_alignment_discovery._chain_quality",
        lambda *_args, **_kwargs: {"chain_passed": True, "safe": True},
    )
    trajectory = {"shooter_ball_contact_foot": np.asarray((0, -1, 1), dtype=np.int64)}
    quality = strict_intended_contact_quality(
        result=SimpleNamespace(),
        trajectory=trajectory,
        quality_config=CausalTransitionGrowthConfig(),
        intended_contact_foot=1,
    )
    assert quality["first_shooter_contact_foot"] == -1
    assert quality["intended_foot_contact"] is False
    assert quality["strict_chain_passed"] is False
