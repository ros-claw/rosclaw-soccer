from __future__ import annotations

from rosclaw_soccer.training.causal_transition_growth import CausalTransitionContext
from rosclaw_soccer.training.recovered_target_contact_discovery import (
    default_recovered_target_contact_probes,
)


def _context(index: int) -> CausalTransitionContext:
    return CausalTransitionContext(
        f"test.s130.{index}",
        (5.1, -0.164, 0.0),
        -0.11 + 0.005 * index,
        1.34 - 0.005 * index,
        (1.19 + 0.002 * index, -0.15),
        0.8,
        0.09 + 0.002 * index,
    )


def test_recovered_contact_probes_cover_every_consumed_context_and_action() -> None:
    contexts = tuple(_context(index) for index in range(6))
    probes = default_recovered_target_contact_probes(contexts)
    assert len(probes) == 24
    assert len({probe.probe_hash for probe in probes}) == 24
    assert {probe.context.context_hash for probe in probes} == {
        context.context_hash for context in contexts
    }
    for context in contexts:
        selected = [probe for probe in probes if probe.context == context]
        assert len(selected) == 4
        assert len({probe.action.action_hash for probe in selected}) == 4
        assert all(probe.action.activation_ceiling == "SIM_ONLY" for probe in selected)
