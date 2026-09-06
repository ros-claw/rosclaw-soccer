from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.causal_skill_transition import (
    load_causal_skill_transition_actor,
)
from rosclaw_soccer.growth.causal_skill_transition_risk import (
    G1CausalSkillTransitionMemoryActor,
    save_causal_skill_transition_memory_actor,
)
from rosclaw_soccer.sim.contracts import hash_json


def _memory_actor() -> G1CausalSkillTransitionMemoryActor:
    contexts = tuple(f"context-{index:02d}" for index in range(16))
    features = tuple(tuple([0.1 * index] + [0.0] * 13) for index in range(len(contexts)))
    safe = []
    chain = []
    for _ in contexts:
        safe.append((False, True, True, True, False))
        chain.append((False, False, False, True, False))
    return G1CausalSkillTransitionMemoryActor(
        source_snapshot_hash=str(hash_json({"source": "probes"})),
        implementation_hash=str(hash_json({"implementation": "memory"})),
        feature_center=tuple([0.75] + [0.0] * 13),
        feature_scale=tuple([0.5] + [1.0] * 13),
        feature_minimum=tuple([0.0] * 14),
        feature_maximum=tuple([1.5] + [0.0] * 13),
        prototype_context_ids=contexts,
        prototype_features=features,
        safe_labels=tuple(safe),
        chain_labels=tuple(chain),
        neighbor_count=3,
        minimum_neighbor_chain_fraction=0.5,
        minimum_chain_advantage=0.1,
        maximum_neighbor_distance=4.0,
    )


def test_memory_actor_selects_only_unanimously_safe_local_phase(tmp_path: Path) -> None:
    actor = _memory_actor()
    observation = np.asarray([0.05] + [0.0] * 13, dtype=np.float64)

    decision = actor.decide(observation)

    assert decision.accepted
    assert decision.trigger_policy_frame == 90
    assert decision.residual_frames == 2
    assert decision.predicted_safe_probability == 1.0
    path = tmp_path / "memory.json"
    save_causal_skill_transition_memory_actor(actor, path)
    assert load_causal_skill_transition_actor(path).actor_hash == actor.actor_hash


def test_memory_actor_falls_back_when_one_nearest_memory_is_unsafe() -> None:
    actor = _memory_actor()
    unsafe = [list(row) for row in actor.safe_labels]
    chain = [list(row) for row in actor.chain_labels]
    unsafe[0][3] = False
    chain[0][3] = False
    actor = replace(
        actor,
        safe_labels=tuple(tuple(row) for row in unsafe),
        chain_labels=tuple(tuple(row) for row in chain),
    )

    decision = actor.decide(np.asarray([0.05] + [0.0] * 13, dtype=np.float64))

    assert decision.trigger_policy_frame == actor.parent_trigger_policy_frame
    assert decision.residual_frames == 0
    assert decision.used_parent_fallback


def test_memory_actor_ood_rejects_learned_authority() -> None:
    actor = _memory_actor()
    observation = np.asarray([100.0] + [0.0] * 13, dtype=np.float64)

    decision = actor.decide(observation)

    assert not decision.accepted
    assert decision.trigger_policy_frame == actor.parent_trigger_policy_frame
    assert decision.used_parent_fallback


def test_memory_actor_rejects_integer_labels() -> None:
    actor = _memory_actor()
    integer_labels = tuple(tuple(int(value) for value in row) for row in actor.safe_labels)

    with pytest.raises(ValueError, match="tensors are invalid"):
        replace(actor, safe_labels=integer_labels)  # type: ignore[arg-type]
