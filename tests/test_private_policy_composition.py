from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.private_policy_composition import assemble_private_candidate
from rosclaw_soccer.training.role_receiving_courses import ROSTER


def policies():
    parent = NearBallResidualPolicy.initialize(ROSTER, "sha256:" + "a" * 64)
    weights = {k: v.copy() for k, v in parent.weights.items()}
    weights["b2"][:] = np.arange(8)[:, None] / 100 + 0.01
    return parent, replace(parent, generation=1, parent_hash=parent.policy_hash, weights=weights)


def test_assembly_copies_only_explicit_actor_and_preserves_sources():
    parent, source = policies()
    before = parent.policy_hash, source.policy_hash
    child, record = assemble_private_candidate(
        parent, {ROSTER[-1]: source}, selection_evidence_hash="sha256:" + "b" * 64
    )
    for k in parent.weights:
        assert np.array_equal(child.weights[k][:-1], parent.weights[k][:-1])
        assert np.array_equal(child.weights[k][-1], source.weights[k][-1])
    assert before == (parent.policy_hash, source.policy_hash)
    assert child.parent_hash == parent.policy_hash
    assert record["requires_team_replay"] and not record["promoted"]
    assert record["optimizer_steps"] == 0 and record["activation_ceiling"] == "SIM_ONLY"
    digest = record.pop("manifest_hash")
    assert digest == hash_json(record)


@pytest.mark.parametrize("fault", ["body", "observation", "role", "hash", "empty", "type"])
def test_assembly_rejects_incompatible_or_unbound_sources(fault):
    parent, source = policies()
    evidence = "sha256:" + "b" * 64
    if fault == "body":
        source = replace(source, body_hash="sha256:" + "c" * 64)
    elif fault == "observation":
        source = source.with_task_geometry()
    elif fault == "hash":
        evidence = "unverified"
    replacements = {"unknown" if fault == "role" else ROSTER[0]: source}
    if fault == "empty":
        replacements = {}
    elif fault == "type":
        replacements = {ROSTER[0]: object()}
    with pytest.raises(ValueError):
        assemble_private_candidate(parent, replacements, selection_evidence_hash=evidence)


def test_explicit_multi_actor_composition_is_deterministic():
    parent, source = policies()
    selected = {ROSTER[0]: source, ROSTER[-1]: source}
    a, report_a = assemble_private_candidate(
        parent, selected, selection_evidence_hash="sha256:" + "b" * 64
    )
    b, report_b = assemble_private_candidate(
        parent, dict(reversed(list(selected.items()))), selection_evidence_hash="sha256:" + "b" * 64
    )
    assert a.policy_hash == b.policy_hash and report_a == report_b
