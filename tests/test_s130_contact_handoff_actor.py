from __future__ import annotations

import json

import pytest

from rosclaw_soccer.growth.contact_handoff_actor import (
    G1ContactHandoffActor,
    load_contact_handoff_actor,
    save_contact_handoff_actor,
)


def _hash(label: str) -> str:
    return "sha256:" + label * 64


def _actor() -> G1ContactHandoffActor:
    return G1ContactHandoffActor(
        body_hash=_hash("a"),
        target_plan_actor_hash=_hash("b"),
        target_contact_actor_hash=_hash("c"),
        source_evidence_hashes=(_hash("d"),),
        selected_offset_frames=0,
        evaluated_offset_frames=(0, 7),
        safe_case_count=6,
        recovered_failure_count=3,
        training_case_count=6,
    )


def test_contact_handoff_actor_is_contact_relative_and_sim_only(tmp_path) -> None:
    actor = _actor()
    decision = actor.decide(contact_policy_frame=248)
    assert decision.accepted
    assert decision.handoff_policy_frame == 248
    assert decision.route == "LEARNED_CONTACT_GATED_HANDOFF"
    assert actor.to_dict()["authority"]["torque_authority"] is False

    path = tmp_path / "actor.json"
    save_contact_handoff_actor(actor, path)
    assert load_contact_handoff_actor(path) == actor


def test_contact_handoff_actor_fails_closed_outside_support() -> None:
    actor = _actor()
    assert not actor.decide(contact_policy_frame=219).accepted
    assert not actor.decide(contact_policy_frame=True).accepted


def test_contact_handoff_actor_detects_tampering(tmp_path) -> None:
    path = tmp_path / "actor.json"
    save_contact_handoff_actor(_actor(), path)
    payload = json.loads(path.read_text())
    payload["selected_offset_frames"] = 7
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_contact_handoff_actor(path)
