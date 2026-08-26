from __future__ import annotations

import hashlib

from rosclaw_soccer.skills.goalkeeper_v2.motion_prior import (
    goalkeeper_motion_prior_contract,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def test_goalkeeper_motion_teacher_is_task_and_region_conditioned() -> None:
    contract = goalkeeper_motion_prior_contract(
        artifact_hash=_hash("motion-library"),
        body_hash=_hash("g1"),
    )

    query = contract.query({"task": "save", "region": "upper_left"})

    assert query.condition_values["task"] == "save"
    assert query.condition_values["region"] == "upper_left"
    assert contract.training_use_only
    assert not contract.deployed_actor_depends_on_teacher
