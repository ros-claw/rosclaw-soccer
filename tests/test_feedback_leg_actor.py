"""SIM_ONLY feedback-leg artifact and per-player contact isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.feedback_leg_actor import FrozenFeedbackLegActor
from rosclaw_soccer.providers.g1.receiving_sonic import (
    ReceivingSonicFeedbackOption,
    ReceivingSonicOption,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import (
    TeamBallContact,
    TeamMotorObservation,
    TeamMotorPhysicsObservation,
    TeamMotorTarget,
)


def _artifact(path: Path) -> Path:
    payload = {
        "schema": "rosclaw_soccer.rsi.mjx_contact_residual_es.v1",
        "activation_ceiling": "SIM_ONLY",
        "parameters": [0.0] * 60 + [1.0, 1.0, 1.0, 1.0],
        "promotion_authorized": False,
        "cpu_holdout_evaluated": False,
    }
    payload["result_hash"] = hash_json(payload)
    path.write_text(json.dumps(payload))
    return path


def _observation(ball_x: float) -> TeamMotorObservation:
    qpos = np.zeros(43, dtype=np.float64)
    qvel = np.zeros(41, dtype=np.float64)
    qpos[2] = 0.75
    qpos[36] = ball_x
    return TeamMotorObservation(
        agent_id="red.finisher",
        frame=0,
        time_sec=0.0,
        intent="shoot",
        prospective_owner=True,
        qpos=tuple(qpos.tolist()),
        qvel=tuple(qvel.tolist()),
        target_position_m=(5.0, 0.0, 0.5),
    )


def test_feedback_actor_is_bounded_and_touch_gated(tmp_path: Path) -> None:
    actor = FrozenFeedbackLegActor.load(_artifact(tmp_path / "actor.json"))
    target = TeamMotorTarget((0.0,) * 29, (100.0,) * 29, (10.0,) * 29)
    far = actor.propose(_observation(2.0), target, foot_seen=False, nonfoot_seen=False)
    near = actor.propose(_observation(0.5), target, foot_seen=False, nonfoot_seen=False)
    assert far == target
    assert [i for i, value in enumerate(near.target_rad) if value] == [0, 3, 6, 9]
    assert all(0 < near.target_rad[i] < 0.12 for i in (0, 3, 6, 9))
    assert not actor.parameters.flags.writeable
    with pytest.raises(ValueError):
        actor.parameters.setflags(write=True)


def test_feedback_actor_rejects_tamper_and_promotion(tmp_path: Path) -> None:
    path = _artifact(tmp_path / "actor.json")
    payload = json.loads(path.read_text())
    payload["parameters"][0] = 5.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sealed"):
        FrozenFeedbackLegActor.load(path)
    payload["result_hash"] = hash_json({k: v for k, v in payload.items() if k != "result_hash"})
    payload["promotion_authorized"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sealed"):
        FrozenFeedbackLegActor.load(path)


def test_receiver_feedback_observes_only_own_ball_contacts(tmp_path: Path) -> None:
    assert not hasattr(ReceivingSonicOption, "observe_physics")
    receiver = object.__new__(ReceivingSonicFeedbackOption)
    receiver.agent_id = "red.finisher"
    receiver.feedback_actor = FrozenFeedbackLegActor.load(_artifact(tmp_path / "actor.json"))
    receiver.feedback_foot_seen = False
    receiver.feedback_nonfoot_seen = False
    receiver.feedback_last_physics_time_sec = -1.0
    receiver.faulted = False
    other = TeamBallContact(1, "blue.finisher", "left_foot", 2.0)
    own = TeamBallContact(2, "red.finisher", "right_foot", 3.0)
    receiver.observe_physics(
        TeamMotorPhysicsObservation(
            time_sec=0.002,
            qpos=(0.0,) * 43,
            qvel=(0.0,) * 41,
            world_bodies_safe=True,
            foot_normal_force_n=3.0,
            other_non_ground_normal_force_n=2.0,
            observer_agent_id="red.finisher",
            contacts_complete=True,
            ball_contacts=(other, own),
        )
    )
    assert receiver.feedback_foot_seen
    assert not receiver.feedback_nonfoot_seen
    with pytest.raises(ValueError, match="consecutive"):
        receiver.observe_physics(
            TeamMotorPhysicsObservation(
                time_sec=0.002,
                qpos=(0.0,) * 43,
                qvel=(0.0,) * 41,
                world_bodies_safe=True,
                foot_normal_force_n=0.0,
                other_non_ground_normal_force_n=0.0,
                observer_agent_id="red.finisher",
                contacts_complete=True,
            )
        )
    assert receiver.faulted
