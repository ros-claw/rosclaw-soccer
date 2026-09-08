from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    _movement_command,
)
from rosclaw_soccer.training.owned_contact_learning import reward


def test_contact_stance_is_half_turn_equivariant():
    policy = OwnedBallContactPolicy()
    red, ryaw = policy.stance((2.0, -0.3), (5.0, 0.4))
    blue, byaw = policy.stance((4.0, 0.3), (1.0, -0.4))
    assert blue == pytest.approx((6.0 - red[0], -red[1]))
    assert np.cos(byaw) == pytest.approx(-np.cos(ryaw))
    assert np.sin(byaw) == pytest.approx(-np.sin(ryaw))


@pytest.mark.parametrize(
    "values",
    [dict(depth_m=0.1), dict(lateral_m=0.5), dict(pass_speed_mps=3.0), dict(depth_m=float("nan"))],
)
def test_policy_cannot_expand_the_search_bounds(values):
    with pytest.raises(ValueError):
        OwnedBallContactPolicy(**values)


def test_learned_stance_replaces_distant_fixed_retreat():
    cell = SimpleNamespace(
        agent_id="red.playmaker",
        self_model=SimpleNamespace(
            primary_role=MatchRole.PLAYMAKER,
            team_id="red",
            teammate_ids=(),
            opponent_ids=(),
        ),
    )
    controller = SimpleNamespace(cell=cell, qpos_base=0, last_world_command=None)
    data = SimpleNamespace(
        qpos=np.array([0.5, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.115]), qvel=np.zeros(3)
    )
    decision = SimpleNamespace(intent=TacticalIntent.PASS, target_position_m=(5.0, 0.0, 0.0))
    config = IndependentTeamWorldConfig(owned_contact_policy=OwnedBallContactPolicy())
    kwargs = dict(
        controller=controller,
        decision=decision,
        positions={"red.playmaker": np.array([0.5, 0.0])},
        data=data,
        ball_qpos=7,
        ball_qvel=0,
        possession_agent_id="red.playmaker",
        committed_receiver=False,
        active_receiver=False,
        post_receive_hold=False,
        receive_foot_lateral_offset_m=0.18,
        strike_target_position_m=None,
    )
    owned = _movement_command(**kwargs, config=config)
    legacy = _movement_command(**kwargs, config=replace(config, owned_contact_policy=None))
    assert legacy[0] < 0.0 < owned[0]
    assert np.linalg.norm(owned[:2]) <= config.maximum_speed_mps
    held = _movement_command(**{**kwargs, "post_receive_hold": True}, config=config)
    np.testing.assert_array_equal(held, np.zeros(3))


def test_reward_does_not_credit_running_or_intentions():
    report = {
        "exact_replay": True,
        "results": [{"safe": True, "pass_intent_count": 1000}],
        "assessment": {"events": [{"skill": "sprint"}]},
    }
    assert reward(report) == (1, 0, 0)
    report["assessment"]["events"].append({"skill": "pass"})
    assert reward(report) == (1, 0, 0)
    report["causal_pass_feedback"] = [{"physical_receive_confirmed": True}]
    report["assessment"]["gates"] = {"foot_only_ball_control": True}
    assert reward(report) == (1, 1, 0)
    report["results"][0]["safe"] = False
    assert reward(report) == (-1, 0, 0)


def _pass_trace():
    n = 31
    trace = {
        "time": np.arange(n) * 0.1,
        "pass_source_agent_code": np.zeros(n),
        "pass_target_agent_code": np.zeros(n),
        "ball_contact_agent_code": np.zeros(n),
        "ball_contact_effector_code": np.ones(n),
        "ball_contact_force_n": np.ones(n),
        "ball_pose": np.zeros((n, 7)),
        "ball_velocity": np.zeros((n, 6)),
        "red_finisher_left_foot_position": np.tile([1.0, 0.0, 0.0], (n, 1)),
        "red_finisher_right_foot_position": np.tile([1.0, 0.1, 0.0], (n, 1)),
    }
    trace["pass_source_agent_code"][4:12] = 2
    trace["pass_target_agent_code"][4:12] = 1
    trace["ball_contact_agent_code"][10] = 2
    trace["ball_contact_agent_code"][20] = 1
    trace["ball_pose"][:, 0] = np.clip((np.arange(n) - 10) / 10.0, 0.0, 1.0)
    trace["ball_velocity"][:, 0] = 1.0
    return trace


def test_pass_feedback_requires_physical_delivery():
    from rosclaw_soccer.growth.pass_failure_feedback import diagnose_passes

    trace = _pass_trace()
    rows = diagnose_passes(trace, ("red.finisher", "red.playmaker"))
    assert rows[0]["physical_receive_confirmed"]
    trace["ball_contact_agent_code"][20] = 0
    assert not diagnose_passes(trace, ("red.finisher", "red.playmaker"))[0][
        "physical_receive_confirmed"
    ]


def test_late_unrelated_contact_and_fractional_codes_are_not_pass_evidence():
    from rosclaw_soccer.growth.pass_failure_feedback import diagnose_passes

    trace = _pass_trace()
    trace["ball_contact_agent_code"][10] = 0
    trace["ball_contact_agent_code"][19] = 2
    row = diagnose_passes(trace, ("red.finisher", "red.playmaker"))[0]
    assert row["failure"] == "NO_POST_COMMITMENT_FOOT_CONTACT"
    trace["pass_source_agent_code"][4] = 1.5
    with pytest.raises(ValueError):
        diagnose_passes(trace, ("red.finisher", "red.playmaker"))


def test_audit_downgrades_historical_postcontact_credit(tmp_path, monkeypatch):
    import json

    import rosclaw_soccer.training.owned_contact_learning as module
    from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

    reports = {}
    sources = []
    for index in range(14):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / "probe.json"
        path.write_text(json.dumps({"index": index}))
        np.savez(directory / "primary.npz", marker=np.zeros(1))
        reports[str(path)] = {
            "exact_replay": True,
            "results": [{"safe": True}],
            "assessment": {"events": [{"skill": "pass"}] if index >= 12 else []},
            "cells": [
                {"self_model": {"agent_id": "red.playmaker"}},
                {"self_model": {"agent_id": "red.finisher"}},
            ],
        }
        sources.append((str(path), hash_bytes(path.read_bytes())))
    monkeypatch.setattr(module, "validate_probe", lambda p: reports[str(p)])
    monkeypatch.setattr(
        module, "diagnose_passes", lambda t, ids: [{"physical_receive_confirmed": False}]
    )
    policy = OwnedBallContactPolicy()
    from dataclasses import asdict

    result = {
        "request": {"policies": [None] + [asdict(policy)] * 4},
        "selection": {
            "training_scores": [[1, 0, 0]] * 5,
            "selected_index": 1,
            "selected_policy": asdict(policy),
            "policy_hash": policy.policy_hash,
        },
        "training_sources": dict(sources[:10]),
        "exam_sources": dict(sources[10:]),
        "exam_scores": [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]],
        "heldout_bilateral_pass_gain": True,
        "status": "PASS_BOUNDED_PASS_EXAM",
        "promotion_eligible": False,
        "hardware_command_sent": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    path = tmp_path / "learning.json"
    path.write_text(json.dumps(result))
    verified = module.audit(path)
    assert verified["original_contract_status"] == "PASS_BOUNDED_PASS_EXAM"
    assert verified["status"] == "REJECTED_PASS_EXAM"
