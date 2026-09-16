from dataclasses import asdict, replace

import numpy as np
import pytest
import test_keeper_distribution_preview as preview

from rosclaw_soccer.growth.goalkeeper_positioning import goalkeeper_cover_target
from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.sim.contracts import hash_json


def arguments():
    return dict(
        ball_position_m=(5.0, -1.5, 0.115),
        ball_velocity_mps=(0.0, 0.0, 0.0),
        own_goal_m=(-1.5, 0.0, 0.0),
        opponent_goal_m=(7.5, 0.0, 0.0),
        depth_m=0.65,
    )


def test_distant_ball_covers_angle_instead_of_shadowing_full_lateral_offset():
    assert goalkeeper_cover_target(**arguments()) == pytest.approx((-0.85, -0.15, 0.0))


def test_incoming_intercept_uses_keeper_plane_not_goal_plane():
    values = arguments()
    values.update(ball_position_m=(3.0, -0.5, 1.0), ball_velocity_mps=(-4.0, 1.0, 0.5))
    # Travel from x=3 to x=-.85 takes .9625 s, not 1.125 s to the goal line.
    assert goalkeeper_cover_target(**values) == pytest.approx((-0.85, 0.4625, 0))


@pytest.mark.parametrize("velocity", [(4.0, 1.0, 0.0), (-0.2, 1.0, 0.0), (0.0, 4.0, 0.0)])
def test_receding_slow_or_sideways_ball_does_not_invent_imminent_interception(velocity):
    values = arguments()
    values["ball_velocity_mps"] = velocity
    assert goalkeeper_cover_target(**values) == pytest.approx((-0.85, -0.15, 0.0))


@pytest.mark.parametrize("angle", [0.0, np.pi / 2, np.pi, -0.47])
@pytest.mark.parametrize("incoming", [False, True])
def test_goal_relative_translation_and_rotation_equivariance(angle, incoming):
    values = arguments()
    if incoming:
        values["ball_velocity_mps"] = (-6.0, 0.6, 1.0)
    before = goalkeeper_cover_target(**values)
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    translation = np.array((9.3, -4.1))
    for name in ("ball_position_m", "own_goal_m", "opponent_goal_m"):
        old = values[name]
        xy = rotation @ old[:2] + translation
        values[name] = (float(xy[0]), float(xy[1]), old[2])
    v = values["ball_velocity_mps"]
    xy = rotation @ v[:2]
    values["ball_velocity_mps"] = (float(xy[0]), float(xy[1]), v[2])
    expected = rotation @ before[:2] + translation
    assert goalkeeper_cover_target(**values) == pytest.approx((*expected, 0.0))


def test_near_or_behind_goal_ball_is_bounded_without_division_by_zero():
    for x in (-1.5, -2.0, -0.9):
        values = arguments()
        values["ball_position_m"] = (x, 30.0, 0.115)
        assert goalkeeper_cover_target(**values) == pytest.approx((-0.85, 1.25, 0))


@pytest.mark.parametrize(
    "change",
    [
        {"depth_m": True},
        {"depth_m": 2},
        {"depth_m": float("nan")},
        {"lateral_limit_m": 0},
        {"interception_horizon_sec": 10},
        {"ball_position_m": (float("nan"), 0, 0)},
        {"ball_velocity_mps": (float("inf"), 0, 0)},
        {"ball_position_m": (1, 2)},
        {"ball_position_m": (10001, 0, 0)},
        {"own_goal_m": (7.5, 0, 0)},
        {"opponent_goal_m": (1000, 0, 0)},
    ],
)
def test_invalid_geometry_and_envelopes_rejected(change):
    with pytest.raises(ValueError):
        goalkeeper_cover_target(**{**arguments(), **change})


@pytest.fixture
def course(monkeypatch):
    return preview.basic.course.__wrapped__(monkeypatch)


@pytest.mark.parametrize("team", ["red", "blue"])
def test_opt_in_changes_cover_target_and_keeps_possession_workflow(course, team):
    cell, value = preview.situation(course, team)
    covering = replace(
        cell, tactical_profile=replace(cell.tactical_profile, goalkeeper_angle_cover=True)
    )
    far = (5.0, -1.5, 0.115) if team == "red" else (1.0, 1.5, 0.115)
    distant = replace(value, ball_position_m=far)
    old, new = cell.decide(distant), covering.decide(distant)
    assert old.intent is new.intent is TacticalIntent.COVER
    assert abs(new.target_position_m[1]) < abs(old.target_position_m[1])
    # An accepted grounded outlet is not replaced by a positioning heuristic.
    assert covering.decide(value).intent is TacticalIntent.PASS
    assert cell.tactical_profile.profile_hash != covering.tactical_profile.profile_hash


def test_default_profile_hash_preserved(course):
    cell, _ = preview.situation(course, enabled=False)
    legacy = asdict(cell.tactical_profile)
    for name in (
        "goalkeeper_angle_cover",
        "keeper_distribution_preview",
        "moving_ball_finish_intent",
        "blocked_shot_layoff",
    ):
        legacy.pop(name)
    assert cell.tactical_profile.profile_hash == hash_json(legacy)
    assert "goalkeeper_angle_cover" not in cell.tactical_profile.to_dict()


@pytest.mark.parametrize("value", [1, None, "true"])
def test_profile_requires_explicit_boolean(course, value):
    cell, _ = preview.situation(course)
    with pytest.raises(ValueError):
        replace(cell.tactical_profile, goalkeeper_angle_cover=value)
