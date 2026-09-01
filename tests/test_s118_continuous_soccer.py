from __future__ import annotations

import hashlib

import pytest

from rosclaw_soccer.growth.first_touch import (
    FirstTouchFailure,
    FirstTouchGateConfig,
    FirstTouchMeasurement,
    build_first_touch_dream,
    evaluate_first_touch,
)
from rosclaw_soccer.growth.tactical_2v1 import (
    MatchedTacticalRollout,
    TacticalAction,
    TacticalRewardWeights,
    TwoVsOneDecisionEvidence,
    TwoVsOneState,
)
from rosclaw_soccer.skills.team.continuous_soccer import (
    ContinuousSoccerConfig,
    ContinuousSoccerTrace,
    SoccerEventKind,
    SoccerMatchEvent,
)


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _event(index: int, kind: SoccerEventKind, actor: str) -> SoccerMatchEvent:
    return SoccerMatchEvent(
        event_id=f"event.{index}",
        kind=kind,
        time_sec=float(index * 10),
        actor_id=actor,
        team_id=("team.blue" if actor in {"agent.defender", "agent.goalkeeper"} else "team.red"),
        ball_state_hash=_hash(f"ball.{index}"),
        self_state_hash=_hash(f"self.{index}"),
        world_state_hash=_hash(f"world.{index}"),
        source_evidence_hash=_hash(f"evidence.{index}"),
    )


def _continuous_trace(events: tuple[SoccerMatchEvent, ...]) -> ContinuousSoccerTrace:
    return ContinuousSoccerTrace(
        match_id="match.s118",
        config=ContinuousSoccerConfig(),
        body_bundle_hash=_hash("bodies"),
        environment_hash=_hash("environment"),
        policy_bundle_hash=_hash("policies"),
        trajectory_hash=_hash("trajectory"),
        clock_id="mujoco.sim.time",
        observed_duration_sec=60.0,
        physics_step_count=30_000,
        events=events,
    )


def test_continuous_match_requires_play_after_an_in_episode_restart() -> None:
    passing = _continuous_trace(
        (
            _event(0, SoccerEventKind.PASS, "agent.playmaker"),
            _event(1, SoccerEventKind.GOAL, "agent.finisher"),
            _event(2, SoccerEventKind.RESTART, "agent.goalkeeper"),
            _event(3, SoccerEventKind.INTERCEPTION, "agent.defender"),
            _event(4, SoccerEventKind.PASS, "agent.playmaker"),
            _event(5, SoccerEventKind.SHOT, "agent.finisher"),
            _event(6, SoccerEventKind.SAVE, "agent.goalkeeper"),
        )
    )
    assert passing.trace_hash.startswith("sha256:")

    with pytest.raises(ValueError, match="ended at a highlight"):
        _continuous_trace(
            (
                _event(0, SoccerEventKind.PASS, "agent.playmaker"),
                _event(1, SoccerEventKind.INTERCEPTION, "agent.defender"),
                _event(2, SoccerEventKind.PASS, "agent.playmaker"),
                _event(3, SoccerEventKind.SHOT, "agent.finisher"),
                _event(4, SoccerEventKind.SAVE, "agent.goalkeeper"),
                _event(5, SoccerEventKind.GOAL, "agent.finisher"),
                _event(6, SoccerEventKind.RESTART, "agent.goalkeeper"),
            )
        )


def test_continuous_match_rejects_idle_gaps_and_delayed_restarts() -> None:
    with pytest.raises(ValueError, match="idle activity gap"):
        _continuous_trace(
            (
                _event(0, SoccerEventKind.PASS, "agent.playmaker"),
                _event(3, SoccerEventKind.INTERCEPTION, "agent.defender"),
                _event(6, SoccerEventKind.SHOT, "agent.finisher"),
            )
        )

    with pytest.raises(ValueError, match="missing its in-episode restart"):
        _continuous_trace(
            (
                _event(0, SoccerEventKind.PASS, "agent.playmaker"),
                _event(1, SoccerEventKind.GOAL, "agent.finisher"),
                _event(2, SoccerEventKind.SAVE, "agent.goalkeeper"),
                _event(3, SoccerEventKind.RESTART, "agent.goalkeeper"),
                _event(4, SoccerEventKind.PASS, "agent.playmaker"),
                _event(5, SoccerEventKind.SHOT, "agent.finisher"),
                _event(6, SoccerEventKind.SAVE, "agent.goalkeeper"),
            )
        )


def test_continuous_match_requires_two_distinct_teams() -> None:
    events = tuple(
        SoccerMatchEvent(
            event_id=f"event.{index}",
            kind=SoccerEventKind.PASS,
            time_sec=float(index * 10),
            actor_id=f"agent.player{index % 2}",
            team_id="team.red",
            ball_state_hash=_hash(f"ball.{index}"),
            self_state_hash=_hash(f"self.{index}"),
            world_state_hash=_hash(f"world.{index}"),
            source_evidence_hash=_hash(f"evidence.{index}"),
        )
        for index in range(7)
    )
    with pytest.raises(ValueError, match="insufficient participating teams"):
        _continuous_trace(events)


def _touch(**overrides: object) -> FirstTouchMeasurement:
    values: dict[str, object] = {
        "sample_id": "touch.s118.1",
        "actor_id": "soccer.playmaker",
        "source_snapshot_hash": _hash("snapshot"),
        "body_hash": _hash("body"),
        "scenario_hash": _hash("scenario"),
        "incoming_speed_mps": 2.0,
        "outgoing_speed_mps": 0.8,
        "target_error_m": 0.10,
        "direction_error_deg": 8.0,
        "next_action_latency_sec": 0.45,
        "minimum_pelvis_height_m": 0.72,
        "maximum_torso_tilt_deg": 12.0,
        "maximum_root_speed_mps": 0.8,
        "contact_detected": True,
        "selected_foot": "left",
        "required_foot": "left",
    }
    values.update(overrides)
    return FirstTouchMeasurement(**values)  # type: ignore[arg-type]


def test_first_touch_passes_only_when_control_balance_and_handoff_pass() -> None:
    result = evaluate_first_touch(_touch(), FirstTouchGateConfig())
    assert result.passed
    assert result.controlled_first_touch
    assert result.successor_action_ready
    assert result.safety_passed


def test_first_touch_attributes_all_failures_and_builds_deterministic_dream() -> None:
    measurement = _touch(
        selected_foot="right",
        outgoing_speed_mps=3.0,
        target_error_m=0.6,
        direction_error_deg=35.0,
        minimum_pelvis_height_m=0.5,
        next_action_latency_sec=1.2,
    )
    result = evaluate_first_touch(measurement)
    assert result.primary_failure is FirstTouchFailure.WRONG_FOOT
    assert result.all_failures == (
        FirstTouchFailure.WRONG_FOOT,
        FirstTouchFailure.LOST_BALANCE,
        FirstTouchFailure.TOUCH_WRONG_DIRECTION,
        FirstTouchFailure.TOUCH_TOO_HARD,
        FirstTouchFailure.TOO_SLOW_TO_NEXT_ACTION,
    )
    dream = build_first_touch_dream(measurement, result, maximum_variants=8)
    assert dream.failure_code == "soccer.wrong_foot"
    assert dream.sample(count=3, seed=118) == dream.sample(count=3, seed=118)
    assert dream.hardware_authorized is False


def test_missed_touch_does_not_claim_a_contradictory_hard_contact() -> None:
    result = evaluate_first_touch(
        _touch(
            contact_detected=False,
            outgoing_speed_mps=0.0,
            target_error_m=1.0,
            direction_error_deg=180.0,
            next_action_latency_sec=1.0,
        )
    )
    assert result.primary_failure is FirstTouchFailure.TOUCH_TOO_SOFT
    assert result.all_failures == (
        FirstTouchFailure.TOUCH_TOO_SOFT,
        FirstTouchFailure.TOO_SLOW_TO_NEXT_ACTION,
    )


def _tactical_state() -> TwoVsOneState:
    return TwoVsOneState(
        state_id="state.s118.1",
        seed=118,
        self_state_hash=_hash("self"),
        world_state_hash=_hash("world"),
        scenario_hash=_hash("scenario"),
        environment_hash=_hash("environment"),
        frozen_foundation_hash=_hash("foundation"),
        frozen_skill_bundle_hash=_hash("skills"),
        frozen_defender_hash=_hash("defender"),
        carrier_pressure=0.9,
        teammate_lane_openness=0.8,
        shot_lane_openness=0.2,
        goal_progress=0.6,
        teammate_progress=0.8,
    )


def test_two_vs_one_credit_requires_a_distinct_matched_ablation() -> None:
    state = _tactical_state()
    rollout = MatchedTacticalRollout(
        state_hash=state.state_hash,
        policy_hash=_hash("tactical-policy"),
        action=TacticalAction.PASS,
        action_trace_hash=_hash("actions"),
        trajectory_hash=_hash("with-playmaker"),
        ablation_action_trace_hash=_hash("ablation-actions"),
        ablated_trajectory_hash=_hash("without-playmaker"),
        team_reward=0.8,
        role_reward=0.5,
        ablated_team_reward=0.2,
        possession_progress=0.7,
        safety_cost=0.0,
    )
    evidence = TwoVsOneDecisionEvidence(state, rollout, TacticalRewardWeights())
    assert evidence.promotion_eligible
    assert rollout.difference_reward == pytest.approx(0.6)
    assert evidence.weighted_score == pytest.approx(1.365)

    with pytest.raises(ValueError, match="distinct physical replay"):
        MatchedTacticalRollout(
            state_hash=state.state_hash,
            policy_hash=_hash("tactical-policy"),
            action=TacticalAction.PASS,
            action_trace_hash=_hash("actions"),
            trajectory_hash=_hash("same"),
            ablation_action_trace_hash=_hash("ablation-actions"),
            ablated_trajectory_hash=_hash("same"),
            team_reward=0.8,
            role_reward=0.5,
            ablated_team_reward=0.2,
            possession_progress=0.7,
            safety_cost=0.0,
        )
