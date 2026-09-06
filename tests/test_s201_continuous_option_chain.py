from __future__ import annotations

from dataclasses import replace

import pytest
from rosclaw.continual.plasticity_lease import AgentUpdateMode

from rosclaw_soccer.growth.continuous_option_chain import (
    ChainRoleBinding,
    ContinuousChainEvent,
    ContinuousChainFailure,
    ContinuousChainPhase,
    ContinuousOptionChainRequest,
    assess_continuous_option_chain,
    build_chain_repair_lease,
)
from rosclaw_soccer.growth.independent_agent_cell import (
    RosclawSoccerAgentCell,
    build_independent_agent_cell,
)
from rosclaw_soccer.growth.pass_aim_calibration import (
    PassAimCalibrationSample,
    train_pass_aim_residual_actor,
)
from rosclaw_soccer.growth.physical_option_router import PhysicalSoccerOption
from rosclaw_soccer.growth.role_option_backend import (
    RoleOptionBackend,
    RoleOptionBackendCandidate,
    select_role_option_backend,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.sim.contracts import hash_json


def _hash(label: str) -> str:
    return str(hash_json({"s201": label}))


def _cells() -> tuple[RosclawSoccerAgentCell, ...]:
    layout = (
        ("red.playmaker", "red", MatchRole.PLAYMAKER, (-1.0, -0.2, 0.0)),
        ("red.finisher", "red", MatchRole.FINISHER, (1.0, 0.0, 0.0)),
        ("blue.goalkeeper", "blue", MatchRole.GOALKEEPER, (7.0, 0.0, 0.0)),
    )
    team_ids = {
        team: tuple(agent_id for agent_id, value, _, _ in layout if value == team)
        for team in ("red", "blue")
    }
    return tuple(
        build_independent_agent_cell(
            agent_id=agent_id,
            team_id=team,
            primary_role=role,
            teammate_ids=tuple(value for value in team_ids[team] if value != agent_id),
            opponent_ids=team_ids["blue" if team == "red" else "red"],
            body_hash=_hash("body"),
            foundation_policy_hash=_hash("foundation"),
            home_position_m=home,
        )
        for agent_id, team, role, home in layout
    )


def _request() -> ContinuousOptionChainRequest:
    cells = _cells()
    by_id = {cell.agent_id: cell for cell in cells}
    return ContinuousOptionChainRequest(
        chain_id="s201.chain.red-attack",
        roster_hash=_hash("roster"),
        ball_id="match.ball.0001",
        initial_ball_state_hash=_hash("ball-0"),
        role_bindings=tuple(
            ChainRoleBinding(
                agent_id=agent_id,
                role=role,
                cell_hash=by_id[agent_id].cell_hash,
                champion_policy_hash=by_id[agent_id].growth_scope.champion_policy.version_hash,
                parent_policy_hash=by_id[agent_id].growth_scope.parent_policy.version_hash,
            )
            for agent_id, role in (
                ("red.playmaker", MatchRole.PLAYMAKER),
                ("red.finisher", MatchRole.FINISHER),
                ("blue.goalkeeper", MatchRole.GOALKEEPER),
            )
        ),
        goal_target_m=(7.5, 0.8, 1.5),
    )


def _events() -> tuple[ContinuousChainEvent, ...]:
    values = (
        (ContinuousChainPhase.PASS, "red.playmaker", "ball-0", "ball-1", 0.0, 1.0, 0.08, 1.4),
        (ContinuousChainPhase.RECEIVE, "red.finisher", "ball-1", "ball-2", 1.0, 1.4, 0.06, 0.9),
        (ContinuousChainPhase.SHOOT, "red.finisher", "ball-2", "ball-3", 1.4, 2.2, 0.07, 8.0),
        (ContinuousChainPhase.SAVE, "blue.goalkeeper", "ball-3", "ball-4", 2.2, 2.6, None, 3.0),
    )
    return tuple(
        ContinuousChainEvent(
            phase=phase,
            agent_id=agent_id,
            ball_id="match.ball.0001",
            input_ball_state_hash=_hash(before),
            output_ball_state_hash=_hash(after),
            decision_hash=_hash(f"decision-{phase.value}"),
            option_request_hash=_hash(f"request-{phase.value}"),
            started_at_sec=start,
            ended_at_sec=end,
            # RECEIVE is a one-touch readiness interval; its first physical
            # contact belongs to the immediately following SHOOT event.
            contact_observed=phase is not ContinuousChainPhase.RECEIVE,
            safe=True,
            phase_ready=True,
            target_error_m=error,
            post_contact_ball_speed_mps=speed,
            exact_replay=True,
        )
        for phase, agent_id, before, after, start, end, error, speed in values
    )


def test_complete_same_ball_chain_passes_strict_gates() -> None:
    assessment = assess_continuous_option_chain(_request(), _events())

    assert assessment.passed
    assert assessment.completed_phase_count == 4
    assert assessment.earliest_failure is None
    assert assessment.ball_lineage_verified
    assert assessment.safe
    assert assessment.exact_replay
    assert not assessment.downstream_credit_blocked


def test_bad_pass_assigns_only_playmaker_credit_and_freezes_everyone_else() -> None:
    events = _events()
    bad_pass = replace(events[0], target_error_m=0.80)
    assessment = assess_continuous_option_chain(_request(), (bad_pass, *events[1:]))

    assert not assessment.passed
    assert assessment.completed_phase_count == 0
    assert assessment.earliest_failure is ContinuousChainFailure.PASS_INACCURATE
    assert assessment.failure_phase is ContinuousChainPhase.PASS
    assert assessment.focal_agent_id == "red.playmaker"
    assert assessment.downstream_credit_blocked

    lease = build_chain_repair_lease(
        cells=_cells(),
        assessment=assessment,
        dataset_manifest_hash=_hash("bad-pass-replay"),
        scenario_contract_hash=_request().request_hash,
        maximum_optimizer_steps=800,
    )
    modes = {binding.agent_id: binding.mode for binding in lease.bindings}
    assert modes["red.playmaker"] is AgentUpdateMode.PLASTIC
    assert modes["red.finisher"] is AgentUpdateMode.FROZEN
    assert modes["blue.goalkeeper"] is AgentUpdateMode.FROZEN


def test_broken_ball_lineage_is_a_system_failure_with_no_learning_lease() -> None:
    events = _events()
    forged_receive = replace(events[1], input_ball_state_hash=_hash("different-ball-state"))
    assessment = assess_continuous_option_chain(
        _request(), (events[0], forged_receive, *events[2:])
    )

    assert assessment.earliest_failure is ContinuousChainFailure.BALL_LINEAGE_BROKEN
    assert assessment.focal_agent_id is None
    assert not assessment.ball_lineage_verified
    with pytest.raises(ValueError, match="no role-local failure"):
        build_chain_repair_lease(
            cells=_cells(),
            assessment=assessment,
            dataset_manifest_hash=_hash("tampered"),
            scenario_contract_hash=_request().request_hash,
            maximum_optimizer_steps=100,
        )


def test_nondeterministic_replay_blocks_plasticity_instead_of_blame() -> None:
    events = _events()
    nondeterministic_shot = replace(events[2], exact_replay=False)
    assessment = assess_continuous_option_chain(
        _request(), (*events[:2], nondeterministic_shot, events[3])
    )

    assert assessment.completed_phase_count == 2
    assert assessment.earliest_failure is ContinuousChainFailure.NONDETERMINISTIC_REPLAY
    assert assessment.focal_agent_id is None
    assert not assessment.exact_replay


def test_safety_failure_is_not_automatically_blame_assigned_to_option_owner() -> None:
    events = _events()
    unsafe = replace(
        events[0],
        safe=False,
        safety_failure_agent_ids=("red.finisher",),
    )

    assessment = assess_continuous_option_chain(_request(), (unsafe, *events[1:]))

    assert assessment.earliest_failure is ContinuousChainFailure.UNSAFE_MOTION
    assert assessment.focal_agent_id == "red.finisher"

    unknown = replace(unsafe, safety_failure_agent_ids=())
    system_assessment = assess_continuous_option_chain(_request(), (unknown, *events[1:]))
    assert system_assessment.focal_agent_id is None


def test_incomplete_prefix_trains_the_next_owner_without_fabricating_contact() -> None:
    events = _events()
    assessment = assess_continuous_option_chain(_request(), events[:2])

    assert assessment.completed_phase_count == 2
    assert assessment.earliest_failure is ContinuousChainFailure.CHAIN_INCOMPLETE
    assert assessment.failure_phase is ContinuousChainPhase.SHOOT
    assert assessment.focal_agent_id == "red.finisher"


def test_phase_owner_and_time_order_are_fail_closed() -> None:
    events = _events()
    forged = replace(events[1], agent_id="red.playmaker")
    with pytest.raises(ValueError, match="phase, owner, ball, or time order"):
        assess_continuous_option_chain(_request(), (events[0], forged, *events[2:]))

    overlapping = replace(events[1], started_at_sec=0.9)
    with pytest.raises(ValueError, match="phase, owner, ball, or time order"):
        assess_continuous_option_chain(_request(), (events[0], overlapping, *events[2:]))


def test_pass_bias_is_learned_from_physics_but_single_sample_stays_preliminary() -> None:
    sample = PassAimCalibrationSample(
        outcome_hash=_hash("outcome"),
        physical_policy_hash=_hash("kick-prior"),
        context_hash=_hash("context"),
        trajectory_digest=_hash("trajectory"),
        requested_aim_m=(3.55, 0.15, 0.115),
        observed_delivery_m=(2.31, -0.17, 0.115),
        contact_observed=True,
        safe=True,
        exact_replay=True,
    )

    actor = train_pass_aim_residual_actor((sample,))

    assert actor.residual_mean_m == pytest.approx((1.24, 0.32, 0.0))
    assert actor.propose_aim((2.31, -0.17, 0.115)) == pytest.approx((3.55, 0.15, 0.115))
    assert not actor.deployment_ready


def test_pass_calibration_rejects_mixed_physical_policy_lineage() -> None:
    first = PassAimCalibrationSample(
        outcome_hash=_hash("outcome-a"),
        physical_policy_hash=_hash("kick-a"),
        context_hash=_hash("context-a"),
        trajectory_digest=_hash("trajectory-a"),
        requested_aim_m=(3.0, 0.0, 0.115),
        observed_delivery_m=(2.0, 0.0, 0.115),
        contact_observed=True,
        safe=True,
        exact_replay=True,
    )
    second = replace(
        first,
        outcome_hash=_hash("outcome-b"),
        physical_policy_hash=_hash("kick-b"),
    )

    with pytest.raises(ValueError, match="cannot mix"):
        train_pass_aim_residual_actor((first, second))


def test_replayed_duplicate_contexts_do_not_fake_deployment_coverage() -> None:
    first = PassAimCalibrationSample(
        outcome_hash=_hash("duplicate-outcome-0"),
        physical_policy_hash=_hash("kick"),
        context_hash=_hash("one-context"),
        trajectory_digest=_hash("one-trajectory"),
        requested_aim_m=(3.0, 0.0, 0.115),
        observed_delivery_m=(2.0, -0.2, 0.115),
        contact_observed=True,
        safe=True,
        exact_replay=True,
    )
    duplicates = tuple(
        replace(first, outcome_hash=_hash(f"duplicate-outcome-{index}")) for index in range(4)
    )

    actor = train_pass_aim_residual_actor(duplicates)

    assert len(actor.source_sample_hashes) == 4
    assert not actor.deployment_ready


def _backend_candidate(
    backend: RoleOptionBackend,
    *,
    contexts: int,
    holdout: bool = True,
) -> RoleOptionBackendCandidate:
    return RoleOptionBackendCandidate(
        backend=backend,
        option=PhysicalSoccerOption.PASS,
        artifact_hash=_hash(f"{backend.value}-actor"),
        evidence_hash=_hash(f"{backend.value}-evidence"),
        distinct_context_count=contexts,
        distinct_trajectory_count=contexts,
        strict_replay=True,
        holdout_passed=holdout,
        parent_retention_passed=True,
    )


def test_playmaker_routes_to_mature_role_backend_not_generic_kick_prior() -> None:
    playmaker = next(cell for cell in _cells() if cell.agent_id == "red.playmaker")
    generic = _backend_candidate(RoleOptionBackend.GENERIC_FREEKICK, contexts=100)
    seed = _backend_candidate(RoleOptionBackend.PASS_AIM_RESIDUAL, contexts=1)
    mature = _backend_candidate(RoleOptionBackend.DYNAMIC_LEAD_PASS, contexts=8)

    route = select_role_option_backend(
        cell=playmaker,
        option=PhysicalSoccerOption.PASS,
        candidates=(generic, seed, mature),
    )

    assert route.accepted
    assert route.selected_backend is RoleOptionBackend.DYNAMIC_LEAD_PASS
    assert route.candidate_hash == mature.candidate_hash


def test_unqualified_seed_fails_closed_and_wrong_role_cannot_request_pass() -> None:
    cells = _cells()
    playmaker = next(cell for cell in cells if cell.agent_id == "red.playmaker")
    finisher = next(cell for cell in cells if cell.agent_id == "red.finisher")
    route = select_role_option_backend(
        cell=playmaker,
        option=PhysicalSoccerOption.PASS,
        candidates=(_backend_candidate(RoleOptionBackend.PASS_AIM_RESIDUAL, contexts=1),),
    )

    assert not route.accepted
    assert route.reason == "NO_ROLE_QUALIFIED_BACKEND"
    with pytest.raises(ValueError, match="does not own"):
        select_role_option_backend(
            cell=finisher,
            option=PhysicalSoccerOption.PASS,
            candidates=(),
        )
