"""Fail-closed validation of one passer/shooter/goalkeeper evidence bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.physics.rolling_authenticity import measure_rolling_authenticity
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest

Array: TypeAlias = NDArray[np.generic]
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024
_ROLES = ("passer", "shooter", "goalkeeper")


@dataclass(frozen=True)
class ThreePlayerEvidenceBundle:
    evidence_path: Path
    request_path: Path
    trajectory_path: Path
    report: dict[str, Any]
    request: dict[str, Any]
    trajectory: dict[str, Array]
    evidence_hash: str
    request_hash: str
    trajectory_hash: str
    trajectory_digest: str


def validate_three_player_evidence(
    evidence_path: Path,
    *,
    source_checkout: Path,
    allow_development_candidate: bool = False,
) -> ThreePlayerEvidenceBundle:
    """Validate immutable shared-world evidence before any rendering starts."""

    checkout = source_checkout.expanduser().resolve()
    if not checkout.is_dir() or not (checkout / "pyproject.toml").is_file():
        raise ValueError("three-player source checkout is missing or invalid")
    evidence = evidence_path.expanduser().resolve()
    _require_outside_checkout(evidence, checkout, "evidence")
    request_path = evidence.parent / "request.json"
    trajectory_path = evidence.parent / "trajectory.npz"
    for path, label in ((request_path, "request"), (trajectory_path, "trajectory")):
        _require_outside_checkout(path, checkout, label)

    report = _load_json(evidence, "evidence")
    request = _load_json(request_path, "request")
    trajectory = load_three_player_trajectory(trajectory_path)
    request_hash = _file_hash(request_path)
    trajectory_hash = _file_hash(trajectory_path)
    digest = trajectory_digest(trajectory)

    if report.get("request_hash") != request_hash:
        raise ValueError("three-player request hash mismatch")
    if report.get("trajectory_hash") != trajectory_hash:
        raise ValueError("three-player trajectory file hash mismatch")
    if report.get("trajectory_digest") != digest:
        raise ValueError("three-player trajectory content digest mismatch")
    _validate_implementation(report, checkout)
    _validate_authority(report, allow_development_candidate=allow_development_candidate)
    _validate_metrics(report, allow_development_candidate=allow_development_candidate)
    _validate_request(request, report)
    _validate_pass_roll(report, request, trajectory)

    return ThreePlayerEvidenceBundle(
        evidence_path=evidence,
        request_path=request_path,
        trajectory_path=trajectory_path,
        report=report,
        request=request,
        trajectory=trajectory,
        evidence_hash=_file_hash(evidence),
        request_hash=request_hash,
        trajectory_hash=trajectory_hash,
        trajectory_digest=digest,
    )


def load_three_player_trajectory(path: Path) -> dict[str, Array]:
    """Load a bounded no-pickle three-G1 trajectory on one strict timeline."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not 1 <= resolved.stat().st_size <= _MAX_TRAJECTORY_BYTES:
        raise ValueError("three-player trajectory is missing, empty, or too large")
    with np.load(resolved, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}
    expected: dict[str, tuple[int, ...]] = {
        "time": (),
        "ball_pose": (7,),
        "ball_velocity": (6,),
    }
    for role in _ROLES:
        expected[f"{role}_pelvis_pose"] = (7,)
        expected[f"{role}_joint_position"] = (29,)
    for name, shape in expected.items():
        array = np.asarray(value.get(name))
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
            raise ValueError(f"three-player trajectory {name} has invalid shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"three-player trajectory {name} is non-finite")
    if len({len(np.asarray(value[name])) for name in expected}) != 1:
        raise ValueError("three-player trajectory arrays do not share one timeline")
    time = np.asarray(value["time"], dtype=np.float64)
    if len(time) < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("three-player trajectory time must be strictly increasing")
    for name in ("ball_pose", *(f"{role}_pelvis_pose" for role in _ROLES)):
        norms = np.linalg.norm(np.asarray(value[name])[:, 3:], axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError(f"three-player trajectory {name} has a zero quaternion")
    return value


def _validate_authority(
    report: dict[str, Any], *, allow_development_candidate: bool = False
) -> None:
    required_true = (
        "strict_replay",
        "simultaneous_three_body_physics",
        "shared_ball_state",
        "unified_physics_and_render_scene",
    )
    if any(report.get(name) is not True for name in required_true):
        raise ValueError("three-player evidence is not a passing strict shared-world replay")
    if report.get("passed") is not True and not allow_development_candidate:
        raise ValueError("three-player evidence is not a passing strict shared-world replay")
    if allow_development_candidate and report.get("promotion_status") != "REJECTED_DEVELOPMENT":
        raise ValueError("three-player development candidate must be explicitly rejected")
    if report.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError("three-player evidence is not SIM_ONLY")
    if report.get("physics_authority") != "CPU_MUJOCO":
        raise ValueError("three-player evidence does not declare CPU MuJoCo authority")
    if report.get("hardware_command_sent") is not False:
        raise ValueError("three-player evidence contains a hardware command claim")
    claims = report.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("three-player evidence claims are missing")
    if claims.get("real_hardware") is not False:
        raise ValueError("three-player evidence contains a real-hardware claim")
    if claims.get("pixels_used_for_promotion") is not False:
        raise ValueError("three-player evidence must not use pixels for scoring")


def _validate_metrics(report: dict[str, Any], *, allow_development_candidate: bool = False) -> None:
    if report.get("schema_version") == "rosclaw_soccer.goalkeeper_block_evidence.v1":
        _validate_goalkeeper_block_metrics(
            report,
            allow_development_candidate=allow_development_candidate,
        )
        return
    if report.get("schema_version") == "rosclaw_soccer.g1_imitation_evidence.v1":
        _validate_imitation_metrics(report)
        return
    if report.get("schema_version") == "rosclaw_soccer.g1_composite_imitation_evidence.v1":
        _validate_composite_imitation_metrics(report)
        return
    if report.get("schema_version") == "rosclaw_soccer.g1_agility_evidence.v1":
        _validate_agility_metrics(report)
        return
    if report.get("schema_version") == "rosclaw_soccer.g1_follow_through_evidence.v1":
        _validate_follow_through_metrics(report)
        return
    result = report.get("result")
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise ValueError("three-player result is not passing")
    required_true = (
        "finite_state",
        "pass_contact_observed",
        "shot_contact_observed",
        "goal_crossed",
        "goalkeeper_enabled",
        "pass_precision_passed",
    )
    if any(result.get(name) is not True for name in required_true):
        raise ValueError("three-player result is missing a required physical event")
    required_false = (
        "joint_limit_violation",
        "torque_limit_violation",
        "actuator_saturation",
        "passer_post_kick_fall",
        "shooter_post_kick_fall",
        "goalkeeper_joint_limit_violation",
    )
    if any(result.get(name) is not False for name in required_false):
        raise ValueError("three-player result violates a safety or stability gate")
    bounds = (
        (report.get("pass_distance_m"), 2.75, "pass distance"),
        (report.get("shot_distance_m"), 6.25, "shot distance"),
        (result.get("pass_delivery_error_m"), 0.10, "pass delivery error", True),
        (result.get("target_error_m"), 0.10, "shot target error", True),
        (result.get("goalkeeper_lateral_displacement_m"), 0.75, "goalkeeper displacement"),
        (result.get("goalkeeper_min_pelvis_height_m"), 0.65, "goalkeeper height"),
    )
    for item in bounds:
        value, limit, label, *upper = item
        number = _finite_number(value, str(label))
        development_keeper_gap = allow_development_candidate and label == "goalkeeper displacement"
        if not np.isfinite(number) or (
            not development_keeper_gap and (number > limit if upper else number < limit)
        ):
            raise ValueError(f"three-player {label} fails its gate")
    if allow_development_candidate:
        if result.get("goalkeeper_ball_contact_observed") is not False:
            raise ValueError("three-player rejected development outcome is inconsistent")
        if (
            _finite_number(
                result.get("goalkeeper_anticipation_active_fraction"),
                "goalkeeper anticipation fraction",
            )
            <= 0.0
        ):
            raise ValueError("three-player development goalkeeper did not anticipate")
    if _finite_number(report.get("pass_speed_max_positive_step_mps"), "pass speed step") > 0.03:
        raise ValueError("three-player pass speed contains a non-physical positive jump")
    pass_time = _finite_number(result.get("pass_contact_time_sec"), "pass contact time")
    shot_time = _finite_number(result.get("shot_contact_time_sec"), "shot contact time")
    if not (np.isfinite(pass_time) and np.isfinite(shot_time) and 0.0 < pass_time < shot_time):
        raise ValueError("three-player contact order is invalid")


def _validate_goalkeeper_block_metrics(
    report: dict[str, Any], *, allow_development_candidate: bool = False
) -> None:
    accepted_rejected_save = bool(
        allow_development_candidate
        and report.get("passed") is False
        and report.get("promotion_status") == "REJECTED_DEVELOPMENT"
    )
    if report.get("passed") is not True and not accepted_rejected_save:
        raise ValueError("goalkeeper block evidence did not pass its development gate")
    if (
        not accepted_rejected_save
        and report.get("promotion_status") != "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED"
    ):
        raise ValueError("goalkeeper block promotion state is invalid")
    result = report.get("result")
    if not isinstance(result, dict) or result.get("passed") is not False:
        raise ValueError("goalkeeper save must remain distinct from attack success")
    required_true = (
        "finite_state",
        "pass_contact_observed",
        "shot_contact_observed",
        "goalkeeper_enabled",
        "pass_precision_passed",
        "goalkeeper_ball_contact_observed",
        "goalkeeper_save_observed",
    )
    if any(result.get(name) is not True for name in required_true):
        raise ValueError("goalkeeper block is missing a required physical event")
    required_false = (
        "goal_crossed",
        "joint_limit_violation",
        "torque_limit_violation",
        "actuator_saturation",
        "passer_post_kick_fall",
        "shooter_post_kick_fall",
        "goalkeeper_joint_limit_violation",
    )
    if any(result.get(name) is not False for name in required_false):
        raise ValueError("goalkeeper block violates its save or safety gate")
    if not accepted_rejected_save:
        if report.get("baseline_goal_crossed") is not True:
            raise ValueError("goalkeeper block is missing a scoring counterfactual")
        if report.get("baseline_goalkeeper_contact_observed") is not False:
            raise ValueError("goalkeeper baseline unexpectedly contacts the ball")
    bounds = (
        (report.get("pass_distance_m"), 2.75, "pass distance", False),
        (report.get("shot_to_block_distance_m"), 5.0, "shot-to-block distance", False),
        (result.get("pass_delivery_error_m"), 0.10, "pass delivery error", True),
        (result.get("goalkeeper_min_pelvis_height_m"), 0.65, "goalkeeper height", False),
    )
    for value, limit, label, upper in bounds:
        number = _finite_number(value, label)
        if number > limit if upper else number < limit:
            raise ValueError(f"goalkeeper block {label} fails its gate")
    search = report.get("search")
    selected = search.get("selected_trial") if isinstance(search, dict) else None
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise ValueError("goalkeeper block has no eligible selected policy")
    if _finite_number(selected.get("safety_cost"), "goalkeeper safety cost") != 0.0:
        raise ValueError("goalkeeper selected policy has a non-zero safety cost")
    if _finite_number(selected.get("post_contact_speed_ratio"), "post-contact speed ratio") > 0.80:
        raise ValueError("goalkeeper block amplifies the post-contact ball speed")
    if selected.get("policy_hash") != report.get("selected_policy_hash"):
        raise ValueError("goalkeeper selected policy hash mismatch")
    claims = report.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("goalkeeper block claims are missing")
    if claims.get("goalkeeper_save_achieved") is not True:
        raise ValueError("goalkeeper save claim is missing")
    if claims.get("candidate_promoted") is not False:
        raise ValueError("goalkeeper development candidate claims promotion")
    pass_time = _finite_number(result.get("pass_contact_time_sec"), "pass contact time")
    shot_time = _finite_number(result.get("shot_contact_time_sec"), "shot contact time")
    save_time = _finite_number(
        result.get("goalkeeper_ball_contact_time_sec"), "goalkeeper contact time"
    )
    if not 0.0 < pass_time < shot_time < save_time:
        raise ValueError("goalkeeper block event order is invalid")


def _validate_imitation_metrics(report: dict[str, Any]) -> None:
    if report.get("passed") is not True:
        raise ValueError("imitation evidence did not pass its development gate")
    if report.get("promotion_status") != "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED":
        raise ValueError("imitation promotion state is invalid")
    result = report.get("result")
    required_true = (
        "passed",
        "finite_state",
        "pass_contact_observed",
        "shot_contact_observed",
        "goal_crossed",
        "goalkeeper_enabled",
        "pass_precision_passed",
    )
    if not isinstance(result, dict) or any(result.get(name) is not True for name in required_true):
        raise ValueError("imitation result is missing a required physical event")
    required_false = (
        "joint_limit_violation",
        "torque_limit_violation",
        "actuator_saturation",
        "passer_post_kick_fall",
        "shooter_post_kick_fall",
        "goalkeeper_joint_limit_violation",
    )
    if any(result.get(name) is not False for name in required_false):
        raise ValueError("imitation result violates a safety or stability gate")
    search = report.get("search")
    selected = search.get("selected_trial") if isinstance(search, dict) else None
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise ValueError("imitation evidence has no eligible selected trial")
    if selected.get("candidate_hash") != report.get("selected_candidate_hash"):
        raise ValueError("imitation selected candidate hash mismatch")
    if result.get("shooter_motion_prior_hash") != report.get("motion_prior_hash"):
        raise ValueError("imitation runtime motion-prior hash mismatch")
    if _finite_number(result.get("target_error_m"), "shot target error") > 0.10:
        raise ValueError("imitation shot target error fails its gate")
    if _finite_number(report.get("pass_speed_max_positive_step_mps"), "pass speed step") > 0.03:
        raise ValueError("imitation pass speed contains a non-physical positive jump")
    claims = report.get("claims")
    if not isinstance(claims, dict) or any(
        claims.get(name) is not True
        for name in (
            "motiondecode_whole_body_position_teacher",
            "motiondecode_whole_body_velocity_teacher",
        )
    ):
        raise ValueError("imitation teacher claims are missing")
    if claims.get("candidate_promoted") is not False:
        raise ValueError("imitation development candidate claims promotion")


def _validate_composite_imitation_metrics(report: dict[str, Any]) -> None:
    if report.get("passed") is not True:
        raise ValueError("composite imitation evidence did not pass its development gate")
    if report.get("promotion_status") != "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED":
        raise ValueError("composite imitation promotion state is invalid")
    result = report["result"]
    search = report["search"]
    selected = search["selected_trial"]
    required_true = (
        "passed",
        "finite_state",
        "pass_contact_observed",
        "shot_contact_observed",
        "goal_crossed",
        "goalkeeper_enabled",
        "pass_precision_passed",
    )
    if not isinstance(result, dict) or any(result.get(name) is not True for name in required_true):
        raise ValueError("composite imitation result is missing a required physical event")
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise ValueError("composite imitation has no eligible selected trial")
    if selected.get("candidate_hash") != report.get("selected_candidate_hash"):
        raise ValueError("composite imitation selected candidate hash mismatch")
    if result.get("shooter_motion_prior_hash") != report.get("motion_prior_hash"):
        raise ValueError("composite imitation runtime motion-prior hash mismatch")
    if result.get("shooter_contact_prior_hash") != report.get("contact_prior_hash"):
        raise ValueError("composite imitation runtime contact-prior hash mismatch")
    if _finite_number(result.get("target_error_m"), "shot target error") > 0.03:
        raise ValueError("composite imitation shot target error fails its 3 cm gate")
    tracking = report.get("candidate_contact_tracking")
    if (
        not isinstance(tracking, dict)
        or _finite_number(tracking.get("active_fraction"), "contact teacher active fraction") <= 0.0
    ):
        raise ValueError("composite imitation contact teacher did not execute")
    if selected.get("omnicontact_tracking") != tracking:
        raise ValueError("composite imitation selected contact metrics mismatch")
    claims = report.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("composite imitation claims are missing")
    if claims.get("omnicontact_train_only_contact_teacher") is not True:
        raise ValueError("composite imitation train-only claim is missing")
    if claims.get("omnicontact_heldout_metrics_accessed") is not False:
        raise ValueError("composite imitation accessed held-out metrics")
    if claims.get("teacher_direct_torque_output") is not False:
        raise ValueError("composite imitation teacher bypasses the PD safety chain")
    if claims.get("candidate_promoted") is not False:
        raise ValueError("composite imitation development candidate claims promotion")


def _validate_agility_metrics(report: dict[str, Any]) -> None:
    if report.get("passed") is not True:
        raise ValueError("agility evidence did not pass its development gate")
    if report.get("promotion_status") != "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED":
        raise ValueError("agility promotion state is invalid")
    result = report.get("result")
    search = report.get("search")
    if not isinstance(result, dict) or not isinstance(search, dict):
        raise ValueError("agility result or search is missing")
    selected = search.get("selected_trial")
    neighborhood = search.get("neighborhood")
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise ValueError("agility search has no eligible selected trial")
    if not isinstance(neighborhood, dict) or neighborhood.get("passed") is not True:
        raise ValueError("agility local neighborhood gate did not pass")
    if selected.get("candidate_hash") != report.get("selected_candidate_hash"):
        raise ValueError("agility selected candidate hash mismatch")
    required_true = (
        "passed",
        "finite_state",
        "pass_contact_observed",
        "shot_contact_observed",
        "goal_crossed",
        "goalkeeper_enabled",
        "pass_precision_passed",
    )
    if any(result.get(name) is not True for name in required_true):
        raise ValueError("agility result is missing a required physical event")
    if result.get("shooter_motion_prior_hash") != report.get("motion_prior_hash"):
        raise ValueError("agility runtime motion-prior hash mismatch")
    if result.get("shooter_contact_prior_hash") != report.get("contact_prior_hash"):
        raise ValueError("agility runtime contact-prior hash mismatch")
    naturalness = report.get("candidate_naturalness")
    if not isinstance(naturalness, dict):
        raise ValueError("agility naturalness metrics are missing")
    bounds = (
        (result.get("target_error_m"), 0.03, "agility shot target error"),
        (
            naturalness.get("post_contact_support_slip_m"),
            0.06,
            "agility support slip",
        ),
        (
            naturalness.get("post_contact_peak_backward_velocity_mps"),
            0.01,
            "agility backward speed",
        ),
    )
    for value, upper, label in bounds:
        if _finite_number(value, label) > upper:
            raise ValueError(f"{label} fails its gate")
    claims = report.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("agility claims are missing")
    for name in (
        "joint_group_position_velocity_authority_separated",
        "counterfactual_parent_retained",
        "local_neighborhood_gate_passed",
        "motiondecode_whole_body_teacher",
        "omnicontact_train_only_contact_teacher",
    ):
        if claims.get(name) is not True:
            raise ValueError(f"agility claim {name} is missing")
    if claims.get("omnicontact_heldout_metrics_accessed") is not False:
        raise ValueError("agility accessed held-out metrics")
    if claims.get("teacher_direct_torque_output") is not False:
        raise ValueError("agility teacher bypasses the PD safety chain")
    if claims.get("candidate_promoted") is not False:
        raise ValueError("agility development candidate claims promotion")


def _validate_follow_through_metrics(report: dict[str, Any]) -> None:
    if report.get("passed") is not True:
        raise ValueError("follow-through evidence did not pass its development gate")
    if report.get("promotion_status") != "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED":
        raise ValueError("follow-through promotion state is invalid")
    result = report.get("result")
    search = report.get("search")
    if not isinstance(result, dict) or not isinstance(search, dict):
        raise ValueError("follow-through result or search is missing")
    selected = search.get("selected_trial")
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise ValueError("follow-through search has no eligible selected trial")
    if selected.get("candidate_hash") != report.get("selected_candidate_hash"):
        raise ValueError("follow-through selected candidate hash mismatch")
    if _finite_number(search.get("neighborhood_eligible_fraction"), "follow-through basin") < 0.8:
        raise ValueError("follow-through local neighborhood gate did not pass")
    if result.get("shooter_motion_prior_hash") != report.get("motion_prior_hash"):
        raise ValueError("follow-through runtime motion-prior hash mismatch")
    if result.get("shooter_contact_prior_hash") != report.get("contact_prior_hash"):
        raise ValueError("follow-through runtime contact-prior hash mismatch")
    if result.get("shooter_agility_prior_hash") != report.get("mosaic_prior_hash"):
        raise ValueError("follow-through runtime MOSAIC-prior hash mismatch")
    for name in (
        "passed",
        "finite_state",
        "pass_contact_observed",
        "shot_contact_observed",
        "goal_crossed",
        "goalkeeper_enabled",
        "pass_precision_passed",
    ):
        if result.get(name) is not True:
            raise ValueError(f"follow-through physical event {name} is missing")
    for name in (
        "joint_limit_violation",
        "torque_limit_violation",
        "actuator_saturation",
        "passer_post_kick_fall",
        "shooter_post_kick_fall",
        "goalkeeper_joint_limit_violation",
    ):
        if result.get(name) is not False:
            raise ValueError(f"follow-through safety gate {name} failed")
    naturalness = report.get("candidate_naturalness")
    parent = report.get("parent_follow_through")
    candidate = report.get("candidate_follow_through")
    if not all(isinstance(value, dict) for value in (naturalness, parent, candidate)):
        raise ValueError("follow-through metrics are missing")
    assert isinstance(naturalness, dict)
    assert isinstance(parent, dict)
    assert isinstance(candidate, dict)
    for value, upper, label in (
        (result.get("target_error_m"), 0.03, "target error"),
        (naturalness.get("post_contact_support_slip_m"), 0.06, "support slip"),
        (naturalness.get("post_contact_peak_backward_velocity_mps"), 0.01, "backward speed"),
    ):
        if _finite_number(value, f"follow-through {label}") > upper:
            raise ValueError(f"follow-through {label} fails its gate")
    if _finite_number(
        candidate.get("arm_excursion_rms_rad"), "candidate arm excursion"
    ) < 1.08 * _finite_number(parent.get("arm_excursion_rms_rad"), "parent arm excursion"):
        raise ValueError("follow-through visible arm-excursion floor did not pass")
    if _finite_number(
        candidate.get("upper_body_motion_energy"), "candidate motion energy"
    ) < 1.10 * _finite_number(parent.get("upper_body_motion_energy"), "parent motion energy"):
        raise ValueError("follow-through visible motion-energy floor did not pass")
    claims = report.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("follow-through claims are missing")
    for name in (
        "semantic_mosaic_soccer_teacher",
        "endpoint_neutral_pose_residual",
        "arm_only_plasticity_boundary",
        "visible_plasticity_floor_passed",
        "counterfactual_parent_retained",
        "local_neighborhood_gate_passed",
    ):
        if claims.get(name) is not True:
            raise ValueError(f"follow-through claim {name} is missing")
    if claims.get("teacher_direct_torque_output") is not False:
        raise ValueError("follow-through teacher bypasses the PD safety chain")
    if claims.get("candidate_promoted") is not False:
        raise ValueError("follow-through development candidate claims promotion")


def _validate_request(request: dict[str, Any], report: dict[str, Any]) -> None:
    if request.get("activation_ceiling") != "SIM_ONLY":
        raise ValueError("three-player request is not SIM_ONLY")
    if request.get("physics_authority") != "CPU_MUJOCO":
        raise ValueError("three-player request does not declare CPU MuJoCo authority")
    if request.get("body_hash") != report.get("body_hash"):
        raise ValueError("three-player request Body hash does not match evidence")
    goal = request.get("goal_spec")
    target = request.get("physical_scoring_target_m")
    if not isinstance(goal, dict) or not isinstance(target, list | tuple) or len(target) != 3:
        raise ValueError("three-player request goal or physical target is missing")
    expected = (
        _finite_number(goal.get("plane_x_m"), "goal plane"),
        _finite_number(goal.get("target_y_m"), "goal target y"),
        _finite_number(goal.get("target_z_m"), "goal target z"),
    )
    actual = tuple(_finite_number(value, "physical scoring target") for value in target)
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-9):
        raise ValueError("three-player physical target does not match the goal contract")
    if report.get("schema_version") == "rosclaw_soccer.three_role_development_evidence.v1":
        policies = request.get("role_policy_hashes")
        if not isinstance(policies, dict) or policies != report.get("role_policy_hashes"):
            raise ValueError("three-player role policy hashes do not match the request")
    if report.get("schema_version") == "rosclaw_soccer.goalkeeper_block_evidence.v1":
        if request.get("schema_version") != "rosclaw_soccer.goalkeeper_block_request.v1":
            raise ValueError("goalkeeper block request schema is invalid")
        if request.get("selected_policy_hash") != report.get("selected_policy_hash"):
            raise ValueError("goalkeeper block request policy hash mismatch")
    if report.get("schema_version") == "rosclaw_soccer.g1_imitation_evidence.v1":
        if request.get("schema_version") != "rosclaw_soccer.g1_imitation_request.v1":
            raise ValueError("imitation request schema is invalid")
        if request.get("selected_candidate_hash") != report.get("selected_candidate_hash"):
            raise ValueError("imitation request candidate hash mismatch")
        if request.get("motion_prior_hash") != report.get("motion_prior_hash"):
            raise ValueError("imitation request motion-prior hash mismatch")
        if request.get("trajectory_digest_commitment") != report.get("trajectory_digest"):
            raise ValueError("imitation request trajectory commitment mismatch")
    if report.get("schema_version") == "rosclaw_soccer.g1_composite_imitation_evidence.v1":
        if request.get("schema_version") != "rosclaw_soccer.g1_composite_imitation_request.v1":
            raise ValueError("composite imitation request schema is invalid")
        if request.get("selected_candidate_hash") != report.get("selected_candidate_hash"):
            raise ValueError("composite imitation request candidate hash mismatch")
        if request.get("motion_prior_hash") != report.get("motion_prior_hash"):
            raise ValueError("composite imitation request motion-prior hash mismatch")
        if request.get("contact_prior_hash") != report.get("contact_prior_hash"):
            raise ValueError("composite imitation request contact-prior hash mismatch")
        if request.get("contact_prior_heldout_metrics_accessed") is not False:
            raise ValueError("composite imitation request accessed held-out metrics")
        if request.get("trajectory_digest_commitment") != report.get("trajectory_digest"):
            raise ValueError("composite imitation request trajectory commitment mismatch")
    if report.get("schema_version") == "rosclaw_soccer.g1_agility_evidence.v1":
        if request.get("schema_version") != "rosclaw_soccer.g1_agility_request.v1":
            raise ValueError("agility request schema is invalid")
        if request.get("selected_candidate_hash") != report.get("selected_candidate_hash"):
            raise ValueError("agility request candidate hash mismatch")
        if request.get("motion_prior_hash") != report.get("motion_prior_hash"):
            raise ValueError("agility request motion-prior hash mismatch")
        if request.get("contact_prior_hash") != report.get("contact_prior_hash"):
            raise ValueError("agility request contact-prior hash mismatch")
        if request.get("contact_prior_heldout_metrics_accessed") is not False:
            raise ValueError("agility request accessed held-out metrics")
        if request.get("trajectory_digest_commitment") != report.get("trajectory_digest"):
            raise ValueError("agility request trajectory commitment mismatch")
    if report.get("schema_version") == "rosclaw_soccer.g1_follow_through_evidence.v1":
        if request.get("schema_version") != "rosclaw_soccer.g1_follow_through_request.v1":
            raise ValueError("follow-through request schema is invalid")
        if request.get("selected_candidate_hash") != report.get("selected_candidate_hash"):
            raise ValueError("follow-through request candidate hash mismatch")
        for name in ("motion_prior_hash", "contact_prior_hash", "mosaic_prior_hash"):
            if request.get(name) != report.get(name):
                raise ValueError(f"follow-through request {name} mismatch")
        if request.get("numerical_thread_contract") != report.get("numerical_thread_contract"):
            raise ValueError("follow-through numerical thread contract mismatch")
        if request.get("trajectory_digest_commitment") != report.get("trajectory_digest"):
            raise ValueError("follow-through request trajectory commitment mismatch")


def _validate_implementation(report: dict[str, Any], checkout: Path) -> None:
    schema = report.get("schema_version")
    if schema not in {
        "rosclaw_soccer.three_role_development_evidence.v1",
        "rosclaw_soccer.goalkeeper_block_evidence.v1",
        "rosclaw_soccer.g1_imitation_evidence.v1",
        "rosclaw_soccer.g1_composite_imitation_evidence.v1",
        "rosclaw_soccer.g1_agility_evidence.v1",
        "rosclaw_soccer.g1_follow_through_evidence.v1",
    }:
        return
    digest = hashlib.sha256()
    relatives = (
        (
            Path("src/rosclaw_soccer/skills/team/development_evidence.py"),
            Path("src/rosclaw_soccer/skills/team/shared_world.py"),
        )
        if schema == "rosclaw_soccer.three_role_development_evidence.v1"
        else (
            Path("src/rosclaw_soccer/skills/team/goalkeeper_evidence.py"),
            Path("src/rosclaw_soccer/skills/team/goalkeeper_learning.py"),
            Path("src/rosclaw_soccer/skills/team/shared_world.py"),
            Path("src/rosclaw_soccer/world/field.py"),
        )
        if schema == "rosclaw_soccer.goalkeeper_block_evidence.v1"
        else (
            Path("src/rosclaw_soccer/skills/team/imitation_evidence.py"),
            Path("src/rosclaw_soccer/skills/team/imitation_learning.py"),
            Path("src/rosclaw_soccer/skills/team/shared_world.py"),
        )
        if schema == "rosclaw_soccer.g1_imitation_evidence.v1"
        else (
            Path("src/rosclaw_soccer/skills/team/agility_evidence.py"),
            Path("src/rosclaw_soccer/skills/team/agility_growth.py"),
            Path("src/rosclaw_soccer/skills/team/shared_world.py"),
        )
        if schema == "rosclaw_soccer.g1_agility_evidence.v1"
        else (
            Path("src/rosclaw_soccer/skills/team/follow_through_evidence.py"),
            Path("src/rosclaw_soccer/skills/team/follow_through_growth.py"),
            Path("src/rosclaw_soccer/skills/team/agility_growth.py"),
            Path("src/rosclaw_soccer/skills/team/shared_world.py"),
            Path("src/rosclaw_soccer/growth/mosaic_agility_prior.py"),
        )
        if schema == "rosclaw_soccer.g1_follow_through_evidence.v1"
        else (
            Path("src/rosclaw_soccer/skills/team/composite_imitation_evidence.py"),
            Path("src/rosclaw_soccer/skills/team/composite_imitation.py"),
            Path("src/rosclaw_soccer/skills/team/shared_world.py"),
        )
    )
    for relative in relatives:
        path = checkout / relative
        if not path.is_file():
            raise ValueError("three-player implementation source is missing")
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    expected = "sha256:" + digest.hexdigest()
    if report.get("implementation_hash") != expected:
        raise ValueError("three-player implementation hash mismatch")


def _validate_pass_roll(
    report: dict[str, Any],
    request: dict[str, Any],
    trajectory: dict[str, Array],
) -> None:
    result = report["result"]
    pass_time = _finite_number(result.get("pass_contact_time_sec"), "pass contact time")
    shot_time = _finite_number(result.get("shot_contact_time_sec"), "shot contact time")
    goal = request["goal_spec"]
    radius = _finite_number(goal.get("ball_radius_m", 0.115), "ball radius")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    start = int(np.searchsorted(time, pass_time + 0.10, side="left"))
    end = int(np.searchsorted(time, shot_time - 0.15, side="right"))
    if end - start < 2:
        raise ValueError("three-player pass roll interval is too short")
    metrics, _ = measure_rolling_authenticity(
        time=time[start:end],
        ball_pose=np.asarray(trajectory["ball_pose"], dtype=np.float64)[start:end],
        ball_velocity=np.asarray(trajectory["ball_velocity"], dtype=np.float64)[start:end],
        ball_radius_m=radius,
        ignore_initial_sec=0.0,
    )
    if not metrics.passed:
        raise ValueError(
            "three-player pass is sliding rather than rolling "
            f"(median_slip_ratio={metrics.median_slip_ratio:.3f})"
        )


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"three-player {label} is missing or invalid")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"three-player {label} is non-finite")
    return converted


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or not 1 <= path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError(f"three-player {label} is missing, empty, or too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"three-player {label} must be a JSON object")
    return value


def _require_outside_checkout(path: Path, checkout: Path, label: str) -> None:
    if path == checkout or checkout in path.parents:
        raise ValueError(f"three-player raw {label} must be outside the source checkout")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "ThreePlayerEvidenceBundle",
    "load_three_player_trajectory",
    "validate_three_player_evidence",
]
