from __future__ import annotations

import json

import numpy as np
import pytest

from rosclaw_soccer.growth.runtime_contact_mode_actor import (
    G1RuntimeContactModeActor,
    RuntimeContactModeAction,
    RuntimeContactModeMemory,
    load_runtime_contact_mode_actor,
    save_runtime_contact_mode_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.runtime_contact_mode_growth import (
    train_runtime_contact_mode_actor,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64


def _memory(
    index: int, velocity_x: float, action: RuntimeContactModeAction
) -> RuntimeContactModeMemory:
    digit = format(index, "x")
    return RuntimeContactModeMemory(
        context_hash="sha256:" + digit * 64,
        trajectory_hash="sha256:" + "f" * 63 + digit,
        features=(3.65, 0.0, velocity_x, 0.0, 1.0, 0.0, 0.1),
        action=action,
    )


def _actor() -> G1RuntimeContactModeActor:
    early = RuntimeContactModeAction(0, 0.12)
    late = RuntimeContactModeAction(12, -0.04)
    return G1RuntimeContactModeActor(
        body_hash=_HASH_A,
        kick_prior_hash=_HASH_B,
        source_discovery_hashes=(_HASH_C,),
        training_snapshot_hash=_HASH_D,
        feature_center=(0.0,) * 7,
        feature_scale=(1.0,) * 7,
        successful_memories=(
            _memory(1, -1.50, early),
            _memory(2, -1.60, early),
            _memory(3, -2.20, late),
            _memory(4, -2.30, late),
        ),
        failed_memories=(
            _memory(5, -0.50, early),
            _memory(6, -0.60, early),
            _memory(7, -3.20, late),
            _memory(8, -3.30, late),
        ),
    )


def test_runtime_contact_mode_routes_stance_and_timing_from_measured_arrival() -> None:
    actor = _actor()
    decision = actor.decide((3.65, 0.0, -2.25, 0.0, 1.0, 0.0, 0.1))
    rejected = actor.decide((3.65, 0.0, -3.25, 0.0, 1.0, 0.0, 0.1))

    assert decision.accepted
    assert decision.action is not None
    assert decision.action.maximum_arrival_advance_frames == 12
    assert decision.action.stance_offset_x_m == -0.04
    assert not rejected.accepted
    assert rejected.route == "RUNTIME_BODY_CONTACT_FAILURE_FALLBACK"


def test_runtime_contact_mode_artifact_is_content_bound(tmp_path) -> None:
    actor = _actor()
    path = tmp_path / "actor.json"
    save_runtime_contact_mode_actor(actor, path)

    assert load_runtime_contact_mode_actor(path) == actor
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"maximum_support_distance": 2.5', '"maximum_support_distance": 2.4'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_runtime_contact_mode_actor(path)


def test_runtime_contact_mode_action_rejects_unsafe_authority() -> None:
    with pytest.raises(ValueError, match="SIM-only envelope"):
        RuntimeContactModeAction(0, 0.20)
    with pytest.raises(ValueError, match="SIM-only envelope"):
        RuntimeContactModeAction(6, 0.0)


def test_runtime_contact_mode_training_uses_exact_measured_arrivals(tmp_path) -> None:
    discovery = tmp_path / "discovery"
    discovery.mkdir()
    rows = []
    for index in range(8):
        path = discovery / f"probe-{index:03d}.npz"
        count = np.arange(6, dtype=np.int64)
        np.savez_compressed(
            path,
            shooter_causal_strike_option_incoming_observation_count=count,
            shooter_causal_strike_option_ball_arrival_eta_sec=np.linspace(1.5, 1.0, 6),
            shooter_ball_local_position=np.column_stack(
                (np.linspace(4.0, 3.5, 6), np.full(6, index * 0.01), np.full(6, 0.115))
            ),
            shooter_ball_local_velocity=np.column_stack(
                (np.full(6, -1.5 - index * 0.1), np.zeros(6), np.zeros(6))
            ),
            shooter_pelvis_local_position=np.column_stack(
                (np.zeros(6), np.zeros(6), np.full(6, 0.78))
            ),
            shooter_joint_velocity=np.full((6, 29), 0.1 + index * 0.01),
        )
        rows.append(
            {
                "probe_hash": "sha256:" + format(index + 1, "x") * 64,
                "probe": {
                    "context": {"receiver_lane_m": 0.0},
                    "maximum_arrival_advance_frames": 0 if index < 4 else 12,
                    "stance_offset_x_m": -0.12 + index * 0.03,
                    "stance_offset_y_m": -0.06,
                    "contact_policy_frame": 248,
                },
                "quality": {"chain_passed": index < 4, "safe": index != 7},
                "trajectory": {"file": path.name, "file_hash": hash_bytes(path.read_bytes())},
            }
        )
    report = {
        "schema_version": "rosclaw.growth.three_axis_contact_discovery.v1",
        "status": "PASS_THREE_AXIS_CONTACT_DISCOVERY",
        "body_hash": _HASH_A,
        "kick_prior_hash": _HASH_B,
        "rows": rows,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    report_path = discovery / "discovery-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    actor, training = train_runtime_contact_mode_actor(
        discovery_report_paths=(report_path,), output_dir=tmp_path / "training"
    )

    assert len(actor.successful_memories) == 4
    assert len(actor.failed_memories) == 4
    assert training["trajectory_count"] == 8
    assert training["decision_clock"].startswith("FIFTH_CONSECUTIVE")
