from __future__ import annotations

import json

import pytest

from rosclaw_soccer.growth.role_option_backend import RoleOptionBackend
from rosclaw_soccer.media.role_backend_continuity_video import (
    validate_role_backend_continuity_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_backend_continuity_evidence import (
    build_dynamic_lead_pass_candidate,
)


def _hash(label: str) -> str:
    return str(hash_json({"s202": label}))


def _source(*, duplicate_contexts: bool = False, parent_passed: bool = True) -> dict:
    discovery = {}
    for index in range(8):
        phase = 1.96 if duplicate_contexts else 1.80 + index * 0.02
        yaw = 0.0 if duplicate_contexts else -0.06 + index * 0.015
        discovery[f"sample-{index}"] = {
            "sample": {
                "receiver_phase_start_sec": phase,
                "passer_yaw_delta_rad": yaw,
            },
            "trajectory_digest": _hash(f"trajectory-{index}"),
        }
    return {
        "policy_hash": _hash("policy"),
        "evidence_hash": _hash("evidence"),
        "discovery": discovery,
        "holdouts": {
            "sealed-left": {
                "strict_replay": True,
                "passed": True,
                "gates": {"beats_fixed_parent": parent_passed},
            },
            "sealed-right": {
                "strict_replay": True,
                "passed": True,
                "gates": {"beats_fixed_parent": parent_passed},
            },
        },
    }


def test_s95_adapter_becomes_role_qualified_only_with_distinct_physics() -> None:
    candidate = build_dynamic_lead_pass_candidate(_source())

    assert candidate.backend is RoleOptionBackend.DYNAMIC_LEAD_PASS
    assert candidate.distinct_context_count == 8
    assert candidate.distinct_trajectory_count == 8
    assert candidate.strict_replay
    assert candidate.holdout_passed
    assert candidate.parent_retention_passed
    assert candidate.evidence_ready


def test_duplicate_context_labels_cannot_fake_backend_coverage() -> None:
    candidate = build_dynamic_lead_pass_candidate(_source(duplicate_contexts=True))

    assert candidate.distinct_context_count == 1
    assert candidate.distinct_trajectory_count == 8
    assert not candidate.evidence_ready


def test_failed_parent_comparison_blocks_role_backend() -> None:
    candidate = build_dynamic_lead_pass_candidate(_source(parent_passed=False))

    assert not candidate.parent_retention_passed
    assert not candidate.evidence_ready


def test_video_manifest_detects_changed_video_bytes(tmp_path) -> None:
    video = tmp_path / "review.mp4"
    video.write_bytes(b"visualization-only")
    manifest = {
        "schema_version": "rosclaw_soccer.role_backend_continuity_video.v1",
        "claim": "S202_ROLE_QUALIFIED_PASS_AND_CAUSAL_HANDOFF_VISUALIZATION",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "width": 1920,
        "height": 1080,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    manifest["manifest_hash"] = hash_json(manifest)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        validate_role_backend_continuity_video_manifest(path)["video_hash"]
        == manifest["video_hash"]
    )
    video.write_bytes(b"edited")
    with pytest.raises(ValueError, match="integrity"):
        validate_role_backend_continuity_video_manifest(path)
