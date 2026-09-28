"""Reset training design needs exact initial state and cleared contacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi import statistical_reset_contract as contract
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    folder = tmp_path / "reset"
    folder.mkdir()
    frames, envs = 30, 2
    base = {"frames": frames, "environments": [{}, {}], "report_hash": "base"}
    first = {
        "report_hash": "first",
        "minimum_replay_pelvis_z_m": 0.7,
    }
    (folder / "report.json").write_text(json.dumps(base), encoding="utf-8")
    (folder / "reset_report.json").write_text(json.dumps(first), encoding="utf-8")
    state = {}
    second_state = {}
    for name, width in contract.STATE_WIDTHS.items():
        values = np.zeros((frames, envs, width), dtype=np.float32)
        if name == "root":
            values[:, :, 2] = 0.7
        state[f"{name}_before"] = values
        state[f"{name}_after"] = values.copy()
        second_state[name] = values.copy()
    np.savez_compressed(folder / "reset_state_probe.npz", **state)
    np.savez_compressed(folder / "second_reset_state_probe.npz", **second_state)
    forces = np.zeros((frames, envs, 6), dtype=np.float32)
    forces[25, 0, 0] = 5.0
    forces[26, 1, 1] = 5.0
    trace = {
        "ball_position_m": np.zeros((frames, envs, 3), dtype=np.float32),
        "ball_angular_velocity_rad_s": np.zeros((frames, envs, 3), dtype=np.float32),
        "ball_body_contact_force_peak_n": forces,
    }
    for filename in ("trace.npz", "reset_replay.npz", "second_reset_replay.npz"):
        np.savez_compressed(folder / filename, **trace)
    second = {
        "schema": "rsi_isaac_vector_first_touch_second_reset_v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "first_reset_report_hash": "first",
        "second_trace_hash": hash_bytes((folder / "second_reset_replay.npz").read_bytes()),
        "second_state_hash": hash_bytes((folder / "second_reset_state_probe.npz").read_bytes()),
        "first_reset_contact_frames": [25, 26],
        "second_reset_contact_frames": [25, 26],
        "first_reset_contact_bodies": [[0], [1]],
        "second_reset_contact_bodies": [[0], [1]],
        "second_minimum_pelvis_z_m": 0.7,
    }
    second["report_hash"] = hash_json(second)
    (folder / "second_reset_report.json").write_text(json.dumps(second), encoding="utf-8")
    monkeypatch.setattr(
        contract,
        "audit_reset_replay",
        lambda _: {"reset_report_hash": "first", "reset_verified": False},
    )
    return folder


def test_statistical_reset_accepts_stochastic_future_not_new_courses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = _fixture(monkeypatch, tmp_path)
    report = contract.audit_statistical_reset(folder)
    assert report["training_environment_design_ready"] is True
    assert report["strict_trajectory_replay_passed"] is False
    assert report["episode_count"] == 6
    assert report["unique_course_count"] == 2
    assert report["learning_authorized"] is False


def test_stale_contact_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = _fixture(monkeypatch, tmp_path)
    with np.load(folder / "reset_replay.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    trace["ball_body_contact_force_peak_n"][0, 0, 0] = 5.0
    np.savez_compressed(folder / "reset_replay.npz", **trace)
    with pytest.raises(ValueError, match="stale"):
        contract.audit_statistical_reset(folder)
