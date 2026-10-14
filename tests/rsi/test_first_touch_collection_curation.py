import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_collection_curation import curate_collection
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
from rosclaw_soccer.rsi.first_touch_recovery import FirstTouchRecoveryLedger, file_hash
from rosclaw_soccer.rsi.public_first_touch_training_plan import PublicFirstTouchTrainingPlan
from rosclaw_soccer.sim.contracts import hash_json


def receipt(path: Path, **value):
    value.update(promotion_authorized=False, hardware_authorized=False)
    value["report_hash"] = hash_json(value)
    path.write_text(json.dumps(value))


def bank(tmp_path: Path):
    plan = PublicFirstTouchTrainingPlan((20261060,), "sha256:" + "a" * 64, 18)
    ledger = FirstTouchRecoveryLedger(tmp_path / "ledger", plan)
    force = np.zeros((300, 1, 6))
    force[260, 0, 0] = 2
    trace = dict(
        force_n=force,
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
    )
    raw = tmp_path / "trace.npz"
    np.savez_compressed(raw, **trace)
    label = tmp_path / "label.json"
    receipt(
        label,
        label=label_first_touch(trace).to_dict(),
        physical_report_hash="physical",
        outcome_contract_hash=outcome_contract()["contract_hash"],
    )
    replay = tmp_path / "replay.json"
    receipt(
        replay,
        reviewed_report_hash="physical",
        physical_substeps=3000,
        actual_recurrent_sampling_draws_reconstructed=270,
        actual_mujoco_dynamics_replayed=True,
        actual_pd_torque_reconstructed=True,
        high_quality=False,
    )
    foundation = tmp_path / "foundation.json"
    receipt(
        foundation,
        physical_report_hash="physical",
        physical_trace_hash=file_hash(raw),
        recomputed_executed_calls=300,
    )
    view = tmp_path / "view.json"
    view.write_text("{}")
    paths = dict(trace=raw, label=label, replay=replay, foundation=foundation, sampling_view=view)
    return ledger, paths


def test_all_censored_failures_are_retained(tmp_path: Path):
    ledger, paths = bank(tmp_path)
    for group in range(16):
        ledger.record_episode(group, paths)
    report = curate_collection(ledger)
    assert report["event_counts"] == {"POST_CONTACT_CENSORED": 16}
    assert [s["episodes"] for s in report["strata"]] == [4] * 4
    assert [s["high_quality"] for s in report["strata"]] == [0] * 4
    assert report["independent_contexts"] == 4
    assert report["complete_learning_bank"] is False
    assert report["new_physics_executions"] == 0


def test_partial_collection_cannot_be_curated(tmp_path: Path):
    ledger, paths = bank(tmp_path)
    ledger.record_episode(0, paths)
    with pytest.raises(ValueError, match="whole"):
        curate_collection(ledger)


def test_consistent_seals_do_not_hide_cross_episode_report_mismatch(tmp_path: Path):
    ledger, paths = bank(tmp_path)
    value = json.loads(paths["foundation"].read_text())
    del value["report_hash"]
    value["physical_report_hash"] = "different"
    receipt(paths["foundation"], **value)
    for group in range(16):
        ledger.record_episode(group, paths)
    with pytest.raises(ValueError, match="witnesses"):
        curate_collection(ledger)
