from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_snapshot_sampling_frontier import snapshot
from tests.rsi.test_sampling_frontier_diagnostics import record


def sealed(value):
    return {**value, "report_hash": hash_json(value)}


def fixture(root: Path, *, wrong_context=False, fresh=False, changed_frames=False):
    folder = root / "sample-0"
    folder.mkdir()
    row = record()
    row.update(
        folder=str(folder),
        seed=123,
        lane=0,
        sampling_seed=234,
        sample_index=0,
        view_hash="sha256:" + "a" * 64,
        actual_behavior_model_hash="sha256:" + "b" * 64,
        behavior_generation=1,
        execution_origin="NEW",
        report_hash="sha256:" + "c" * 64,
    )
    review = sealed(
        {
            **{k: v for k, v in row["outcome"].items() if k != "reward"},
            "physical_substeps": 3000,
            "actual_mujoco_dynamics_replayed": True,
            "actual_pd_torque_reconstructed": True,
            "neural_target_reconstructed": True,
            "hardware_authorized": False,
            "promotion_authorized": False,
        }
    )
    write_once(folder / "review.json", review)
    row["outcome"] = {**review, "reward": 3.0}
    row["review_hash"] = review["report_hash"]
    foundation = sealed(
        dict(
            physical_report_hash=row["report_hash"],
            recomputed_executed_calls=300,
            hardware_authorized=False,
            promotion_authorized=False,
        )
    )
    write_once(root / "foundation-review-0.json", foundation)
    row["executed_torch_review_hash"] = foundation["report_hash"]
    # Synthetic tests exercise byte integrity, not a claimed physics execution.
    frames = root / "learning-frames-0.npz"
    frames.write_bytes(b"synthetic-test-bytes")
    row["learning_frames_path"] = str(frames)
    row["learning_frames_hash"] = hash_bytes(b"other" if changed_frames else frames.read_bytes())
    job = {
        k: row[k]
        for k in (
            "group",
            "seed",
            "lane",
            "sampling_seed",
            "baseline_course_index",
            "sample_index",
            "view_hash",
        )
    }
    if wrong_context:
        row["baseline_course_index"] = 1
    write_once(root / "row-0.json", row)
    write_once(
        root / "commitment.json",
        sealed(
            dict(
                schema="soccer.rsi.current_proposal_collection.v1",
                private_fresh_accessed=fresh,
                hardware_authorized=False,
                behavior_model_hash=row["actual_behavior_model_hash"],
                behavior_generation=1,
                jobs=[{**job, "group": i} for i in range(160)],
            )
        ),
    )


def test_snapshot_retains_source_and_is_not_new_physics(tmp_path):
    fixture(tmp_path)
    result = snapshot(tmp_path, tmp_path / "snapshot.json")
    assert result["completed_episodes"] == 1
    assert result["new_physical_executions"] == 0
    assert result["sealed_original_reviews_checked"] is True
    assert result["source_physics_validated_here"] is False
    assert len(result["source_file_hashes"]) == 4
    assert snapshot(tmp_path, tmp_path / "snapshot.json") == result


@pytest.mark.parametrize(
    "options,message",
    [
        ({"wrong_context": True}, "declared job"),
        ({"fresh": True}, "consumed SIM"),
        ({"changed_frames": True}, "frames changed"),
    ],
)
def test_snapshot_rejects_wrong_context_fresh_or_changed_data(tmp_path, options, message):
    fixture(tmp_path, **options)
    with pytest.raises(ValueError, match=message):
        snapshot(tmp_path, tmp_path / "snapshot.json")
    assert not (tmp_path / "snapshot.json").exists()
