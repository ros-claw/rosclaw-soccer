from copy import deepcopy

import pytest

from scripts.rsi_sampling_recovery_contract import (
    check_recovery_job,
    describe_recovery,
    inspect_failed_attempt,
)


def fixture(tmp_path):
    courses = [dict(seed=100 + i, lane=(i % 4) * 2) for i in range(13)]
    views = [f"fixture-{i}" for i in range(208)]
    commitment = dict(
        schema="soccer.rsi.failure_step_exploration_commitment.v1",
        partition="TRAIN_CONSUMED",
        exploration_stream=2,
        sampling_rho=0.9,
        std_raw=0.1,
        execution_timeout_s=600,
        samples_per_course=16,
        courses=[[c["seed"], c["lane"]] for c in courses],
        sampling_view_hashes=views,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    job = dict(
        gpu=0,
        arm="candidate",
        courses=courses,
        view_hashes=views,
        args=dict(
            output_root=tmp_path,
            resume=True,
            samples_per_course=16,
            shared_sampling_models=True,
            compressed_sampling_models=True,
        ),
    )
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs/seed108-lane0-sample-3-actor.log"
    log.write_text("native startup failed, zero physics reports")
    return job, commitment, log


def test_plan_is_read_only_preserves_complete_shard_and_no_authority(tmp_path):
    job, commitment, log = fixture(tmp_path)
    before = log.read_bytes()
    failed = inspect_failed_attempt(tmp_path, seed=108, lane=0, sample=3)
    declaration = describe_recovery(job, commitment, failed)
    assert declaration["course_indices"] == [0, 4, 8, 12]
    assert declaration["completed_reports_require_full_reaudit"] is True
    assert declaration["reused_reports_are_not_new_physics"] is True
    assert declaration["automatic_retry"] is False
    assert declaration["promotion_authorized"] is declaration["hardware_authorized"] is False
    assert log.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["logs"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("exploration_stream", 1),
        ("std_raw", 0.2),
        ("sampling_rho", 0.0),
        ("samples_per_course", 8),
        ("hardware_authorized", True),
        ("execution_timeout_s", 900),
    ],
)
def test_changed_sampling_protocol_cannot_be_called_same_source_recovery(tmp_path, key, value):
    job, commitment, _ = fixture(tmp_path)
    commitment[key] = value
    with pytest.raises(ValueError, match="13x16"):
        check_recovery_job(job, commitment)


def test_dropped_or_reordered_views_and_courses_rejected(tmp_path):
    job, commitment, _ = fixture(tmp_path)
    for key in ("courses", "view_hashes"):
        altered = deepcopy(job)
        altered[key] = altered[key][:-1]
        with pytest.raises(ValueError):
            check_recovery_job(altered, commitment)
        altered = deepcopy(job)
        altered[key][0], altered[key][1] = altered[key][1], altered[key][0]
        with pytest.raises(ValueError):
            check_recovery_job(altered, commitment)


@pytest.mark.parametrize("gpu", [True, -1, 4])
def test_invalid_gpu_rejected(tmp_path, gpu):
    job, commitment, _ = fixture(tmp_path)
    job["gpu"] = gpu
    with pytest.raises(ValueError):
        check_recovery_job(job, commitment)


def test_existing_report_or_symlink_log_is_not_archivable_failure(tmp_path):
    _, _, log = fixture(tmp_path)
    folder = tmp_path / "seed108-lane0-sample-3-actor"
    folder.mkdir()
    report = folder / "report.json.gz"
    report.write_bytes(b"not read by preflight")
    with pytest.raises(ValueError, match="missing physical"):
        inspect_failed_attempt(tmp_path, seed=108, lane=0, sample=3)
    report.unlink()
    real = tmp_path / "retained-log"
    log.rename(real)
    log.symlink_to(real)
    with pytest.raises(ValueError, match="missing physical"):
        inspect_failed_attempt(tmp_path, seed=108, lane=0, sample=3)


def test_wrong_shard_failure_cannot_authorize_another_shard(tmp_path):
    job, commitment, _ = fixture(tmp_path)
    failed = inspect_failed_attempt(tmp_path, seed=108, lane=0, sample=3)
    job["gpu"] = 1
    with pytest.raises(ValueError, match="original native failure"):
        describe_recovery(job, commitment, failed)
