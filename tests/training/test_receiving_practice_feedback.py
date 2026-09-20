import numpy as np
import pytest

from rosclaw_soccer.training.receiving_practice_feedback import receiving_feedback_payloads


def trace():
    return {
        "time": np.array([0.02, 0.04]),
        "blue_finisher_joint_position": np.full((2, 29), 0.2),
        "blue_finisher_applied_pd_target": np.full((2, 29), 0.3),
        "blue_finisher_joint_torque": np.full((2, 29), 1.5),
    }


def convert(data, **kwargs):
    return receiving_feedback_payloads(
        data,
        agent_id="blue.finisher",
        body_id="g1_sim",
        frame_prefix="receiving",
        source_file_hash=kwargs.get("source_file_hash", "sha256:" + "a" * 64),
    )


def test_recorded_commands_and_completed_positions_remain_distinct():
    data = trace()
    snapshot = {k: v.copy() for k, v in data.items()}
    rows = convert(data)
    assert len(rows) == 2
    assert set(rows[0]["actual"].values()) == {0.2}
    assert set(rows[0]["target"].values()) == {0.3}
    assert list(rows[0]["position_error"].values()) == pytest.approx([0.1] * 29)
    assert set(rows[0]["metadata"]["commanded_torque_nm"].values()) == {1.5}
    assert not rows[0]["metadata"]["complete_command_history"]
    assert "not_measured_torque" in rows[0]["metadata"]["torque_semantics"]
    assert all(np.array_equal(v, snapshot[k]) for k, v in data.items())
    rows[0]["target"]["left_knee_joint"] = 99
    assert np.array_equal(
        data["blue_finisher_applied_pd_target"], snapshot["blue_finisher_applied_pd_target"]
    )


def test_missing_commands_stay_unknown_not_zero_or_measured_pose():
    data = trace()
    del data["blue_finisher_applied_pd_target"], data["blue_finisher_joint_torque"]
    row = convert(data)[0]
    for field in ("target", "position_error"):
        assert len(row[field]) == 29 and set(row[field].values()) == {None}
    assert set(row["metadata"]["commanded_torque_nm"].values()) == {None}
    assert not row["metadata"]["target_recorded"]


@pytest.mark.parametrize("field", ["joint_position", "applied_pd_target", "joint_torque"])
@pytest.mark.parametrize(
    "bad",
    [np.zeros((1, 29)), np.full((2, 29), np.nan), np.zeros((2, 12)), np.ones((2, 29), dtype=bool)],
)
def test_misaligned_or_invalid_measurements_rejected(field, bad):
    data = trace()
    data["blue_finisher_" + field] = bad
    with pytest.raises(ValueError):
        convert(data)


@pytest.mark.parametrize("time", [[], [0.02, 0.02], [0.02, 0.06], [float("nan")], [-0.02]])
def test_invalid_timeline(time):
    data = trace()
    data["time"] = np.asarray(time)
    with pytest.raises(ValueError):
        convert(data)


def test_content_identity_required():
    with pytest.raises(ValueError):
        convert(trace(), source_file_hash="unverified")


def test_error_overflow_rejected_not_exported_as_nonfinite():
    data = trace()
    data["blue_finisher_joint_position"][:] = -1e308
    data["blue_finisher_applied_pd_target"][:] = 1e308
    with pytest.raises(ValueError, match="finite derived"):
        convert(data)


def test_integer_subtraction_does_not_wrap():
    data = trace()
    data["blue_finisher_joint_position"] = np.full((2, 29), -(2**62), dtype=np.int64)
    data["blue_finisher_applied_pd_target"] = np.full((2, 29), 2**62, dtype=np.int64)
    assert set(convert(data)[0]["position_error"].values()) == {float(2**63)}
