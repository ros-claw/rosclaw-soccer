import numpy as np
import pytest

from scripts import rsi_review_receiver_transition_pair as review


def test_target_jump_measures_actual_executed_targets(monkeypatch, tmp_path):
    target = np.zeros((14, 29))
    target[3:, 0] = 0.1
    target[4:, 0] = 0.14
    p = tmp_path / "physics.npz"
    np.savez(p, target_b=target)
    monkeypatch.setattr(review, "bind", lambda *args: p)
    result = review.transition_target_metrics(tmp_path, {"handoff": {"control": 3}})
    assert result["first_target_jump_rad"] == pytest.approx(0.1)
    assert result["transition_max_target_jump_rad"] == pytest.approx(0.1)
    assert result["observed_transition_controls"] == 10


def test_feasibility_uses_recorded_reference_and_actual_previous_target(monkeypatch, tmp_path):
    target = np.zeros((5, 29))
    target[:3, 0] = 0.7
    target[3:, 0] = 0.2
    physics = tmp_path / "physics.npz"
    neural = tmp_path / "neural.npz"
    np.savez(physics, target_b=target)
    np.savez(neural, control=np.array([3]), NN_reference_nominal=np.zeros((1, 29)))
    monkeypatch.setattr(
        review, "bind", lambda root, row, key: physics if key == "physics" else neural
    )
    codec = (np.zeros(29), np.ones(29), np.arange(29))
    result = review.transition_target_metrics(tmp_path, {"handoff": {"control": 3}}, codec)
    assert result["first_target_jump_rad"] == pytest.approx(0.5)
    assert result["unavoidable_first_jump_lower_bound_rad"] == pytest.approx(0.49998)
    assert result["limiting_joint"] == review.G1_DDS_JOINT_NAMES[0]
    np.savez(neural, control=np.array([4]), NN_reference_nominal=np.zeros((1, 29)))
    with pytest.raises(ValueError, match="first handoff"):
        review.transition_target_metrics(tmp_path, {"handoff": {"control": 3}}, codec)


def test_tracking_binds_sequential_commands_to_own_physics(monkeypatch, tmp_path):
    from types import SimpleNamespace

    physics = tmp_path / "physics.npz"
    neural = tmp_path / "neural.npz"
    qpos = np.zeros((2, 10, 7))
    qpos[:, :, 3] = 1
    qvel = np.zeros((2, 10, 6))
    qvel[:, -1, 0] = 0.5
    forces = np.zeros((2, 10, 2, 3))
    forces[0, 3, 0, 0] = 11
    np.savez(physics, qpos=qpos, qvel=qvel, ball_forces=forces)
    np.savez(
        neural,
        control=np.arange(2),
        command=np.tile([0.5, 0, 0], (2, 1)),
        contact_flags_before=np.zeros((2, 2), bool),
    )
    monkeypatch.setattr(
        review, "bind", lambda root, row, key: physics if key == "physics" else neural
    )
    result = review.receiver_tracking(tmp_path, {}, SimpleNamespace(rq=0, rv=0))
    assert result["all_before_receiver_touch"][
        "precontact_velocity_error_mean_mps"
    ] == pytest.approx(0)
    assert result["after_passer_foot_before_receiver_touch"]["precontact_observations"] == 1
    assert result["first_passer_foot_control"] == 0


def test_no_handoff_cannot_hide_a_different_physical_history():
    base = {
        "handoff": None,
        "artifacts": {
            name: {"hash": "same"}
            for name in ("initial", "physics", "receiver_velocity", "receiver_shadow")
        },
    }
    candidate = {
        "handoff": None,
        "artifacts": {name: dict(item) for name, item in base["artifacts"].items()},
    }
    assert review.verify_before_handoff(None, base, candidate) is None
    candidate["artifacts"]["physics"]["hash"] = "different"
    with pytest.raises(ValueError):
        review.verify_before_handoff(None, base, candidate)
