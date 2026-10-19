"""Actual synthetic toy integration, never genuine G1 skill evidence."""

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.sim.contracts import hash_bytes
from tests.physics.test_g1_feedback_forecast import world
from tests.physics.test_recorded_early_episode import recorded  # noqa: F401


def body(*, record_all_ball_contacts=False):
    model, _, _ = world()
    model.dof_armature[6:35] = 0.1
    binary = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    return G1TargetEpisode(
        model,
        world_hash=hash_bytes(binary.tobytes()),
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        course=(2.0, 1.0, 0.2),
        defaults=np.zeros(29),
        record_all_ball_contacts=record_all_ball_contacts,
    )


@pytest.mark.parametrize("recorded", [True], indirect=True)
def test_whole_plain_pd_matches_separate_recorded_body(recorded):  # noqa: F811
    _, trace = recorded
    episode = body()
    for frame in range(300):
        row = episode.step(frame, trace["joint_target_rad"][frame, 0].copy())
        for key in trace:
            if key in row:
                np.testing.assert_array_equal(row[key], trace[key][frame])
    assert episode.review()["integrated_steps"] == 3000
    assert episode.review()["policy_causality_verified"] is False
    with pytest.raises(ValueError):
        episode.step(300, np.zeros(29))


@pytest.mark.parametrize(
    "target",
    [np.zeros(28), np.zeros(29, dtype=np.float32), np.full(29, np.nan), np.full(29, np.inf)],
)
def test_bad_target_rejected_before_charging(target):
    episode = body()
    with pytest.raises(ValueError):
        episode.step(0, target)
    assert episode.attempted_steps == episode.integrated_steps == 0
    assert episode.next_frame == 0


def test_handoff_keeps_state_and_clock_no_new_body():
    episode = body()
    first = episode.step(0, np.zeros(29))
    second = episode.step(1, np.full(29, 0.01))
    np.testing.assert_array_equal(
        second["canonical_qpos"][0], first["target_body_canonical_qpos_after_substep"][0, -1]
    )
    np.testing.assert_array_equal(
        second["canonical_qvel"][0], first["target_body_canonical_qvel_after_substep"][0, -1]
    )
    assert episode.attempted_steps == episode.integrated_steps == 20
    with pytest.raises(ValueError):
        episode.step(1, np.zeros(29))
    with pytest.raises(ValueError):
        episode.review()


def test_partial_failure_is_terminal(monkeypatch):
    episode = body()

    def broken(*args):
        raise RuntimeError("synthetic integration fault")

    monkeypatch.setattr(mujoco, "mj_step", broken)
    with pytest.raises(RuntimeError):
        episode.step(0, np.zeros(29))
    assert episode.failed and episode.attempted_steps == 1 and episode.integrated_steps == 0
    with pytest.raises(ValueError):
        episode.step(0, np.zeros(29))


def test_review_checks_both_native_counters():
    episode = body()
    episode.next_frame = 300
    episode.attempted_steps = episode.integrated_steps = 10
    with pytest.raises(ValueError):
        episode.review()


def test_optional_full_contact_capture_preserves_every_default_physical_field():
    plain = body()
    captured = body(record_all_ball_contacts=True)
    ground_contact_seen = False
    for frame in range(300):
        target = np.zeros(29)
        original = plain.step(frame, target)
        observed = captured.step(frame, target)
        for key, value in original.items():
            np.testing.assert_array_equal(value, observed[key])
        assert "ball_contact_force_by_body_n" not in original
        contact = observed["ball_contact_force_by_body_n"]
        assert contact.shape == (1, 10, captured.model.nbody)
        ground_contact_seen |= bool(np.any(contact[:, :, 0] > 0))
    assert ground_contact_seen
    assert plain.review() == captured.review()


def test_nonfinite_captured_contact_is_terminal_not_silently_zero(monkeypatch):
    episode = body(record_all_ball_contacts=True)

    def invalid_contact(model, data, contact_id, wrench):
        wrench[:] = np.nan

    monkeypatch.setattr(mujoco, "mj_contactForce", invalid_contact)
    with pytest.raises(ValueError, match="nonfinite native ball contact"):
        for frame in range(20):
            episode.step(frame, np.zeros(29))
    assert episode.failed
    with pytest.raises(ValueError):
        episode.step(episode.next_frame, np.zeros(29))
