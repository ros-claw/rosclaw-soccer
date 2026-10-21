from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_standing_lifecycle_audit import audit_ball_drop, audit_kick_entry


def fixture(performed):
    entry = {
        key: np.zeros(shape)
        for key, shape in (
            ("qpos", 43),
            ("qvel", 41),
            ("ctrl", 29),
            ("qacc_warmstart", 41),
            ("mocap_pos", (1, 3)),
            ("mocap_quat", (1, 4)),
            ("qfrc_applied", 41),
            ("xfrc_applied", (33, 6)),
            ("act", 0),
            ("time", ()),
        )
    }
    entry["qpos"][36] = 3 if performed else 1
    entry["mocap_pos"][0] = [6, 0, 0.11]
    raw = {"before_" + key: value.copy() for key, value in entry.items()}
    raw["performed"] = np.array(performed)
    if performed:
        after = {key: value.copy() for key, value in entry.items()}
        after["qpos"][36:43] = [0.8, -0.15, 0.4, 1, 0, 0, 0]
        raw.update({"after_" + key: value for key, value in after.items()})
    auditor = SimpleNamespace(
        canonical=lambda q, v: (q.copy(), v.copy()),
        goal_mocap_id=0,
        layout=SimpleNamespace(bq=36, bv=35),
    )
    return auditor, entry, raw


@pytest.mark.parametrize("performed", [False, True])
def test_ball_return_is_explicit_fixture_not_robot_reset(performed):
    a, entry, raw = fixture(performed)
    result = audit_ball_drop(a, raw, entry, "motion-right.npz")
    assert result["qpos"][36] == (0.8 if performed else 1)
    assert np.array_equal(result["qpos"][:36], entry["qpos"][:36])


@pytest.mark.parametrize("fault", ["robot", "time", "warmstart", "goal", "ball", "velocity"])
def test_fixture_rejects_non_ball_writes_or_wrong_declared_return(fault):
    a, entry, raw = fixture(True)
    key, index = {
        "robot": ("qpos", 0),
        "time": ("time", ()),
        "warmstart": ("qacc_warmstart", 0),
        "goal": ("mocap_pos", (0, 0)),
        "ball": ("qpos", 36),
        "velocity": ("qvel", 35),
    }[fault]
    raw["after_" + key][index] += 0.01
    with pytest.raises(ValueError):
        audit_ball_drop(a, raw, entry, "motion-right.npz")


def test_ball_return_cannot_be_fabricated_when_near_robot():
    a, entry, raw = fixture(False)
    raw["performed"] = np.array(True)
    with pytest.raises(ValueError, match="eligibility"):
        audit_ball_drop(a, raw, entry, "motion-right.npz")


def test_legacy_kick_seven_field_entry_is_not_fabricated_as_full_snapshot():
    _, entry, _ = fixture(False)
    keys = {"qpos", "qvel", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "time"}
    raw = {"initial_" + key: entry[key].copy() for key in keys}
    audit_kick_entry(raw, entry)
    raw["initial_qpos"][0] += 0.01
    with pytest.raises(ValueError):
        audit_kick_entry(raw, entry)
    raw["initial_qpos"][0] -= 0.01
    del raw["initial_ctrl"]
    with pytest.raises(ValueError, match="schema"):
        audit_kick_entry(raw, entry)
