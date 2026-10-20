import numpy as np
import pytest

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision
from rosclaw_soccer.rsi.entry_readiness_diagnostic import diagnose_readiness
from rosclaw_soccer.rsi.full_entry_forecast import feedback_entry_laws


def fixture():
    entry = SkillEntryAffordance((0.2, 0.03), 150, 1.0, "sha256:" + "a" * 64)
    q, v = np.zeros((300, 43)), np.zeros((300, 41))
    q[:, 2], q[:, 3], q[:, 36:39] = 0.793, 1.0, [0.3, 0.3, 0.13]
    raw = dict(
        canonical_qpos=q,
        canonical_qvel=v,
        entry_stage=np.where(np.arange(300) > 90, 2, 0).astype(np.int64),
    )
    decisions = [
        ball_entry_decision(a, b, frame=i, affordance=entry)
        for i, (a, b) in enumerate(zip(q, v, strict=True))
    ]
    raw["entry_error_m"] = np.array([d.horizontal_error_m for d in decisions])
    raw["entry_target_root_xy_m"] = np.array([d.target_root_xy_m for d in decisions])
    raw["entry_command"] = np.array(
        [
            d.command
            if i < 20
            else feedback_entry_laws()[0].command(q[i], v[i], frame=i, affordance=entry)
            for i, d in enumerate(decisions)
        ]
    )
    return raw, entry


def test_readiness_failure_is_before_contact_geometry_test():
    raw, e = fixture()
    d = diagnose_readiness(raw, e, law_index=0)
    assert d["compared_navigation_frames"] == 300 and d["readiness_window_frames"] == 61
    assert d["distance_predicate_pass_frames"] == 0
    assert (
        d["yaw_predicate_pass_frames"]
        == d["height_predicate_pass_frames"]
        == d["upright_predicate_pass_frames"]
        == 61
    )
    assert d["failure_stage"] == "navigation_before_reference_handoff"


@pytest.mark.parametrize(
    "key",
    ["entry_stage", "entry_error_m", "entry_command", "entry_target_root_xy_m", "canonical_qpos"],
)
def test_raw_decision_disagreement_rejected(key):
    raw, e = fixture()
    raw[key][40] += 1
    with pytest.raises(ValueError):
        diagnose_readiness(raw, e, law_index=0)
