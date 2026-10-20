from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix


def fixture():
    names = [
        side + suffix
        for side in ("left", "right")
        for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
    ] + ["pelvis"]
    m = SimpleNamespace(
        nbody=6,
        body_rootid=np.array([0, 5, 5, 5, 5, 5]),
        body=lambda n: SimpleNamespace(id=names.index(n) + 1),
    )
    q = np.zeros((20, 10, 43))
    q[:, :, 2] = 0.8
    q[:, :, 3] = 1
    return m, dict(
        ball_contact_force_by_body_n=np.zeros((20, 10, 6)),
        target_body_canonical_qpos_after_substep=q,
    )


def test_prefix_safety_and_early_contact_attribution_are_separate():
    m, p = fixture()
    result = diagnose_entry_prefix(m, p)
    assert result["prefix_physical_safe"] and result["post_boundary_control_can_cause_first_touch"]
    p["ball_contact_force_by_body_n"][15, 3, 1] = 2
    result = diagnose_entry_prefix(m, p)
    assert (
        result["prefix_physical_safe"] and not result["post_boundary_control_can_cause_first_touch"]
    )
    assert result["prefix_foot_contact"]["first_contact_global_frame"] == 15
    p["ball_contact_force_by_body_n"][1, 0, 5] = 2
    assert not diagnose_entry_prefix(m, p)["prefix_physical_safe"]


@pytest.mark.parametrize("kind", ["nan", "short", "negative"])
def test_invalid_native_prefix_not_assumed_clean(kind):
    m, p = fixture()
    if kind == "nan":
        p["target_body_canonical_qpos_after_substep"][0, 0, 2] = np.nan
    elif kind == "short":
        p["ball_contact_force_by_body_n"] = p["ball_contact_force_by_body_n"][:-1]
    else:
        p["ball_contact_force_by_body_n"][0, 0, 1] = -1
    with pytest.raises(ValueError):
        diagnose_entry_prefix(m, p)
