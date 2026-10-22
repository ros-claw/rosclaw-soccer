import copy

import pytest

from rosclaw_soccer.rsi.paid_canonical_behavior import quality_score


def repeat(lateral=0.4):
    return dict(
        kick=dict(
            body_failed=False,
            forbidden=False,
            postcontact=dict(
                correct_foot_first=True,
                post_contact_quality_observed=True,
                other_foot_contact_in_window=False,
                goal_lateral_1p2s_m=lateral,
                goal_progress_1p2s_m=3.0,
                maximum_pelvis_retreat_after_contact_m=0.01,
            ),
        ),
        stand=dict(healthy=True, stability_observed=True),
    )


def test_quality_selects_worst_repeat_not_cherry_picked_best():
    assert quality_score([repeat(0.1), repeat(0.6)]) == (-0.6, 3.0)
    assert quality_score([repeat(-0.4), repeat(0.4)]) == (-0.4, 3.0)
    assert quality_score([repeat(0.75), repeat(0.751)]) is None
    with pytest.raises(ValueError, match="two whole"):
        quality_score([repeat()])


@pytest.mark.parametrize(
    "fault", ["failed", "forbidden", "unobserved", "foot", "stand", "retreat", "nan"]
)
def test_failed_or_censored_whole_behavior_cannot_be_selected(fault):
    row = repeat()
    p = row["kick"]["postcontact"]
    if fault in ("failed", "forbidden"):
        row["kick"]["body_failed" if fault == "failed" else "forbidden"] = True
    elif fault == "unobserved":
        p["post_contact_quality_observed"] = False
    elif fault == "foot":
        p["other_foot_contact_in_window"] = True
    elif fault == "stand":
        row["stand"]["healthy"] = False
    elif fault == "retreat":
        p["maximum_pelvis_retreat_after_contact_m"] = 0.051
    else:
        p["goal_lateral_1p2s_m"] = float("nan")
    if fault == "nan":
        with pytest.raises(ValueError, match="finite"):
            quality_score([row, copy.deepcopy(row)])
    else:
        assert quality_score([row, copy.deepcopy(row)]) is None


def test_canonical_config_is_explicit_and_old_schema_remains_unchanged():
    from rosclaw_soccer.rsi.paid_cardinal_memory import cardinal_config

    old = cardinal_config()
    new = cardinal_config(canonical_behavior=True)
    assert "canonical_behavior" not in old
    assert new["canonical_behavior"] is True
    assert old["protocol"] != new["protocol"]
    assert new["final_R2_bound_rad"] == old["final_R2_bound_rad"] == 0.2
    with pytest.raises(ValueError, match="typed"):
        cardinal_config(canonical_behavior=1)


def test_dirty_canonical_curator_never_allocates_output(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_curate_paid_canonical_behavior as runner

    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    for name in ("original", "teacher", "teacher-audit", "replan", "replan-audit"):
        args += ["--" + name + "-root", str(tmp_path / name), "--" + name + "-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
