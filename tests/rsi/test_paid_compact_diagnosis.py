import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_compact_diagnosis import summarize_compact_trace


def trace():
    return dict(
        features=np.zeros((4, 288), np.float32),
        gain=np.array([1, 0.7, 0.2, 0], np.float32),
        delta=np.zeros((4, 29), np.float32),
    )


def test_support_exit_and_actual_contact_not_confused_with_learning():
    t = trace()
    t["delta"][1, 0] = 0.1
    r = summarize_compact_trace(t, first_foot_substep=25)
    assert r["first_support_below_half_control"] == 2
    assert r["first_zero_support_control"] == 3
    assert r["first_contact_control"] == 2
    assert r["support_at_first_foot_contact"] == pytest.approx(0.2)
    assert not r["training_labels_created"] and not r["physical_replay_performed"]


def test_same_index_teacher_difference_is_not_an_own_state_expert_label():
    t = trace()
    teacher = t["features"].copy()
    teacher[1, 0] = 1
    r = summarize_compact_trace(t, teacher_features=teacher, teacher_delta=t["delta"].copy())
    s = r["same_index_teacher_comparison"]
    assert s["first_feature_difference_control"] == 1
    assert not s["is_expert_label_on_student_state"]


@pytest.mark.parametrize("bad", ["nan", "gain", "authority", "shape", "contact", "teacher"])
def test_invalid_trace_does_not_produce_a_progress_report(bad):
    t = trace()
    kwargs = {}
    if bad == "nan":
        t["features"][0, 0] = np.nan
    elif bad == "gain":
        t["gain"][0] = 1.1
    elif bad == "authority":
        t["delta"][0, 0] = 0.3
    elif bad == "shape":
        t["features"] = t["features"][:, :5]
    elif bad == "contact":
        kwargs["first_foot_substep"] = True
    else:
        kwargs["teacher_features"] = t["features"].copy()
    with pytest.raises(ValueError):
        summarize_compact_trace(t, **kwargs)


@pytest.mark.parametrize("fault", [None, "before", "after", "action"])
def test_trace_recomputes_own_recurrence_and_actual_action(fault):
    from types import SimpleNamespace

    import torch
    from test_paid_successor_features import fixture

    from rosclaw_soccer.rsi.paid_ancestor_router import AncestralResidualRouter
    from rosclaw_soccer.rsi.paid_compact_diagnosis import compact_own_trace
    from rosclaw_soccer.rsi.paid_compact_student import FrozenCompactResidual
    from rosclaw_soccer.rsi.paid_compact_support import CompactSupportedHead, CompactTeacherSupport
    from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
    from rosclaw_soccer.rsi.paid_progressive_contact import ProgressiveContactResidual

    base, raw = fixture()
    parent = AncestralResidualRouter(
        ProgressiveContactResidual(DecoupledContactResidual(), DecoupledContactResidual())
    )
    head = CompactSupportedHead(
        CompactTeacherSupport(torch.zeros(1, 288), torch.ones(1, dtype=torch.float64))
    )
    actor = SimpleNamespace(
        base=base,
        residual=FrozenCompactResidual(parent, head).eval(),
        action_scale=torch.ones(29),
        training=False,
    )
    if fault in ("before", "after"):
        raw["NN_h_" + fault][1, 0, 0, 0] += 1
    elif fault == "action":
        raw["NN_action"][1, 0] = 0.1
    if fault:
        with pytest.raises(ValueError):
            compact_own_trace(actor, raw)
    else:
        t = compact_own_trace(actor, raw)
        assert t["features"].shape == (3, 288)
        assert np.allclose(t["features"][:, 160], [0.1, 0.2, 0.3])


def test_diagnostic_requires_clean_source_before_wait_or_output(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_diagnose_paid_compact_policy as runner

    args = [
        "run",
        "--data-hash",
        "hash",
        "--producer-head",
        "producer",
        "--expected-head",
        "frozen",
    ]
    for name in ("exam-root", "audit-root", "data-root", "output"):
        args += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "output").exists()
