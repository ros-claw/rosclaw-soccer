"""Transaction/budget fixtures only, not native G1 success evidence."""

from types import SimpleNamespace as NS

import numpy as np
import pytest


def fixture(
    tmp_path, monkeypatch, *, veto=False, fail=False, protocol="original9", window=(286, 767)
):
    from rosclaw_soccer.physics import g1_target_episode as bm
    from rosclaw_soccer.providers.g1 import sonic_navigation as nm
    from rosclaw_soccer.rsi import bilateral_entry as bi
    from rosclaw_soccer.rsi import entry_prefix_diagnostics as di
    from rosclaw_soccer.rsi import foundation_motor_codec as cm
    from rosclaw_soccer.rsi import full_entry_forecast as fm
    from rosclaw_soccer.rsi import student_intervention_score as sm
    from scripts import rsi_warm_student_intervention as wm
    from scripts.rsi_student_intervention_course import run_course

    calls = []

    def arrays(i, n):
        return {"fixture_trace": np.full((n, 1), float(i))}

    class Nav:
        def __init__(self, *args):
            self._next_frame = 0

    class Body:
        def __init__(self, *args, **kwargs):
            self.next_frame = self.integrated_steps = self.attempted_steps = 0

    def warm(m, c, n, b, p, e, **kwargs):
        assert b.integrated_steps == 0 and n._next_frame == 0
        b.integrated_steps = b.attempted_steps = 300
        b.next_frame = n._next_frame = 30
        return arrays(-1, 30), {"teacher_label": np.zeros((30, 1), np.float32)}

    def context(poses, prior, foot):
        return np.full((2, 36), float(foot == "left")), prior, None

    def index(kw):
        rank = (0, 3, 4).index(kw["law"].feedback_index)
        return (
            6 + rank
            if kw["source_poses"][0, 0]
            else 2 * rank + int(any(kw["contact_reference_offset_m"]))
        )

    class Forecaster:
        def __init__(self, **kw):
            assert kw["max_forecasts"] == (9 if protocol == "original9" else 6)
            self.integrated_native_steps = self.attempted_native_steps = 0

        def forecast(self, n, b, **kw):
            assert b.integrated_steps == 300 and b.next_frame == n._next_frame == 30
            assert kw["start_frame"] == 30
            assert kw["source_window"] == window
            self.integrated_native_steps += 2700
            self.attempted_native_steps += 2700
            i = index(kw)
            calls.append(("future", i))
            return NS(
                arrays=arrays(i, 270),
                report=dict(option=i),
                donor_body_hash="fixture",
                donor_navigation_hash="fixture",
                new_integrated_native_steps=2700,
                new_attempted_native_steps=2700,
            )

    def suffix(n, b, **kw):
        assert b.next_frame == 30 and b.integrated_steps == 300 and kw["start_frame"] == 30
        i = index(kw)
        calls.append(("actual", i))
        if fail:
            b.integrated_steps += 17
            b.attempted_steps += 17
            kw["step_observer"](arrays(i, 1))
            raise ValueError("fixture partial real-continuation failure")
        b.next_frame = n._next_frame = 300
        b.integrated_steps = b.attempted_steps = 3000
        return arrays(i, 270), dict(option=i)

    monkeypatch.setattr(nm, "G1SonicNavigation", Nav)
    monkeypatch.setattr(bm, "G1TargetEpisode", Body)
    monkeypatch.setattr(cm, "FoundationMotorCodec", lambda b: object())
    monkeypatch.setattr(wm, "warm_student30", warm)
    monkeypatch.setattr(
        di,
        "diagnose_entry_prefix",
        lambda m, p, **kw: dict(
            prefix_physical_safe=not veto, post_boundary_control_can_cause_first_touch=True
        ),
    )
    monkeypatch.setattr(bi, "contact_foot_context", context)
    monkeypatch.setattr(bi, "observed_contact_foot", lambda *a, **kw: dict(fixture=True))
    monkeypatch.setattr(fm, "FullEntryForecaster", Forecaster)
    monkeypatch.setattr(fm, "run_entry_suffix", suffix)
    monkeypatch.setattr(
        sm,
        "fuse_physical_intervention30",
        lambda m, p, s: arrays(int(s["fixture_trace"][0, 0]), 300),
    )
    monkeypatch.setattr(
        sm,
        "score_student_intervention30",
        lambda m, p, s: dict(
            qualified=bool(s["fixture_trace"][0, 0] == 6),
            cost=0.1 if s["fixture_trace"][0, 0] == 6 else None,
        ),
    )
    world = tmp_path / "fixture.mjb"
    world.write_bytes(b"fixture not physical")
    args = NS(output=tmp_path / "output", warm=tmp_path / "warm", model_root=tmp_path, world=world)
    args.output.mkdir()
    warmdir = args.warm / "course000"
    warmdir.mkdir(parents=True)
    np.savez_compressed(warmdir / "actual-student30.npz", **arrays(-1, 30))
    np.savez_compressed(
        warmdir / "shadow-labels30.npz", teacher_label=np.zeros((30, 1), np.float32)
    )
    row = run_course(
        args,
        NS(course_index=0, agent_id="fixture", course=(0.0, 0.0, 0.0, 0.0)),
        object(),
        np.zeros((1, 36)),
        None,
        NS(calls=0),
        NS(kp=np.ones(29), kd=np.ones(29)),
        {},
        "fixture",
        {},
        option_protocol=protocol,
        source_window=window,
    )
    return row, calls


def test_all9_futures_and_actual_from_same30_no_extra_baseline(tmp_path, monkeypatch):
    row, calls = fixture(tmp_path, monkeypatch)
    assert calls == [("future", i) for i in range(9)] + [("actual", 6)]
    assert row["new_native_steps"] == 27300
    assert row["executed_native_steps"] == 3000 and row["forecast_native_steps"] == 24300
    assert row["selected"] == 6 and row["causal_qualified"]


def test_irreversible_history_stays_in_denominator_without_fake_futures(tmp_path, monkeypatch):
    row, calls = fixture(tmp_path, monkeypatch, veto=True)
    assert calls == [] and row["denominator_retained"]
    assert row["new_native_steps"] == 300
    assert not row["causal_qualified"] and row["selected"] is None


def test_partial_actual_failure_keeps_forecast_and_body_charges(tmp_path, monkeypatch):
    import json

    with pytest.raises(ValueError, match="fixture partial"):
        fixture(tmp_path, monkeypatch, fail=True)
    d = json.loads((tmp_path / "output/course000/failure.json").read_text())
    assert d["forecast_integrated_steps"] == 24300 and d["actual_integrated_steps"] == 317
    assert not d["automatic_retry_authorized"]


@pytest.mark.parametrize("window", [(286, 767), (466, 767)])
def test_six_options_preserve_actual30_and_charge_only_six_futures(tmp_path, monkeypatch, window):
    row, calls = fixture(tmp_path, monkeypatch, protocol="zero_c_six", window=window)
    assert calls == [("future", i) for i in (0, 2, 4, 6, 7, 8)] + [("actual", 6)]
    assert row["new_native_steps"] == 19200
    assert row["forecast_native_steps"] == 16200 and row["executed_native_steps"] == 3000
    assert row["selected"] == 3 and row["causal_qualified"]
    assert row["source_window"] == list(window) and row["selected_report"] == dict(option=6)
