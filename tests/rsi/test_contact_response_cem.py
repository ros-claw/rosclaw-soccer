"""Bounded optimizer fixtures, not neural training or physical success."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_response_cem import (
    bounded_offset,
    elite_update,
    response_merit,
    sample_four,
)
from rosclaw_soccer.rsi.full_entry_forecast import FullEntryForecaster, feedback_entry_laws


def score(**patch):
    return dict(
        dict(
            qualified=False,
            physical_prefix_and_suffix_safe=True,
            complete_post60_observable=True,
            maximum_nonfoot_force_n=0.0,
            minimum_pelvis_z_m=0.8,
            minimum_upright=1.0,
            forward_to_task_end_m=0.3,
            lateral_to_task_end_m=0.1,
            maximum_ball_vx_mps=0.5,
            post_touch_maximum_root_drift_m=0.1,
            cost=None,
        ),
        **patch,
    )


def test_merit_priority_and_fixed_formula():
    safe = response_merit(score())
    assert safe == pytest.approx(102.49)
    assert response_merit(score(qualified=True, cost=0.01)) == 999999.99
    assert (
        response_merit(score(physical_prefix_and_suffix_safe=False, maximum_nonfoot_force_n=12.0))
        == -1000012
    )
    assert response_merit(score(qualified=True, cost=0.01)) > safe > -1000012


@pytest.mark.parametrize(
    "patch",
    [
        dict(qualified=1),
        dict(cost=np.nan, qualified=True),
        dict(minimum_upright=np.inf),
        dict(post_touch_maximum_root_drift_m=-1),
    ],
)
def test_invalid_response_not_optimizer_input(patch):
    with pytest.raises(ValueError):
        response_merit(score(**patch))


def test_sample_trace_determinism_and_no_clipping():
    mean, std = np.array([0.0, -0.025, -0.025]), np.full(3, 0.015)
    a, log = sample_four(np.random.default_rng(14), mean, std)
    assert (a, log) == sample_four(np.random.default_rng(14), mean, std)
    assert len(a) == 4 and 4 <= len(log) <= 256
    assert sum(r["accepted"] for r in log) == 4
    assert [list(x) for x in a] == [r["offset_root_m"] for r in log if r["accepted"]]
    for x in a:
        bounded_offset(x)


def test_exhaustion_does_not_silently_clip_or_return_partial():
    class AlwaysOutside:
        def normal(self, *args):
            return np.array([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="exhausted"):
        sample_four(AlwaysOutside(), np.zeros(3), np.full(3, 0.015))


def test_elite_update_population_std_floor_and_stable_ties():
    points = [
        (0.0, 0.0, 0.0),
        (0.02, 0.0, 0.0),
        (0.03, 0.0, 0.0),
        (0.01, 0.01, 0.0),
        (0.0, -0.025, -0.025),
    ]
    mean, std, elite = elite_update(points, [4.0, 4.0, 2.0, 1.0, 0.0])
    assert elite == [0, 1]
    np.testing.assert_array_equal(mean, [0.01, 0.0, 0.0])
    np.testing.assert_array_equal(std, [0.01, 0.003, 0.003])


@pytest.mark.parametrize("budget", [0, 33, True, 1.5])
def test_forecast_budget_requires_explicit_bounded_integer(budget):
    with pytest.raises(ValueError, match="budget1..32"):
        FullEntryForecaster(world_hash="unused", max_forecasts=budget)


@pytest.mark.parametrize("budget", [7, 30])
def test_every_preview_charged_and_last_call_rejected_before_clone(monkeypatch, budget):
    from rosclaw_soccer.rsi import full_entry_forecast as module

    body = SimpleNamespace(
        failed=False,
        next_frame=20,
        record_all_ball_contacts=True,
        model=object(),
        data=object(),
        kp=np.ones(29),
        kd=np.ones(29),
        attempted_steps=200,
        integrated_steps=200,
    )
    nav = SimpleNamespace(_next_frame=20)
    monkeypatch.setattr(module, "G1PDTorqueForecaster", lambda *a, **k: None)
    monkeypatch.setattr(module, "integration_hash", lambda *a: "body")
    monkeypatch.setattr(module, "navigation_state_hash", lambda *a: "nav")
    monkeypatch.setattr(module, "body_fields_hash", lambda *a: "fields")
    clones = []
    monkeypatch.setattr(module, "clone_navigation", lambda n: object())

    def clone(b):
        fresh = SimpleNamespace(attempted_steps=200, integrated_steps=200)
        clones.append(fresh)
        return fresh

    monkeypatch.setattr(module, "_clone_body", clone)

    def suffix(controller, owned, **kwargs):
        owned.attempted_steps += 2800
        owned.integrated_steps += 2800
        return {}, {}

    monkeypatch.setattr(module, "run_entry_suffix", suffix)
    reader = FullEntryForecaster(
        world_hash="unused", **({} if budget == 7 else dict(max_forecasts=budget))
    )
    assert reader.max_forecasts == budget
    with pytest.raises(AttributeError):
        reader.max_forecasts = 32
    args = dict(
        model_root=Path("unused"),
        source_poses=np.zeros((993, 36)),
        source_evidence_hash="unused",
        affordance=object(),
        law=feedback_entry_laws()[0],
    )
    for _ in range(budget):
        reader.forecast(nav, body, **args)
    assert reader.attempted_native_steps == reader.integrated_native_steps == budget * 2800
    with pytest.raises(ValueError, match="remaining full-task budget"):
        reader.forecast(nav, body, **args)
    assert len(clones) == budget and body.integrated_steps == 200
