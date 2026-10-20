"""Synthetic orchestration/accounting only; never physical skill evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest


def run_fixture(tmp_path, monkeypatch, *, shared=False, fail=False):
    from rosclaw_soccer.physics import g1_target_episode as body_module
    from rosclaw_soccer.physics import sonic_navigation_forecast as clone_module
    from rosclaw_soccer.providers.g1 import sonic_navigation as nav_module
    from rosclaw_soccer.rsi import entry_prefix_diagnostics as prefix_module
    from rosclaw_soccer.rsi import full_entry_forecast as entry_module
    from rosclaw_soccer.rsi import full_task_branch as integration_module
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.training.expanded_entry_courses import expanded_entry_courses
    from scripts.rsi_expanded_bilateral_course import run_course

    calls = []

    class Nav:
        def __init__(self, *a, **k):
            self.backend = SimpleNamespace(kp=np.ones(29), kd=np.ones(29))
            self.state = 0

    class Body:
        def __init__(self, *a, **k):
            self.attempted_steps = self.integrated_steps = 0
            self.data = SimpleNamespace(time=0.0, qpos=np.zeros(43), qvel=np.zeros(41))

    names = [s + t for s in ("left", "right") for t in ("_ankle_roll_link", "_ankle_pitch_link")]
    model = SimpleNamespace(nbody=5, body=lambda name: SimpleNamespace(id=names.index(name) + 1))

    def arrays(index, frames):
        return dict(
            trace=np.full((frames, 1), float(index)),
            ball_contact_force_by_body_n=np.zeros((frames, 10, 5)),
        )

    def prefix(nav, body, affordance, **kwargs):
        body.attempted_steps = body.integrated_steps = 200
        body.data.time = 0.4
        nav.state = 1
        return arrays(-1, 20)

    def index_of(kwargs):
        rank = (0, 3, 4).index(kwargs["law"].feedback_index)
        return (
            6 + rank
            if kwargs["source_poses"][0, 7] == 0
            else 2 * rank + int(any(kwargs["contact_reference_offset_m"]))
        )

    class Forecaster:
        def __init__(self, **kw):
            assert kw["max_forecasts"] == 9
            self.attempted_native_steps = self.integrated_native_steps = 0

        def forecast(self, nav, body, **kwargs):
            assert body.integrated_steps == 200 and nav.state == 1
            i = index_of(kwargs)
            calls.append(("forecast", i))
            self.attempted_native_steps += 2800
            self.integrated_native_steps += 2800
            return SimpleNamespace(
                arrays=arrays(i, 280),
                report=dict(index=i),
                new_integrated_native_steps=2800,
                new_attempted_native_steps=2800,
                donor_body_hash="fixture",
                donor_navigation_hash="fixture",
            )

    def suffix(nav, body, **kwargs):
        i = index_of(kwargs)
        calls.append(("actual", i))
        if fail:
            body.attempted_steps += 42
            body.integrated_steps += 42
            body.data.time += 0.084
            kwargs["step_observer"](arrays(i, 1))
            raise ValueError("fixture partial native failure")
        body.attempted_steps += 2800
        body.integrated_steps += 2800
        body.data.time = 6.0
        nav.state = 2
        return arrays(i, 280), dict(index=i)

    def score(model, a):
        i = int(a["trace"][0, 0])
        return dict(
            qualified=i in (1, 6),
            cost=(0.05 if shared else 0.2) if i == 1 else 0.1,
            physical_prefix_and_suffix_safe=True,
        )

    monkeypatch.setattr(body_module, "G1TargetEpisode", Body)
    monkeypatch.setattr(nav_module, "G1SonicNavigation", Nav)
    monkeypatch.setattr(entry_module, "FullEntryForecaster", Forecaster)
    monkeypatch.setattr(entry_module, "nominal_prefix20", prefix)
    monkeypatch.setattr(entry_module, "run_entry_suffix", suffix)
    monkeypatch.setattr(entry_module, "score_full_entry", score)
    monkeypatch.setattr(clone_module, "_clone_body", copy.deepcopy)
    monkeypatch.setattr(clone_module, "clone_navigation", copy.deepcopy)
    monkeypatch.setattr(clone_module, "navigation_state_hash", lambda nav: nav.state)
    monkeypatch.setattr(integration_module, "integration_hash", lambda model, data: data.time)
    monkeypatch.setattr(
        prefix_module,
        "diagnose_entry_prefix",
        lambda model, prefix: dict(
            prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
        ),
    )
    world = tmp_path / "world.mjb"
    world.write_bytes(b"fixture-not-a-native-model")
    args = SimpleNamespace(output=tmp_path, model_root=tmp_path, world=world)
    source = np.zeros((993, 36))
    source[:, 2:4] = [0.8, 1]
    source[:, 7] = 0.1
    prior = SkillEntryAffordance((0.21, 0.034), 150, 1.7, "sha256:" + "a" * 64)
    result = run_course(
        args, expanded_entry_courses()[0], model, source, prior, "sha256:" + "b" * 64, {}
    )
    return result, calls


@pytest.mark.parametrize("shared", [False, True])
def test_own_nine_futures_two_endpoints_or_single_shared_execution(tmp_path, monkeypatch, shared):
    result, calls = run_fixture(tmp_path, monkeypatch, shared=shared)
    assert calls[:9] == [("forecast", i) for i in range(9)]
    assert calls[9:] == ([("actual", 1)] if shared else [("actual", 1), ("actual", 6)])
    assert result["forecast_native_steps"] == 25200
    assert result["endpoint_native_steps"] == (3000 if shared else 5800)
    assert result["shared_endpoint"] == shared
    assert result["baseline"]["selected_index"] == 1
    assert result["selected"]["selected_index"] == (1 if shared else 6)
    assert (tmp_path / "course00/complete.json").is_file()


def test_partial_actual_is_saved_and_not_retried_or_marked_complete(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="partial native"):
        run_fixture(tmp_path, monkeypatch, fail=True)
    failure = json.loads((tmp_path / "course00/failure.json").read_text())
    assert failure["complete_candidates"] == 9 and failure["incomplete_endpoint_steps"] == 42
    assert (
        failure["current_body_attempted_steps"] == failure["current_body_integrated_steps"] == 242
    )
    assert not (tmp_path / "course00/complete.json").exists()
    assert (tmp_path / "course00/incomplete-observed-control.npz").is_file()
