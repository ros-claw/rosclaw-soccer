"""Boundary math/transaction fixtures, not G1 physical recovery evidence."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi import full_entry_forecast as module


@pytest.mark.parametrize("bad", [-1, 0, 19, 300, True, 30.0, float("nan")])
def test_boundary_rejected_before_body_or_model_access(bad):
    kw = dict(
        model_root=Path("unused"),
        source_poses=np.empty((0, 36)),
        source_evidence_hash="unused",
        affordance=None,
        law=None,
        start_frame=bad,
    )
    with pytest.raises(ValueError, match="start frame"):
        module.run_entry_suffix(None, None, **kw)
    with pytest.raises(ValueError, match="start frame"):
        module.FullEntryForecaster(world_hash="unused").forecast(None, None, **kw)


@pytest.mark.parametrize("start", [20, 30, 299])
def test_actual_clock_preserved_and_short_suffix_cannot_bypass_call_budget(monkeypatch, start):
    body = SimpleNamespace(
        failed=False,
        next_frame=start,
        record_all_ball_contacts=True,
        model=object(),
        data=object(),
        kp=np.ones(29),
        kd=np.ones(29),
        attempted_steps=10 * start,
        integrated_steps=10 * start,
    )
    nav = SimpleNamespace(_next_frame=start)
    monkeypatch.setattr(module, "G1PDTorqueForecaster", lambda *a, **kw: None)
    monkeypatch.setattr(module, "integration_hash", lambda *a: "body")
    monkeypatch.setattr(module, "navigation_state_hash", lambda *a: "nav")
    monkeypatch.setattr(module, "body_fields_hash", lambda *a: "fields")
    clones = []
    monkeypatch.setattr(module, "clone_navigation", lambda n: object())

    def clone(b):
        branch = SimpleNamespace(
            attempted_steps=b.attempted_steps, integrated_steps=b.integrated_steps
        )
        clones.append(branch)
        return branch

    monkeypatch.setattr(module, "_clone_body", clone)

    def suffix(n, b, **kw):
        assert kw.get("start_frame", 20) == start
        steps = (300 - start) * 10
        b.attempted_steps += steps
        b.integrated_steps += steps
        return {}, {}

    monkeypatch.setattr(module, "run_entry_suffix", suffix)
    f = module.FullEntryForecaster(world_hash="unused", max_forecasts=1)
    args = dict(
        model_root=Path("unused"),
        source_poses=np.zeros((993, 36)),
        source_evidence_hash="unused",
        affordance=object(),
        law=module.feedback_entry_laws()[0],
        start_frame=start,
    )
    r = f.forecast(nav, body, **args)
    assert r.new_integrated_native_steps == r.new_attempted_native_steps == (300 - start) * 10
    assert body.next_frame == start and body.integrated_steps == 10 * start
    with pytest.raises(ValueError, match="aligned"):
        f.forecast(nav, body, **args)
    assert len(clones) == 1
    body.integrated_steps -= 10
    with pytest.raises(ValueError, match="aligned"):
        module.FullEntryForecaster(world_hash="unused").forecast(nav, body, **args)
    assert len(clones) == 1
