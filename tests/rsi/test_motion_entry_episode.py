"""Episode orchestration mocks, not native G1 motor or policy evidence."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest


def setup(tmp_path, monkeypatch, *, fail=False, binding_fail=False):
    from rosclaw_soccer.physics import g1_target_episode as bm
    from rosclaw_soccer.providers.g1 import sonic_navigation as nm
    from rosclaw_soccer.rsi import bilateral_entry as cm
    from rosclaw_soccer.rsi import entry_prefix_diagnostics as dm
    from rosclaw_soccer.rsi import full_entry_forecast as fm
    from rosclaw_soccer.rsi import motion_entry_profile as pm
    from rosclaw_soccer.rsi import recorded_entry_review as rm
    from scripts.rsi_motion_entry_episode import run_episode

    events = []

    class Binding:
        contract_hash = "sha256:" + "a" * 64

        def validate(self, *args, **kwargs):
            events.append("binding")
            if binding_fail:
                raise ValueError("fixture wrong motion/entry binding")

        def report(self):
            return dict(fixture=True)

    entry = NS(contract_hash="sha256:" + "b" * 64, ball_offset_xy_m=(0.5, -0.17))
    monkeypatch.setattr(
        pm, "build_motion_entry", lambda *args, **kwargs: (Binding(), entry, np.zeros((800, 36)))
    )

    class Nav:
        def __init__(self, *args, **kwargs):
            events.append("nav")
            self.backend = NS(kp=np.ones(29), kd=np.ones(29))

    class Body:
        def __init__(self, *args, **kwargs):
            events.append("body")
            self.next_frame = self.integrated_steps = self.attempted_steps = 0

    def arrays(n):
        return dict(fixture=np.zeros((n, 1)))

    def prefix(n, b, e, **kwargs):
        assert events[:3] == ["binding", "nav", "body"] and e is entry
        assert b.next_frame == 0 and b.integrated_steps == 0
        b.next_frame, b.integrated_steps, b.attempted_steps = 20, 200, 200
        events.append("prefix")
        return arrays(20)

    def suffix(n, b, **kwargs):
        assert b.next_frame == 20 and b.integrated_steps == 200
        assert kwargs["affordance"] is entry and kwargs["source_window"] == (466, 767)
        assert kwargs["contact_reference_offset_m"] == (0.0, 0.0, 0.0)
        if fail:
            b.integrated_steps += 17
            b.attempted_steps += 17
            kwargs["step_observer"](arrays(1))
            raise ValueError("fixture native continuation failed")
        b.next_frame, b.integrated_steps, b.attempted_steps = 300, 3000, 3000
        events.append("suffix")
        return arrays(280), dict(body=dict(frames=300), handoff_frame=50)

    monkeypatch.setattr(nm, "G1SonicNavigation", Nav)
    monkeypatch.setattr(bm, "G1TargetEpisode", Body)
    monkeypatch.setattr(fm, "nominal_prefix20", prefix)
    monkeypatch.setattr(fm, "run_entry_suffix", suffix)
    monkeypatch.setattr(
        fm,
        "score_full_entry",
        lambda *args: dict(
            qualified=True,
            cost=0.1,
            physical_prefix_and_suffix_safe=True,
            first_foot_contact_global_frame=100,
        ),
    )
    monkeypatch.setattr(
        dm,
        "diagnose_entry_prefix",
        lambda *args: dict(
            prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
        ),
    )
    monkeypatch.setattr(cm, "observed_contact_foot", lambda *args, **kwargs: dict(fixture=True))
    monkeypatch.setattr(rm, "validate_recording", lambda *args, **kwargs: None)
    world = tmp_path / "fixture.mjb"
    csv = tmp_path / "fixture.csv"
    world.write_bytes(b"fixture not a physical model")
    csv.write_bytes(b"fixture not real motion")
    a = NS(world=world, motion_csv=csv, model_root=tmp_path, output=tmp_path / "evidence")

    def call():
        return run_episode(
            a,
            NS(agent_id="tracker", course=(0.4, -0.2, 0.0)),
            NS(nbody=46),
            None,
            None,
            arm="compact_transported",
            side="right",
            law_index=0,
            evidence="fixture",
            runtime={},
        )

    return call, events, a


def test_profile_known_before_neural_and_body_allocation_from0(tmp_path, monkeypatch):
    call, events, a = setup(tmp_path, monkeypatch)
    row = call()
    assert events == ["binding", "nav", "body", "prefix", "suffix", "binding"]
    assert row["new_native_steps"] == 3000 and row["known_intent_before_frame0"]
    assert not row["initial_state_adopted_from_other_episode"]
    assert (a.output / "right-law0/native300.npz").is_file()


def test_binding_failure_cannot_allocate_controller_or_body(tmp_path, monkeypatch):
    call, events, a = setup(tmp_path, monkeypatch, binding_fail=True)
    with pytest.raises(ValueError, match="wrong motion/entry"):
        call()
    assert events == ["binding"] and not a.output.exists()


def test_partial_continuation_preserves_actual_counts_not_fake_h300(tmp_path, monkeypatch):
    call, _, a = setup(tmp_path, monkeypatch, fail=True)
    with pytest.raises(ValueError, match="native continuation failed"):
        call()
    d = json.loads((a.output / "right-law0/failure.json").read_text())
    assert d["native_integrated_steps"] == d["native_attempted_steps"] == 217
    assert not d["automatic_retry_authorized"]
