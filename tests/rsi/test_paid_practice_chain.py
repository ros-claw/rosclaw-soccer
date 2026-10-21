from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_practice_chain import (
    advance_practice_boundary,
    require_identical_snapshot,
    run_practice_chain,
)


def test_snapshot_does_not_accept_missing_or_changed_integration_fields():
    s = dict(qpos=np.zeros(43), qvel=np.zeros(41), time=np.array(1.0))
    require_identical_snapshot(s, {k: v.copy() for k, v in s.items()})
    changed = {k: v.copy() for k, v in s.items()}
    changed["qpos"][0] = 0.01
    with pytest.raises(ValueError):
        require_identical_snapshot(s, changed)
    with pytest.raises(ValueError):
        require_identical_snapshot(s, {"qpos": s["qpos"]})


@pytest.mark.parametrize("count", [0, 1, 5, 100])
def test_unbounded_chains_rejected_before_world_allocation(count):
    with pytest.raises(ValueError, match="bounded"):
        run_practice_chain(None, {}, [None] * count, store=lambda *_: None)


@pytest.mark.parametrize("source", ["nofoot.npz", "left_right.npz"])
def test_ambiguous_contact_reference_rejected_before_physics(source):
    with pytest.raises(ValueError, match="contact foot"):
        run_practice_chain(None, dict(source=source), [None, None], store=lambda *_: None)


def test_failed_skill_cannot_be_silently_restarted():
    ep = SimpleNamespace(termination_reason="BODY_FAILURE")
    with pytest.raises(ValueError, match="healthy"):
        advance_practice_boundary(ep, None, foot="left", store=lambda *_: None, prefix="b")


def mock_chain(monkeypatch, *, erase=False, fail_on=None):
    """Unit fixture only: no native physics or policy performance claim."""
    import rosclaw_soccer.rsi.paid_practice_chain as chain

    data = SimpleNamespace(
        **{
            k: np.zeros(1)
            for k in (
                "qpos",
                "qvel",
                "ctrl",
                "qacc_warmstart",
                "mocap_pos",
                "mocap_quat",
                "qfrc_applied",
                "xfrc_applied",
                "act",
            )
        },
        time=0.0,
    )
    ep = SimpleNamespace(
        env=SimpleNamespace(data=data),
        native_steps=0,
        last_action=np.zeros(29, np.float32),
        goal=np.array([6.0, 0, 0.11]),
        termination_reason="REFERENCE_COMPLETE",
    )
    starts = []

    def begin(clip, goal):
        starts.append(ep.last_action.copy())
        if erase:
            ep.last_action = np.zeros(29, np.float32)

    ep.begin_current_state = begin
    monkeypatch.setattr(
        chain, "practice_ball_drop", lambda *_, **__: dict(performed=np.array(False))
    )

    class Bench:
        clips = {"left.npz": object()}
        makes = 0
        calls = 0

        def make(self, case):
            self.makes += 1
            return ep

        def collect_current(self, current, actor):
            self.calls += 1
            current.native_steps += 10
            data.time += 0.02
            current.last_action = np.full(29, self.calls, np.float32)
            failed = self.calls == fail_on
            raw = dict(
                NN_action=current.last_action[None], NN_nominal_action=current.last_action[None]
            )
            return raw, dict(body_failed=failed, native_steps=10, neural_rows=1)

    bench = Bench()
    return bench, starts


def test_three_skills_have_one_setup_and_preserve_actual_previous_actions(monkeypatch):
    bench, starts = mock_chain(monkeypatch)
    artifacts = []

    def store(name, data):
        artifacts.append(name)
        return dict(raw_path=name, raw_hash="fixture")

    result = run_practice_chain(bench, dict(source="left.npz"), [None] * 3, store=store)
    assert bench.makes == 1
    assert result["executed_skills"] == 3
    assert result["native_steps"] == 30
    assert result["body_resets_between_skills"] == 0
    assert result["world_stable"]
    assert len(starts) == 2
    assert (starts[0] == 1).all() and (starts[1] == 2).all()
    assert artifacts == [
        "skill0",
        "terminal0",
        "between0-1-drop",
        "between0-1-boundary",
        "skill1",
        "terminal1",
        "between1-2-drop",
        "between1-2-boundary",
        "skill2",
        "terminal2",
    ]


def test_boundary_that_erases_previous_executed_action_is_rejected(monkeypatch):
    bench, _ = mock_chain(monkeypatch, erase=True)
    with pytest.raises(ValueError, match="erased executed action"):
        run_practice_chain(bench, dict(source="left.npz"), [None] * 3, store=lambda *_: {})


def test_failure_is_retained_without_attempting_remaining_skills(monkeypatch):
    bench, _ = mock_chain(monkeypatch, fail_on=2)
    result = run_practice_chain(bench, dict(source="left.npz"), [None] * 4, store=lambda *_: {})
    assert result["executed_skills"] == 2
    assert result["requested_skills"] == 4
    assert not result["world_stable"]
    assert result["status"] == "SKILL_BODY_FAILURE"
    assert bench.makes == 1 and bench.calls == 2
