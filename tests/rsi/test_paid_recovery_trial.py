from types import SimpleNamespace

import numpy as np
import pytest

import rosclaw_soccer.rsi.paid_recovery_trial as trial


@pytest.mark.parametrize("gain", [np.zeros((8, 4)), np.full((9, 4), np.nan), np.full((9, 4), 2.0)])
def test_invalid_common_feedback_rejected_before_world_allocation(gain):
    with pytest.raises(ValueError):
        trial.run_recovery_trial(None, {}, None, None, gain, store=lambda *_: {})


@pytest.mark.parametrize("failure", ["first", "recovery", None])
def test_ordinary_failures_retained_and_no_unqualified_next_skill(monkeypatch, failure):
    """Flow-only fixture. No native physical outcomes are claimed."""
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
    ep = SimpleNamespace(native_steps=0, env=SimpleNamespace(data=data), clip=None)
    calls = []

    class Bench:
        neural_rows = 0

        @property
        def native_steps(self):
            return ep.native_steps

        def make(self, case):
            return ep

        def collect_current(self, ep, actor, *, stop_at_reference):
            calls.append(stop_at_reference)
            ep.native_steps += 10
            data.time += 0.02
            self.neural_rows += 1
            return dict(NN_action=np.zeros((1, 29)), NN_nominal_action=np.zeros((1, 29))), dict(
                body_failed=failure == "first"
            )

    def recovery(ep, latent):
        ep.native_steps += 10
        data.time += 0.02
        return dict(target=np.zeros((1, 29))), dict(healthy=failure != "recovery")

    monkeypatch.setattr(
        trial, "advance_practice_boundary", lambda *args, **kwargs: dict(status="READY")
    )
    monkeypatch.setattr(trial, "collect_balance_recovery", recovery)
    keys = []

    def store(key, raw):
        keys.append(key)
        return dict(raw_path=key, raw_hash="fixture")

    result = trial.run_recovery_trial(
        Bench(), dict(source="clip-right.npz"), None, None, np.zeros((9, 4)), store=store
    )
    if failure == "first":
        assert result["status"] == "FIRST_BODY_FAILURE" and calls == [False]
    elif failure == "recovery":
        assert result["status"] == "RECOVERY_BODY_FAILURE" and calls == [False, True]
        assert "skill2" not in keys
    else:
        assert result["stable"] and calls == [False, True, False]
        assert "recovery" in keys and "skill2" in keys
    assert result["body_resets"] == 0
