from types import SimpleNamespace

import numpy as np

from rosclaw_soccer.rsi import paid_successor_trial as module


def test_failed_counterfactual_branch_stops_before_standing(monkeypatch):
    """Flow-only test; no physical success is inferred."""
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
        time=12.0,
    )
    ep = SimpleNamespace(env=SimpleNamespace(data=data))
    monkeypatch.setattr(module, "begin_successor_branch", lambda *args: ep)
    raw = {"NN_h_after": np.zeros((1, 2, 1, 128)), "NN_c_after": np.zeros((1, 2, 1, 128))}

    def collect(actual, actor, *, transform, stop_at_reference):
        assert actual is ep and transform is None and stop_at_reference
        return raw, dict(body_failed=True)

    def unexpected(*args, **kwargs):
        raise AssertionError("failed body cannot restart standing")

    monkeypatch.setattr(module, "NeuralStandingTask", unexpected)
    recorded = []
    result = module.run_successor_trial(
        SimpleNamespace(collect_current=collect),
        {},
        None,
        entry={},
        previous=None,
        transform=None,
        store=lambda key, value: recorded.append(key) or {"raw_path": key},
    )
    assert result["stand"] is None and recorded == ["kick", "kick-terminal"]
