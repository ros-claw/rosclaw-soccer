from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1RollingOptionBridgeConfig
from rosclaw_soccer.providers.g1 import kick_warmstart
from rosclaw_soccer.sim.contracts import hash_json


def policy(monkeypatch):
    monkeypatch.setattr(
        kick_warmstart.importlib, "import_module", lambda _: SimpleNamespace(NPZ_ANCHOR_IDX=0)
    )
    return SimpleNamespace(
        use_body_frame_ball=False,
        runtime_mode="sim",
        motion_body_pos=np.array([[[0.0, 0.0, 1.0]], [[0.4, -0.2, 0.9]]]),
        _init_to_world=np.diag([-1.0, -1.0, 1.0]),
        _ref_anchor_world_origin=np.array([0.0, 0.0, 1.0]),
        state_cmd=SimpleNamespace(q=np.ones(29)),
        last_action_il=np.ones(29),
        history=[np.arange(3.0)],
    )


def test_only_reference_origin_changes_without_fabricated_history(monkeypatch):
    p = policy(monkeypatch)
    old = p.__dict__.copy()
    kick_warmstart.rebase_kick_reference(p, entry_frame=1)
    np.testing.assert_allclose(p._ref_anchor_world_origin, [-0.4, 0.2, 0.9])
    assert all(v is p.__dict__[k] for k, v in old.items() if k != "_ref_anchor_world_origin")
    np.testing.assert_array_equal(p.state_cmd.q, np.ones(29))
    np.testing.assert_array_equal(p.history[0], np.arange(3.0))


@pytest.mark.parametrize("frame", [-1, 2, True, 0.5])
def test_bad_frame_never_mutates_reference(monkeypatch, frame):
    p = policy(monkeypatch)
    before = p._ref_anchor_world_origin
    with pytest.raises(ValueError):
        kick_warmstart.rebase_kick_reference(p, entry_frame=frame)
    assert p._ref_anchor_world_origin is before


@pytest.mark.parametrize("fault", ["real", "reflection", "nan"])
def test_invalid_contract_rejected(monkeypatch, fault):
    p = policy(monkeypatch)
    if fault == "real":
        p.runtime_mode = "real"
    elif fault == "reflection":
        p._init_to_world = -np.eye(3)
    else:
        p.motion_body_pos[1, 0, 0] = np.nan
    before = p._ref_anchor_world_origin
    with pytest.raises(ValueError):
        kick_warmstart.rebase_kick_reference(p, entry_frame=1)
    assert p._ref_anchor_world_origin is before


def test_rebase_is_explicit_and_legacy_config_hash_unchanged():
    base = G1RollingOptionBridgeConfig()
    legacy = asdict(base)
    for key in (
        "reference_rebase_only",
        "per_player_options_enabled",
        "continuous_rearm_enabled",
        "task_context_bound",
    ):
        legacy.pop(key)
    assert base.config_hash == hash_json(legacy)
    assert replace(base, reference_rebase_only=True).config_hash != base.config_hash
    with pytest.raises(ValueError):
        replace(base, reference_rebase_only=True, observation_warmstart=True)
    with pytest.raises(ValueError):
        replace(base, reference_rebase_only=1)
