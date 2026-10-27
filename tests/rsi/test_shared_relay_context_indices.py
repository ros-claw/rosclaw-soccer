from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.shared_relay_setup import SharedRelaySetup, validate_relay_context_indices


def test_explicit_relay_contexts_are_unique_and_in_range():
    validate_relay_context_indices((2, 3, 5, 7), 22)
    validate_relay_context_indices((1, 4, 20, 21), 22)
    for indices in ((2, 2), (True,), (-1,), (22,), (), [2, 3]):
        with pytest.raises(ValueError):
            validate_relay_context_indices(indices, 22)


def test_setup_loads_declared_assets_only(monkeypatch, tmp_path):
    import mujoco

    import rosclaw_soccer.physics.native_ball_dimensions as dimensions
    import rosclaw_soccer.rsi.shared_native_scene as scene

    @dataclass
    class BallDimensions:
        size_and_mass_in_ifab_range: bool = True

    model = SimpleNamespace(geom=lambda _: SimpleNamespace(id=0))
    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_xml_string=lambda _: model))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *_: None)
    monkeypatch.setattr(
        dimensions, "inspect_native_ball_dimensions", lambda *_, **__: BallDimensions()
    )
    monkeypatch.setattr(scene, "shared_native_scene", lambda *_, **__: "scene")
    monkeypatch.setattr(SharedRelaySetup, "velocity", lambda _: SimpleNamespace(params=object()))
    xml = tmp_path / "cached-standard" / "declared-scene.xml"
    xml.parent.mkdir()
    xml.write_text("fixture")
    pins = {
        str(xml): "fixture",
        "assets/velocity/v0/exported/policy.onnx": "fixture",
        "assets/velocity/v0/params/deploy.yaml": "fixture",
    }
    contexts = [{"case": i, "backoff_m": 0.1} for i in range(22)]
    seen = []

    def make(case):
        seen.append(case)
        q = np.zeros(43)
        q[3] = 1
        return SimpleNamespace(
            layout=SimpleNamespace(rq=0, canonical=lambda _: (q, np.zeros(41))),
            env=SimpleNamespace(data=SimpleNamespace(qpos=q), model=model),
        )

    setup = SharedRelaySetup(
        pins, contexts, bench=SimpleNamespace(make=make), actor=None, context_indices=(2, 3, 5, 7)
    )
    assert seen == [2, 3, 5, 7]
    assert set(setup.episodes) == {2, 3, 5, 7}
