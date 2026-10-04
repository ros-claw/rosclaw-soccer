from types import SimpleNamespace

import pytest

from scripts.rsi_mujoco_motor_transfer import main


@pytest.mark.parametrize("factory", [None, SimpleNamespace(restore_envelope=lambda _: {})])
def test_cached_transport_rejects_missing_or_arbitrary_factory_before_scene_open(factory):
    args = [
        "--scene",
        "/nonexistent/scene.xml",
        "--model-root",
        "/nonexistent/model",
        "--late-swing-policy",
        "/nonexistent/late.json",
        "--output-root",
        "/nonexistent/output",
        "--step-model",
        "/nonexistent/view.json",
        "--cached-proposal-envelope",
        "--seed",
        "1",
        "--lane",
        "0",
    ]
    with pytest.raises(SystemExit) as error:
        main(args, sampling_factory=factory)
    assert error.value.code == 2
