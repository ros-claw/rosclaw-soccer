from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1 import sonic_navigation as nav


def configuration(mode):
    return nav.SonicNavigationConfig(
        model_variant="low_latency",
        experimental_maximum_speed_mps=1.5,
        experimental_locomotion_mode=mode,
    )


@pytest.mark.parametrize("value", [True, False, 1, 4, 3.0, "3"])
def test_only_explicit_integer_walk_or_run_modes_allowed(value):
    with pytest.raises(ValueError, match="mode 2/3"):
        configuration(value)


@pytest.mark.parametrize(
    "change",
    [
        {"model_variant": "sonic_v1_1"},
        {"experimental_maximum_speed_mps": None},
        {"experimental_maximum_speed_mps": 2.0},
        {"experimental_command_replanning": True},
    ],
)
def test_probe_cannot_widen_envelope_or_mix_other_mechanisms(change):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(configuration(3), **change)


class Planner:
    def run(self, names, feed):
        self.feed = {k: v.copy() for k, v in feed.items()}
        ref = np.zeros((1, 4, 36), dtype=np.float32)
        ref[:, :, 3] = 1
        return ref, np.asarray([4], dtype=np.int64)


def plan(mode, speed):
    backend = object.__new__(nav._StreamingBackend)
    backend.navigation = configuration(mode)
    backend.command = (speed, 0.0, 0.0)
    backend.facing = 0.0
    backend.planner_calls = 0
    backend.events = []
    backend._planner = Planner()
    context = np.zeros((4, 36))
    context[:, 3] = 1
    backend.plan(context, 0)
    return backend._planner.feed, backend.events


@pytest.mark.parametrize(
    "speed,legacy_mode,run_mode",
    [
        (0.0, 0, 0),
        (0.019, 0, 0),
        (0.02, 1, 1),
        (0.8, 1, 1),
        (0.80001, 2, 3),
        (1.4, 2, 3),
    ],
)
def test_only_high_speed_mode_input_changes_and_actual_mode_recorded(speed, legacy_mode, run_mode):
    default, events = plan(None, speed)
    baseline, before = plan(2, speed)
    probe, after = plan(3, speed)
    assert default["mode"][0] == baseline["mode"][0] == legacy_mode
    assert probe["mode"][0] == run_mode
    assert "planner_mode" not in events[0]
    assert before[0]["planner_mode"] == legacy_mode
    assert after[0]["planner_mode"] == run_mode
    for key in default:
        np.testing.assert_array_equal(default[key], baseline[key])
        if key != "mode":
            np.testing.assert_array_equal(default[key], probe[key])


def test_default_motor_contract_hashes_match_pre_probe_frozen_source(monkeypatch):
    class Backend:
        def __init__(self, *args):
            self.qualification = SimpleNamespace(qualification_hash="sha256:" + "a" * 64)

    monkeypatch.setattr(nav, "_StreamingBackend", Backend)
    legacy = nav.G1SonicNavigation(None, "red.defender")
    bounded = nav.G1SonicNavigation(None, "red.defender", configuration(None))
    # Independently measured using the frozen pre-probe provider source.
    assert (
        legacy.contract_hash
        == "sha256:a568f499236301e6405b63573f2891c886992a898af11c6a72973142143beca9"
    )
    assert (
        bounded.contract_hash
        == "sha256:5f3ad59a8bd2d34b3daee9831dbefd12022aa3f1bc24e13773d7f8f5c6f84a49"
    )
    walk = nav.G1SonicNavigation(None, "red.defender", configuration(2))
    run = nav.G1SonicNavigation(None, "red.defender", configuration(3))
    assert len({bounded.contract_hash, walk.contract_hash, run.contract_hash}) == 3
