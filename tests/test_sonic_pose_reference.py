from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_latent import SonicLatentSchedule
from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig, _StreamingBackend
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference


def reference():
    pose = (0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0) + (0.0,) * 29
    return SonicPoseReference((pose,) * 10, "sha256:" + "1" * 64)


def test_entry_binding_terminal_hold_and_independent_storage():
    ref = reference()
    result = ref.initialize(np.asarray(ref.poses[0]), frames=60)
    assert result.shape == (60, 36)
    np.testing.assert_array_equal(result[-1], ref.poses[-1])
    result[:] = 0
    assert ref.initialize(np.asarray(ref.poses[0]), frames=60)[0, 2] == 0.75
    assert (
        replace(ref, source_evidence_hash="sha256:" + "2" * 64).contract_hash != ref.contract_hash
    )
    bad = np.asarray(ref.poses[0])
    bad[7] = 0.001
    with pytest.raises(ValueError, match="measured pose"):
        ref.initialize(bad, frames=60)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_variant": "sonic_v1_1"},
        {"latent_schedule": SonicLatentSchedule(((0.0,) * 64,))},
        {"experimental_maximum_speed_mps": 1.0},
        {"maximum_frames": "100"},
    ],
)
def test_fixed_reference_is_explicit_and_not_mixed_with_latent_search(kwargs):
    options = dict(model_variant="low_latency", pose_reference=reference())
    options.update(kwargs)
    with pytest.raises(ValueError):
        SonicNavigationConfig(**options)


def test_fixed_reference_never_calls_navigation_planner(monkeypatch):
    backend = object.__new__(_StreamingBackend)
    backend.navigation = SonicNavigationConfig(
        model_variant="low_latency", pose_reference=reference()
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed reference was silently replanned")

    monkeypatch.setattr(backend, "plan", forbidden)
    values = backend._generate_reference(np.asarray(reference().poses[0]))
    assert values.shape == (710, 36)
    calls = []
    monkeypatch.setattr(backend, "_update_from_reference", lambda state, frame: calls.append(frame))
    backend.navigation_tick(None, 20)
    assert calls == [20]


@pytest.mark.parametrize(
    "column,value", [(2, 0.1), (3, 0.0), (7, 4.0), (0, 100.0), (7, float("nan"))]
)
def test_malformed_pose_rejected(column, value):
    values = np.asarray(reference().poses).copy()
    values[1, column] = value
    with pytest.raises(ValueError):
        SonicPoseReference(
            tuple(tuple(float(v) for v in row) for row in values), "sha256:" + "1" * 64
        )


def test_pose_jump_is_not_silently_smoothed():
    values = np.asarray(reference().poses).copy()
    values[1, 7] = 0.3
    with pytest.raises(ValueError, match="continuous"):
        SonicPoseReference(
            tuple(tuple(float(v) for v in row) for row in values), "sha256:" + "1" * 64
        )
