import mujoco
import numpy as np
import pytest

from rosclaw_soccer.rsi.mesh_surface_swing_probe import (
    G1MeshSurfaceSwingProbe,
    inactive_surface_probe_record,
)
from rosclaw_soccer.sim.contracts import hash_bytes


def test_inactive_capture_has_explicit_clock_and_no_fabricated_point():
    record = inactive_surface_probe_record(81)
    assert record.shape == (39,) and record.dtype == np.float64
    np.testing.assert_array_equal(record[:3], [81, -1, 0])
    assert not record[3:].any()
    assert not np.shares_memory(record, inactive_surface_probe_record(81))


@pytest.mark.parametrize("frame", [-1, 300, True, 1.5])
def test_invalid_clock_rejected(frame):
    with pytest.raises(ValueError):
        inactive_surface_probe_record(frame)


def test_wrong_body_rejected_before_hull_allocation():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError, match="canonical"):
        G1MeshSurfaceSwingProbe(model, expected_world_hash="sha256:" + "a" * 64)


def test_changed_compiled_world_rejected_before_hull_allocation():
    # Only the topology precondition is emulated; no G1 success is claimed.
    class WrongWorld:
        nq, nv, nu = 43, 41, 29

    # A structurally plausible object is not accepted as a serialized MjModel.
    with pytest.raises(TypeError):
        G1MeshSurfaceSwingProbe(WrongWorld(), expected_world_hash=hash_bytes(b"not a model"))
