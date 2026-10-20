"""Compile the exact optional force function on CPU, not a Python imitation."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from mujoco_warp._src.types import vec5  # noqa: E402

from scripts.rsi_mjwarp_addressed_contact_force import addressed_contact_force  # noqa: E402


@wp.kernel
def probe(
    frame: wp.array(dtype=wp.mat33),
    friction: wp.array(dtype=vec5),
    dim: wp.array(dtype=int),
    address: wp.array2d(dtype=int),
    force: wp.array2d(dtype=float),
    count: wp.array(dtype=int),
    contact_id: int,
    out: wp.array(dtype=wp.spatial_vector),
):
    out[0] = addressed_contact_force(
        0, frame, friction, dim, address, force, 32, count, 0, contact_id, False
    )


@pytest.mark.parametrize("contact_id,dim", [(0, 1), (0, 3), (0, 6), (-1, 3), (1, 3)])
def test_noncontiguous_pyramid_edges_and_inactive_contact(contact_id, dim):
    addresses = np.array([[18, 19, 4, 5, 9, 8, 3, 2, 14, 15]], dtype=np.int32)
    force = np.zeros((1, 32), dtype=np.float32)
    force[0, 20] = 117.47256  # stale adjacent row, NEVER part of this contact
    width = 1 if dim == 1 else 2 * (dim - 1)
    force[0, addresses[0, :width]] = np.arange(1, width + 1)
    friction = np.array([[0.8, 0.7, 0.1, 0.1, 0.1]], dtype=np.float32)
    out = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
    wp.launch(
        probe,
        dim=1,
        inputs=[
            wp.array(np.eye(3, dtype=np.float32)[None], dtype=wp.mat33, device="cpu"),
            wp.array(friction, dtype=vec5, device="cpu"),
            wp.array([dim], dtype=int, device="cpu"),
            wp.array(addresses, dtype=int, device="cpu"),
            wp.array(force, dtype=float, device="cpu"),
            wp.array([1], dtype=int, device="cpu"),
            contact_id,
        ],
        outputs=[out],
        device="cpu",
    )
    expected = np.zeros(6)
    if contact_id == 0:
        values = force[0, addresses[0, :width]]
        expected[0] = values.sum()
        if dim > 1:
            expected[1:dim] = (values[::2] - values[1::2]) * friction[0, : dim - 1]
    np.testing.assert_allclose(out.numpy()[0], expected, atol=1e-6)
