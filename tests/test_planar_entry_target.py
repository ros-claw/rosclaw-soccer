"""Frame-correct geometry cannot grant a motor action."""

import math

import pytest

from rosclaw_soccer.world.planar_entry_target import planar_entry_root_target


@pytest.mark.parametrize("angle", [-math.pi, -1.2, 0.0, 0.9, math.pi])
@pytest.mark.parametrize("offset", [(0.35, 0.0), (0.2, -0.12), (0.0, 0.0)])
def test_requested_contact_is_reconstructed(angle, offset):
    contact = (2.5, -1.3)
    root = planar_entry_root_target(
        contact_xy=contact, root_to_contact_body_xy=offset, heading_rad=angle
    )
    c, s = math.cos(angle), math.sin(angle)
    assert (
        root[0] + c * offset[0] - s * offset[1],
        root[1] + s * offset[0] + c * offset[1],
    ) == pytest.approx(contact)


def test_translation_and_rotation_equivariance():
    root = planar_entry_root_target(
        contact_xy=(3.0, 1.0), root_to_contact_body_xy=(0.3, -0.1), heading_rad=0.0
    )
    rotated = planar_entry_root_target(
        contact_xy=(-1.0, 3.0), root_to_contact_body_xy=(0.3, -0.1), heading_rad=math.pi / 2
    )
    assert rotated == pytest.approx((-root[1], root[0]))
    shifted = planar_entry_root_target(
        contact_xy=(13.0, -4.0), root_to_contact_body_xy=(0.3, -0.1), heading_rad=0.0
    )
    assert shifted == pytest.approx((root[0] + 10.0, root[1] - 5.0))


@pytest.mark.parametrize(
    "bad",
    [(math.nan, 0.0), (0.0, math.inf), (True, 0.0), [0.0, 0.0], (0.0,), (0.0, "0"), (1001.0, 0.0)],
)
def test_bad_contact_is_rejected(bad):
    with pytest.raises(ValueError):
        planar_entry_root_target(
            contact_xy=bad, root_to_contact_body_xy=(0.3, 0.0), heading_rad=0.0
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, "0", math.pi + 0.001, -math.pi - 0.001])
def test_bad_heading_is_rejected(bad):
    with pytest.raises(ValueError):
        planar_entry_root_target(
            contact_xy=(0.0, 0.0), root_to_contact_body_xy=(0.3, 0.0), heading_rad=bad
        )


@pytest.mark.parametrize("bad", [(2.01, 0.0), (0.0, math.nan), [0.3, 0.0], (False, 0.0)])
def test_bad_skill_offset_is_rejected(bad):
    with pytest.raises(ValueError):
        planar_entry_root_target(
            contact_xy=(0.0, 0.0), root_to_contact_body_xy=bad, heading_rad=0.0
        )
