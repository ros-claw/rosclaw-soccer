"""Contact labels used by the prospective snapshot/full-episode transfer gate."""

from __future__ import annotations

import numpy as np

from rosclaw_soccer.rsi.snapshot_transfer_evidence import _clean, _first


def test_first_contact_uses_first_physical_frame_only() -> None:
    force = np.zeros((4, 6))
    force[1, 1] = 1.1
    force[3, 5] = 5.0
    assert _first(force) == (1, [1])
    assert not _clean(force)


def test_clean_foot_requires_contact_and_no_knee_or_shank() -> None:
    force = np.zeros((3, 6))
    assert _first(force) == (None, [])
    assert not _clean(force)
    force[1, 0] = 2.0
    force[2, 1] = 3.0
    assert _clean(force)
    force[2, 4] = 1.01
    assert not _clean(force)
