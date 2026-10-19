import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import (
    WRIST_REFERENCE_COLUMNS,
    quarantine_wrist_reference,
)
from rosclaw_soccer.rsi.source_foot_event import forward_foot_event


def source():
    entry = np.array([0, 0, 0.793, 1, 0, 0, 0] + [0.0] * 29)
    return np.repeat(entry[None], 481, axis=0), entry


def prepare(poses, entry):
    return prepare_sonic_reference(
        poses,
        entry,
        source_frequency_hz=120,
        entry_intervals=50,
        source_evidence_hash="sha256:" + "1" * 64,
    )


def test_only_six_named_wrist_channels_changed_and_original_not_repaired():
    poses, entry = source()
    poses[240:, WRIST_REFERENCE_COLUMNS] = 1
    original = poses.copy()
    with pytest.raises(ValueError, match="continuous"):
        prepare(poses, entry)
    held, receipt = quarantine_wrist_reference(poses, entry)
    assert len(WRIST_REFERENCE_COLUMNS) == 6
    keep = [i for i in range(36) if i not in WRIST_REFERENCE_COLUMNS]
    np.testing.assert_array_equal(held[:, keep], poses[:, keep])
    np.testing.assert_array_equal(poses, original)
    assert receipt.original_source_hash != receipt.quarantined_source_hash
    assert receipt.maximum_removed_adjacent_change_rad == (1.0,) * 6
    assert receipt.contract_hash.startswith("sha256:")
    assert len(prepare(held, entry).reference.poses) == 251


@pytest.mark.parametrize("column", [0, 7, 22])
def test_root_leg_and_shoulder_jumps_are_not_quarantined(column):
    poses, entry = source()
    poses[240:, column] += 1
    held, _ = quarantine_wrist_reference(poses, entry)
    with pytest.raises(ValueError, match="continuous"):
        prepare(held, entry)


def test_nonfinite_wrist_not_hidden_by_quarantine():
    poses, entry = source()
    poses[200, WRIST_REFERENCE_COLUMNS[0]] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        quarantine_wrist_reference(poses, entry)


def test_forward_foot_cue_is_source_motion_not_height_or_root_position():
    p = np.zeros((20, 2, 3))
    p[8:, 1, 0] = 0.02
    p[5:, 0, 2] = 10
    event = forward_foot_event(p)
    assert event.interval == 7 and event.foot == "right"
    assert event.root_relative_forward_speed_mps == pytest.approx(2.4)
    with pytest.raises(ValueError, match="lacks"):
        forward_foot_event(np.zeros((10, 2, 3)))
