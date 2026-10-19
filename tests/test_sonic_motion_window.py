import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_motion_window import select_dynamic_leg_window


def inputs():
    entry = np.array([0, 0, 0.793, 1, 0, 0, 0] + [0.0] * 29)
    source = np.repeat(entry[None], 1000, axis=0)
    source[600:, 7] = 0.1
    return source, entry


def select(source, entry):
    return select_dynamic_leg_window(source, entry, source_evidence_hash="sha256:" + "1" * 64)


def test_selects_source_leg_motion_not_wrist_motion_or_outcome():
    source, entry = inputs()
    source[800:, 30] = 0.2
    result = select(source, entry)
    assert result.source_peak_leg_interval == 599
    assert result.source_start_frame == 359
    assert result.source_stop_frame_exclusive == 840
    assert result.source_peak_leg_speed_rad_per_sec == pytest.approx(12)
    assert len(result.prepared.reference.poses) == 251
    assert source[600, 7] == 0.1


def test_invalid_selected_window_is_not_replaced_with_quiet_segment():
    source, entry = inputs()
    source[500:, 26] = 1
    with pytest.raises(ValueError, match="continuous"):
        select(source, entry)


def test_short_source_not_padded_to_create_fake_motion():
    source, entry = inputs()
    with pytest.raises(ValueError, match="at least4seconds"):
        select(source[:480], entry)


def test_edge_peak_has_full_four_seconds_without_stretch():
    source, entry = inputs()
    source[:, 7] = 0
    source[1:, 7] = 0.1
    result = select(source, entry)
    assert result.source_start_frame == 0
    assert result.source_stop_frame_exclusive == 481
    assert result.prepared.sampled_duration_sec == 4
