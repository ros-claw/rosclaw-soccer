from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.sim.contact_training_samples import BODY_ORDER, extract_precontact_samples


def _episode() -> dict[str, np.ndarray]:
    ball = np.zeros((30, 3), dtype=np.float64)
    ball[:, 0] = 2.5
    body = np.zeros((30, 1, 4, 3), dtype=np.float64)
    body[:, 0, 1, 1] = -0.12
    force = np.zeros((30, 10, 6), dtype=np.float64)
    force[20, 3, 1] = 10.0
    return {
        "ball_observation_position_m": ball,
        "contact_body_position_m": body,
        "contact_body_velocity_m_s": np.zeros_like(body),
        "ball_body_contact_force_micro_n": force,
    }


def test_extracts_only_preimpact_frames_and_clean_label() -> None:
    episode = _episode()
    sample = extract_precontact_samples(episode, body_names=BODY_ORDER, window_frames=4)
    assert sample.first_contact_microstep == 203
    assert sample.clean_foot_only
    assert sample.frame_indices.tolist() == [16, 17, 18, 19]
    assert sample.features.shape == (4, 18)
    assert sample.features[0, 3:6].tolist() == pytest.approx([2.5, 0.12, 0.0])
    episode["ball_body_contact_force_micro_n"][23, 0, 5] = 3.0
    assert not extract_precontact_samples(episode, body_names=BODY_ORDER).clean_foot_only


def test_rejects_missing_geometry_and_nonfinite_values() -> None:
    episode = _episode()
    with pytest.raises(ValueError):
        extract_precontact_samples(episode, body_names=BODY_ORDER[::-1])
    episode["contact_body_position_m"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        extract_precontact_samples(episode, body_names=BODY_ORDER)
