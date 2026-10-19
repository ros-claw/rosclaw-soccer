import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference


def inputs():
    entry = np.array([0, 0, 0.793, 1, 0, 0, 0] + [0.0] * 29)
    source = np.repeat(entry[None], 121, axis=0)
    source[:, :3] += [1, 2, -0.05]
    source[:, 7:] = np.linspace(0, 0.3, 121)[:, None] * np.linspace(0.1, 1, 29)
    return source, entry


def prepare(source, entry, **kwargs):
    options = dict(
        source_frequency_hz=120, entry_intervals=50, source_evidence_hash="sha256:" + "1" * 64
    )
    options.update(kwargs)
    return prepare_sonic_reference(source, entry, **options)


def test_explicit_timing_complete_channels_entry_and_no_input_mutation():
    source, entry = inputs()
    original = source.copy()
    result = prepare(source, entry)
    values = result.reference.initialize(entry, frames=110)
    assert len(result.reference.poses) == 101
    assert result.sampled_frames == 51
    assert result.source_duration_sec == result.sampled_duration_sec == 1
    np.testing.assert_array_equal(values[0], entry)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_allclose(values[100, 7:], source[-1, 7:], atol=0, rtol=0)
    np.testing.assert_allclose(values[50:, :3], np.repeat(entry[None, :3], 60, axis=0))
    assert result.contract_hash.startswith("sha256:")
    assert prepare(source, entry, source_frequency_hz=60).contract_hash != result.contract_hash


def test_yaw_and_translation_are_reference_metadata_not_body_reset():
    source, entry = inputs()
    source[:, 3:7] = Rotation.from_euler("Z", np.pi / 2).as_quat()[[3, 0, 1, 2]]
    source[:, 1] += np.linspace(0, 0.1, len(source))
    result = prepare(source, entry)
    out = np.asarray(result.reference.poses)
    np.testing.assert_array_equal(out[0], entry)
    np.testing.assert_allclose(out[-1, :3], entry[:3] + [0.1, 0, 0], atol=1e-14)
    assert result.heading_rotation_rad == pytest.approx(-np.pi / 2)
    np.testing.assert_array_equal(entry[7:], np.zeros(29))


def test_antipodal_quaternions_do_not_cause_rotation_jump():
    source, entry = inputs()
    source[1::2, 3:7] *= -1
    out = np.asarray(prepare(source, entry).reference.poses)
    np.testing.assert_allclose(abs(out[:, 3]), 1)
    np.testing.assert_allclose(out[:, 4:7], 0)


@pytest.mark.parametrize("fps", [0, 1200, float("nan"), True])
def test_bad_source_timing_rejected(fps):
    with pytest.raises(ValueError):
        prepare(*inputs(), source_frequency_hz=fps)


@pytest.mark.parametrize("kind", ["missing_joints", "nonfinite", "quaternion", "jump", "too_long"])
def test_unqualified_source_not_padded_clipped_or_silently_smoothed(kind):
    source, entry = inputs()
    if kind == "missing_joints":
        source = source[:, :-6]
    elif kind == "nonfinite":
        source[3, 8] = float("inf")
    elif kind == "quaternion":
        source[0, 3] = 2
    elif kind == "jump":
        source[::2, 7] = 3
    elif kind == "too_long":
        source = np.repeat(source, 100, axis=0)
    with pytest.raises(ValueError):
        prepare(source, entry)


def test_source_end_is_not_stretched_and_entry_is_explicit():
    source, entry = inputs()
    result = prepare(source[:-1], entry)
    assert 0 <= result.source_duration_sec - result.sampled_duration_sec < 0.02
    assert result.sampled_frames == 50
    changed = entry.copy()
    changed[7] = 0.01
    with pytest.raises(ValueError, match="measured pose"):
        result.reference.initialize(changed, frames=110)
