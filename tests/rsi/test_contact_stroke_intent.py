import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_stroke_intent import contact_stroke_delta


def test_zero_and_outside_support_preserve_baseline_exactly():
    for frame in range(300):
        assert not contact_stroke_delta(
            foot="left", frame=frame, center=104, latent=np.zeros((2, 6))
        ).any()
    for frame in (0, 79, 80, 85, 86, 114, 115, 299):
        assert not contact_stroke_delta(
            foot="left", frame=frame, center=104, latent=np.ones((2, 6))
        ).any()


def test_overlapping_pulses_stay_bounded_and_affect_only_the_selected_leg():
    for foot in ("left", "right"):
        rows = np.array(
            [
                contact_stroke_delta(foot=foot, frame=f, center=104, latent=np.ones((2, 6)))
                for f in range(300)
            ]
        )
        assert rows.shape == (300, 29)
        assert abs(rows).max() < 0.2
        assert np.count_nonzero(rows.any(axis=0)) == 6
    left = contact_stroke_delta(foot="left", frame=104, center=104, latent=np.ones((2, 6)))
    right = contact_stroke_delta(foot="right", frame=104, center=104, latent=np.ones((2, 6)))
    assert not (left * right).any()


def test_retraction_pulse_is_available_before_contact_and_can_reverse_stroke():
    latent = np.stack([np.ones(6), -np.ones(6)])
    early = contact_stroke_delta(foot="left", frame=95, center=104, latent=latent)
    impact = contact_stroke_delta(foot="left", frame=104, center=104, latent=latent)
    assert early.max() > 0 and impact.min() < 0


@pytest.mark.parametrize("fault", ["nan", "shape", "bound", "center", "frame", "foot"])
def test_invalid_search_does_not_silently_clip_or_shift(fault):
    kwargs = dict(foot="left", frame=104, center=104, latent=np.zeros((2, 6)))
    if fault == "nan":
        kwargs["latent"][0, 0] = np.nan
    elif fault == "shape":
        kwargs["latent"] = np.zeros(12)
    elif fault == "bound":
        kwargs["latent"][0, 0] = 1.01
    elif fault == "center":
        kwargs["center"] = 90
    elif fault == "frame":
        kwargs["frame"] = 300
    else:
        kwargs["foot"] = "best"
    with pytest.raises(ValueError):
        contact_stroke_delta(**kwargs)
