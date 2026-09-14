import numpy as np
import pytest

from rosclaw_soccer.sim.causal_reference_pulse import (
    CausalReferencePulse,
    CausalReferencePulseConfig,
)


def config(**overrides: object) -> CausalReferencePulseConfig:
    values = dict(
        amplitude_rad=(0.1, -0.1),
        rise_sec=0.02,
        hold_end_sec=0.06,
        fade_sec=0.04,
        maximum_combined_delta_rad=0.25,
    )
    values.update(overrides)
    return CausalReferencePulseConfig(**values)


def test_profile_latches_event_and_does_not_rearm() -> None:
    pulse = CausalReferencePulse(config())
    existing = np.asarray([0.02, 0.01])
    existing.flags.writeable = False
    for time, event, expected in [
        (0.0, False, 0.0),
        (0.20, True, 0.0),
        (0.21, False, 0.5),
        (0.23, False, 1.0),
        (0.28, False, 0.5),
        (0.31, False, 0.0),
        (0.40, True, 0.0),
    ]:
        result = pulse.sample(
            timestamp_sec=time, observed_trigger=event, existing_delta_rad=existing
        )
        assert result.weight == pytest.approx(expected)
        np.testing.assert_allclose(result.delta_rad, np.asarray([0.1, -0.1]) * expected)
        np.testing.assert_allclose(result.combined_delta_rad, existing + result.delta_rad)
        assert not result.delta_rad.flags.writeable
        assert not result.combined_delta_rad.flags.writeable
    np.testing.assert_array_equal(existing, [0.02, 0.01])
    assert result.trigger_timestamp_sec == 0.20


def test_roles_have_independent_event_state() -> None:
    first = CausalReferencePulse(config())
    second = CausalReferencePulse(config())
    for pulse, event in [(first, True), (second, False)]:
        pulse.sample(timestamp_sec=0.0, observed_trigger=event, existing_delta_rad=np.zeros(2))
    a = first.sample(timestamp_sec=0.03, observed_trigger=False, existing_delta_rad=np.zeros(2))
    b = second.sample(timestamp_sec=0.03, observed_trigger=False, existing_delta_rad=np.zeros(2))
    assert a.weight == 1.0 and b.weight == 0.0


@pytest.mark.parametrize(
    "override",
    [
        {"amplitude_rad": ()},
        {"amplitude_rad": (np.nan,)},
        {"amplitude_rad": (True,)},
        {"amplitude_rad": (0.26,)},
        {"rise_sec": 0.0},
        {"rise_sec": 0.08},
        {"hold_end_sec": 0.01},
        {"fade_sec": np.inf},
        {"fade_sec": 0.0},
        {"rise_sec": 10**1000},
        {"maximum_combined_delta_rad": 0.0},
        {"maximum_combined_delta_rad": True},
        {"maximum_combined_delta_rad": np.nan},
        {"activation_ceiling": "REAL"},
    ],
)
def test_rejects_invalid_config(override: dict) -> None:
    with pytest.raises(ValueError):
        config(**override)


@pytest.mark.parametrize(
    "time,event,existing",
    [
        (-1.0, False, [0.0, 0.0]),
        (np.nan, False, [0.0, 0.0]),
        (True, False, [0.0, 0.0]),
        (10**1000, False, [0.0, 0.0]),
        (0.0, 1, [0.0, 0.0]),
        (0.0, False, [0.0]),
        (0.0, False, [np.inf, 0.0]),
        (0.0, False, [0.26, 0.0]),
        (0.0, False, [False, False]),
        (0.0, False, [0j, 0j]),
    ],
)
def test_invalid_samples_latch_fault(time: float, event: bool, existing: list) -> None:
    pulse = CausalReferencePulse(config())
    with pytest.raises(ValueError):
        pulse.sample(
            timestamp_sec=time, observed_trigger=event, existing_delta_rad=np.asarray(existing)
        )
    with pytest.raises(ValueError, match="fault-latched"):
        pulse.sample(timestamp_sec=1.0, observed_trigger=False, existing_delta_rad=np.zeros(2))


@pytest.mark.parametrize("time", [0.0, 0.01])
def test_rejects_duplicate_or_reversed_time(time: float) -> None:
    pulse = CausalReferencePulse(config())
    pulse.sample(timestamp_sec=0.01, observed_trigger=False, existing_delta_rad=np.zeros(2))
    with pytest.raises(ValueError):
        pulse.sample(timestamp_sec=time, observed_trigger=False, existing_delta_rad=np.zeros(2))


def test_combined_bound_checked_instead_of_only_each_term() -> None:
    pulse = CausalReferencePulse(config())
    pulse.sample(timestamp_sec=0.0, observed_trigger=True, existing_delta_rad=np.zeros(2))
    with pytest.raises(ValueError, match="combined"):
        pulse.sample(
            timestamp_sec=0.03, observed_trigger=False, existing_delta_rad=np.ones(2) * 0.2
        )


def test_hash_binds_timing_and_amplitudes() -> None:
    assert config().config_hash == config().config_hash
    assert config().config_hash != config(rise_sec=0.01).config_hash
    assert config().config_hash != config(amplitude_rad=(0.1, 0.0)).config_hash


def test_numpy_scalars_have_serializable_normalized_config() -> None:
    value = config(rise_sec=np.float32(0.02), amplitude_rad=(np.float32(0.1), 0.0))
    assert value.config_hash.startswith("sha256:")
    assert type(value.rise_sec) is float and type(value.amplitude_rad[0]) is float
