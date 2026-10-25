import numpy as np
import pytest
from test_g1_velocity_policy import parameters

from rosclaw_soccer.providers.g1.experimental_velocity_envelope import experimental_observation98
from rosclaw_soccer.providers.g1.velocity_policy import observation98, phase_delta


def state():
    q, v = np.zeros(43), np.zeros(41)
    q[3] = q[39] = 1
    return q, v, np.zeros(3), np.zeros(29, np.float32), phase_delta(), parameters()


def test_experimental_codec_is_bit_exact_in_original_domain_over_whole_phase_history():
    q, v, angular, action, phase, params = state()
    rng = np.random.default_rng(123)
    for frame in range(300):
        command = rng.uniform([-0.5, -0.5, -1], [1, 0.5, 1]).astype(np.float32)
        if frame % 10 == 0:
            command[:] = 0.01
        q[7:36] = rng.uniform(-0.1, 0.1, 29)
        v[6:35] = rng.uniform(-0.5, 0.5, 29)
        old, clock = observation98(q, v, angular, command, action, phase, params)
        new, next_clock = experimental_observation98(q, v, angular, command, action, phase, params)
        np.testing.assert_array_equal(old.view(np.uint32), new.view(np.uint32))
        assert clock == next_clock
        new[:] = 100
        assert not np.any(old == 100)
        phase = clock
        action = rng.uniform(-1, 1, 29).astype(np.float32)


@pytest.mark.parametrize("command", [(1.5, 0, 0), (2, 0, 0), (0, 0.8, 0), (0, -0.8, 0)])
def test_expanded_command_is_explicit_and_original_validation_still_rejects(command):
    q, v, angular, action, phase, params = state()
    c = np.array(command, np.float32)
    with pytest.raises(ValueError):
        observation98(q, v, angular, c, action, phase, params)
    new, _ = experimental_observation98(q, v, angular, c, action, phase, params)
    np.testing.assert_array_equal(new[6:9], c)
    assert np.any(new[9:11])


@pytest.mark.parametrize("command", [(2.01, 0, 0), (0, 1.01, 0), (0, 0, float("nan"))])
def test_experimental_domain_is_not_arbitrary_or_nonfinite(command):
    q, v, angular, action, phase, params = state()
    with pytest.raises(ValueError):
        experimental_observation98(
            q, v, angular, np.array(command, np.float32), action, phase, params
        )


def test_experimental_stream_has_private_history_and_latches_sequence_fault(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rosclaw_soccer.providers.g1 import experimental_velocity_envelope as module
    from rosclaw_soccer.providers.g1.velocity_policy import file_hash

    class Session:
        def run(self, outputs, inputs):
            return [np.zeros((1, 29), np.float32)]

    monkeypatch.setattr(
        module,
        "G1VelocityPolicy",
        lambda *args, **kw: SimpleNamespace(
            params=parameters(), session=Session(), contract_hash="sha256:" + "a" * 64
        ),
    )
    reference = tmp_path / "reference.txt"
    reference.write_text("synthetic reference, not a trained curriculum claim")
    kwargs = dict(
        expected_model_hash="sha256:" + "b" * 64,
        expected_config_hash="sha256:" + "c" * 64,
        reference_source=reference,
        expected_reference_hash=file_hash(reference),
        horizon_controls=600,
    )
    first = module.ExperimentalVelocityEnvelopeStream(
        tmp_path / "model", tmp_path / "config", **kwargs
    )
    second = module.ExperimentalVelocityEnvelopeStream(
        tmp_path / "model", tmp_path / "config", **kwargs
    )
    q, v, angular, _, _, _ = state()
    target = first.propose(q, v, angular, np.array([2, 0, 0], np.float32), frame=0)
    assert target.shape == (29,) and target.dtype == np.float64
    assert first.neural_calls == 1 and second.neural_calls == 0
    assert first.last_action is not second.last_action
    assert not first.physically_qualified
    for frame in (0, 1):
        with pytest.raises(ValueError):
            first.propose(q, v, angular, np.zeros(3, np.float32), frame=frame)
    assert first.faulted and first.neural_calls == 1
