import numpy as np
import pytest

from scripts.rsi_diagnose_memory_learning_transfer import transfer_statistics


def test_exact_zero_gate_and_action_envelope_statistics():
    noise = np.full((2, 12), 0.1)
    shift = np.vstack((np.zeros(12), np.full(12, 0.025)))
    report = transfer_statistics(noise, shift, [0, 1], [0.1, 0.1], residual_cap=0.05)
    assert report["fraction_noise_components_outside_residual_envelope"] == 1
    assert report["gate_quantiles"][0] == 0
    assert report["exact_mean_latent_kl"] == pytest.approx(0.1875)


def test_exploration_need_not_exceed_actor_capacity():
    report = transfer_statistics(
        np.zeros((1, 12)), np.zeros((1, 12)), [1], [0.1], residual_cap=0.05
    )
    assert report["fraction_noise_components_outside_residual_envelope"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_noise_rejected(bad):
    with pytest.raises(ValueError):
        transfer_statistics(np.full((1, 12), bad), np.zeros((1, 12)), [1], [0.1], residual_cap=0.05)


def test_shift_outside_memory_gated_capacity_rejected():
    with pytest.raises(ValueError):
        transfer_statistics(
            np.zeros((1, 12)), np.full((1, 12), 0.01), [0], [0.1], residual_cap=0.05
        )
