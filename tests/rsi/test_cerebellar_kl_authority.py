import numpy as np
import pytest

from scripts.rsi_audit_cerebellar_kl_authority import budget_report


def test_exact_existing_anchor_caps_imply_small_same_state_learned_torque_ceiling():
    report = budget_report()
    rows = {r["gate"]: r for r in report["rows"]}
    assert rows["maximum_anchor_state_kl"]["torque_vector_l2_bound_nm"] == pytest.approx(0.02)
    assert rows["maximum_anchor_mean_kl"]["torque_vector_l2_bound_nm"] == pytest.approx(
        0.00447213595499958
    )
    assert not report["physical_ineffectiveness_proven"]
    assert not report["limits_changed"]


def test_tanh_same_noise_and_same_feature_change_respects_gaussian_bound():
    rng = np.random.default_rng(1028)
    report = budget_report()
    caps = np.asarray(report["all_29_caps_nm"])
    for row in report["rows"]:
        mean = rng.normal(size=(1000, 29))
        delta = rng.normal(size=(1000, 29))
        delta *= row["latent_mean_l2_bound"] / np.linalg.norm(delta, axis=1, keepdims=True)
        shared_noise = 0.05 * rng.normal(size=(1000, 29))
        difference = (np.tanh(mean + delta + shared_noise) - np.tanh(mean + shared_noise)) * caps
        assert (
            np.max(np.linalg.norm(difference, axis=1)) <= row["torque_vector_l2_bound_nm"] + 1e-14
        )
