import numpy as np
import pytest

from scripts.rsi_recover_archived_receding_expert import require_actual_zero_control


@pytest.mark.parametrize("tamper", [None, "mean", "applied", "sampling", "optimizer", "identity"])
def test_archived_success_is_not_mislabeled_as_trained_or_random_controller(tamper):
    receipt = dict(
        job="zero_control-2026101801-context-7",
        group=77,
        cerebellar_model_optimizer_steps=0,
        cerebellar_sampling_draws=0,
        cerebellar_inference_calls=3000,
    )
    trace = {
        k: np.zeros((300, 1, 10, 29))
        for k in (
            "cerebellar_sampling_mean",
            "cerebellar_sampling_latent_action",
            "requested_cerebellar_torque_offset_nm",
            "applied_cerebellar_torque_offset_nm",
        )
    }
    trace["cerebellar_sampling_performed"] = np.zeros((300, 1, 10, 1))
    if tamper == "mean":
        trace["cerebellar_sampling_mean"][0, 0, 0, 0] = 0.001
    elif tamper == "applied":
        trace["applied_cerebellar_torque_offset_nm"][0, 0, 0, 0] = 0.1
    elif tamper == "sampling":
        trace["cerebellar_sampling_performed"][0, 0, 0, 0] = 1
    elif tamper == "optimizer":
        receipt["cerebellar_model_optimizer_steps"] = 2
    elif tamper == "identity":
        receipt["job"] = "collection-context-7"
    if tamper:
        with pytest.raises(ValueError):
            require_actual_zero_control(receipt, trace)
    else:
        require_actual_zero_control(receipt, trace)
