import argparse
import copy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.cerebellar_torque_audit import reconstruct_native_cerebellar
from rosclaw_soccer.rsi.cerebellar_torque_policy import initialize
from rosclaw_soccer.rsi.native_cerebellar_torque import (
    WIDTHS,
    NativeCerebellarTorque,
    validate_native_request,
)
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS

WORLD = "sha256:" + "b" * 64


def body():
    q, v = np.zeros(43), np.zeros(41)
    q[3] = 1.0
    return q, v, np.zeros(29), np.zeros(29)


def test_default_request_preserves_existing_entry_without_new_model():
    validate_native_request(
        None,
        None,
        partition="old",
        captured=False,
        foundation_only=True,
        recurrent_factory=None,
        mixed_probe=True,
    )
    with pytest.raises(ValueError, match="explicit model"):
        validate_native_request(
            None,
            42,
            partition="old",
            captured=False,
            foundation_only=True,
            recurrent_factory=None,
            mixed_probe=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"partition": "REAL"},
        {"record_foundation_observation": False},
        {"foundation_only": True},
        {"locomotion_mode_probe": 2},
        {"body_response_bundle": Path("unused")},
    ],
)
def test_invalid_native_request_rejected_before_model_or_output_allocation(kwargs, tmp_path):
    arguments = dict(
        record_foundation_observation=True, foundation_only=False, body_response_bundle=None
    )
    options = dict(partition="DECLARED_DEVELOPMENT_DIAGNOSTIC", locomotion_mode_probe=None)
    for key, value in kwargs.items():
        (options if key in options else arguments)[key] = value
    with pytest.raises(ValueError):
        run_native_first_touch_episode(
            argparse.Namespace(**arguments),
            course=(2.3, 0.0, -0.4),
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            cerebellar_model=initialize(seed=1, world_hash=WORLD),
            **options,
        )
    assert not list(tmp_path.iterdir())


def test_world_binding_and_episode_owned_parameters():
    model = initialize(seed=1, world_hash=WORLD)
    with pytest.raises(ValueError, match="actual compiled world"):
        NativeCerebellarTorque(model, world_hash="sha256:" + "c" * 64, sampling_seed=None)
    layer = NativeCerebellarTorque(model, world_hash=WORLD, sampling_seed=None)
    original = layer.contract()
    model["layers"][-1]["bias"][0] = 10
    assert layer.contract() == original
    torque, record = layer.advance(*body())
    np.testing.assert_array_equal(torque, np.zeros(29))
    assert set(record) == set(WIDTHS)
    assert record["cerebellar_sampling_sample_index"].tolist() == [-1]
    assert not record["cerebellar_sampling_performed"].any()
    assert layer.calls == 1


def test_history_is_actual_after_clipping_and_rng_continues_across_calls():
    layer = NativeCerebellarTorque(
        initialize(seed=1, world_hash=WORLD), world_hash=WORLD, sampling_seed=191
    )
    q, v, target, _ = body()
    for index in range(12):
        torque, record = layer.advance(q, v, target, np.full(29, 1000.0))
        np.testing.assert_array_equal(torque, G1_HARD_TORQUE_LIMITS)
        assert record["requested_cerebellar_torque_offset_nm"].any()
        assert not record["applied_cerebellar_torque_offset_nm"].any()
        assert not record["cerebellar_input_features"][-29:].any()
        assert record["cerebellar_sampling_sample_index"].tolist() == [index]
    assert layer.calls == 12


@pytest.fixture(scope="module", params=[(None, False), (192, False), (None, True), (192, True)])
def numeric_capture(request):
    """Synthetic numeric contract fixture; NOT a physical evidence fixture."""
    seed, time_aware = request.param
    layer = NativeCerebellarTorque(
        initialize(seed=1, world_hash=WORLD, time_aware=time_aware),
        world_hash=WORLD,
        sampling_seed=seed,
    )
    widths = layer.capture_widths()
    records = {key: [] for key in widths}
    for _ in range(3000):
        _, row = layer.advance(*body())
        for key, value in row.items():
            records[key].append(value)
    trace = {
        key: np.asarray(value).reshape(300, 1, 10, widths[key]) for key, value in records.items()
    }
    trace["joint_target_rad"] = np.zeros((300, 1, 29))
    commitment = dict(
        schema="soccer.rsi.cpu_motor_transfer.v2",
        compiled_model_hash=WORLD,
        cerebellar_torque_contract=layer.contract(),
    )
    return commitment, trace, seed


def test_independent_capture_reconstructs_math_but_does_not_claim_dynamics(numeric_capture):
    commitment, trace, seed = numeric_capture
    result = reconstruct_native_cerebellar(commitment, commitment, trace)
    assert result["independently_recomputed_substeps"] == 3000
    assert result["sampling_draws"] == (0 if seed is None else 3000)
    assert result["numpy_torch_maximum_mean_error"] == 0


@pytest.mark.parametrize("kind", ["feature", "request", "density", "counter", "missing", "extra"])
def test_corrupt_capture_cannot_get_independent_reconstruction(numeric_capture, kind):
    commitment, original, _ = numeric_capture
    trace = copy.deepcopy(original)
    if kind == "feature":
        trace["cerebellar_input_features"][20, 0, 3, 0] += 0.01
    elif kind == "request":
        trace["requested_cerebellar_torque_offset_nm"][20, 0, 3, 0] += 0.01
    elif kind == "density":
        trace["cerebellar_sampling_log_probability"][20, 0, 3, 0] += 1
    elif kind == "counter":
        trace["cerebellar_sampling_sample_index"][20, 0, 3, 0] = 0
    elif kind == "missing":
        del trace["cerebellar_sampling_mean"]
    else:
        trace["cerebellar_unknown"] = np.zeros(1)
    with pytest.raises(ValueError):
        reconstruct_native_cerebellar(commitment, commitment, trace)


def test_uncommitted_or_one_sided_capture_rejected(numeric_capture):
    commitment, trace, _ = numeric_capture
    with pytest.raises(ValueError, match="execution commitment"):
        reconstruct_native_cerebellar({}, {}, trace)
    with pytest.raises(ValueError, match="contract"):
        reconstruct_native_cerebellar({}, commitment, trace)
    assert reconstruct_native_cerebellar({}, {}, {}) is None


def test_v2_clock_cannot_be_replayed_reset_or_forged_as_float(numeric_capture):
    commitment, original, _ = numeric_capture
    key = "cerebellar_before_integration_substep_index"
    if key not in original:
        return
    for corruption in ("value", "dtype", "feature"):
        trace = copy.deepcopy(original)
        if corruption == "value":
            trace[key][7, 0, 0, 0] = 0
        elif corruption == "dtype":
            trace[key] = trace[key].astype(np.float64)
        else:
            trace["cerebellar_input_features"][7, 0, 0, -1] = 1
        with pytest.raises(ValueError):
            reconstruct_native_cerebellar(commitment, commitment, trace)
