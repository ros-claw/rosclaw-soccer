"""Independent full-episode neural/latent reconstruction before CPU dynamics.

Does not call the producer bridge, feature function, or compiled policy forward.
Recorded substep state is subsequently checked against native dynamics by the
CPU replay. This function alone is not an execution or physical qualification.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import cerebellar_torque_policy as model_module
from rosclaw_soccer.rsi import native_cerebellar_torque as bridge_module
from rosclaw_soccer.rsi.cerebellar_torque_policy import validate_model
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json


def reconstruct_native_cerebellar(
    report: dict[str, Any], commitment: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any] | None:
    names = {
        key
        for key in trace
        if key.startswith(("cerebellar_", "requested_cerebellar_", "applied_cerebellar_"))
    }
    contract = commitment.get("cerebellar_torque_contract")
    if contract is None:
        if (
            names
            or "cerebellar_torque_contract" in report
            or report.get("schema") == "soccer.rsi.cpu_motor_transfer.v2"
        ):
            raise ValueError("neural torque capture requires its explicit execution commitment")
        return None
    time_aware = (
        type(contract) is dict and contract.get("schema") == bridge_module.TIME_AWARE_SCHEMA
    )
    widths = dict(bridge_module.WIDTHS)
    if time_aware:
        widths["cerebellar_input_features"] = 127
        widths["cerebellar_before_integration_substep_index"] = 1
    expected_names = set(widths)
    if (
        type(contract) is not dict
        or report.get("cerebellar_torque_contract") != contract
        or report.get("schema") != "soccer.rsi.cpu_motor_transfer.v2"
        or commitment.get("schema") != report["schema"]
        or contract.get("schema") not in (bridge_module.SCHEMA, bridge_module.TIME_AWARE_SCHEMA)
        or names != expected_names
        or contract.get("contract_hash")
        != hash_json({k: v for k, v in contract.items() if k != "contract_hash"})
        or contract.get("world_hash") != commitment.get("compiled_model_hash")
        or contract.get("foundation_control_dt_s") != 0.02
        or contract.get("torque_control_dt_s") != 0.002
        or contract.get("activation_ceiling") != "SIM_ONLY"
        or any(
            contract.get(key) is not False
            for key in (
                "runtime_execution_authorized",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            contract.get(key) is not True
            for key in (
                "foundation_and_parent_motor_use_current_body",
                "episode_applied_history_initial_zero",
                "deterministic_logp_zero_is_placeholder_not_density",
            )
        )
        or contract.get("source_hashes")
        != dict(
            bridge=hash_bytes(Path(bridge_module.__file__).read_bytes()),
            policy=hash_bytes(Path(model_module.__file__).read_bytes()),
        )
    ):
        raise ValueError("complete source-bound SIM neural torque contract required")
    model = contract.get("model")
    if type(model) is not dict:
        raise ValueError("complete explicit neural torque parameter dictionary required")
    validate_model(model)
    if (
        contract.get("model_hash") != model["model_hash"]
        or model["world_hash"] != contract["world_hash"]
        or (model["schema"] == model_module.TIME_AWARE_SCHEMA) != time_aware
    ):
        raise ValueError("actual neural torque parameters/world binding differ")
    sampling = contract.get("sampling")
    if (
        type(sampling) is not dict
        or sampling.get("schema") != "soccer.rsi.cerebellar_latent_sampling.v1"
        or sampling.get("model_hash") != model["model_hash"]
        or sampling.get("latent_std") != 0.05
        or sampling.get("generator") != "numpy_default_rng_PCG64"
        or sampling.get("distribution")
        != "IID Gaussian latent before tanh and original hard torque projection"
        or any(
            sampling.get(key) is not False
            for key in (
                "actuator_density_claimed",
                "reset_per_control_frame",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("exact declared latent sampling contract required")
    seed = sampling.get("sampling_seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("explicit uint32 neural torque RNG seed required")
    arrays = {}
    for key, width in widths.items():
        raw = np.asarray(trace[key])
        dtype = (
            np.int64
            if key.endswith(("sample_index", "substep_index"))
            else np.bool_
            if key.endswith("performed")
            else np.float64
        )
        if raw.shape != (300, 1, 10, width) or raw.dtype != dtype or not np.isfinite(raw).all():
            raise ValueError("whole typed finite 3000-substep neural torque capture required")
        if dtype == np.float64 and np.max(np.abs(raw)) > 1e6:
            raise ValueError("bounded neural torque capture required")
        arrays[key] = raw.reshape(3000, width)
    q, v = (arrays[f"cerebellar_canonical_{name}_before_substep"] for name in ("qpos", "qvel"))
    if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("unit current-body neural torque quaternions required")
    target = np.asarray(trace.get("joint_target_rad"))
    if target.shape != (300, 1, 29) or target.dtype != np.float64 or not np.isfinite(target).all():
        raise ValueError("complete recorded joint target required")
    target = np.repeat(target[:, 0], 10, axis=0)
    caps = np.minimum(2, 0.04 * np.asarray(G1_HARD_TORQUE_LIMITS))
    applied = arrays["applied_cerebellar_torque_offset_nm"]
    if np.any(np.abs(applied) > caps + 1e-8):
        raise ValueError("actual neural torque residual exceeded unchanged caps")
    previous = np.vstack((np.zeros((1, 29)), applied[:-1]))
    quaternion = q[:, 3:7].copy()
    first = np.argmax(quaternion != 0, axis=1)
    quaternion *= np.where(quaternion[np.arange(3000), first] < 0, -1, 1)[:, None]
    features = np.clip(
        np.concatenate(
            (
                (target - q[:, 7:36]) / 0.5,
                v[:, 6:35] / 10,
                quaternion,
                v[:, :6] / [2, 2, 2, 3, 3, 3],
                previous / caps,
            ),
            axis=1,
        ),
        -10,
        10,
    )
    if time_aware:
        clock = arrays["cerebellar_before_integration_substep_index"][:, 0]
        if not np.array_equal(clock, np.arange(3000, dtype=np.int64)):
            raise ValueError("actual before-integration task clock must advance once per substep")
        features = np.concatenate(
            (features, np.clip(q[:, 7:36] / 2, -10, 10), ((3000 - clock) / 3000)[:, None]),
            axis=1,
        )
    if not np.allclose(features, arrays["cerebellar_input_features"], rtol=0, atol=1e-12):
        raise ValueError("independent current-body features/applied history differ")
    import torch

    value = torch.tensor(features, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        for index, layer in enumerate(model["layers"]):
            value = torch.nn.functional.linear(
                value,
                torch.tensor(layer["weight"], dtype=torch.float64, device="cpu"),
                torch.tensor(layer["bias"], dtype=torch.float64, device="cpu"),
            )
            if index < 2:
                value = torch.tanh(value)
    mean = arrays["cerebellar_sampling_mean"]
    error = float(np.max(np.abs(value.numpy() - mean)))
    if error > 1e-10:
        raise ValueError("independent Torch neural torque mean differs")
    latent = arrays["cerebellar_sampling_latent_action"]
    if seed is None:
        expected_latent = mean
        expected_logp = np.zeros(3000)
        expected_index = np.full(3000, -1, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        expected_latent = mean + 0.05 * np.stack([rng.normal(size=29) for _ in range(3000)])
        expected_logp = np.sum(
            -0.5 * ((latent - mean) / 0.05) ** 2 - np.log(0.05) - 0.5 * np.log(2 * np.pi), axis=1
        )
        expected_index = np.arange(3000)
    if (
        not np.array_equal(latent, expected_latent)
        or not np.allclose(
            arrays["cerebellar_sampling_log_probability"][:, 0], expected_logp, rtol=0, atol=1e-10
        )
        or not np.array_equal(arrays["cerebellar_sampling_sample_index"][:, 0], expected_index)
        or not np.all(arrays["cerebellar_sampling_performed"] == (seed is not None))
    ):
        raise ValueError("actual latent draws/density/counter differ from frozen episode law")
    offset = np.tanh(latent) * caps
    if not np.allclose(offset, arrays["requested_cerebellar_torque_offset_nm"], rtol=0, atol=1e-10):
        raise ValueError("actual sampled neural torque request differs")
    return dict(
        requested_offsets=offset.reshape(300, 10, 29),
        model_hash=model["model_hash"],
        sampling_draws=0 if seed is None else 3000,
        independently_recomputed_substeps=3000,
        numpy_torch_maximum_mean_error=error,
    )
