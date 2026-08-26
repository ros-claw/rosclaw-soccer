"""Four-GPU MJX recovery learning around an immutable OpenTrack teacher.

This module is an explicit training-host integration and is intentionally not
imported by :mod:`rosclaw_soccer.training`.  OpenTrack supplies a privileged,
reference-conditioned teacher during simulation only.  The trainable actor
sees a finite history of deployable pelvis-IMU and joint proprioception frames
and emits a small residual in the teacher's joint-target space.  The history
makes impact and body drift more observable without leaking MuJoCo root
velocity.  A later
DAgger/distillation stage must remove the teacher and reference trajectory
before a candidate can enter the independent CPU-MuJoCo promotion exam.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo_train

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.opentrack_recovery_mjx_ppo import (
    _make_recovery_ppo_networks,
)
from rosclaw_soccer.training.recovery_mjx import (
    RecoveryMJXTeacherResidualPPOConfig,
    _atomic_json,
    compiled_mujoco_model_contract,
    validate_recovery_mjx_directional_curriculum,
    validate_recovery_mjx_failure_state_manifest,
    validate_recovery_mjx_teacher_residual_report,
)
from rosclaw_soccer.training.recovery_mjx_routes import (
    resolve_recovery_mjx_route_group,
    validate_recovery_mjx_route_manifest,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)

_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MOTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOINT_COUNT = 29
_PRIVILEGED_CRITIC_AUXILIARY_DIM = 8
_FAILURE_TEMPORAL_BIN_COUNT = 6
_PELVIS_IMU_FRAME_DIM = 96
_PELVIS_IMU_VELOCITY_FRAME_DIM = 99


def _tree_hash(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "hash": hash_bytes(path.read_bytes()),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    if not rows:
        raise ValueError("checkpoint tree is empty")
    return str(hash_json(rows)), rows


def _upright_projection(quaternion_wxyz: jax.Array) -> jax.Array:
    _, x, y, _ = quaternion_wxyz
    return 1.0 - 2.0 * (x * x + y * y)


def _quaternion_rotation_matrix(quaternion_wxyz: jax.Array) -> jax.Array:
    w, x, y, z = quaternion_wxyz
    return jnp.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=jnp.float32,
    )


def _potential(data: Any) -> jax.Array:
    upright = jnp.clip(_upright_projection(data.qpos[3:7]), -1.0, 1.0)
    height = jnp.clip((data.qpos[2] - 0.08) / 0.62, 0.0, 1.0)
    return (
        1.5 * height
        + 0.75 * (upright + 1.0)
        - 0.10 * jnp.tanh(jnp.linalg.norm(data.qvel[:3]))
        - 0.10 * jnp.tanh(0.5 * jnp.linalg.norm(data.qvel[3:6]))
    )


def _root_velocity_diagnostics(teacher_environment: Any, data: Any) -> dict[str, jax.Array]:
    """Expose signed, non-actor balance diagnostics in the pelvis frame.

    Norm-only telemetry cannot distinguish backward recoil from forward drift
    or lateral stumbling.  These values are metrics only: they are never
    appended to the deployable actor observation.  The pelvis gyroscope is
    used instead of privileged generalized angular velocity so the diagnostic
    axes match the IMU axes seen by the actor.
    """

    rotation = _quaternion_rotation_matrix(data.qpos[3:7])
    body_linear_velocity = rotation.T @ data.qvel[:3]
    pelvis_angular_velocity = teacher_environment.get_gyro(data, "pelvis")
    return {
        "root_body_forward_velocity": body_linear_velocity[0],
        "root_body_lateral_velocity": body_linear_velocity[1],
        "root_body_vertical_velocity": body_linear_velocity[2],
        "root_body_backward_speed": jnp.maximum(-body_linear_velocity[0], 0.0),
        "root_body_lateral_speed": jnp.abs(body_linear_velocity[1]),
        "pelvis_roll_rate": pelvis_angular_velocity[0],
        "pelvis_pitch_rate": pelvis_angular_velocity[1],
        "pelvis_yaw_rate": pelvis_angular_velocity[2],
        "pelvis_yaw_speed": jnp.abs(pelvis_angular_velocity[2]),
    }


def _temporal_failure_metrics(
    *, wrapper_step: jax.Array, diagnostics: dict[str, jax.Array], episode_length: int
) -> dict[str, jax.Array]:
    temporal_bin = jnp.minimum(
        wrapper_step * _FAILURE_TEMPORAL_BIN_COUNT // episode_length,
        _FAILURE_TEMPORAL_BIN_COUNT - 1,
    )
    sources = {
        "root_body_backward_speed": diagnostics["root_body_backward_speed"],
        "root_body_lateral_speed": diagnostics["root_body_lateral_speed"],
        "pelvis_yaw_speed": diagnostics["pelvis_yaw_speed"],
    }
    return {
        f"{name}_phase_{index}": value * (temporal_bin == index).astype(jnp.float32)
        for name, value in sources.items()
        for index in range(_FAILURE_TEMPORAL_BIN_COUNT)
    }


def _select_tree(predicate: jax.Array, new: Any, old: Any) -> Any:
    return jax.tree_util.tree_map(lambda left, right: jnp.where(predicate, left, right), new, old)


def _append_velocity_to_proprioception_history(
    history: jax.Array,
    velocity_estimate: jax.Array,
) -> jax.Array:
    """Append velocity without changing any legacy IMU or joint feature slot."""

    if history.shape[-1] != _PELVIS_IMU_FRAME_DIM or velocity_estimate.shape[-1] != 3:
        raise ValueError("recovery MJX observation expansion shape is invalid")
    velocity = jnp.broadcast_to(velocity_estimate, history.shape[:-1] + (3,))
    return jnp.concatenate((history[..., :9], velocity, history[..., 9:]), axis=-1)


def _expand_actor_vector(
    value: jax.Array,
    *,
    history_frames: int,
    velocity_fill: jax.Array,
) -> jax.Array:
    expected = history_frames * _PELVIS_IMU_FRAME_DIM
    if value.shape != (expected,):
        raise ValueError("recovery MJX actor vector migration shape is invalid")
    frames = value.reshape((history_frames, _PELVIS_IMU_FRAME_DIM))
    velocity = jnp.broadcast_to(velocity_fill, (history_frames, 3))
    return _append_velocity_to_proprioception_history(frames, velocity).reshape((-1,))


def _expand_observation_normalizer(
    normalizer: Any,
    *,
    history_frames: int,
) -> Any:
    """Expand running statistics while retaining every old normalized feature."""

    old_actor_dim = history_frames * _PELVIS_IMU_FRAME_DIM
    auxiliary_dim = _PRIVILEGED_CRITIC_AUXILIARY_DIM

    def expand_leaf(value: jax.Array, *, privileged: bool, velocity_fill: jax.Array) -> jax.Array:
        expected = old_actor_dim + (auxiliary_dim if privileged else 0)
        if value.shape != (expected,):
            raise ValueError("recovery MJX normalizer migration shape is invalid")
        actor = _expand_actor_vector(
            value[:old_actor_dim],
            history_frames=history_frames,
            velocity_fill=velocity_fill,
        )
        return jnp.concatenate((actor, value[old_actor_dim:]), axis=0)

    mean = {
        "state": expand_leaf(
            normalizer.mean["state"], privileged=False, velocity_fill=jnp.zeros((3,))
        ),
        "privileged_state": expand_leaf(
            normalizer.mean["privileged_state"],
            privileged=True,
            velocity_fill=jnp.zeros((3,)),
        ),
    }
    std = {
        "state": expand_leaf(
            normalizer.std["state"], privileged=False, velocity_fill=jnp.ones((3,))
        ),
        "privileged_state": expand_leaf(
            normalizer.std["privileged_state"],
            privileged=True,
            velocity_fill=jnp.ones((3,)),
        ),
    }
    # Treat each new, already bounded velocity feature as a unit-variance prior.
    # This preserves the old shared sample count without collapsing the new
    # feature scale to the running-statistics minimum on the first update.
    variance_fill = jnp.broadcast_to(normalizer.count, (3,))
    summed_variance = {
        "state": expand_leaf(
            normalizer.summed_variance["state"],
            privileged=False,
            velocity_fill=variance_fill,
        ),
        "privileged_state": expand_leaf(
            normalizer.summed_variance["privileged_state"],
            privileged=True,
            velocity_fill=variance_fill,
        ),
    }
    return normalizer.replace(mean=mean, std=std, summed_variance=summed_variance)


def _expand_first_layer_kernel(
    kernel: jax.Array,
    *,
    history_frames: int,
    privileged: bool,
) -> jax.Array:
    old_actor_dim = history_frames * _PELVIS_IMU_FRAME_DIM
    new_actor_dim = history_frames * _PELVIS_IMU_VELOCITY_FRAME_DIM
    auxiliary_dim = _PRIVILEGED_CRITIC_AUXILIARY_DIM if privileged else 0
    if kernel.ndim != 2 or kernel.shape[0] != old_actor_dim + auxiliary_dim:
        raise ValueError("recovery MJX first-layer migration shape is invalid")
    expanded = jnp.zeros((new_actor_dim + auxiliary_dim, kernel.shape[1]), dtype=kernel.dtype)
    for frame_index in range(history_frames):
        old_start = frame_index * _PELVIS_IMU_FRAME_DIM
        new_start = frame_index * _PELVIS_IMU_VELOCITY_FRAME_DIM
        expanded = expanded.at[new_start : new_start + 9].set(kernel[old_start : old_start + 9])
        expanded = expanded.at[new_start + 12 : new_start + _PELVIS_IMU_VELOCITY_FRAME_DIM].set(
            kernel[old_start + 9 : old_start + _PELVIS_IMU_FRAME_DIM]
        )
    if privileged:
        expanded = expanded.at[new_actor_dim:].set(kernel[old_actor_dim:])
    return expanded


def _migrate_checkpoint_with_appended_velocity(
    params: Any,
    *,
    history_frames: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Zero-expand a legacy checkpoint and prove its initial behavior parity."""

    if not isinstance(params, (tuple, list)) or len(params) != 3:
        raise ValueError("recovery MJX checkpoint migration payload is invalid")
    normalizer, parent_policy, value = params
    value = copy.deepcopy(value)
    old_actor_dim = history_frames * _PELVIS_IMU_FRAME_DIM
    new_actor_dim = history_frames * _PELVIS_IMU_VELOCITY_FRAME_DIM
    old_networks = _make_recovery_ppo_networks(
        {"state": old_actor_dim, "privileged_state": old_actor_dim + 8},
        _JOINT_COUNT,
        running_statistics.normalize,
    )
    new_networks = _make_recovery_ppo_networks(
        {"state": new_actor_dim, "privileged_state": new_actor_dim + 8},
        _JOINT_COUNT,
        running_statistics.normalize,
    )
    policy = copy.deepcopy(new_networks.policy_network.init(jax.random.PRNGKey(5400)))
    try:
        parent_policy_layers = parent_policy["params"]
        migrated_policy_layers = policy["params"]
        value_kernel = value["params"]["hidden_0"]["kernel"]
    except (KeyError, TypeError) as exc:
        raise ValueError("recovery MJX checkpoint migration network is incompatible") from exc
    parent_layer_names = ("Dense_0", "Dense_1", "Dense_2", "location", "scale")
    if any(name not in parent_policy_layers for name in parent_layer_names):
        raise ValueError("recovery MJX checkpoint migration parent trunk is incomplete")
    for name in parent_layer_names:
        migrated_policy_layers[name] = copy.deepcopy(parent_policy_layers[name])
    migrated_policy_layers["parent_normalizer_mean"] = copy.deepcopy(normalizer.mean["state"])
    migrated_policy_layers["parent_normalizer_std"] = copy.deepcopy(normalizer.std["state"])
    value["params"]["hidden_0"]["kernel"] = _expand_first_layer_kernel(
        value_kernel, history_frames=history_frames, privileged=True
    )
    expanded_normalizer = _expand_observation_normalizer(normalizer, history_frames=history_frames)

    old_actor = jnp.linspace(-0.75, 0.75, old_actor_dim, dtype=jnp.float32)
    velocity = jnp.asarray((0.61, -0.37, 0.19), dtype=jnp.float32)
    new_actor = _expand_actor_vector(
        old_actor, history_frames=history_frames, velocity_fill=velocity
    )
    auxiliary = jnp.linspace(-0.3, 0.3, _PRIVILEGED_CRITIC_AUXILIARY_DIM, dtype=jnp.float32)
    old_observation = {
        "state": old_actor[jnp.newaxis, :],
        "privileged_state": jnp.concatenate((old_actor, auxiliary))[jnp.newaxis, :],
    }
    new_observation = {
        "state": new_actor[jnp.newaxis, :],
        "privileged_state": jnp.concatenate((new_actor, auxiliary))[jnp.newaxis, :],
    }
    old_policy_output = old_networks.policy_network.apply(normalizer, params[1], old_observation)
    new_policy_output = new_networks.policy_network.apply(
        expanded_normalizer, policy, new_observation
    )
    policy_gradients = jax.grad(
        lambda policy_params: jnp.sum(
            new_networks.policy_network.apply(expanded_normalizer, policy_params, new_observation)[
                ..., :_JOINT_COUNT
            ]
        )
    )(policy)
    frozen_parent_names = parent_layer_names + (
        "parent_normalizer_mean",
        "parent_normalizer_std",
    )
    parent_gradient_max = max(
        float(jnp.max(jnp.abs(leaf)))
        for name in frozen_parent_names
        for leaf in jax.tree_util.tree_leaves(policy_gradients["params"][name])
    )
    adapter_output_gradient_l2 = float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(leaf))
                for leaf in jax.tree_util.tree_leaves(
                    policy_gradients["params"]["velocity_adapter_location"]
                )
            )
        )
    )
    old_value_output = old_networks.value_network.apply(normalizer, params[2], old_observation)
    new_value_output = new_networks.value_network.apply(expanded_normalizer, value, new_observation)
    drifted_normalizer = expanded_normalizer.replace(
        mean=jax.tree_util.tree_map(lambda leaf: leaf + 0.37, expanded_normalizer.mean),
        std=jax.tree_util.tree_map(lambda leaf: leaf * 1.73, expanded_normalizer.std),
    )
    drifted_policy_output = new_networks.policy_network.apply(
        drifted_normalizer, policy, new_observation
    )
    policy_error = float(jnp.max(jnp.abs(old_policy_output - new_policy_output)))
    value_error = float(jnp.max(jnp.abs(old_value_output - new_value_output)))
    parent_normalizer_drift_error = float(
        jnp.max(jnp.abs(new_policy_output - drifted_policy_output))
    )
    if (
        policy_error > 1.0e-6
        or value_error > 1.0e-6
        or parent_normalizer_drift_error > 1.0e-6
        or parent_gradient_max != 0.0
        or adapter_output_gradient_l2 <= 0.0
    ):
        raise RuntimeError("recovery MJX observation migration changed parent behavior")
    contract = {
        "strategy": "FROZEN_PARENT_TRUNK_ZERO_OUTPUT_VELOCITY_ADAPTER",
        "source_actor_observation_dim": old_actor_dim,
        "target_actor_observation_dim": new_actor_dim,
        "source_frame_dim": _PELVIS_IMU_FRAME_DIM,
        "target_frame_dim": _PELVIS_IMU_VELOCITY_FRAME_DIM,
        "appended_feature": "ONBOARD_BASE_VELOCITY_ESTIMATE",
        "legacy_accelerometer_slots_preserved": True,
        "parent_policy_trunk_frozen": True,
        "parent_observation_normalizer_frozen": True,
        "parent_normalizer_drift_invariance_max_abs_error": (parent_normalizer_drift_error),
        "new_adapter_output_weights_initialized_to_zero": True,
        "adapter_location_limit": 0.25,
        "parent_policy_gradient_max_abs": parent_gradient_max,
        "adapter_output_gradient_l2": adapter_output_gradient_l2,
        "new_value_input_weights_initialized_to_zero": True,
        "policy_output_max_abs_error": policy_error,
        "value_output_max_abs_error": value_error,
        "behavior_preserved": True,
    }
    return [expanded_normalizer, policy, value], contract


def _named_array_leaves(value: Any, *, prefix: str) -> dict[str, np.ndarray]:
    """Flatten one immutable policy component into content-addressable arrays."""

    if isinstance(value, Mapping):
        leaves: dict[str, np.ndarray] = {}
        for name in sorted(value):
            leaves.update(_named_array_leaves(value[name], prefix=f"{prefix}/{name}"))
        return leaves
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.kind not in "biufc":
        raise ValueError("frozen parent policy contains a non-numeric leaf")
    return {prefix: array}


def _frozen_parent_arrays(params: Any, *, migrated: bool) -> dict[str, np.ndarray]:
    if not isinstance(params, (tuple, list)) or len(params) != 3:
        raise ValueError("frozen parent retention checkpoint is invalid")
    normalizer, policy, _value = params
    try:
        layers = policy["params"]
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen parent retention policy is invalid") from exc
    arrays: dict[str, np.ndarray] = {}
    for name in ("Dense_0", "Dense_1", "Dense_2", "location", "scale"):
        if name not in layers:
            raise ValueError("frozen parent retention trunk is incomplete")
        arrays.update(_named_array_leaves(layers[name], prefix=f"policy/{name}"))
    if migrated:
        for name in ("parent_normalizer_mean", "parent_normalizer_std"):
            if name not in layers:
                raise ValueError("frozen parent retention normalizer is incomplete")
            arrays.update(_named_array_leaves(layers[name], prefix=f"policy/{name}"))
    else:
        arrays.update(
            _named_array_leaves(normalizer.mean["state"], prefix="policy/parent_normalizer_mean")
        )
        arrays.update(
            _named_array_leaves(normalizer.std["state"], prefix="policy/parent_normalizer_std")
        )
    return arrays


def _array_set_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return str(
        hash_json(
            [
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "hash": hash_bytes(array.tobytes(order="C")),
                }
                for name, array in sorted(arrays.items())
            ]
        )
    )


def _verify_frozen_parent_retention(
    *,
    source_params: Any,
    checkpoint_dir: Path,
    checkpoint_loader: Any,
) -> dict[str, Any]:
    """Prove every persisted generation retained the old actor bit-for-bit."""

    source = _frozen_parent_arrays(source_params, migrated=False)
    source_hash = _array_set_hash(source)
    results: list[dict[str, Any]] = []
    paths = sorted(
        (path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not paths:
        raise ValueError("frozen parent retention has no persisted checkpoint")
    for path in paths:
        candidate = _frozen_parent_arrays(checkpoint_loader(path), migrated=True)
        same_keys = candidate.keys() == source.keys()
        max_error = float("inf")
        exact = False
        if same_keys and all(candidate[name].shape == source[name].shape for name in source):
            max_error = max(
                float(np.max(np.abs(candidate[name] - source[name]), initial=0.0))
                for name in source
            )
            exact = all(np.array_equal(candidate[name], source[name]) for name in source)
        results.append(
            {
                "step": int(path.name),
                "frozen_state_hash": _array_set_hash(candidate),
                "exact_equal": exact,
                "maximum_absolute_error": max_error,
            }
        )
    if not all(row["exact_equal"] for row in results):
        raise RuntimeError("PPO changed the frozen parent actor or normalizer")
    return {
        "schema_version": "rosclaw_soccer.frozen_parent_actor_retention.v1",
        "source_frozen_state_hash": source_hash,
        "frozen_components": [
            "DENSE_TRUNK",
            "ACTION_LOCATION_HEAD",
            "ACTION_SCALE_HEAD",
            "OBSERVATION_NORMALIZER_MEAN",
            "OBSERVATION_NORMALIZER_STD",
        ],
        "checkpoint_results": results,
        "all_checkpoints_exact": True,
    }


class OpenTrackRecoveryMJXTeacherResidualEnv(Env):
    """Brax facade over a modern OpenTrack MJX closed-loop teacher."""

    def __init__(
        self,
        *,
        teacher_environment: Any,
        trajectory_data: Any,
        teacher_policy: Any,
        snapshots: tuple[RecoverySnapshot, ...],
        time_dilation: int,
        terminal_balance_reference_frame: int | None,
        directional_curriculum: dict[str, Any] | None,
        failure_state_bank: dict[str, np.ndarray[Any, Any]] | None,
        config: RecoveryMJXTeacherResidualPPOConfig,
        parent_residual_policy: Any | None = None,
        diagnostic_failure_state_reset_fraction: float | None = None,
        diagnostic_adapter_gain: float = 1.0,
    ) -> None:
        self._teacher_environment = teacher_environment
        self._trajectory_data = trajectory_data
        self._teacher_policy = teacher_policy
        self._snapshots = snapshots
        if time_dilation not in (1, 2, 3, 4):
            raise ValueError("teacher residual time dilation is invalid")
        self._time_dilation = time_dilation
        self._config = config
        self._parent_residual_policy = parent_residual_policy
        if (parent_residual_policy is not None) is not config.regularize_velocity_adapter_only:
            raise ValueError("velocity-adapter regularization parent policy is inconsistent")
        if (
            not math.isfinite(diagnostic_adapter_gain)
            or not 1.0 <= diagnostic_adapter_gain <= 4.0
            or (
                diagnostic_adapter_gain != 1.0
                and (
                    diagnostic_failure_state_reset_fraction != 1.0 or parent_residual_policy is None
                )
            )
        ):
            raise ValueError("diagnostic adapter gain requires an exact failure-state exam")
        self._diagnostic_adapter_gain = diagnostic_adapter_gain
        model = teacher_environment.mj_model
        if (model.nq, model.nv, model.nu) != (36, 35, _JOINT_COUNT):
            raise ValueError("teacher residual environment requires OpenTrack G1 29-DoF")
        if config.use_pelvis_imu_observation:
            if not callable(
                getattr(teacher_environment, "get_accelerometer", None)
            ) or not callable(getattr(teacher_environment, "get_gyro", None)):
                raise ValueError("teacher residual environment lacks deployable pelvis IMU access")
            try:
                accelerometer_sensor = model.sensor("accelerometer_pelvis")
                gyroscope_sensor = model.sensor("gyro_pelvis")
            except KeyError as exc:
                raise ValueError(
                    "teacher residual environment lacks required pelvis IMU sensors"
                ) from exc
            if accelerometer_sensor.dim != 3 or gyroscope_sensor.dim != 3:
                raise ValueError("teacher residual pelvis IMU sensor dimensions are invalid")
        self._snapshot_qpos = jnp.asarray(np.stack([item.qpos for item in snapshots]))
        self._snapshot_qvel = jnp.asarray(np.stack([item.qvel for item in snapshots]))
        self._default = jnp.asarray(teacher_environment._default_qpos)
        self._joint_lower = jnp.asarray(teacher_environment._lowers)
        self._joint_upper = jnp.asarray(teacher_environment._uppers)
        self._residual_limits = jnp.asarray(config.residual_limits_rad)
        self._mjx_env = importlib.import_module("mujoco_playground._src.mjx_env")
        self._terminal_balance_reference_frame = terminal_balance_reference_frame
        self._terminal_balance_reference = None
        self._terminal_body_linear_velocity_bias = jnp.zeros((3,), dtype=jnp.float32)
        self._terminal_pelvis_yaw_rate_bias = jnp.zeros((3,), dtype=jnp.float32)
        if directional_curriculum is not None:
            self._terminal_body_linear_velocity_bias = jnp.asarray(
                directional_curriculum["terminal_body_linear_velocity_bias_mps"],
                dtype=jnp.float32,
            )
            self._terminal_pelvis_yaw_rate_bias = jnp.asarray(
                (0.0, 0.0, directional_curriculum["terminal_pelvis_yaw_rate_bias_rad_s"]),
                dtype=jnp.float32,
            )
        if config.terminal_balance_reset_fraction > 0.0:
            split_points = np.asarray(trajectory_data.split_points, dtype=np.int64)
            if (
                terminal_balance_reference_frame is None
                or split_points.shape != (2,)
                or not 0 <= terminal_balance_reference_frame < int(split_points[1]) - 1
            ):
                raise ValueError("terminal balance curriculum reference frame is invalid")
            self._terminal_balance_reference = trajectory_data.get(
                0, terminal_balance_reference_frame, jnp
            )
        self._failure_state_bank = (
            {name: jnp.asarray(value) for name, value in failure_state_bank.items()}
            if failure_state_bank is not None
            else None
        )
        self._failure_state_has_policy_context = bool(
            self._failure_state_bank is not None
            and {
                "last_motor_targets",
                "last_teacher_action",
                "last_residual",
                "proprioception_history",
                "phase_repeat",
            }.issubset(self._failure_state_bank)
        )
        if diagnostic_failure_state_reset_fraction is not None and (
            diagnostic_failure_state_reset_fraction != 1.0
            or config.failure_state_reset_fraction != 0.0
            or config.terminal_balance_reset_fraction != 0.0
            or not self._failure_state_has_policy_context
        ):
            raise ValueError("diagnostic failure-state reset requires an exact exclusive bank")
        self._failure_state_reset_fraction = (
            diagnostic_failure_state_reset_fraction
            if diagnostic_failure_state_reset_fraction is not None
            else config.failure_state_reset_fraction
        )
        if self._failure_state_reset_fraction > 0.0:
            if self._failure_state_bank is None:
                raise ValueError("failure-state curriculum bank is absent")
            failure_count = int(self._failure_state_bank["qpos"].shape[0])
            if (
                self._failure_state_bank["qpos"].shape != (failure_count, 36)
                or self._failure_state_bank["qvel"].shape != (failure_count, 35)
                or any(
                    self._failure_state_bank[name].shape != (failure_count,)
                    for name in (
                        "handoff_frozen",
                        "trajectory_step",
                        "trajectory_initial_step",
                    )
                )
            ):
                raise ValueError("failure-state curriculum bank is invalid")
            if self._failure_state_has_policy_context and (
                self._failure_state_bank["last_motor_targets"].shape != (failure_count, 29)
                or self._failure_state_bank["last_teacher_action"].shape != (failure_count, 29)
                or self._failure_state_bank["last_residual"].shape != (failure_count, 29)
                or self._failure_state_bank["proprioception_history"].shape
                != (
                    failure_count,
                    config.proprioception_history_frames,
                    (
                        _PELVIS_IMU_FRAME_DIM
                        if config.preserve_pelvis_accelerometer_observation
                        else config.actor_proprioception_frame_dim
                    ),
                )
                or self._failure_state_bank["phase_repeat"].shape != (failure_count,)
            ):
                raise ValueError("failure-state policy context is invalid")

    @property
    def observation_size(self) -> int | dict[str, int]:
        actor_size = (
            self._config.actor_proprioception_frame_dim * self._config.proprioception_history_frames
        )
        if not self._config.use_asymmetric_critic:
            return actor_size
        return {
            "state": actor_size,
            "privileged_state": actor_size + _PRIVILEGED_CRITIC_AUXILIARY_DIM,
        }

    @property
    def action_size(self) -> int:
        return _JOINT_COUNT

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def mj_model(self) -> Any:
        return self._teacher_environment.mj_model

    def _actor_proprioception_frame(self, data: Any, last_motor_targets: jax.Array) -> jax.Array:
        rotation = _quaternion_rotation_matrix(data.qpos[3:7])
        gravity_body = rotation.T @ jnp.asarray((0.0, 0.0, -1.0), dtype=jnp.float32)
        if self._config.use_pelvis_imu_observation:
            gyro_body = self._teacher_environment.get_gyro(data, "pelvis")
            accelerometer_body = (
                jnp.clip(
                    self._teacher_environment.get_accelerometer(data, "pelvis"),
                    -self._config.pelvis_accelerometer_clip_mps2,
                    self._config.pelvis_accelerometer_clip_mps2,
                )
                / self._config.pelvis_accelerometer_clip_mps2
            )
            if self._config.use_base_velocity_estimate_observation:
                velocity_estimate_body = (
                    jnp.clip(
                        rotation.T @ data.qvel[:3],
                        -self._config.base_velocity_estimate_clip_mps,
                        self._config.base_velocity_estimate_clip_mps,
                    )
                    / self._config.base_velocity_estimate_clip_mps
                )
                linear_motion_body = (
                    jnp.concatenate((accelerometer_body, velocity_estimate_body))
                    if self._config.preserve_pelvis_accelerometer_observation
                    else velocity_estimate_body
                )
            else:
                linear_motion_body = accelerometer_body
        else:
            # Compatibility path for content-bound v1-v4 experiments only.
            gyro_body = rotation.T @ data.qvel[3:6]
            linear_motion_body = jnp.zeros((0,), dtype=jnp.float32)
        result = jnp.concatenate(
            (
                gravity_body,
                gyro_body * 0.05,
                linear_motion_body,
                data.qpos[7:] - self._default,
                data.qvel[6:] * 0.05,
                last_motor_targets,
            )
        )
        return jnp.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def _initial_proprioception_history(
        self, data: Any, last_motor_targets: jax.Array
    ) -> jax.Array:
        frame = self._actor_proprioception_frame(data, last_motor_targets)
        return jnp.repeat(frame[jnp.newaxis, :], self._config.proprioception_history_frames, axis=0)

    def _actor_critic_observation(
        self,
        data: Any,
        proprioception_history: jax.Array,
        failure_state_reset: jax.Array,
    ) -> jax.Array | dict[str, jax.Array]:
        actor_observation = proprioception_history.reshape((-1,))
        if not self._config.use_asymmetric_critic:
            return actor_observation
        if self._config.failure_state_conditioned_critic:
            velocity_diagnostics = _root_velocity_diagnostics(self._teacher_environment, data)
            privileged_auxiliary = jnp.stack(
                (
                    velocity_diagnostics["root_body_forward_velocity"],
                    velocity_diagnostics["root_body_lateral_velocity"],
                    velocity_diagnostics["root_body_vertical_velocity"],
                    velocity_diagnostics["pelvis_roll_rate"],
                    velocity_diagnostics["pelvis_pitch_rate"],
                    velocity_diagnostics["pelvis_yaw_rate"],
                    data.qpos[2],
                    failure_state_reset.astype(jnp.float32),
                )
            )
        else:
            privileged_auxiliary = jnp.concatenate(
                (
                    data.qvel[:6],
                    jnp.stack((data.qpos[2], _upright_projection(data.qpos[3:7]))),
                )
            )
        privileged_observation = jnp.nan_to_num(
            jnp.concatenate((actor_observation, privileged_auxiliary)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return {
            "state": actor_observation,
            "privileged_state": privileged_observation,
        }

    def _legacy_parent_actor_observation(self, observation: Any) -> dict[str, jax.Array]:
        """Remove appended velocity while retaining raw v11 critic context."""

        if not isinstance(observation, dict):
            raise ValueError("velocity-adapter parent requires asymmetric observations")
        actor = observation["state"]
        frames = actor.reshape((self._config.proprioception_history_frames, 99))
        legacy_actor = jnp.concatenate((frames[:, :9], frames[:, 12:]), axis=-1).reshape((-1,))
        privileged_auxiliary = observation["privileged_state"][-_PRIVILEGED_CRITIC_AUXILIARY_DIM:]
        return {
            "state": legacy_actor,
            "privileged_state": jnp.concatenate((legacy_actor, privileged_auxiliary)),
        }

    def _inject_snapshot(
        self, base_state: Any, rng: jax.Array
    ) -> tuple[
        Any,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        (
            rng,
            index_rng,
            joint_rng,
            velocity_rng,
            curriculum_rng,
            failure_index_rng,
            failure_curriculum_rng,
        ) = jax.random.split(rng, 7)
        index = jax.random.randint(index_rng, (), 0, self._snapshot_qpos.shape[0])
        snapshot_qpos = self._snapshot_qpos[index].at[:2].set(base_state.data.qpos[:2])
        position_noise = jax.random.uniform(
            joint_rng,
            (_JOINT_COUNT,),
            minval=-self._config.joint_position_noise_rad,
            maxval=self._config.joint_position_noise_rad,
        )
        snapshot_qpos = snapshot_qpos.at[7:].set(
            jnp.clip(snapshot_qpos[7:] + position_noise, self._joint_lower, self._joint_upper)
        )
        velocity_noise = jax.random.uniform(velocity_rng, (35,), minval=-1.0, maxval=1.0)
        snapshot_qvel = self._snapshot_qvel[index]
        snapshot_qvel = snapshot_qvel.at[:3].add(
            velocity_noise[:3] * self._config.root_linear_velocity_noise_mps
        )
        snapshot_qvel = snapshot_qvel.at[3:6].add(
            velocity_noise[3:6] * self._config.root_angular_velocity_noise_rad_s
        )
        snapshot_qvel = snapshot_qvel.at[6:].add(
            velocity_noise[6:] * self._config.joint_velocity_noise_rad_s
        )
        terminal_reset = jax.random.uniform(curriculum_rng, ()) < (
            self._config.terminal_balance_reset_fraction
        )
        failure_reset = jax.random.uniform(failure_curriculum_rng, ()) < (
            self._failure_state_reset_fraction
        )
        if self._terminal_balance_reference is None:
            terminal_qpos = snapshot_qpos
            terminal_qvel = snapshot_qvel
            terminal_reset = jnp.asarray(False)
        else:
            terminal_qpos = self._terminal_balance_reference.qpos
            terminal_qpos = terminal_qpos.at[7:].set(
                jnp.clip(
                    terminal_qpos[7:] + position_noise,
                    self._joint_lower,
                    self._joint_upper,
                )
            )
            terminal_qvel = self._terminal_balance_reference.qvel
            terminal_rotation = _quaternion_rotation_matrix(terminal_qpos[3:7])
            terminal_qvel = terminal_qvel.at[:3].add(
                terminal_rotation @ self._terminal_body_linear_velocity_bias
            )
            terminal_qvel = terminal_qvel.at[3:6].add(
                terminal_rotation @ self._terminal_pelvis_yaw_rate_bias
            )
            terminal_qvel = terminal_qvel.at[:3].add(
                velocity_noise[:3] * self._config.terminal_balance_root_linear_velocity_noise_mps
            )
            terminal_qvel = terminal_qvel.at[3:6].add(
                velocity_noise[3:6]
                * self._config.terminal_balance_root_angular_velocity_noise_rad_s
            )
            terminal_qvel = terminal_qvel.at[6:].add(
                velocity_noise[6:] * self._config.joint_velocity_noise_rad_s
            )
        qpos = jnp.where(terminal_reset, terminal_qpos, snapshot_qpos)
        qvel = jnp.where(terminal_reset, terminal_qvel, snapshot_qvel)
        if self._failure_state_bank is None:
            failure_qpos = qpos
            failure_qvel = qvel
            failure_handoff_frozen = jnp.asarray(False)
            failure_trajectory_step = jnp.asarray(0, dtype=jnp.int32)
            failure_trajectory_initial_step = jnp.asarray(0, dtype=jnp.int32)
            failure_motor_targets = qpos[7:]
            failure_teacher_action = jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32)
            failure_last_residual = jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32)
            failure_proprioception_history = jnp.zeros(
                (
                    self._config.proprioception_history_frames,
                    self._config.actor_proprioception_frame_dim,
                ),
                dtype=jnp.float32,
            )
            failure_phase_repeat = jnp.asarray(0, dtype=jnp.int32)
            failure_index = jnp.asarray(-1, dtype=jnp.int32)
            failure_reset = jnp.asarray(False)
        else:
            failure_index = jax.random.randint(
                failure_index_rng, (), 0, self._failure_state_bank["qpos"].shape[0]
            )
            failure_qpos = self._failure_state_bank["qpos"][failure_index]
            failure_qvel = self._failure_state_bank["qvel"][failure_index]
            failure_handoff_frozen = self._failure_state_bank["handoff_frozen"][failure_index]
            failure_trajectory_step = self._failure_state_bank["trajectory_step"][failure_index]
            failure_trajectory_initial_step = self._failure_state_bank["trajectory_initial_step"][
                failure_index
            ]
            if self._failure_state_has_policy_context:
                failure_motor_targets = self._failure_state_bank["last_motor_targets"][
                    failure_index
                ]
                failure_teacher_action = self._failure_state_bank["last_teacher_action"][
                    failure_index
                ]
                failure_last_residual = self._failure_state_bank["last_residual"][failure_index]
                failure_proprioception_history = self._failure_state_bank["proprioception_history"][
                    failure_index
                ]
                failure_phase_repeat = self._failure_state_bank["phase_repeat"][failure_index]
            else:
                # Compatibility for the v1 qpos/qvel-only curriculum.  It is
                # still loadable for evidence validation, but is explicitly
                # reconstructed rather than presented as an exact continuation.
                failure_motor_targets = failure_qpos[7:]
                failure_teacher_action = jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32)
                failure_last_residual = jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32)
                failure_proprioception_history = jnp.zeros(
                    (
                        self._config.proprioception_history_frames,
                        self._config.actor_proprioception_frame_dim,
                    ),
                    dtype=jnp.float32,
                )
                failure_phase_repeat = jnp.asarray(0, dtype=jnp.int32)
        qpos = jnp.where(failure_reset, failure_qpos, qpos)
        qvel = jnp.where(failure_reset, failure_qvel, qvel)
        motor_targets = jnp.where(failure_reset, failure_motor_targets, qpos[7:])
        teacher_last_action = jnp.where(
            failure_reset, failure_teacher_action, jnp.zeros_like(failure_teacher_action)
        )
        initial_last_residual = jnp.where(
            failure_reset, failure_last_residual, jnp.zeros_like(failure_last_residual)
        )
        initial_phase_repeat = jnp.where(
            failure_reset, failure_phase_repeat, jnp.asarray(0, dtype=jnp.int32)
        )
        data = self._mjx_env.init(
            self._teacher_environment.mjx_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=motor_targets,
        )
        info = dict(base_state.info)
        if self._terminal_balance_reference_frame is not None:
            terminal_traj_state = info["traj_info"].traj_state.replace(
                subtraj_step_no=self._terminal_balance_reference_frame + 1,
                subtraj_step_no_init=self._terminal_balance_reference_frame,
            )
            terminal_carry = info["traj_info"].replace(traj_state=terminal_traj_state)
            info["traj_info"] = _select_tree(terminal_reset, terminal_carry, info["traj_info"])
        failure_traj_state = info["traj_info"].traj_state.replace(
            subtraj_step_no=failure_trajectory_step,
            subtraj_step_no_init=failure_trajectory_initial_step,
        )
        failure_carry = info["traj_info"].replace(traj_state=failure_traj_state)
        info["traj_info"] = _select_tree(failure_reset, failure_carry, info["traj_info"])
        info.update(
            last_motor_targets=motor_targets,
            last_action=teacher_last_action,
            last_root_pos=data.qpos[:3],
            last_root_ori=data.qpos[3:7],
            last_dof_pos=data.qpos[7:],
            last_rigid_body_pos=data.xpos,
            last_rigid_body_ori=data.xquat,
            last_joint_vel=data.qvel[6:],
            truncation=jnp.zeros((), dtype=jnp.float32),
        )
        reference = self._teacher_environment.th.get_current_traj_data_with_trajectory(
            self._trajectory_data, info["traj_info"]
        )
        teacher_observation, _ = self._teacher_environment._get_obs(data, reference, info)
        reconstructed_history = self._initial_proprioception_history(data, motor_targets)
        if (
            self._failure_state_has_policy_context
            and self._config.use_base_velocity_estimate_observation
        ):
            # The v2 bank stores accelerometer channels in slots 6:9.  Keep
            # its joint/action history.  Replacement-mode experiments rewrite
            # slots 6:9; append-mode migrations preserve those slots and add
            # an independent velocity estimate after them.
            actor_frame = self._actor_proprioception_frame(data, motor_targets)
            if self._config.preserve_pelvis_accelerometer_observation:
                failure_proprioception_history = _append_velocity_to_proprioception_history(
                    failure_proprioception_history,
                    actor_frame[9:12],
                )
            else:
                failure_proprioception_history = failure_proprioception_history.at[:, 6:9].set(
                    actor_frame[6:9][jnp.newaxis, :]
                )
        initial_proprioception_history = jnp.where(
            failure_reset & self._failure_state_has_policy_context,
            failure_proprioception_history,
            reconstructed_history,
        )
        initial_handoff_frozen = terminal_reset | (
            failure_reset & failure_handoff_frozen.astype(jnp.bool_)
        )
        return (
            base_state.replace(data=data, obs=teacher_observation, info=info),
            rng,
            terminal_reset,
            failure_reset,
            initial_handoff_frozen,
            initial_proprioception_history,
            initial_phase_repeat,
            initial_last_residual,
            failure_index,
        )

    def reset(self, rng: jax.Array) -> State:
        base_rng, injection_rng = jax.random.split(rng)
        inner = self._teacher_environment.reset(base_rng, self._trajectory_data)
        (
            inner,
            rng,
            terminal_balance_reset,
            failure_state_reset,
            initial_handoff_frozen,
            proprioception_history,
            initial_phase_repeat,
            initial_last_residual,
            selected_failure_state_index,
        ) = self._inject_snapshot(inner, injection_rng)
        zero = jnp.zeros((), dtype=jnp.float32)
        velocity_diagnostics = _root_velocity_diagnostics(self._teacher_environment, inner.data)
        zero_temporal_metrics = {
            f"{name}_phase_{index}": zero
            for name in (
                "root_body_backward_speed",
                "root_body_lateral_speed",
                "pelvis_yaw_speed",
            )
            for index in range(_FAILURE_TEMPORAL_BIN_COUNT)
        }
        return State(
            pipeline_state=inner,
            obs=self._actor_critic_observation(
                inner.data, proprioception_history, failure_state_reset
            ),
            reward=zero,
            done=zero,
            metrics={
                "reward": zero,
                "success": zero,
                "stable": zero,
                "ready": zero,
                "linear_speed_safe": zero,
                "angular_speed_safe": zero,
                "maximum_stable_streak": zero,
                "handoff_frozen": zero,
                "pelvis_height": inner.data.qpos[2],
                "upright": _upright_projection(inner.data.qpos[3:7]),
                "root_linear_speed": jnp.linalg.norm(inner.data.qvel[:3]),
                "root_angular_speed": jnp.linalg.norm(inner.data.qvel[3:6]),
                **velocity_diagnostics,
                **zero_temporal_metrics,
                "residual_rms": zero,
                "adapter_residual_rms": zero,
                "adapter_motor_target_delta_rms_rad": zero,
                "residual_active": zero,
                "momentum_cost": zero,
                "directional_momentum_cost": zero,
                "stable_streak_fraction": zero,
                "handoff_regression": zero,
                "terminal_balance_reset": terminal_balance_reset.astype(jnp.float32),
                "failure_state_reset": failure_state_reset.astype(jnp.float32),
                "failure_state_target_active": zero,
                "failure_state_directional_cost": zero,
                "failure_state_horizon_complete": zero,
            },
            info={
                "rng": rng,
                "handoff_streak": jnp.zeros((), dtype=jnp.int32),
                "stable_streak": jnp.zeros((), dtype=jnp.int32),
                "maximum_stable_streak": jnp.zeros((), dtype=jnp.int32),
                "handoff_frozen": initial_handoff_frozen,
                "frozen_carry": inner.info["traj_info"],
                "phase_repeat": initial_phase_repeat,
                "initial_phase_repeat": initial_phase_repeat,
                "last_residual": initial_last_residual,
                "initial_last_residual": initial_last_residual,
                "selected_failure_state_index": selected_failure_state_index,
                "terminal_balance_reset": terminal_balance_reset,
                "failure_state_reset": failure_state_reset,
                "initial_handoff_frozen": initial_handoff_frozen,
                "proprioception_history": proprioception_history,
                "initial_proprioception_history": proprioception_history,
                "potential": _potential(inner.data),
                "initial_potential": _potential(inner.data),
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        inner = state.pipeline_state
        wrapper_step = state.info.get("steps", jnp.zeros((), dtype=jnp.int32))
        new_episode = wrapper_step == 0
        handoff_streak = jnp.where(new_episode, 0, state.info["handoff_streak"])
        stable_streak = jnp.where(new_episode, 0, state.info["stable_streak"])
        previous_maximum_stable_streak = jnp.where(
            new_episode, 0, state.info["maximum_stable_streak"]
        )
        was_frozen = jnp.where(
            new_episode,
            state.info["initial_handoff_frozen"],
            state.info["handoff_frozen"],
        )
        phase_repeat = jnp.where(
            new_episode, state.info["initial_phase_repeat"], state.info["phase_repeat"]
        )
        frozen_carry = _select_tree(
            new_episode, inner.info["traj_info"], state.info["frozen_carry"]
        )
        previous_residual = jnp.where(
            new_episode, state.info["initial_last_residual"], state.info["last_residual"]
        )
        proprioception_history = jnp.where(
            new_episode,
            state.info["initial_proprioception_history"],
            state.info["proprioception_history"],
        )
        previous_potential = jnp.where(new_episode, _potential(inner.data), state.info["potential"])

        rng, teacher_rng, parent_rng = jax.random.split(state.info["rng"], 3)
        teacher_action, _ = self._teacher_policy(inner.obs, teacher_rng)
        bounded_residual = jnp.clip(action, -1.0, 1.0)
        pre_upright = _upright_projection(inner.data.qpos[3:7])
        pre_posture_ready = (inner.data.qpos[2] >= self._config.ready_pelvis_height_m) & (
            pre_upright >= self._config.ready_upright_projection
        )
        residual_active = (
            jnp.asarray(True)
            if not self._config.posture_gated_residual
            else (was_frozen | pre_posture_ready)
        )
        applied_residual = jnp.where(
            residual_active, bounded_residual, jnp.zeros_like(bounded_residual)
        )
        if self._parent_residual_policy is not None:
            parent_residual, _ = self._parent_residual_policy(
                self._legacy_parent_actor_observation(state.obs), parent_rng
            )
            applied_parent_residual = jnp.where(
                residual_active,
                jnp.clip(parent_residual, -1.0, 1.0),
                jnp.zeros_like(parent_residual),
            )
        else:
            applied_parent_residual = jnp.zeros_like(applied_residual)
        # Reachability audits may amplify only the *new adapter increment*
        # around the immutable parent.  This is deliberately unavailable in
        # training and normal/full-route evaluation: it is an exact-bank,
        # SIM_ONLY counterfactual for distinguishing insufficient authority
        # from a topologically invalid residual policy class.
        applied_residual = jnp.clip(
            applied_parent_residual
            + self._diagnostic_adapter_gain * (applied_residual - applied_parent_residual),
            -1.0,
            1.0,
        )
        adapter_residual = applied_residual - applied_parent_residual
        residual_rad = applied_residual * self._residual_limits
        parent_residual_rad = applied_parent_residual * self._residual_limits
        combined_action = jnp.clip(teacher_action + residual_rad, -1.0, 1.0)
        parent_combined_action = jnp.clip(teacher_action + parent_residual_rad, -1.0, 1.0)
        previous_carry = inner.info["traj_info"]
        inner = self._teacher_environment.step(inner, combined_action, self._trajectory_data)

        # The CPU bridge may only succeed with a slower reference.  OpenTrack's
        # environment advances one reference frame on every step, so retain the
        # previous carry until the declared number of control ticks has elapsed.
        # The observation is rebuilt against that retained phase before the
        # next frozen-teacher action is requested.
        phase_repeat = phase_repeat + 1
        advance_phase = phase_repeat >= self._time_dilation
        dilated_carry = _select_tree(advance_phase, inner.info["traj_info"], previous_carry)
        phase_repeat = jnp.where(advance_phase, 0, phase_repeat)
        dilated_info = dict(inner.info)
        dilated_info["traj_info"] = dilated_carry
        dilated_reference = self._teacher_environment.th.get_current_traj_data_with_trajectory(
            self._trajectory_data, dilated_carry
        )
        dilated_observation, _ = self._teacher_environment._get_obs(
            inner.data, dilated_reference, dilated_info
        )
        inner = inner.replace(info=dilated_info, obs=dilated_observation)

        upright = _upright_projection(inner.data.qpos[3:7])
        linear_speed = jnp.linalg.norm(inner.data.qvel[:3])
        angular_speed = jnp.linalg.norm(inner.data.qvel[3:6])
        velocity_diagnostics = _root_velocity_diagnostics(self._teacher_environment, inner.data)
        temporal_failure_metrics = _temporal_failure_metrics(
            wrapper_step=wrapper_step,
            diagnostics=velocity_diagnostics,
            episode_length=self._config.episode_length,
        )
        ready = (inner.data.qpos[2] >= self._config.ready_pelvis_height_m) & (
            upright >= self._config.ready_upright_projection
        )
        linear_speed_safe = linear_speed <= self._config.handoff_maximum_linear_speed_mps
        angular_speed_safe = angular_speed <= self._config.handoff_maximum_angular_speed_rad_s
        momentum_safe = linear_speed_safe & angular_speed_safe
        handoff_safe = ready & momentum_safe
        handoff_streak = jnp.where(handoff_safe, handoff_streak + 1, 0)
        just_frozen = (~was_frozen) & (handoff_streak >= self._config.handoff_stable_steps)
        handoff_frozen = was_frozen | just_frozen
        phase_repeat = jnp.where(handoff_frozen, 0, phase_repeat)
        frozen_carry = _select_tree(just_frozen, inner.info["traj_info"], frozen_carry)
        selected_carry = _select_tree(handoff_frozen, frozen_carry, inner.info["traj_info"])
        inner_info = dict(inner.info)
        inner_info["traj_info"] = selected_carry
        frozen_reference = self._teacher_environment.th.get_current_traj_data_with_trajectory(
            self._trajectory_data, selected_carry
        )
        frozen_observation, _ = self._teacher_environment._get_obs(
            inner.data, frozen_reference, inner_info
        )
        inner = inner.replace(
            info=inner_info,
            obs=_select_tree(handoff_frozen, frozen_observation, inner.obs),
        )

        stable = ready & momentum_safe
        stable_streak = jnp.where(stable, stable_streak + 1, 0)
        maximum_stable_streak = jnp.maximum(previous_maximum_stable_streak, stable_streak)
        maximum_stable_streak_increment = maximum_stable_streak - previous_maximum_stable_streak
        success = stable_streak >= self._config.success_stable_steps
        finite = jnp.all(jnp.isfinite(inner.data.qpos)) & jnp.all(jnp.isfinite(inner.data.qvel))
        current_potential = _potential(inner.data)
        adapter_residual_rms = jnp.sqrt(jnp.mean(jnp.square(adapter_residual)))
        adapter_motor_target_delta_rms_rad = jnp.sqrt(
            jnp.mean(jnp.square(combined_action - parent_combined_action))
        )
        residual_rms = (
            adapter_residual_rms
            if self._config.regularize_velocity_adapter_only
            else jnp.sqrt(jnp.mean(jnp.square(applied_residual)))
        )
        residual_delta_rms = jnp.sqrt(jnp.mean(jnp.square(applied_residual - previous_residual)))
        teacher_deviation_rms = jnp.sqrt(jnp.mean(jnp.square(residual_rad)))
        height_readiness = jnp.clip((inner.data.qpos[2] - 0.08) / 0.62, 0.0, 1.0)
        upright_readiness = jnp.clip((upright + 1.0) * 0.5, 0.0, 1.0)
        dense_posture = 0.04 * (height_readiness + upright_readiness)
        balance_activation = height_readiness * upright_readiness
        momentum_cost = balance_activation * jnp.clip(
            jnp.square(linear_speed / self._config.handoff_maximum_linear_speed_mps)
            + 0.5 * jnp.square(angular_speed / self._config.handoff_maximum_angular_speed_rad_s),
            0.0,
            10.0,
        )
        directional_momentum_cost = balance_activation * jnp.clip(
            jnp.square(
                velocity_diagnostics["root_body_backward_speed"]
                / self._config.handoff_maximum_linear_speed_mps
            )
            + 0.5
            * jnp.square(
                velocity_diagnostics["root_body_lateral_speed"]
                / self._config.handoff_maximum_linear_speed_mps
            )
            + 0.25
            * jnp.square(
                velocity_diagnostics["pelvis_yaw_rate"]
                / self._config.handoff_maximum_angular_speed_rad_s
            ),
            0.0,
            10.0,
        )
        normalized_failure_velocities = (
            velocity_diagnostics["root_body_backward_speed"]
            / self._config.handoff_maximum_linear_speed_mps,
            velocity_diagnostics["root_body_lateral_speed"]
            / self._config.handoff_maximum_linear_speed_mps,
            velocity_diagnostics["pelvis_yaw_rate"]
            / self._config.handoff_maximum_angular_speed_rad_s,
        )
        if (
            self._config.failure_state_directional_cost_mode
            == "LEGACY_BALANCE_GATED_CLIPPED_SQUARE"
        ):
            failure_state_directional_cost = balance_activation * jnp.clip(
                self._config.failure_state_backward_cost_weight
                * jnp.square(normalized_failure_velocities[0])
                + self._config.failure_state_lateral_cost_weight
                * jnp.square(normalized_failure_velocities[1])
                + self._config.failure_state_yaw_cost_weight
                * jnp.square(normalized_failure_velocities[2]),
                0.0,
                10.0,
            )
        else:
            # The legacy cost could be reduced by lowering posture readiness,
            # and clipping removed its gradient on severe failures.  A
            # pseudo-Huber cost stays active independent of posture, remains
            # quadratic close to zero, and keeps a bounded non-zero slope for
            # hard failures.
            pseudo_huber_delta = 0.25

            def pseudo_huber(value: jax.Array) -> jax.Array:
                scaled = value / pseudo_huber_delta
                return jnp.square(pseudo_huber_delta) * (jnp.sqrt(1.0 + jnp.square(scaled)) - 1.0)

            failure_state_directional_cost = (
                self._config.failure_state_backward_cost_weight
                * pseudo_huber(normalized_failure_velocities[0])
                + self._config.failure_state_lateral_cost_weight
                * pseudo_huber(normalized_failure_velocities[1])
                + self._config.failure_state_yaw_cost_weight
                * pseudo_huber(normalized_failure_velocities[2])
            )
        failure_state_target_active = state.info["failure_state_reset"] & (
            wrapper_step < self._config.failure_state_target_horizon_steps
        )
        failure_state_target_weight = failure_state_target_active.astype(jnp.float32)
        failure_state_horizon_complete = (
            state.info["failure_state_reset"]
            & self._config.terminate_failure_state_episode_at_target_horizon
            & (wrapper_step + 1 >= self._config.failure_state_target_horizon_steps)
        )
        stable_streak_fraction = jnp.clip(
            stable_streak.astype(jnp.float32) / self._config.success_stable_steps,
            0.0,
            1.0,
        )
        handoff_regression = handoff_frozen & (~ready)
        reward = (
            8.0 * (current_potential - previous_potential)
            + dense_posture
            + 0.75 * stable.astype(jnp.float32)
            + self._config.stable_streak_reward_scale * jnp.square(stable_streak_fraction)
            + 100.0 * success.astype(jnp.float32)
            - self._config.residual_penalty_scale * residual_rms
            - self._config.action_delta_penalty_scale * residual_delta_rms
            - self._config.teacher_deviation_penalty_scale * teacher_deviation_rms
            - self._config.ready_momentum_penalty_scale * momentum_cost
            - self._config.directional_momentum_penalty_scale * directional_momentum_cost
            + self._config.failure_state_stable_streak_reward_scale
            * failure_state_target_weight
            * jnp.square(stable_streak_fraction)
            - self._config.failure_state_directional_penalty_scale
            * failure_state_target_weight
            * failure_state_directional_cost
            - self._config.handoff_regression_penalty_scale * handoff_regression.astype(jnp.float32)
            - 50.0 * (~finite).astype(jnp.float32)
        )
        reward = jnp.nan_to_num(reward, nan=-50.0, posinf=-50.0, neginf=-50.0)
        done = success | (~finite) | failure_state_horizon_complete
        actor_frame = self._actor_proprioception_frame(inner.data, inner.info["last_motor_targets"])
        proprioception_history = jnp.concatenate(
            (proprioception_history[1:], actor_frame[jnp.newaxis, :]), axis=0
        )
        actor_critic_observation = self._actor_critic_observation(
            inner.data,
            proprioception_history,
            state.info["failure_state_reset"],
        )
        info = dict(state.info)
        info.update(
            rng=rng,
            handoff_streak=handoff_streak,
            stable_streak=stable_streak,
            maximum_stable_streak=maximum_stable_streak,
            handoff_frozen=handoff_frozen,
            frozen_carry=frozen_carry,
            phase_repeat=phase_repeat,
            last_residual=applied_residual,
            proprioception_history=proprioception_history,
            potential=current_potential,
        )
        return state.replace(
            pipeline_state=inner,
            obs=actor_critic_observation,
            reward=reward,
            done=done.astype(jnp.float32),
            metrics={
                "reward": reward,
                "success": success.astype(jnp.float32),
                "stable": stable.astype(jnp.float32),
                "ready": ready.astype(jnp.float32),
                "linear_speed_safe": linear_speed_safe.astype(jnp.float32),
                "angular_speed_safe": angular_speed_safe.astype(jnp.float32),
                "maximum_stable_streak": maximum_stable_streak_increment.astype(jnp.float32),
                "handoff_frozen": handoff_frozen.astype(jnp.float32),
                "pelvis_height": inner.data.qpos[2],
                "upright": upright,
                "root_linear_speed": linear_speed,
                "root_angular_speed": angular_speed,
                **velocity_diagnostics,
                **temporal_failure_metrics,
                "residual_rms": residual_rms,
                "adapter_residual_rms": adapter_residual_rms,
                "adapter_motor_target_delta_rms_rad": adapter_motor_target_delta_rms_rad,
                "residual_active": residual_active.astype(jnp.float32),
                "momentum_cost": momentum_cost,
                "directional_momentum_cost": directional_momentum_cost,
                "stable_streak_fraction": stable_streak_fraction,
                "handoff_regression": handoff_regression.astype(jnp.float32),
                "terminal_balance_reset": state.info["terminal_balance_reset"].astype(jnp.float32),
                "failure_state_reset": state.info["failure_state_reset"].astype(jnp.float32),
                "failure_state_target_active": failure_state_target_weight,
                "failure_state_directional_cost": failure_state_directional_cost,
                "failure_state_horizon_complete": (
                    failure_state_horizon_complete.astype(jnp.float32)
                ),
            },
            info=info,
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "shape"):
        array = np.asarray(value)
        return float(array) if array.shape == () else array.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def train_opentrack_recovery_mjx_teacher_residual_ppo(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    motion_dataset_id: str,
    motion_id: str,
    entry_frame: int,
    snapshot_manifest_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    snapshot_indices: tuple[int, ...] | None = None,
    time_dilation: int = 1,
    terminal_balance_reference_frame: int | None = None,
    route_manifest_hash: str | None = None,
    route_group_hash: str | None = None,
    parent_training_report_hash: str | None = None,
    restore_checkpoint_path: Path | None = None,
    directional_curriculum_manifest_path: Path | None = None,
    failure_state_manifest_path: Path | None = None,
    config: RecoveryMJXTeacherResidualPPOConfig | None = None,
) -> dict[str, Any]:
    """Train a proprioceptive residual without mutating the frozen teacher."""

    active = config or RecoveryMJXTeacherResidualPPOConfig()
    root = opentrack_root.expanduser().resolve()
    teacher_checkpoint = teacher_checkpoint_path.expanduser().resolve()
    teacher_config = teacher_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    restore_checkpoint = (
        restore_checkpoint_path.expanduser().resolve()
        if restore_checkpoint_path is not None
        else None
    )
    directional_curriculum_path = (
        directional_curriculum_manifest_path.expanduser().resolve()
        if directional_curriculum_manifest_path is not None
        else None
    )
    directional_curriculum = (
        validate_recovery_mjx_directional_curriculum(directional_curriculum_path)
        if directional_curriculum_path is not None
        else None
    )
    failure_manifest_path = (
        failure_state_manifest_path.expanduser().resolve()
        if failure_state_manifest_path is not None
        else None
    )
    failure_manifest = (
        validate_recovery_mjx_failure_state_manifest(failure_manifest_path)
        if failure_manifest_path is not None
        else None
    )
    failure_state_bank: dict[str, np.ndarray[Any, Any]] | None = None
    if failure_manifest is not None and failure_manifest_path is not None:
        archive_path = failure_manifest_path.parent / str(failure_manifest["state_archive"])
        context_features = tuple(
            str(value) for value in failure_manifest.get("context_features_collected", ())
        )
        archive_names = (
            "qpos",
            "qvel",
            "handoff_frozen",
            "trajectory_step",
            "trajectory_initial_step",
        ) + tuple(
            name
            for name in (
                "last_motor_targets",
                "last_teacher_action",
                "last_residual",
                "proprioception_history",
                "phase_repeat",
            )
            if name in context_features
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            failure_state_bank = {
                name: np.array(archive[name], copy=True) for name in archive_names
            }
    if (
        not root.is_dir()
        or not teacher_checkpoint.is_dir()
        or not teacher_config.is_file()
        or not snapshot_path.is_file()
    ):
        raise FileNotFoundError("MJX teacher-residual PPO inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("MJX teacher-residual PPO output must be new and external")
    if restore_checkpoint is not None and (
        not restore_checkpoint.is_dir()
        or not (restore_checkpoint / "ppo_network_config.json").is_file()
        or restore_checkpoint == checkout
        or checkout in restore_checkpoint.parents
    ):
        raise ValueError("MJX teacher-residual restore checkpoint is invalid")
    if (
        not _DATASET_ID.fullmatch(motion_dataset_id)
        or not _MOTION_ID.fullmatch(motion_id)
        or not 0 <= entry_frame < 10_000_000
        or time_dilation not in (1, 2, 3, 4)
        or (
            active.terminal_balance_reset_fraction > 0.0
            and terminal_balance_reference_frame is None
        )
        or (route_manifest_hash is None) != (route_group_hash is None)
        or (restore_checkpoint is None and parent_training_report_hash is not None)
        or (active.preserve_pelvis_accelerometer_observation and restore_checkpoint is None)
        or (
            route_manifest_hash is not None
            and (restore_checkpoint is None) != (parent_training_report_hash is None)
        )
        or (
            route_manifest_hash is not None
            and (
                not re.fullmatch(r"sha256:[0-9a-f]{64}", route_manifest_hash)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(route_group_hash))
            )
        )
        or (
            directional_curriculum is not None
            and (
                active.terminal_balance_reset_fraction <= 0.0
                or route_manifest_hash is None
                or directional_curriculum.get("source_route_manifest_hash") != route_manifest_hash
                or directional_curriculum.get("source_route_group_hash") != route_group_hash
            )
        )
        or (failure_manifest is None) != (active.failure_state_reset_fraction == 0.0)
        or (
            failure_manifest is not None
            and (
                restore_checkpoint is None
                or route_manifest_hash is None
                or failure_manifest.get("source_route_manifest_hash") != route_manifest_hash
                or failure_manifest.get("source_route_group_hash") != route_group_hash
                or _tree_hash(restore_checkpoint)[0]
                != failure_manifest.get("source_actor_checkpoint_hash")
            )
        )
    ):
        raise ValueError("MJX teacher-residual route is invalid")
    motion_path = (
        root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1" / f"{motion_id}.npz"
    )
    if not motion_path.is_file():
        raise FileNotFoundError("MJX teacher-residual motion archive is missing")
    devices = tuple(jax.devices())
    if len(devices) < active.required_gpu_count:
        raise RuntimeError(
            "MJX teacher-residual PPO requires "
            f"{active.required_gpu_count} GPUs, found {len(devices)}"
        )

    os.environ.setdefault("GLI_PATH", str(root))
    tmj = importlib.import_module("track_mj")
    importlib.import_module("track_mj.envs.g1_tracking.train.g1_env_tracking_general")
    checkpoint = importlib.import_module("brax.training.agents.ppo.checkpoint")
    restore_params: Any | None = None
    actor_observation_migration: dict[str, Any] | None = None
    if active.preserve_pelvis_accelerometer_observation:
        assert restore_checkpoint is not None
        restore_params, actor_observation_migration = _migrate_checkpoint_with_appended_velocity(
            checkpoint.load(restore_checkpoint),
            history_frames=active.proprioception_history_frames,
        )
    payload = json.loads(teacher_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("env_config"), dict):
        raise ValueError("OpenTrack teacher config has no environment contract")
    environment_config = copy.deepcopy(
        tmj.registry.get("G1TrackingGeneral", "tracking_config").env_config
    )
    environment_config.update(payload["env_config"])
    environment_config.reference_traj_config.name = {motion_dataset_id: [motion_id]}
    environment_config.reference_traj_config.random_start = False
    environment_config.reference_traj_config.fixed_start_frame = entry_frame
    environment_config.noise_config.level = 0.0
    environment_config.push_config.enable = False
    environment_config.episode_length = max(3_000, active.episode_length + 100)
    # Disable OpenTrack's internal auto-reset.  The outer Brax wrapper owns the
    # episode boundary, and success is evaluated by the stricter recovery gate.
    environment_config.termination_config.root_height_threshold = 100.0
    environment_config.termination_config.rigid_body_dif_threshold = 100.0
    environment_config.termination_config.diff_gvec_threshold = 100.0
    environment_class = tmj.registry.get("G1TrackingGeneral", "tracking_train_env_class")
    previous_directory = Path.cwd()
    try:
        os.chdir(root)
        teacher_environment = environment_class(
            terrain_type="flat_terrain", config=environment_config
        )
        trajectory_data = teacher_environment.prepare_trajectory(
            environment_config.reference_traj_config.name
        )
    finally:
        os.chdir(previous_directory)
    teacher_policy = checkpoint.load_policy(teacher_checkpoint, deterministic=True)
    parent_residual_policy: Any | None = None
    if active.regularize_velocity_adapter_only:
        assert restore_checkpoint is not None
        parent_residual_policy = checkpoint.load_policy(
            restore_checkpoint,
            network_factory=_make_recovery_ppo_networks,
            deterministic=True,
        )
    all_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    if not all_snapshots:
        raise ValueError("MJX teacher-residual snapshot corpus is empty")
    selected_indices = (
        snapshot_indices if snapshot_indices is not None else tuple(range(len(all_snapshots)))
    )
    if (
        not selected_indices
        or len(set(selected_indices)) != len(selected_indices)
        or any(index < 0 or index >= len(all_snapshots) for index in selected_indices)
    ):
        raise ValueError("MJX teacher-residual snapshot selection is invalid")
    snapshots = tuple(all_snapshots[index] for index in selected_indices)
    environment = OpenTrackRecoveryMJXTeacherResidualEnv(
        teacher_environment=teacher_environment,
        trajectory_data=trajectory_data,
        teacher_policy=teacher_policy,
        snapshots=snapshots,
        time_dilation=time_dilation,
        terminal_balance_reference_frame=terminal_balance_reference_frame,
        directional_curriculum=directional_curriculum,
        failure_state_bank=failure_state_bank,
        config=active,
        parent_residual_policy=parent_residual_policy,
    )
    evaluation_environment = OpenTrackRecoveryMJXTeacherResidualEnv(
        teacher_environment=teacher_environment,
        trajectory_data=trajectory_data,
        teacher_policy=teacher_policy,
        snapshots=snapshots,
        time_dilation=time_dilation,
        terminal_balance_reference_frame=terminal_balance_reference_frame,
        directional_curriculum=None,
        failure_state_bank=None,
        config=replace(
            active,
            terminal_balance_reset_fraction=0.0,
            failure_state_reset_fraction=0.0,
            terminate_failure_state_episode_at_target_horizon=False,
            failure_state_directional_penalty_scale=0.0,
            failure_state_stable_streak_reward_scale=0.0,
            failure_state_conditioned_critic=False,
        ),
        parent_residual_policy=parent_residual_policy,
    )
    destination.mkdir(parents=True)
    progress: list[dict[str, Any]] = []

    def progress_fn(step: int, metrics: Any) -> None:
        row = {"step": int(step), "metrics": _jsonable(metrics)}
        progress.append(row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    checkpoint_dir = destination / "checkpoints"
    started = time.perf_counter()
    _make_policy, _params, final_metrics = ppo_train.train(
        environment=environment,
        num_timesteps=active.total_timesteps,
        max_devices_per_host=active.required_gpu_count,
        num_envs=active.num_envs,
        episode_length=active.episode_length,
        action_repeat=1,
        learning_rate=active.learning_rate,
        entropy_cost=active.entropy_cost,
        discounting=active.discounting,
        unroll_length=active.unroll_length,
        batch_size=active.batch_size,
        num_minibatches=active.num_minibatches,
        num_updates_per_batch=active.num_updates_per_batch,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=active.clipping_epsilon,
        gae_lambda=active.gae_lambda,
        max_grad_norm=active.maximum_gradient_norm,
        network_factory=_make_recovery_ppo_networks,
        seed=active.random_seed,
        num_evals=active.num_evals,
        num_eval_envs=active.num_eval_envs,
        deterministic_eval=True,
        eval_env=evaluation_environment,
        progress_fn=progress_fn,
        save_checkpoint_path=str(checkpoint_dir),
        restore_checkpoint_path=(
            str(restore_checkpoint)
            if restore_checkpoint is not None and restore_params is None
            else None
        ),
        restore_params=restore_params,
    )
    training_sec = time.perf_counter() - started
    candidate_hash, candidate_files = _tree_hash(checkpoint_dir)
    teacher_hash, teacher_files = _tree_hash(teacher_checkpoint)
    parent_hash: str | None = None
    parent_files: list[dict[str, Any]] = []
    parent_actor_retention: dict[str, Any] | None = None
    if restore_checkpoint is not None:
        parent_hash, parent_files = _tree_hash(restore_checkpoint)
    if actor_observation_migration is not None:
        assert restore_checkpoint is not None
        parent_actor_retention = _verify_frozen_parent_retention(
            source_params=checkpoint.load(restore_checkpoint),
            checkpoint_dir=checkpoint_dir,
            checkpoint_loader=checkpoint.load,
        )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_report.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "parallelization": "BRAX_PPO_JAX_PMAP_VMAP",
        "devices": [str(device) for device in devices[: active.required_gpu_count]],
        "compiled_model_contract": compiled_mujoco_model_contract(environment.mj_model),
        "teacher_checkpoint_hash": teacher_hash,
        "teacher_checkpoint_files": teacher_files,
        "teacher_frozen": True,
        "teacher_role": "PRIVILEGED_REFERENCE_CONDITIONED_SIMULATION_TEACHER",
        "motion_archive_hash": hash_bytes(motion_path.read_bytes()),
        "motion_dataset_id": motion_dataset_id,
        "motion_id": motion_id,
        "entry_frame": entry_frame,
        "time_dilation": time_dilation,
        "route_binding_enforced": route_manifest_hash is not None,
        "route_manifest_hash": route_manifest_hash,
        "route_group_hash": route_group_hash,
        "snapshot_manifest_hash": hash_bytes(snapshot_path.read_bytes()),
        "snapshot_count": len(snapshots),
        "snapshot_indices": list(selected_indices),
        "snapshot_hashes": [item.snapshot_hash for item in snapshots],
        "cross_scene_transfer_required": True,
        "actor_observation_dim": (
            active.actor_proprioception_frame_dim * active.proprioception_history_frames
        ),
        "actor_proprioception_frame_dim": active.actor_proprioception_frame_dim,
        "actor_proprioception_history_frames": active.proprioception_history_frames,
        "actor_observation": (
            (
                (
                    "DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY"
                    if active.proprioception_history_frames > 1
                    else ("DEPLOYABLE_PELVIS_IMU_AND_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_ONLY")
                )
                if active.preserve_pelvis_accelerometer_observation
                else (
                    "DEPLOYABLE_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_HISTORY_ONLY"
                    if active.proprioception_history_frames > 1
                    else "DEPLOYABLE_ESTIMATED_BASE_VELOCITY_PROPRIOCEPTION_ONLY"
                )
            )
            if active.use_base_velocity_estimate_observation
            else (
                (
                    "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_HISTORY_ONLY"
                    if active.proprioception_history_frames > 1
                    else "DEPLOYABLE_PELVIS_IMU_PROPRIOCEPTION_ONLY"
                )
                if active.use_pelvis_imu_observation
                else (
                    "DEPLOYABLE_PROPRIOCEPTION_HISTORY_ONLY"
                    if active.proprioception_history_frames > 1
                    else "DEPLOYABLE_PROPRIOCEPTION_ONLY"
                )
            )
        ),
        "actor_pelvis_imu_contract": (
            {
                **(
                    {
                        **(
                            {
                                "accelerometer_clip_mps2": (active.pelvis_accelerometer_clip_mps2),
                                "accelerometer_sensor": "accelerometer_pelvis",
                                "accelerometer_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
                            }
                            if active.preserve_pelvis_accelerometer_observation
                            else {}
                        ),
                        "linear_motion_feature": "BASE_VELOCITY_ESTIMATE",
                        "linear_motion_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
                    }
                    if active.use_base_velocity_estimate_observation
                    else {
                        "accelerometer_clip_mps2": active.pelvis_accelerometer_clip_mps2,
                        "accelerometer_sensor": "accelerometer_pelvis",
                        "accelerometer_scale": "CLIP_THEN_DIVIDE_BY_CLIP",
                    }
                ),
                "gyroscope_scale": 0.05,
                "gyroscope_sensor": "gyro_pelvis",
            }
            if active.use_pelvis_imu_observation
            else None
        ),
        "actor_base_velocity_estimator_contract": (
            {
                "clip_mps": active.base_velocity_estimate_clip_mps,
                "deployment_source_required": "ONBOARD_STATE_ESTIMATOR",
                "ground_truth_hardware_velocity_authorized": False,
                "simulation_proxy": "MUJOCO_ROOT_QVEL_ROTATED_TO_PELVIS",
            }
            if active.use_base_velocity_estimate_observation
            else None
        ),
        "actor_observation_migration": actor_observation_migration,
        "parent_actor_retention": parent_actor_retention,
        "residual_regularization_target": (
            "VELOCITY_ADAPTER_INCREMENT_ONLY"
            if active.regularize_velocity_adapter_only
            else "TOTAL_TEACHER_RESIDUAL"
        ),
        "critic_observation": (
            "SIMULATION_PRIVILEGED_VALUE_FUNCTION_ONLY"
            if active.use_asymmetric_critic
            else "SAME_AS_ACTOR"
        ),
        "critic_privileged_auxiliary_dim": (
            _PRIVILEGED_CRITIC_AUXILIARY_DIM if active.use_asymmetric_critic else 0
        ),
        "critic_privileged_features": (
            (
                [
                    "root_body_linear_velocity",
                    "pelvis_angular_velocity",
                    "pelvis_height",
                    "failure_state_reset_source",
                ]
                if active.failure_state_conditioned_critic
                else [
                    "root_linear_velocity",
                    "root_angular_velocity",
                    "pelvis_height",
                    "upright",
                ]
            )
            if active.use_asymmetric_critic
            else []
        ),
        "critic_exported_with_actor": False,
        "failure_state_targeted_reward": {
            "actor_observation_features": [],
            "critic_failure_source_indicator": active.failure_state_conditioned_critic,
            "directional_penalty_scale": active.failure_state_directional_penalty_scale,
            "directional_cost_weights": {
                "backward": active.failure_state_backward_cost_weight,
                "lateral": active.failure_state_lateral_cost_weight,
                "yaw": active.failure_state_yaw_cost_weight,
            },
            **(
                {"directional_cost_mode": active.failure_state_directional_cost_mode}
                if active.schema_version
                == "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16"
                else {}
            ),
            "evaluation_active": False,
            "stable_streak_reward_scale": active.failure_state_stable_streak_reward_scale,
            "target_horizon_steps": active.failure_state_target_horizon_steps,
            "training_scope": "FAILURE_STATE_RESET_EPISODES_ONLY",
            "failure_episode_boundary": (
                "TARGET_HORIZON"
                if active.terminate_failure_state_episode_at_target_horizon
                else "OUTER_EPISODE_OR_SUCCESS"
            ),
        },
        "signed_velocity_diagnostics": {
            "actor_observation_features": [],
            "angular_frame": "PELVIS_IMU_SENSOR_FRAME",
            "linear_frame": "PELVIS_BODY_FRAME",
            "metrics": [
                "root_body_forward_velocity",
                "root_body_lateral_velocity",
                "root_body_vertical_velocity",
                "root_body_backward_speed",
                "root_body_lateral_speed",
                "pelvis_roll_rate",
                "pelvis_pitch_rate",
                "pelvis_yaw_rate",
                "pelvis_yaw_speed",
            ],
            "temporal_bin_count": _FAILURE_TEMPORAL_BIN_COUNT,
            "temporal_bin_semantics": "EQUAL_WIDTH_BY_WRAPPER_CONTROL_STEP",
            "temporal_metrics": [
                "root_body_backward_speed",
                "root_body_lateral_speed",
                "pelvis_yaw_speed",
            ],
            "use": "EVALUATION_AND_CURRICULUM_DIAGNOSTICS_ONLY",
        },
        "actor_forbidden_features": [
            "external_reference_phase",
            "teacher_identity",
            "future_reference_state",
            "ground_truth_base_velocity_without_state_estimator",
            "privileged_state",
        ],
        "actor_initialization": (
            "FROZEN_PARENT_TRUNK_ZERO_OUTPUT_VELOCITY_ADAPTER"
            if actor_observation_migration is not None
            else (
                "PARENT_CHECKPOINT_CONTINUATION"
                if restore_checkpoint is not None
                else "EXACT_ZERO_DETERMINISTIC_RESIDUAL"
            )
        ),
        "action_semantics": (
            "POSTURE_GATED_BOUNDED_RESIDUAL_AROUND_FROZEN_CLOSED_LOOP_TEACHER"
            if active.posture_gated_residual
            else "BOUNDED_RESIDUAL_AROUND_FROZEN_CLOSED_LOOP_TEACHER"
        ),
        "residual_activation_gate": (
            "POSTURE_READY_OR_HANDOFF_FROZEN" if active.posture_gated_residual else "ALWAYS_ACTIVE"
        ),
        "momentum_gated_handoff": True,
        "terminal_balance_curriculum": {
            "evaluation_reset_fraction": 0.0,
            "reference_frame": terminal_balance_reference_frame,
            "root_angular_velocity_noise_rad_s": (
                active.terminal_balance_root_angular_velocity_noise_rad_s
            ),
            "root_linear_velocity_noise_mps": (
                active.terminal_balance_root_linear_velocity_noise_mps
            ),
            "training_reset_fraction": active.terminal_balance_reset_fraction,
            "directional_curriculum_manifest_hash": (
                hash_bytes(directional_curriculum_path.read_bytes())
                if directional_curriculum_path is not None
                else None
            ),
            "directional_curriculum_report_hash": (
                directional_curriculum["report_hash"]
                if directional_curriculum is not None
                else None
            ),
            "terminal_body_linear_velocity_bias_mps": (
                directional_curriculum["terminal_body_linear_velocity_bias_mps"]
                if directional_curriculum is not None
                else [0.0, 0.0, 0.0]
            ),
            "terminal_pelvis_yaw_rate_bias_rad_s": (
                directional_curriculum["terminal_pelvis_yaw_rate_bias_rad_s"]
                if directional_curriculum is not None
                else 0.0
            ),
        },
        "failure_state_curriculum": {
            "evaluation_reset_fraction": 0.0,
            "training_reset_fraction": active.failure_state_reset_fraction,
            "training_episode_boundary": (
                "TARGET_HORIZON"
                if active.terminate_failure_state_episode_at_target_horizon
                else "OUTER_EPISODE_OR_SUCCESS"
            ),
            "expected_targeted_training_transition_fraction": (
                active.expected_failure_target_transition_fraction
            ),
            **(
                {
                    "reset_source_resampling": "FIXED_PER_PARALLEL_ENVIRONMENT_AUTORESET",
                    "minimum_completed_target_horizon_cycles": (
                        active.total_timesteps
                        // (active.num_envs * active.failure_state_target_horizon_steps)
                    ),
                }
                if active.schema_version
                in {
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v15",
                    "rosclaw_soccer.recovery_mjx_teacher_residual_ppo_config.v16",
                }
                else {}
            ),
            "failure_state_manifest_hash": (
                failure_manifest["report_hash"] if failure_manifest is not None else None
            ),
            "failure_state_manifest_file_hash": (
                hash_bytes(failure_manifest_path.read_bytes())
                if failure_manifest_path is not None
                else None
            ),
            "failure_state_archive_hash": (
                failure_manifest["state_archive_hash"] if failure_manifest is not None else None
            ),
            "failure_state_count": (
                failure_manifest["collected_state_count"] if failure_manifest is not None else 0
            ),
            "context_features_restored": (
                failure_manifest.get("context_features_collected")
                if failure_manifest is not None
                and isinstance(failure_manifest.get("context_features_collected"), list)
                else [
                    "qpos",
                    "qvel",
                    "trajectory_step",
                    "trajectory_initial_step",
                    "handoff_frozen",
                ]
            ),
            "observation_context_adapter": (
                ("BASE_VELOCITY_ESTIMATE_APPENDED_AFTER_PRESERVED_ACCELEROMETER_CHANNELS_6_TO_8")
                if active.preserve_pelvis_accelerometer_observation
                else "BASE_VELOCITY_ESTIMATE_REPLACES_ACCELEROMETER_CHANNELS_6_TO_8"
                if active.use_base_velocity_estimate_observation
                else None
            ),
        },
        "training_sec": training_sec,
        "progress": progress,
        "final_metrics": _jsonable(final_metrics),
        "candidate_checkpoint_hash": candidate_hash,
        "candidate_checkpoint_files": candidate_files,
        "parent_checkpoint_hash": parent_hash,
        "parent_checkpoint_files": parent_files,
        "parent_training_report_hash": parent_training_report_hash,
        "continued_from_parent": restore_checkpoint is not None,
        "deployment_candidate": False,
        "requires_reference_free_distillation": True,
        "requires_independent_cpu_mujoco_exam": True,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "training-report.json", report)
    return report


def _verified_route_restore_checkpoint(
    *,
    restore_checkpoint: Path,
    route_manifest_hash: str,
    route_group_hash: str,
    expected_proprioception_history_frames: int,
    expected_use_pelvis_imu_observation: bool,
    expected_use_base_velocity_estimate_observation: bool,
    expected_preserve_pelvis_accelerometer_observation: bool,
    expected_use_asymmetric_critic: bool,
) -> str:
    """Bind a continued generation to its exact parent report and files."""

    checkpoint = restore_checkpoint.expanduser().resolve()
    if checkpoint.parent.name != "checkpoints" or not checkpoint.name.isdigit():
        raise ValueError("recovery MJX route restore layout is invalid")
    parent_report_path = checkpoint.parent.parent / "training-report.json"
    if not parent_report_path.is_file():
        raise FileNotFoundError("recovery MJX route parent training report is absent")
    parent = validate_recovery_mjx_teacher_residual_report(parent_report_path)
    if (
        parent.get("route_binding_enforced") is not True
        or parent.get("route_manifest_hash") != route_manifest_hash
        or parent.get("route_group_hash") != route_group_hash
    ):
        raise ValueError("recovery MJX route parent lineage differs")
    parent_config = parent.get("config")
    if not isinstance(parent_config, dict):
        raise ValueError("recovery MJX route parent configuration is absent")
    parent_history_frames = parent_config.get("proprioception_history_frames", 1)
    if parent_history_frames != expected_proprioception_history_frames:
        raise ValueError("recovery MJX route parent observation contract differs")
    if (
        parent_config.get("use_pelvis_imu_observation", False)
        is not expected_use_pelvis_imu_observation
    ):
        raise ValueError("recovery MJX route parent pelvis IMU contract differs")
    parent_uses_velocity = parent_config.get("use_base_velocity_estimate_observation", False)
    parent_preserves_accelerometer = parent_config.get(
        "preserve_pelvis_accelerometer_observation", False
    )
    if expected_preserve_pelvis_accelerometer_observation:
        if parent_uses_velocity is not False or parent_preserves_accelerometer is not False:
            raise ValueError("recovery MJX route migration parent is not legacy pelvis IMU")
    elif (
        parent_uses_velocity is not expected_use_base_velocity_estimate_observation
        or parent_preserves_accelerometer is not False
    ):
        raise ValueError("recovery MJX route parent base-velocity contract differs")
    if parent_config.get("use_asymmetric_critic", False) is not expected_use_asymmetric_critic:
        raise ValueError("recovery MJX route parent critic contract differs")
    prefix = f"{checkpoint.name}/"
    raw_rows = parent.get("candidate_checkpoint_files")
    if not isinstance(raw_rows, list):
        raise ValueError("recovery MJX route parent file evidence is absent")
    expected = {
        str(row["path"])[len(prefix) :]: (row.get("hash"), row.get("size_bytes"))
        for row in raw_rows
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and str(row["path"]).startswith(prefix)
    }
    actual = {
        path.relative_to(checkpoint).as_posix(): (
            hash_bytes(path.read_bytes()),
            path.stat().st_size,
        )
        for path in sorted(item for item in checkpoint.rglob("*") if item.is_file())
    }
    if not expected or actual != expected:
        raise ValueError("recovery MJX route parent checkpoint integrity failed")
    return str(parent["report_hash"])


def train_opentrack_recovery_mjx_route_group_ppo(
    *,
    opentrack_root: Path,
    teacher_checkpoint_path: Path,
    teacher_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    route_manifest_path: Path,
    route_group_index: int,
    output_dir: Path,
    source_checkout_path: Path,
    restore_checkpoint_path: Path | None = None,
    directional_curriculum_manifest_path: Path | None = None,
    failure_state_manifest_path: Path | None = None,
    config: RecoveryMJXTeacherResidualPPOConfig | None = None,
) -> dict[str, Any]:
    """Train exactly one content-bound route group from the CPU manifest."""

    root = opentrack_root.expanduser().resolve()
    checkpoint = teacher_checkpoint_path.expanduser().resolve()
    if not root.is_dir() or not checkpoint.is_dir():
        raise FileNotFoundError("recovery MJX route training roots are incomplete")
    if not _DATASET_ID.fullmatch(motion_dataset_id):
        raise ValueError("recovery MJX route motion dataset id is invalid")
    motion_root = root / "storage" / "data" / "mocap" / motion_dataset_id / "UnitreeG1"
    # The motion id is read from the verified manifest, so first validate all
    # bindings except the motion file, then resolve that final path without
    # accepting user-controlled route coordinates.
    validate_route_path = route_manifest_path.expanduser().resolve()
    manifest = validate_recovery_mjx_route_manifest(validate_route_path)
    groups = manifest["route_groups"]
    if not isinstance(route_group_index, int) or not 0 <= route_group_index < len(groups):
        raise ValueError("recovery MJX route group index is invalid")
    group = groups[route_group_index]
    motion_path = (motion_root / f"{group['motion_id']}.npz").resolve()
    if motion_path.parent != motion_root.resolve():
        raise ValueError("recovery MJX route motion path escapes its dataset")
    job = resolve_recovery_mjx_route_group(
        route_manifest_path=validate_route_path,
        route_group_index=route_group_index,
        snapshot_manifest_path=snapshot_manifest_path,
        teacher_policy_path=checkpoint / "policy.onnx",
        teacher_config_path=teacher_config_path,
        motion_archive_path=motion_path,
    )
    active = config or RecoveryMJXTeacherResidualPPOConfig()
    if active.episode_length < int(job["minimum_episode_length"]):
        raise ValueError("recovery MJX route episode is shorter than its CPU-demonstrated duration")
    parent_report_hash = None
    if restore_checkpoint_path is not None:
        parent_report_hash = _verified_route_restore_checkpoint(
            restore_checkpoint=restore_checkpoint_path,
            route_manifest_hash=str(job["route_manifest_hash"]),
            route_group_hash=str(job["route_group_hash"]),
            expected_proprioception_history_frames=(active.proprioception_history_frames),
            expected_use_pelvis_imu_observation=active.use_pelvis_imu_observation,
            expected_use_base_velocity_estimate_observation=(
                active.use_base_velocity_estimate_observation
            ),
            expected_preserve_pelvis_accelerometer_observation=(
                active.preserve_pelvis_accelerometer_observation
            ),
            expected_use_asymmetric_critic=active.use_asymmetric_critic,
        )
    return train_opentrack_recovery_mjx_teacher_residual_ppo(
        opentrack_root=root,
        teacher_checkpoint_path=checkpoint,
        teacher_config_path=teacher_config_path,
        motion_dataset_id=motion_dataset_id,
        motion_id=str(job["motion_id"]),
        entry_frame=int(job["entry_frame"]),
        snapshot_manifest_path=snapshot_manifest_path,
        output_dir=output_dir,
        source_checkout_path=source_checkout_path,
        snapshot_indices=tuple(int(value) for value in job["snapshot_indices"]),
        time_dilation=int(job["time_dilation"]),
        terminal_balance_reference_frame=int(job["successor_end_frame"]),
        route_manifest_hash=str(job["route_manifest_hash"]),
        route_group_hash=str(job["route_group_hash"]),
        parent_training_report_hash=parent_report_hash,
        restore_checkpoint_path=restore_checkpoint_path,
        directional_curriculum_manifest_path=directional_curriculum_manifest_path,
        failure_state_manifest_path=failure_state_manifest_path,
        config=active,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a G1 recovery residual around an immutable MJX teacher"
    )
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-config", required=True, type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--motion-id")
    parser.add_argument("--entry-frame", type=int)
    parser.add_argument("--time-dilation", default=1, choices=(1, 2, 3, 4), type=int)
    parser.add_argument("--route-manifest", type=Path)
    parser.add_argument("--route-group-index", type=int)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--snapshot-index", action="append", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--restore-checkpoint", type=Path)
    parser.add_argument("--directional-curriculum-manifest", type=Path)
    parser.add_argument("--failure-state-manifest", type=Path)
    parser.add_argument("--total-timesteps", default=4_194_304, type=int)
    parser.add_argument("--num-envs", default=256, type=int)
    parser.add_argument("--episode-length", default=1_200, type=int)
    parser.add_argument("--num-evals", default=4, type=int)
    parser.add_argument("--num-eval-envs", default=64, type=int)
    parser.add_argument("--learning-rate", default=1.0e-4, type=float)
    parser.add_argument("--entropy-cost", default=5.0e-5, type=float)
    parser.add_argument("--residual-penalty-scale", default=0.25, type=float)
    parser.add_argument("--ready-momentum-penalty-scale", default=0.15, type=float)
    parser.add_argument("--directional-momentum-penalty-scale", default=0.0, type=float)
    parser.add_argument("--failure-state-directional-penalty-scale", default=0.0, type=float)
    parser.add_argument("--failure-state-backward-cost-weight", default=3.0, type=float)
    parser.add_argument("--failure-state-lateral-cost-weight", default=0.25, type=float)
    parser.add_argument("--failure-state-yaw-cost-weight", default=0.10, type=float)
    parser.add_argument(
        "--failure-state-directional-cost-mode",
        choices=("LEGACY_BALANCE_GATED_CLIPPED_SQUARE", "ALWAYS_ON_PSEUDO_HUBER"),
        default="ALWAYS_ON_PSEUDO_HUBER",
    )
    parser.add_argument("--failure-state-stable-streak-reward-scale", default=0.0, type=float)
    parser.add_argument("--failure-state-target-horizon-steps", default=400, type=int)
    parser.add_argument("--stable-streak-reward-scale", default=1.0, type=float)
    parser.add_argument("--proprioception-history-frames", default=4, type=int)
    parser.add_argument("--disable-pelvis-imu", action="store_true")
    parser.add_argument("--use-base-velocity-estimate", action="store_true")
    parser.add_argument("--preserve-pelvis-accelerometer", action="store_true")
    parser.add_argument("--regularize-velocity-adapter-only", action="store_true")
    parser.add_argument("--base-velocity-estimate-clip-mps", default=2.0, type=float)
    parser.add_argument("--terminal-balance-reset-fraction", default=0.0, type=float)
    parser.add_argument("--failure-state-reset-fraction", default=0.0, type=float)
    parser.add_argument("--terminate-failure-state-at-target-horizon", action="store_true")
    parser.add_argument("--symmetric-critic", action="store_true")
    parser.add_argument("--failure-state-conditioned-critic", action="store_true")
    parser.add_argument("--ungated-residual", action="store_true")
    parser.add_argument("--seed", default=5411, type=int)
    args = parser.parse_args()
    active_config = RecoveryMJXTeacherResidualPPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        residual_penalty_scale=args.residual_penalty_scale,
        ready_momentum_penalty_scale=args.ready_momentum_penalty_scale,
        directional_momentum_penalty_scale=args.directional_momentum_penalty_scale,
        failure_state_directional_penalty_scale=(args.failure_state_directional_penalty_scale),
        failure_state_backward_cost_weight=args.failure_state_backward_cost_weight,
        failure_state_lateral_cost_weight=args.failure_state_lateral_cost_weight,
        failure_state_yaw_cost_weight=args.failure_state_yaw_cost_weight,
        failure_state_directional_cost_mode=args.failure_state_directional_cost_mode,
        failure_state_stable_streak_reward_scale=(args.failure_state_stable_streak_reward_scale),
        failure_state_target_horizon_steps=args.failure_state_target_horizon_steps,
        stable_streak_reward_scale=args.stable_streak_reward_scale,
        proprioception_history_frames=args.proprioception_history_frames,
        use_pelvis_imu_observation=not args.disable_pelvis_imu,
        use_base_velocity_estimate_observation=args.use_base_velocity_estimate,
        preserve_pelvis_accelerometer_observation=args.preserve_pelvis_accelerometer,
        regularize_velocity_adapter_only=args.regularize_velocity_adapter_only,
        base_velocity_estimate_clip_mps=args.base_velocity_estimate_clip_mps,
        terminal_balance_reset_fraction=args.terminal_balance_reset_fraction,
        failure_state_reset_fraction=args.failure_state_reset_fraction,
        terminate_failure_state_episode_at_target_horizon=(
            args.terminate_failure_state_at_target_horizon
        ),
        use_asymmetric_critic=not args.symmetric_critic,
        failure_state_conditioned_critic=args.failure_state_conditioned_critic,
        posture_gated_residual=not args.ungated_residual,
        random_seed=args.seed,
    )
    if args.route_manifest is not None or args.route_group_index is not None:
        if args.route_manifest is None or args.route_group_index is None:
            parser.error("--route-manifest and --route-group-index must be supplied together")
        if args.motion_id is not None or args.entry_frame is not None or args.snapshot_index:
            parser.error("route-manifest mode does not accept manual route coordinates")
        result = train_opentrack_recovery_mjx_route_group_ppo(
            opentrack_root=args.opentrack_root,
            teacher_checkpoint_path=args.teacher_checkpoint,
            teacher_config_path=args.teacher_config,
            motion_dataset_id=args.motion_dataset_id,
            snapshot_manifest_path=args.snapshot_manifest,
            route_manifest_path=args.route_manifest,
            route_group_index=args.route_group_index,
            output_dir=args.output_dir,
            source_checkout_path=args.source_checkout,
            restore_checkpoint_path=args.restore_checkpoint,
            directional_curriculum_manifest_path=args.directional_curriculum_manifest,
            failure_state_manifest_path=args.failure_state_manifest,
            config=active_config,
        )
    else:
        if args.motion_id is None or args.entry_frame is None:
            parser.error("manual mode requires --motion-id and --entry-frame")
        result = train_opentrack_recovery_mjx_teacher_residual_ppo(
            opentrack_root=args.opentrack_root,
            teacher_checkpoint_path=args.teacher_checkpoint,
            teacher_config_path=args.teacher_config,
            motion_dataset_id=args.motion_dataset_id,
            motion_id=args.motion_id,
            entry_frame=args.entry_frame,
            time_dilation=args.time_dilation,
            snapshot_manifest_path=args.snapshot_manifest,
            output_dir=args.output_dir,
            source_checkout_path=args.source_checkout,
            snapshot_indices=(
                tuple(args.snapshot_index) if args.snapshot_index is not None else None
            ),
            restore_checkpoint_path=args.restore_checkpoint,
            directional_curriculum_manifest_path=args.directional_curriculum_manifest,
            failure_state_manifest_path=args.failure_state_manifest,
            config=active_config,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "OpenTrackRecoveryMJXTeacherResidualEnv",
    "train_opentrack_recovery_mjx_route_group_ppo",
    "train_opentrack_recovery_mjx_teacher_residual_ppo",
]
