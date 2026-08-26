"""Four-GPU MJX residual PPO around frozen G1 recovery skill memory.

This executable module intentionally depends on the external OpenTrack/JAX
training environment. It is not imported by the default Soccer runtime. The
actor sees only deployable proprioception and its current internal memory
target error; reference phase, teacher identity, and future states stay hidden.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from brax.training import distribution as brax_distribution
from brax.training import networks as brax_networks
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo_train
from flax import linen
from mujoco import mjx

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_mjx import (
    _KDS,
    _KPS,
    _TORQUE_LIMIT,
    RecoveryMJXPPOConfig,
    _atomic_json,
    compiled_mujoco_model_contract,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)
from rosclaw_soccer.training.recovery_student import (
    RecoveryTeacherEpisode,
    load_recovery_distillation_corpus,
    recovery_teacher_episodes_from_corpus,
)

_ACTOR_OBSERVATION_DIM = 122
_JOINT_COUNT = 29


class _ZeroResidualPolicy(linen.Module):
    """Policy head whose deterministic initial action is exactly zero."""

    action_size: int

    @linen.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        hidden = observation
        for width in (256, 256, 128):
            hidden = linen.swish(linen.Dense(width)(hidden))
        location = linen.Dense(
            self.action_size,
            name="location",
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )(hidden)

        def scale_bias(
            key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32
        ) -> jax.Array:
            del key
            return jnp.full(shape, -1.5, dtype=dtype)

        scale = linen.Dense(
            self.action_size,
            name="scale",
            kernel_init=jax.nn.initializers.zeros,
            bias_init=scale_bias,
        )(hidden)
        return jnp.concatenate((location, scale), axis=-1)


class _FrozenParentVelocityAdapterPolicy(linen.Module):
    """Keep the complete legacy actor immutable and learn a bounded adapter.

    Brax updates its observation running statistics during PPO.  Freezing only
    the dense trunk is therefore insufficient: changed normalization can still
    change the parent action.  The legacy mean/std live in this module and are
    stop-gradient constants, while the velocity adapter consumes the current
    normalized 99-D/frame observation.
    """

    action_size: int
    history_frames: int

    @linen.compact
    def __call__(
        self,
        raw_observation: jax.Array,
        normalized_observation: jax.Array,
    ) -> jax.Array:
        expected_size = self.history_frames * 99
        if (
            raw_observation.shape[-1] != expected_size
            or normalized_observation.shape[-1] != expected_size
        ):
            raise ValueError("velocity-adapter observation width is invalid")
        frames = raw_observation.reshape(raw_observation.shape[:-1] + (self.history_frames, 99))
        legacy_frames = jnp.concatenate((frames[..., :9], frames[..., 12:]), axis=-1)
        legacy_raw_observation = legacy_frames.reshape(
            raw_observation.shape[:-1] + (self.history_frames * 96,)
        )
        legacy_size = self.history_frames * 96
        parent_mean = self.param(
            "parent_normalizer_mean", jax.nn.initializers.zeros, (legacy_size,)
        )
        parent_std = self.param("parent_normalizer_std", jax.nn.initializers.ones, (legacy_size,))
        legacy_observation = (
            legacy_raw_observation - jax.lax.stop_gradient(parent_mean)
        ) / jax.lax.stop_gradient(parent_std)

        hidden = legacy_observation
        for index, width in enumerate((256, 256, 128)):
            hidden = linen.swish(linen.Dense(width, name=f"Dense_{index}")(hidden))
        parent_location = linen.Dense(self.action_size, name="location")(hidden)
        parent_scale = linen.Dense(self.action_size, name="scale")(hidden)

        adapter = linen.swish(linen.Dense(128, name="velocity_adapter_0")(normalized_observation))
        adapter = linen.swish(linen.Dense(64, name="velocity_adapter_1")(adapter))
        adapter_delta = linen.Dense(
            self.action_size,
            name="velocity_adapter_location",
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )(adapter)
        location = jax.lax.stop_gradient(parent_location) + 0.25 * jnp.tanh(adapter_delta)
        scale = jax.lax.stop_gradient(parent_scale)
        return jnp.concatenate((location, scale), axis=-1)


def _make_recovery_ppo_networks(
    observation_size: Any,
    action_size: int,
    preprocess_observations_fn: Any,
) -> ppo_networks.PPONetworks:
    def flat_size(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 1 and isinstance(value[0], int):
            return value[0]
        return None

    observation_is_mapping = isinstance(observation_size, Mapping)
    if observation_is_mapping:
        actor_observation_size = flat_size(observation_size.get("state"))
        critic_observation_size = flat_size(observation_size.get("privileged_state"))
        if actor_observation_size is None or critic_observation_size is None:
            raise ValueError("recovery MJX actor-critic observation contract is invalid")
        flat_observation_size = actor_observation_size
    else:
        scalar_size = flat_size(observation_size)
        if scalar_size is None:
            raise ValueError("recovery MJX actor requires one flat proprioceptive observation")
        flat_observation_size = scalar_size
    appended_velocity_history_frames = (
        flat_observation_size // 99 if flat_observation_size % 99 == 0 else None
    )
    policy_module = (
        _FrozenParentVelocityAdapterPolicy(
            action_size=action_size,
            history_frames=appended_velocity_history_frames,
        )
        if appended_velocity_history_frames is not None
        else _ZeroResidualPolicy(action_size=action_size)
    )

    def apply(processor_params: Any, policy_params: Any, observation: Any) -> Any:
        processed = preprocess_observations_fn(observation, processor_params)
        actor_observation = processed["state"] if observation_is_mapping else processed
        if appended_velocity_history_frames is not None:
            raw_actor_observation = observation["state"] if observation_is_mapping else observation
            return policy_module.apply(policy_params, raw_actor_observation, actor_observation)
        return policy_module.apply(policy_params, actor_observation)

    dummy = jnp.zeros((1, flat_observation_size), dtype=jnp.float32)

    def init_policy(key: jax.Array) -> Any:
        if appended_velocity_history_frames is not None:
            return policy_module.init(key, dummy, dummy)
        return policy_module.init(key, dummy)

    policy_network = brax_networks.FeedForwardNetwork(
        init=init_policy,
        apply=apply,
    )
    standard = ppo_networks.make_ppo_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(256, 256, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        activation=linen.swish,
        policy_obs_key="state",
        value_obs_key="privileged_state" if observation_is_mapping else "state",
    )
    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=standard.value_network,
        parametric_action_distribution=brax_distribution.NormalTanhDistribution(
            event_size=action_size,
            var_scale=0.5,
        ),
    )


def _projected_gravity_body(quaternion_wxyz: jax.Array) -> jax.Array:
    w, x, y, z = quaternion_wxyz
    return jnp.asarray(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=jnp.float32,
    )


def _upright_projection(quaternion_wxyz: jax.Array) -> jax.Array:
    _, x, y, _ = quaternion_wxyz
    return 1.0 - 2.0 * (x * x + y * y)


def _potential(data: Any) -> jax.Array:
    upright = jnp.clip(_upright_projection(data.qpos[3:7]), -1.0, 1.0)
    height = jnp.clip((data.qpos[2] - 0.08) / 0.62, 0.0, 1.0)
    return (
        1.5 * height
        + 0.75 * (upright + 1.0)
        - 0.10 * jnp.tanh(jnp.linalg.norm(data.qvel[:3]))
        - 0.10 * jnp.tanh(0.5 * jnp.linalg.norm(data.qvel[3:6]))
    )


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
        raise ValueError("recovery MJX PPO produced no checkpoint files")
    return str(hash_json(rows)), rows


def _select_frozen_memories(
    *,
    snapshots: tuple[RecoverySnapshot, ...],
    episodes: tuple[RecoveryTeacherEpisode, ...],
    episode_length: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    memories: list[np.ndarray] = []
    lengths: list[int] = []
    episode_hashes: list[str] = []
    for snapshot in snapshots:
        matches = tuple(
            episode
            for episode in episodes
            if episode.base_snapshot_hash == snapshot.snapshot_hash
            and episode.initial_snapshot_hash == snapshot.snapshot_hash
            and episode.rollout_succeeded
        )
        if len(matches) != 1:
            raise ValueError("frozen recovery memory requires one exact successful route per state")
        episode = matches[0]
        target = np.asarray(episode.absolute_motor_targets_rad, dtype=np.float32)
        active_length = min(target.shape[0], episode_length)
        if target.shape[0] < episode_length:
            target = np.concatenate(
                (target, np.repeat(target[-1:], episode_length - target.shape[0], axis=0)),
                axis=0,
            )
        memories.append(target[:episode_length])
        lengths.append(active_length)
        episode_hashes.append(episode.episode_hash)
    return (
        np.stack(memories).astype(np.float32),
        np.asarray(lengths, dtype=np.int32),
        tuple(episode_hashes),
    )


class RecoveryMJXResidualEnv(Env):
    """Unbatched MJX environment; Brax owns vmap, pmap, and auto-reset."""

    def __init__(
        self,
        *,
        scene_xml_path: Path,
        snapshots: tuple[RecoverySnapshot, ...],
        memory_targets: np.ndarray,
        memory_lengths: np.ndarray,
        default_joint_position_rad: np.ndarray,
        joint_lower_rad: np.ndarray,
        joint_upper_rad: np.ndarray,
        config: RecoveryMJXPPOConfig,
    ) -> None:
        self._config = config
        self._mj_model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
        self._mj_model.opt.timestep = 0.002
        if (self._mj_model.nq, self._mj_model.nv, self._mj_model.nu) != (36, 35, 29):
            raise ValueError("recovery MJX PPO requires the OpenTrack G1 29-DoF model")
        self._mjx_model = mjx.put_model(self._mj_model)
        self._snapshot_qpos = jnp.asarray(np.stack([item.qpos for item in snapshots]))
        self._snapshot_qvel = jnp.asarray(np.stack([item.qvel for item in snapshots]))
        self._memory_targets = jnp.asarray(memory_targets)
        self._memory_lengths = jnp.asarray(memory_lengths)
        self._default = jnp.asarray(default_joint_position_rad)
        self._joint_lower = jnp.asarray(joint_lower_rad)
        self._joint_upper = jnp.asarray(joint_upper_rad)
        self._residual_limits = jnp.asarray(config.residual_limits_rad)
        self._kp = jnp.asarray(_KPS)
        self._kd = jnp.asarray(_KDS)
        self._torque_limit = jnp.asarray(_TORQUE_LIMIT)

    @property
    def observation_size(self) -> int:
        return _ACTOR_OBSERVATION_DIM

    @property
    def action_size(self) -> int:
        return _JOINT_COUNT

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    def _observation(
        self, data: Any, last_target: jax.Array, memory_target: jax.Array
    ) -> jax.Array:
        proprio = jnp.concatenate(
            (
                _projected_gravity_body(data.qpos[3:7]),
                data.qvel[3:6] * 0.05,
                data.qpos[7:] - self._default,
                data.qvel[6:] * 0.05,
                last_target,
            )
        )
        result = jnp.concatenate((proprio, memory_target - data.qpos[7:]))
        return jnp.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, rng: jax.Array) -> State:
        rng, index_rng, joint_rng, velocity_rng = jax.random.split(rng, 4)
        base_index = jax.random.randint(
            index_rng, (), minval=0, maxval=self._snapshot_qpos.shape[0]
        )
        qpos = self._snapshot_qpos[base_index].at[:2].set(0.0)
        joint_noise = jax.random.uniform(
            joint_rng,
            (_JOINT_COUNT,),
            minval=-self._config.joint_position_noise_rad,
            maxval=self._config.joint_position_noise_rad,
        )
        qpos = qpos.at[7:].set(
            jnp.clip(qpos[7:] + joint_noise, self._joint_lower, self._joint_upper)
        )
        velocity_noise = jax.random.uniform(velocity_rng, (35,), minval=-1.0, maxval=1.0)
        qvel = self._snapshot_qvel[base_index]
        qvel = qvel.at[:3].add(velocity_noise[:3] * self._config.root_linear_velocity_noise_mps)
        qvel = qvel.at[3:6].add(
            velocity_noise[3:6] * self._config.root_angular_velocity_noise_rad_s
        )
        qvel = qvel.at[6:].add(velocity_noise[6:] * self._config.joint_velocity_noise_rad_s)
        data = mjx.make_data(self._mjx_model).replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32),
        )
        data = mjx.forward(self._mjx_model, data)
        initial_target = qpos[7:]
        memory_target = self._memory_targets[base_index, 0]
        potential = _potential(data)
        zero = jnp.zeros((), dtype=jnp.float32)
        return State(
            pipeline_state=data,
            obs=self._observation(data, initial_target, memory_target),
            reward=zero,
            done=zero,
            metrics={
                "reward": zero,
                "success": zero,
                "stable": zero,
                "pelvis_height": data.qpos[2],
                "upright": _upright_projection(data.qpos[3:7]),
                "residual_rms": zero,
                "torque_saturation": zero,
            },
            info={
                "rng": rng,
                "base_index": base_index,
                "memory_step": jnp.zeros((), dtype=jnp.int32),
                "stable_streak": jnp.zeros((), dtype=jnp.int32),
                "last_target": initial_target,
                "initial_target": initial_target,
                "last_action": jnp.zeros((_JOINT_COUNT,), dtype=jnp.float32),
                "potential": potential,
                "initial_potential": potential,
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        data = state.pipeline_state
        base_index = state.info["base_index"]
        memory_step = jnp.minimum(state.info["memory_step"], self._memory_lengths[base_index] - 1)
        memory_target = self._memory_targets[base_index, memory_step]
        bounded_action = jnp.clip(action, -1.0, 1.0)
        desired_target = jnp.clip(
            memory_target + bounded_action * self._residual_limits,
            self._joint_lower,
            self._joint_upper,
        )
        target_delta = jnp.clip(
            desired_target - state.info["last_target"],
            -self._config.maximum_target_step_rad,
            self._config.maximum_target_step_rad,
        )
        motor_target = state.info["last_target"] + target_delta

        def simulation_step(current: Any, unused: Any) -> tuple[Any, jax.Array]:
            del unused
            raw_torque = self._kp * (motor_target - current.qpos[7:]) + self._kd * (
                -current.qvel[6:]
            )
            saturation = jnp.mean((jnp.abs(raw_torque) > self._torque_limit).astype(jnp.float32))
            torque = jnp.clip(raw_torque, -self._torque_limit, self._torque_limit)
            return mjx.step(self._mjx_model, current.replace(ctrl=torque)), saturation

        data, saturation_trace = jax.lax.scan(simulation_step, data, None, length=10)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        data = data.replace(
            qpos=jnp.nan_to_num(data.qpos, nan=0.0, posinf=0.0, neginf=0.0),
            qvel=jnp.nan_to_num(data.qvel, nan=0.0, posinf=0.0, neginf=0.0),
        )
        upright = _upright_projection(data.qpos[3:7])
        stable = (
            (data.qpos[2] >= 0.62)
            & (upright >= 0.75)
            & (jnp.linalg.norm(data.qvel[:3]) <= 0.50)
            & (jnp.linalg.norm(data.qvel[3:6]) <= 1.50)
        )
        stable_streak = jnp.where(stable, state.info["stable_streak"] + 1, 0)
        success = stable_streak >= self._config.success_stable_steps
        current_potential = _potential(data)
        residual_rms = jnp.sqrt(jnp.mean(jnp.square(bounded_action)))
        action_delta_rms = jnp.sqrt(
            jnp.mean(jnp.square(bounded_action - state.info["last_action"]))
        )
        tracking_rms = jnp.sqrt(jnp.mean(jnp.square(data.qpos[7:] - memory_target)))
        torque_saturation = jnp.mean(saturation_trace)
        dense_posture = 0.02 * (
            jnp.clip((data.qpos[2] - 0.08) / 0.62, 0.0, 1.0)
            + jnp.clip((upright + 1.0) * 0.5, 0.0, 1.0)
        )
        reward = (
            8.0 * (current_potential - state.info["potential"])
            + dense_posture
            + 0.50 * stable.astype(jnp.float32)
            + 80.0 * success.astype(jnp.float32)
            - self._config.tracking_penalty_scale * tracking_rms
            - self._config.residual_penalty_scale * residual_rms
            - self._config.action_delta_penalty_scale * action_delta_rms
            - self._config.torque_saturation_penalty_scale * torque_saturation
            - 40.0 * (~finite).astype(jnp.float32)
        )
        reward = jnp.nan_to_num(reward, nan=-40.0, posinf=-40.0, neginf=-40.0)
        done = success | (~finite)
        wrapper_step = state.info.get("steps", jnp.zeros((), dtype=jnp.int32))
        reset_info = done | (wrapper_step + 1 >= self._config.episode_length)
        next_memory_step = jnp.where(reset_info, 0, state.info["memory_step"] + 1)
        next_target = jnp.where(reset_info, state.info["initial_target"], motor_target)
        next_memory_index = jnp.minimum(next_memory_step, self._memory_lengths[base_index] - 1)
        obs = self._observation(
            data, next_target, self._memory_targets[base_index, next_memory_index]
        )
        info = dict(state.info)
        info.update(
            memory_step=next_memory_step,
            stable_streak=jnp.where(reset_info, 0, stable_streak),
            last_target=next_target,
            last_action=jnp.where(reset_info, jnp.zeros_like(bounded_action), bounded_action),
            potential=jnp.where(reset_info, state.info["initial_potential"], current_potential),
        )
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done.astype(jnp.float32),
            metrics={
                "reward": reward,
                "success": success.astype(jnp.float32),
                "stable": stable.astype(jnp.float32),
                "pelvis_height": data.qpos[2],
                "upright": upright,
                "residual_rms": residual_rms,
                "torque_saturation": torque_saturation,
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


def train_recovery_mjx_residual_ppo(
    *,
    scene_xml_path: Path,
    snapshot_manifest_path: Path,
    teacher_corpus_manifest_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: RecoveryMJXPPOConfig | None = None,
) -> dict[str, Any]:
    active = config or RecoveryMJXPPOConfig()
    scene_path = scene_xml_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    corpus_path = teacher_corpus_manifest_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if any(not path.is_file() for path in (scene_path, snapshot_path, corpus_path)):
        raise FileNotFoundError("recovery MJX PPO inputs are incomplete")
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("recovery MJX PPO output must be new and external")
    if jax.device_count() < 4:
        raise RuntimeError("recovery MJX PPO requires four visible GPUs")
    destination.mkdir(parents=True)

    snapshots = load_recovery_snapshot_corpus(snapshot_path)
    corpus = load_recovery_distillation_corpus(corpus_path)
    episodes = recovery_teacher_episodes_from_corpus(corpus)
    memory_targets, memory_lengths, memory_episode_hashes = _select_frozen_memories(
        snapshots=snapshots, episodes=episodes, episode_length=active.episode_length
    )
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    scene_hash = hash_bytes(scene_path.read_bytes())
    if corpus_payload.get("physics_scene_hash") != scene_hash:
        raise ValueError("frozen recovery memory and MJX scene are not content-bound")

    environment = RecoveryMJXResidualEnv(
        scene_xml_path=scene_path,
        snapshots=snapshots,
        memory_targets=memory_targets,
        memory_lengths=memory_lengths,
        default_joint_position_rad=corpus.default_joint_position_rad,
        joint_lower_rad=corpus.joint_lower_rad,
        joint_upper_rad=corpus.joint_upper_rad,
        config=active,
    )
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
        max_devices_per_host=4,
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
        progress_fn=progress_fn,
        save_checkpoint_path=str(checkpoint_dir),
    )
    training_sec = time.perf_counter() - started
    checkpoint_hash, checkpoint_files = _tree_hash(checkpoint_dir)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_mjx_ppo_training_report.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "rollout_backend": "MUJOCO_MJX",
        "parallelization": "BRAX_PPO_JAX_PMAP_VMAP",
        "devices": [str(device) for device in jax.devices()[:4]],
        "scene_xml_hash": scene_hash,
        "compiled_model_contract": compiled_mujoco_model_contract(environment.mj_model),
        "snapshot_manifest_hash": hash_bytes(snapshot_path.read_bytes()),
        "teacher_corpus_manifest_hash": corpus.manifest_hash,
        "frozen_memory_episode_hashes": list(memory_episode_hashes),
        "frozen_memory_hash": hash_bytes(memory_targets.tobytes()),
        "frozen_memory_unchanged": True,
        "actor_observation_dim": _ACTOR_OBSERVATION_DIM,
        "actor_initialization": "EXACT_ZERO_DETERMINISTIC_RESIDUAL",
        "actor_forbidden_features": [
            "external_reference_phase",
            "teacher_identity",
            "future_reference_state",
        ],
        "action_semantics": "BOUNDED_PD_TARGET_RESIDUAL_AROUND_FROZEN_SKILL_MEMORY",
        "training_sec": training_sec,
        "progress": progress,
        "final_metrics": _jsonable(final_metrics),
        "checkpoint_tree_hash": checkpoint_hash,
        "checkpoint_files": checkpoint_files,
        "sealed_holdout_reports_loaded": 0,
        "sealed_holdout_states_read": 0,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "GPU rollout evidence only; independent CPU-MuJoCo exam is required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination / "training-report.json", report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Train recovery residual PPO on modern MJX")
    parser.add_argument("--scene-xml", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--teacher-corpus-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-checkout", required=True, type=Path)
    parser.add_argument("--total-timesteps", default=2_097_152, type=int)
    parser.add_argument("--num-envs", default=256, type=int)
    parser.add_argument("--episode-length", default=1_800, type=int)
    parser.add_argument("--num-evals", default=3, type=int)
    parser.add_argument("--num-eval-envs", default=64, type=int)
    parser.add_argument("--seed", default=5401, type=int)
    args = parser.parse_args()
    result = train_recovery_mjx_residual_ppo(
        scene_xml_path=args.scene_xml,
        snapshot_manifest_path=args.snapshot_manifest,
        teacher_corpus_manifest_path=args.teacher_corpus_manifest,
        output_dir=args.output_dir,
        source_checkout_path=args.source_checkout,
        config=RecoveryMJXPPOConfig(
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
            num_evals=args.num_evals,
            num_eval_envs=args.num_eval_envs,
            random_seed=args.seed,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = ["RecoveryMJXResidualEnv", "train_recovery_mjx_residual_ppo"]
