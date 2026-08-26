"""Fail-closed OpenTrack adapter export bridge.

OpenTrack's August 2026 adapter exporter queries ``env.observation_size`` only
after restoring the model-based PPO trainer.  The environment uses mutable
trajectory bookkeeping during shape evaluation, so that ordering can leak a
JAX tracer and abort export.  This optional bridge seals the environment IO
contract before restore, then delegates parameter reconstruction and ONNX
conversion to OpenTrack's own implementation.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def export_opentrack_adapter_onnx(
    *,
    opentrack_root: Path,
    checkpoint_path: Path,
    sanitized_config_path: Path,
    output_path: Path,
    task: str = "G1TrackingGeneralDR",
) -> dict[str, Any]:
    """Export a trained adapter without mutating its checkpoint or config."""

    root = opentrack_root.expanduser().resolve()
    checkpoint = checkpoint_path.expanduser().resolve()
    config_path = sanitized_config_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not root.is_dir() or not checkpoint.is_dir() or not config_path.is_file():
        raise FileNotFoundError("OpenTrack root, adapter checkpoint, and config must exist")
    if root not in checkpoint.parents:
        raise ValueError("adapter checkpoint must belong to the pinned OpenTrack checkout")
    if output.exists():
        raise ValueError("OpenTrack adapter export refuses to overwrite an existing policy")
    if output.parent != checkpoint:
        raise ValueError("OpenTrack adapter policy must be exported beside its checkpoint")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,127}", task):
        raise ValueError("OpenTrack task name is not normalized")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("OpenTrack adapter config must be a JSON object")
    for section in ("env_config", "policy_config", "mbppo_policy_config"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"OpenTrack adapter config is missing {section}")
    callbacks = {"progress_fn", "randomization_fn", "wrap_env_fn"}
    if callbacks.intersection(config["policy_config"]):
        raise ValueError("OpenTrack adapter config still contains serialized callbacks")

    os.environ.setdefault("GLI_PATH", str(root))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    tmj = importlib.import_module("track_mj")
    export_module = importlib.import_module("track_mj.eval.adapter.export_onnx")
    converter = importlib.import_module("track_mj.eval.adapter.brax2onnx")
    wrap_module = importlib.import_module("track_mj.envs.g1_tracking.utils.wrapper")
    mbppo = importlib.import_module(
        "track_mj.learning.policy.model_based_ppo.train_model_based_ppo"
    )

    task_cfg = tmj.registry.get(task, "tracking_adapter_config")
    env_cfg = copy.deepcopy(task_cfg.env_config)
    policy_config = copy.deepcopy(task_cfg.policy_config)
    mbppo_policy_config = copy.deepcopy(task_cfg.mbppo_policy_config)
    env_cfg.update(config["env_config"])
    policy_config.update(config["policy_config"])
    mbppo_policy_config.update(config["mbppo_policy_config"])
    env_class = tmj.registry.get(task, "tracking_adapter_train_env_class")
    env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    env.prepare_trajectory(env_cfg.reference_traj_config.name)

    # Important ordering: seal shapes before the trainer stages transformed
    # environment functions.  This avoids OpenTrack's UnexpectedTracerError.
    observation_size = copy.deepcopy(env.observation_size)
    action_size = int(env.action_size)
    history_len = int(env_cfg.history_len)
    if history_len <= 0 or "history_state" not in observation_size:
        raise ValueError("OpenTrack adapter export requires a positive history contract")

    train_params = export_module._prepare_mbppo_export_params(
        policy_config, mbppo_policy_config, checkpoint
    )
    train_params["wrap_env_fn"] = wrap_module.wrap_fn
    train_fn = functools.partial(mbppo.train, **train_params)
    make_inference_fn, params, _ = train_fn(environment=env)
    inference_fn = make_inference_fn(params, deterministic=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    converter.convert_jax2onnx_with_history(
        output_path=str(output),
        inference_fn=inference_fn,
        policy_network_cfg=policy_config.network_factory,
        mbppo_network_cfg=mbppo_policy_config.network_factory,
        obs_size=observation_size,
        action_size=action_size,
        history_len=history_len,
        jax_params=params,
        use_adapter=bool(mbppo_policy_config.use_adapter),
        activation="swish",
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("OpenTrack adapter exporter did not produce a policy")
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.opentrack_adapter_export.v1",
        "opentrack_commit": _git_head(root),
        "checkpoint_hash": _tree_hash(checkpoint, exclude={output.name}),
        "sanitized_config_hash": _file_hash(config_path),
        "policy_hash": _file_hash(output),
        "task": task,
        "observation_size": _jsonable(observation_size),
        "action_size": action_size,
        "history_len": history_len,
        "export_order": "seal_io_then_restore_then_convert",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tree_hash(path: Path, *, exclude: set[str]) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        if item.name in exclude:
            continue
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(_file_hash(item).removeprefix("sha256:")))
    return "sha256:" + digest.hexdigest()


def _git_head(root: Path) -> str:
    head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        head = (root / ".git" / head.removeprefix("ref: ")).read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("OpenTrack checkout must have a readable pinned commit")
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an OpenTrack history adapter")
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--sanitized-config-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--task", default="G1TrackingGeneralDR")
    args = parser.parse_args()
    report = export_opentrack_adapter_onnx(
        opentrack_root=args.opentrack_root,
        checkpoint_path=args.checkpoint_path,
        sanitized_config_path=args.sanitized_config_path,
        output_path=args.output_path,
        task=args.task,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["export_opentrack_adapter_onnx"]
