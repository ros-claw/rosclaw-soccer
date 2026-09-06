"""Distill role-local playmaker discovery into a failure-aware actor."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.growth.playmaker_pass_actor import (
    G1PlaymakerPassActor,
    PlaymakerPassMemory,
    load_playmaker_pass_actor,
    playmaker_pass_features,
    save_playmaker_pass_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.neural_contact_holdout_exam import (
    validate_neural_contact_holdout_exam,
)
from rosclaw_soccer.training.playmaker_pass_discovery import (
    PlaymakerPassProbeAction,
    validate_playmaker_pass_discovery,
)


def train_playmaker_pass_actor(
    *,
    discovery_report_path: Path,
    source_holdout_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a bounded nearest-memory actor from real success and failure traces."""

    discovery_path = discovery_report_path.expanduser().resolve()
    holdout_path = source_holdout_report_path.expanduser().resolve()
    discovery = validate_playmaker_pass_discovery(discovery_path)
    holdout = validate_neural_contact_holdout_exam(holdout_path)
    if (
        discovery.get("status") != "PASS_PLAYMAKER_PASS_DISCOVERY"
        or discovery.get("frozen_finisher_actor_hash") != holdout.get("actor_hash")
        or discovery.get("promotion_eligible") is not False
    ):
        raise ValueError("playmaker distillation lineage changed")
    output = _new_external_output(output_dir)
    threshold = float(
        _read_object(discovery_path.parent / "request.json")["maximum_delivery_error_m"]
    )
    base_action = PlaymakerPassProbeAction()
    successes: list[PlaymakerPassMemory] = []
    failures: list[PlaymakerPassMemory] = []
    holdout_request = _read_object(holdout_path.parent / "request.json")
    holdout_contexts = {
        str(hash_json(value)): _context_from_dict(value) for value in holdout_request["contexts"]
    }

    for row in holdout["rows"]:
        context = holdout_contexts[row["context_hash"]]
        result = row["primary"]["result"]
        quality = row["primary"]["quality"]
        memory = _memory(
            context=context,
            action=base_action,
            result=result,
            quality=quality,
            trajectory_hash=row["primary"]["trajectory"]["file_hash"],
        )
        (successes if _qualified(memory, threshold) else failures).append(memory)
    for row in discovery["rows"]:
        context = _context_from_dict(row["context"])
        action = PlaymakerPassProbeAction(**row["action"])
        memory = _memory(
            context=context,
            action=action,
            result=row["result"],
            quality=row["quality"],
            trajectory_hash=row["trajectory"]["file_hash"],
        )
        (successes if _qualified(memory, threshold) else failures).append(memory)
    successes = _distinct(successes)
    failures = _distinct(failures)
    matrix = np.asarray([memory.features for memory in (*successes, *failures)], dtype=np.float64)
    center = np.mean(matrix, axis=0)
    scale = np.maximum(np.std(matrix, axis=0), np.asarray((0.003, 0.003, 0.001, 0.001, 0.0007)))
    request = {
        "schema_version": "rosclaw_soccer.playmaker_pass_distillation_request.v1",
        "discovery_report_hash": discovery["report_hash"],
        "discovery_file_hash": hash_bytes(discovery_path.read_bytes()),
        "source_holdout_report_hash": holdout["report_hash"],
        "source_holdout_file_hash": hash_bytes(holdout_path.read_bytes()),
        "success_memory_hashes": [memory.memory_hash for memory in successes],
        "failure_memory_hashes": [memory.memory_hash for memory in failures],
        "maximum_delivery_error_m": threshold,
        "plastic_agent_id": "red.playmaker",
        "frozen_finisher_actor_hash": discovery["frozen_finisher_actor_hash"],
        "frozen_goalkeeper_policy_hash": discovery["frozen_goalkeeper_policy_hash"],
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    actor = G1PlaymakerPassActor(
        body_hash=str(_read_object(discovery_path.parent / "request.json")["body_hash"]),
        source_discovery_hash=str(discovery["report_hash"]),
        source_holdout_hash=str(holdout["report_hash"]),
        frozen_finisher_actor_hash=str(discovery["frozen_finisher_actor_hash"]),
        frozen_goalkeeper_policy_hash=str(discovery["frozen_goalkeeper_policy_hash"]),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        successful_memories=tuple(successes),
        failed_memories=tuple(failures),
        maximum_delivery_error_m=threshold,
    )
    actor_path = output / "playmaker-pass-actor.json"
    save_playmaker_pass_actor(actor, actor_path)
    loaded = load_playmaker_pass_actor(actor_path)
    accepted_successes = sum(int(loaded.decide(memory.features).accepted) for memory in successes)
    rejected_failures = sum(int(not loaded.decide(memory.features).accepted) for memory in failures)
    gates = {
        "all_success_memories_recalled": accepted_successes == len(successes),
        "minimum_failure_support": len(failures) >= 8,
        "actor_round_trip": loaded.actor_hash == actor.actor_hash,
        "roles_frozen": bool(
            actor.frozen_finisher_actor_hash == discovery["frozen_finisher_actor_hash"]
            and actor.frozen_goalkeeper_policy_hash == discovery["frozen_goalkeeper_policy_hash"]
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.playmaker_pass_distillation.v1",
        "status": (
            "PASS_PLAYMAKER_PASS_DISTILLATION"
            if all(gates.values())
            else "REJECTED_PLAYMAKER_PASS_DISTILLATION"
        ),
        "promotion_eligible": False,
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "metrics": {
            "success_memory_count": len(successes),
            "failure_memory_count": len(failures),
            "accepted_success_memory_count": accepted_successes,
            "rejected_failure_memory_count": rejected_failures,
        },
        "gates": gates,
        "plastic_agent_id": "red.playmaker",
        "frozen_finisher_actor_hash": actor.frozen_finisher_actor_hash,
        "frozen_goalkeeper_policy_hash": actor.frozen_goalkeeper_policy_hash,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "training-report.json", report)
    return report


def _memory(
    *,
    context: Any,
    action: PlaymakerPassProbeAction,
    result: dict[str, Any],
    quality: dict[str, Any],
    trajectory_hash: str,
) -> PlaymakerPassMemory:
    error = result.get("pass_delivery_error_m")
    return PlaymakerPassMemory(
        context_hash=context.context_hash,
        trajectory_hash=trajectory_hash,
        features=playmaker_pass_features(context),
        action=action,
        delivery_error_m=(
            float(error)
            if isinstance(error, int | float)
            and not isinstance(error, bool)
            and math.isfinite(error)
            else 5.0
        ),
        safe=quality.get("safe") is True,
        ordered_contacts=quality.get("ordered_contacts") is True,
    )


def _qualified(memory: PlaymakerPassMemory, threshold: float) -> bool:
    return bool(memory.safe and memory.ordered_contacts and memory.delivery_error_m <= threshold)


def _distinct(memories: list[PlaymakerPassMemory]) -> list[PlaymakerPassMemory]:
    return list({memory.memory_hash: memory for memory in memories}.values())


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("playmaker distillation output must use a new external directory")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["train_playmaker_pass_actor"]
