"""Strict research-only learned-head checkpoint admission; not old BC evidence."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
from rosclaw_soccer.rsi.refinement_onpolicy_learning import (
    COMMITMENTS,
    PEER_KIND,
    SCHEMA,
    STATUS,
    config_for_kind,
    kind_for_config,
    tensor_state_hash,
    validate_tensor_map,
)


def file_hash(path: Path, *, maximum: int = 128 * 1024 * 1024) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded ordinary on-policy checkpoint/receipt required")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint(
    state: Any,
    behavior: RefinementMotorAdapter,
    critic: MotorCritic,
    *,
    config: dict[str, Any],
    commitments: dict[str, str],
    steps: int,
) -> None:
    if (
        type(state) is not dict
        or state.get("schema") != SCHEMA
        or state.get("kind") != kind_for_config(config)
        or state.get("config") != config
        or state.get("commitments") != commitments
        or set(commitments) != COMMITMENTS
        or type(steps) is not int
        or not 1 <= steps <= 4
        or state.get("actor_steps") != steps
        or state.get("critic_steps") != steps
        or tensor_state_hash(dict(behavior.parent.state_dict()))
        != commitments["frozen_parent_hash"]
    ):
        raise ValueError("exact actual bounded on-policy update metadata required")
    head = dict(behavior.head.state_dict())
    value = dict(critic.state_dict())
    for key, expected in (
        ("initial_head", head),
        ("head", head),
        ("initial_critic", value),
        ("critic", value),
    ):
        validate_tensor_map(state.get(key), expected)
    if any(not torch.equal(v, state["initial_head"][k]) for k, v in head.items()):
        raise ValueError("initial head must equal whole actual behavior checkpoint")
    if any(not torch.equal(v, state["initial_critic"][k]) for k, v in value.items()):
        raise ValueError("actual declared zero-value critic initialization required")
    if not any(not torch.equal(v, state["head"][k]) for k, v in head.items()):
        raise ValueError("an unchanged actor is not a learned refinement checkpoint")
    if not any(not torch.equal(v, state["critic"][k]) for k, v in value.items()):
        raise ValueError("an unchanged critic is not a learned value function")
    optimizer = state.get("optimizer")
    if (
        type(optimizer) is not dict
        or set(optimizer) != {"state", "param_groups"}
        or len(optimizer["param_groups"]) != 2
    ):
        raise ValueError("explicit full actor+critic Adam state required")
    params = [*head.values(), *value.values()]
    if set(optimizer["state"]) != set(range(len(params))) or [
        g["params"] for g in optimizer["param_groups"]
    ] != [list(range(4)), list(range(4, 8))]:
        raise ValueError("exact unique optimizer parameter ownership required")
    for i, p in enumerate(params):
        s = optimizer["state"][i]
        if set(s) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("complete Adam moments required")
        for key in ("exp_avg", "exp_avg_sq"):
            v = s[key]
            if (
                not isinstance(v, torch.Tensor)
                or v.dtype != p.dtype
                or v.shape != p.shape
                or v.device.type != "cpu"
                or not torch.isfinite(v).all()
            ):
                raise ValueError("finite exact optimizer moments required")
            if key == "exp_avg_sq" and (v < 0).any():
                raise ValueError("nonnegative Adam second moments required")
        if (
            not isinstance(s["step"], torch.Tensor)
            or s["step"].numel() != 1
            or float(s["step"]) != steps
        ):
            raise ValueError("actual accepted Adam step counters required")
    rng = state.get("rng_state")
    if (
        not isinstance(rng, torch.Tensor)
        or rng.device.type != "cpu"
        or rng.dtype != torch.uint8
        or rng.shape != torch.get_rng_state().shape
    ):
        raise ValueError("complete CPU RNG continuation state required")
    for group, rate in zip(
        optimizer["param_groups"], (config["actor_lr"], config["critic_lr"]), strict=True
    ):
        if (
            type(group["lr"]) not in (int, float)
            or not 0 < group["lr"] <= rate
            or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8
            or group["weight_decay"] != 0
        ):
            raise ValueError("declared finite Adam schedule and settings required")


def load_onpolicy_refinement(
    behavior: RefinementMotorAdapter,
    *,
    behavior_training_hash: str,
    root: Path,
    expected_receipt_hash: str,
) -> tuple[RefinementMotorAdapter, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        type(behavior) is not RefinementMotorAdapter
        or root.is_symlink()
        or (root / "failed.json").exists()
        or file_hash(rp) != expected_receipt_hash
    ):
        raise ValueError("exact completed new-kind on-policy refinement required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    cfg = config_for_kind(r["checkpoint_kind"], r["config"]["state_id"])
    if (
        r["status"] != STATUS
        or r["checkpoint_kind"] != kind_for_config(cfg)
        or r["config"] != cfg
        or d["config"] != cfg
        or r["declaration_hash"] != file_hash(dp)
        or r["behavior_training_hash"] != behavior_training_hash
        or any(r[k] != d[k] for k in COMMITMENTS | {"source_head", "input_file_hashes"})
        or r["new_actor_steps"] != r["new_critic_steps"]
        or not 1 <= r["new_actor_steps"] <= 4
        or r["parent_weights_unchanged"] is not True
        or r["on_policy_RL_update"] is not True
        or r["physical_steps_executed"] != 0
        or r["physical_retention_proven"] is not False
        or r["continuous_learning_proven"] is not False
        or len(r["history"]) != 4
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
        or sum(x["update"]["applied"] for x in r["history"]) != r["new_actor_steps"]
    ):
        raise ValueError("exact new actor/critic training contract required, not relabelled BC")
    for i, x in enumerate(r["history"], start=1):
        if (
            x["epoch"] != i
            or type(x["update"]["applied"]) is not bool
            or not 1 <= len(x["update"]["attempts"]) <= 8
        ):
            raise ValueError("all bounded actual epoch/backtrack history required")
        if x["update"]["applied"] and (
            not x["update"]["attempts"][-1]["accepted"]
            or not 0 <= x["update"]["attempts"][-1]["kl"] <= 0.015
        ):
            raise ValueError("accepted update must satisfy original trust region")

    def receipt(h: str) -> dict[str, Any]:
        paths = [
            Path(p)
            for p, v in r["input_file_hashes"].items()
            if v == h and p.endswith("/complete.json")
        ]
        if len(paths) != 1 or file_hash(paths[0]) != h:
            raise ValueError("unique exact on-policy ancestry receipt required")
        value: dict[str, Any] = json.loads(paths[0].read_text())
        return value

    experience, own = receipt(r["experience_hash"]), receipt(r["experience_own_hash"])
    if r["checkpoint_kind"] == PEER_KIND:
        ap = root / "advantages.json"
        if file_hash(ap, maximum=1024 * 1024) != r["advantage_evidence_hash"]:
            raise ValueError("exact peer-baseline derivation evidence required")
        ae = json.loads(ap.read_text())
        if (
            ae["mode"] != "peer-loo-v2"
            or ae["experience_hash"] != r["experience_hash"]
            or ae["initial_state_hashes"]
            != [x["artifacts"]["initial"]["hash"] for x in experience["records"]]
            or ae["group_sizes"] != [8, 8, 8]
            or ae["own_return_in_baseline"] is not False
            or ae["padded_or_held_rows"] != 0
            or len(ae["advantage_vector_hashes"]) != 24
        ):
            raise ValueError("all peer groups must bind actual identical physical initials")
    original = receipt(behavior_training_hash)
    old_own = receipt(experience["expert_own_hash"])
    if (
        experience["status"] != "OWN_STOCHASTIC_REFINEMENT24_COLLECTED_PENDING_NATIVE_LAW_AUDIT"
        or own["status"] != "OWN_STOCHASTIC_REFINEMENT24_ALL_NATIVE_PRIVATE_RNG_LAW_MATCH"
        or own["source_hash"] != r["experience_hash"]
        or own["sampling_law_audit_passed"] is not True
        or own["independent_whole_world_replay"] is not True
        or len(experience["records"]) != 24
        or own["records"]
        != [{k: v for k, v in x.items() if k != "artifacts"} for x in experience["records"]]
        or own["counts"] != experience["counts"]
        or old_own["training_hash"] != behavior_training_hash
        or original["checkpoint_hash"] != r["behavior_checkpoint_hash"]
        or original["parent_lineage_hash"] != r["parent_lineage_hash"]
    ):
        raise ValueError("actual unchanged behavior and ALL native/sampling evidence required")
    cp = Path(r["checkpoint_path"])
    if (
        cp.resolve().parent != root.resolve()
        or file_hash(cp, maximum=16 * 1024 * 1024) != r["checkpoint_hash"]
    ):
        raise ValueError("confined exact tensor-only on-policy checkpoint required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg["seed"])
        critic = MotorCritic()
    state = torch.load(cp, map_location="cpu", weights_only=True)
    validate_checkpoint(
        state,
        behavior,
        critic,
        config=cfg,
        commitments={k: r[k] for k in COMMITMENTS},
        steps=r["new_actor_steps"],
    )
    result = copy.deepcopy(behavior).eval().requires_grad_(False)
    result.head.load_state_dict(state["head"], strict=True)
    return result, r
