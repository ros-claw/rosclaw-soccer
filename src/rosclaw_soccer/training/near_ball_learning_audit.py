"""Recompute private PPO lineage and held-out football outcomes from raw evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import validate_probe
from rosclaw_soccer.training.near_ball_curriculum import examination_courses
from rosclaw_soccer.training.near_ball_plasticity import private_weight_hashes, verify_update_record


def audit(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "training.json").read_text())
    commitment = manifest.pop("manifest_hash")
    if hash_json(manifest) != commitment:
        raise ValueError("training manifest commitment differs")
    initial_generation = manifest.get("initial_generation", 0)
    if type(initial_generation) is not int or not 0 <= initial_generation <= 1000000:
        raise ValueError("initial generation is invalid")
    initial_policy = NearBallResidualPolicy.load(root / f"generation-{initial_generation:03d}.npz")
    if (
        manifest.get("initial_policy_hash", initial_policy.policy_hash)
        != initial_policy.policy_hash
    ):
        raise ValueError("initial checkpoint commitment differs")
    parent = initial_policy
    players = {
        agent: {"active_samples": 0, "actor_squared_delta": 0.0, "critic_squared_delta": 0.0}
        for agent in parent.agent_ids
    }
    paths = {validate_probe(path)["report_hash"]: path for path in root.glob("*/probe.json")}
    verified = 0
    core_verified = 0
    for iteration in manifest["iterations"]:
        child = NearBallResidualPolicy.load(root / f"generation-{iteration['generation']:03d}.npz")
        if (
            child.parent_hash != parent.policy_hash
            or child.generation != parent.generation + 1
            or child.policy_hash != iteration["policy_hash"]
        ):
            raise ValueError("private policy lineage is broken")
        counts = np.zeros(8, dtype=np.int64)
        rollout_digests = []
        if len(iteration["players"]) != 8 or len(set(iteration["rollout_report_hashes"])) != len(
            iteration["rollout_report_hashes"]
        ):
            raise ValueError("private roster or training episode identities are incomplete")
        for digest in iteration["rollout_report_hashes"]:
            source = paths[digest]
            report = validate_probe(source)
            if (
                not report["exact_replay"]
                or not report["near_ball_residual"]["explore"]
                or report["near_ball_residual"]["policy_hash"] != parent.policy_hash
            ):
                raise ValueError("training rollout differs from its parent")
            with np.load(source.parent / "primary.npz", allow_pickle=False) as archive:
                trace = {k: archive[k] for k in archive.files}
            rng = np.random.default_rng(report["near_ball_residual"]["seed"])
            for t, obs in enumerate(trace["residual_observations"]):
                latent, logp, value = parent.act(obs, rng, explore=True)
                for expected, key in (
                    (latent, "residual_latent"),
                    (logp, "residual_log_probability"),
                    (value, "residual_value"),
                ):
                    if not np.array_equal(expected, trace[key][t]):
                        raise ValueError("saved policy does not reproduce actor collection")
            counts += trace["residual_active"].sum(axis=0)
            rollout_digests.append(report["trajectory_digests"][0])
            verified += 1
        dataset_hash = str(hash_json({"rollouts": rollout_digests}))
        context_hash = str(
            hash_json(
                {
                    "body": parent.body_hash,
                    "parent": parent.policy_hash,
                    "roster": parent.agent_ids,
                    "dataset": dataset_hash,
                }
            )
        )
        progressive = {k: v.copy() for k, v in parent.weights.items()}
        requires_core = any("core_plasticity" in row for row in iteration["players"])
        for i, row in enumerate(iteration["players"]):
            agent = parent.agent_ids[i]
            if row["agent_id"] != agent or row["active_samples"] != int(counts[i]):
                raise ValueError("private training sample attribution differs")
            deltas = {
                k: float(np.square(child.weights[k][i] - parent.weights[k][i]).sum())
                for k in parent.weights
            }
            if counts[i] < 32 and any(deltas.values()):
                raise ValueError("unsampled private actor was modified")
            before_hashes = private_weight_hashes(progressive, parent.agent_ids, parent.body_hash)
            for key in progressive:
                progressive[key][i] = child.weights[key][i]
            if "core_plasticity" in row:
                verify_update_record(
                    row["core_plasticity"],
                    before=before_hashes,
                    after=private_weight_hashes(progressive, parent.agent_ids, parent.body_hash),
                    focal=agent,
                    generation=child.generation,
                    dataset_hash=dataset_hash,
                    context_hash=context_hash,
                )
                core_verified += 1
            elif requires_core and counts[i] >= 32:
                raise ValueError("trainable private actor lacks its Core plasticity proof")
            players[agent]["active_samples"] += int(counts[i])
            players[agent]["actor_squared_delta"] += sum(
                deltas[k] for k in ("w1", "b1", "w2", "b2", "log_std")
            )
            players[agent]["critic_squared_delta"] += deltas["wv"] + deltas["bv"]
        parent = child
    outcomes: dict[str, Any] = {}
    baseline_label = manifest.get("baseline_label", "zero")
    if baseline_label not in {"zero", "parent"}:
        raise ValueError("unknown comparison baseline")
    role_curriculum = manifest.get("role_curriculum", False)
    expected_courses = (
        [asdict(c) for c in examination_courses()]
        if role_curriculum
        else [
            {"role": "playmaker", "blue": blue, "offset": offset}
            for blue in (False, True)
            for offset in (-0.08, 0.08)
        ]
    )
    if role_curriculum and manifest.get("evaluation_courses") != expected_courses:
        raise ValueError("role examination contract differs")
    contexts: dict[tuple[str, bool, float], str] = {}
    for label in (baseline_label, "candidate"):
        rows = []
        for item in manifest["evaluation"]:
            if item["label"] != label:
                continue
            report = validate_probe(paths[item["report_hash"]])
            residual = report["near_ball_residual"]
            expected_policy = parent if label == "candidate" else initial_policy
            if residual["explore"] or residual["policy_hash"] != expected_policy.policy_hash:
                raise ValueError("evaluation used exploration or the wrong checkpoint")
            blue = report["scenario"]["scenario_id"].endswith("blue")
            ball_y = report["scenario"]["ball_initial_position_m"][1]
            origin_y = 1.22 if report.get("forward_receiver_lane") else 1.20
            measured_offset = origin_y - ball_y if blue else ball_y + origin_y
            role = item.get("role", "playmaker")
            if role_curriculum:
                if not report.get("basic_ball_play") or report.get("kickoff_role") != role:
                    raise ValueError("role examination did not use its declared skill course")
                if role != "playmaker":
                    actor = f"{'blue' if blue else 'red'}.{role}"
                    player = next(p for p in report["players"] if p["agent_id"] == actor)
                    measured_offset = (ball_y - player["origin_m"][1]) * (
                        -1.0 if blue else 1.0
                    ) + 0.12
            if blue != item["blue"] or abs(measured_offset - item["offset"]) > 1e-9:
                raise ValueError("evaluation context differs from its physical scenario")
            course_key = (role, blue, item["offset"])
            context = str(
                hash_json(
                    {
                        name: report[name]
                        for name in (
                            "fixture_hash",
                            "world_config",
                            "contact_teacher_config",
                            "scenario",
                            "option_config",
                            "strike_phase_config",
                        )
                    }
                )
            )
            if label == baseline_label:
                contexts[course_key] = context
            elif contexts.get(course_key) != context:
                raise ValueError("parent and candidate examination worlds differ")
            events = report["assessment"]["events"]
            pass_events = sum(e["skill"] == "pass" for e in events)
            receives = sum(
                e["physical_receive_confirmed"] for e in report.get("causal_pass_feedback", [])
            )
            clean_control = report["assessment"]["gates"]["foot_only_ball_control"]
            jointly_confirmed = sum(
                any(
                    d["physical_receive_confirmed"]
                    and d["sender"] == e["agent_id"]
                    and d["receiver"] == e["target_agent_id"]
                    and abs(d["foot_contact_time_sec"] - e["time_sec"]) < 1e-9
                    for d in report.get("causal_pass_feedback", [])
                )
                for e in events
                if e["skill"] == "pass"
            )
            rows.append(
                {
                    "blue": item["blue"],
                    "offset": item["offset"],
                    "role": role,
                    "safe": report["results"][0]["safe"],
                    "assessment_pass_events": pass_events,
                    "physical_receive_diagnostics": receives,
                    "foot_only_ball_control": clean_control,
                    "qualified_passes": jointly_confirmed
                    if clean_control and report["results"][0]["safe"]
                    else 0,
                    "full_match_passed": report["assessment"]["passed"],
                    "exact_replay": report["exact_replay"],
                }
            )
        if len(rows) != len(expected_courses) or {
            (r["role"], r["blue"], r["offset"]) for r in rows
        } != {(r["role"], r["blue"], r["offset"]) for r in expected_courses}:
            raise ValueError("bilateral held-out evaluation is incomplete")
        outcomes[label] = rows
    result = {
        "training_manifest_hash": commitment,
        "verified_training_pairs": verified,
        "verified_core_updates": core_verified,
        "players": players,
        "evaluation": outcomes,
        "whole_match_breakthrough": all(
            r["full_match_passed"] and r["exact_replay"] for r in outcomes["candidate"]
        ),
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    result["audit_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    destination = args.root / "learning-audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
