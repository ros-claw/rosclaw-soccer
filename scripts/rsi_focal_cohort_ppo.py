"""Evidence-bound SIM-only PPO on complete focal receiving windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_focal_authority_ppo_pilot import SCOPE, load_arrays, same_arrays
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean


def complete_window_segments(trace, *, agent_ids, agent_id, trials):
    """Exclude incomplete windows; cut GAE at each complete physical window."""
    if agent_id not in SCOPE or agent_id not in agent_ids:
        raise ValueError("focal training scope required")
    n = len(trace["time"])
    column = agent_ids.index(agent_id)
    segments = []
    for trial in trials:
        if trial["agent_id"] != agent_id or trial["censored"]:
            continue
        start, frames = trial["start_frame"], trial["frames"]
        if type(start) is not int or start < 1 or frames != 100 or start + frames > n:
            raise ValueError("complete focal 100-frame window required")
        segment = {
            key: value[start - 1 : start + frames].copy()
            for key, value in trace.items()
            if np.ndim(value) and len(value) == n
        }
        active = segment["residual_active"]
        explore = segment["residual_exploration_mask"]
        injected = segment["residual_pd_target_injected_substeps"]
        overridden = segment["residual_pd_target_override_substeps"]
        if (
            active.shape != (101, len(agent_ids))
            or explore.shape != active.shape
            or injected.shape != active.shape
            or overridden.shape != active.shape
            or active.dtype != np.bool_
            or explore.dtype != np.bool_
            or injected.dtype.kind not in "iu"
            or overridden.dtype.kind not in "iu"
            or np.any(injected < 0)
            or np.any(overridden < 0)
            or np.any(injected + overridden > 10)
        ):
            raise ValueError("invalid frame-aligned physical authority trace")
        mask = np.zeros_like(active)
        mask[1:, column] = True
        segment["residual_learning_mask"] = (
            mask & active & explore & (injected == 10) & (overridden == 0)
        )
        segments.append(segment)
    return segments


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--source-hash", action="append", required=True)
    p.add_argument("--declaration-hash", action="append", required=True)
    for key in ("parent", "protocol", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("parent-hash", "protocol-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--reward-shaping",
        choices=("team_capture_v1", "team_capture_dissipation_v1"),
        default="team_capture_v1",
    )
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.source) != 4
        or len(a.source_hash) != 4
        or len(a.declaration_hash) != 4
        or len(set(a.source)) != 4
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("four distinct sources and unused external output required")
    pins = {a.parent: a.parent_hash, a.protocol: a.protocol_hash}
    for root, source_hash, declaration_hash in zip(
        a.source, a.source_hash, a.declaration_hash, strict=True
    ):
        pins[root / "complete.json"] = source_hash
        pins[root / "declaration.json"] = declaration_hash
    if any(path.is_symlink() or digest(path) != expected for path, expected in pins.items()):
        raise ValueError("source, parent or protocol integrity mismatch")
    from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.near_ball_residual_ppo import (
        _validate_on_policy,
        update_private_actors,
    )
    from rosclaw_soccer.training.role_receiving_courses import ROSTER
    from rosclaw_soccer.training.team_capture_curriculum import team_capture_trials

    if a.parent_hash != "sha256:5c508d0f909ef09e8c3d58695557f1e6b55caead11e0eeb0854f1cb05d9b5477":
        raise ValueError("frozen parent required")
    policy = NearBallResidualPolicy.load(a.parent)
    if policy.agent_ids != ROSTER:
        raise ValueError("policy roster mismatch")
    segments, source_rows = [], []
    for shard, root in enumerate(a.source):
        report = json.loads((root / "complete.json").read_text())
        decl = json.loads((root / "declaration.json").read_text())
        if (
            report.get("status") != "AIRBORNE_RECEIVING_BASELINE_SIM_ONLY"
            or report.get("source_head") != "fa931d053caf7d514dfae51f4cc95118f34d00c6"
            or report.get("protocol_hash")
            != "sha256:554633b2320f523afc252d47b2ce79fcfffa7e3bb3abb46cee107bc5217af0c5"
            or report.get("policy_hash") != a.parent_hash
            or report.get("focal_exploration_seed_base") != 97200
            or report.get("shard") != shard
            or report.get("standard_ball") is not True
            or report.get("focal_nearest") is not True
            or report.get("capture_residual_target_authority") is not True
            or report.get("course_bank_hash") != decl.get("course_bank_hash")
            or report.get("course_bank_hash")
            != "sha256:cf805e207529f64707200810d9ca452e170fec1e847252e537f78c05b1be9221"
            or decl.get("compiled_model_hash")
            != "sha256:476d26071c157eac895a6312801d0d4e5d42d48f3277af517f7b803d1c9f6c8c"
            or decl.get("fixture_hash")
            != "sha256:32d012e49cb5cab6b6d25a4eb1dcf3ab5c9f094adeb7222e66efa78f7f2de7a4"
            or len(report.get("records", [])) != 4
        ):
            raise ValueError("unexpected collection provenance")
        for index in (shard, shard + 4):
            course = decl["courses"][index]
            if course["seed"] != 96400 + index or course["agent_id"] != (
                "red.finisher" if index < 4 else "blue.playmaker"
            ):
                raise ValueError("unexpected frozen course identity")
            pair = []
            for repeat in (0, 1):
                matches = [
                    row
                    for row in report["records"]
                    if row["index"] == index and row["repeat"] == repeat
                ]
                if len(matches) != 1:
                    raise ValueError("one original and one replay required")
                row = matches[0]
                artifact = root / row["artifact"]["file"]
                expected = row["artifact"]["file_hash"]
                if artifact.parent != root or artifact.is_symlink() or digest(artifact) != expected:
                    raise ValueError("source trajectory integrity mismatch")
                pins[artifact] = expected
                trace = load_arrays(artifact)
                if (
                    trace["initial_integration_hash"][0]
                    != decl["initial_integration_hashes"][str(course["seed"])]
                    or trace["initial_physics_hash"][0] != decl["compiled_model_hash"]
                ):
                    raise ValueError("physical initial state or model mismatch")
                n = len(trace["time"])
                aligned = {k: v for k, v in trace.items() if v.ndim and len(v) == n}
                _validate_on_policy(policy, aligned)
                focal = ROSTER.index(course["agent_id"])
                mask = aligned["residual_exploration_mask"]
                if not mask[:, focal].all() or np.delete(mask, focal, axis=1).any():
                    raise ValueError("wrong exploratory agent")
                _, trials = team_capture_trials(aligned, agent_ids=ROSTER)
                if hash_json(trials) != hash_json(row["receiving_trials"]):
                    raise ValueError("trial labels differ from measured trajectory")
                pair.append(trace)
                if repeat == 0:
                    extracted = complete_window_segments(
                        aligned, agent_ids=ROSTER, agent_id=course["agent_id"], trials=trials
                    )
                    for segment in extracted:
                        _validate_on_policy(policy, segment)
                    segments.extend(extracted)
                    focal_trials = [
                        trial for trial in trials if trial["agent_id"] == course["agent_id"]
                    ]
                    source_rows.append(
                        {
                            "shard": shard,
                            "index": index,
                            "agent": course["agent_id"],
                            "windows": len(focal_trials),
                            "censored": sum(t["censored"] for t in focal_trials),
                            "complete_windows": len(extracted),
                            "eligible_samples": sum(
                                int(s["residual_learning_mask"][:, focal].sum()) for s in extracted
                            ),
                        }
                    )
            if not same_arrays(pair[0], pair[1]):
                raise ValueError("fresh physics replay mismatch")
    counts = {
        agent: sum(row["eligible_samples"] for row in source_rows if row["agent"] == agent)
        for agent in SCOPE
    }
    a.output.mkdir(parents=True)
    try:
        if not segments or max(counts.values()) < 32:
            if any(digest(path) != expected for path, expected in pins.items()):
                raise ValueError("source changed during admission")
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                {
                    "status": "NO_ELIGIBLE_PPO_UPDATE",
                    "samples": counts,
                    "sources": source_rows,
                    "promotion_authorized": False,
                    "hardware_authorized": False,
                },
            )
            return
        candidate, rows = update_private_actors(
            policy,
            segments,
            epochs=4,
            gamma=0.99,
            trace_decay=0.95,
            reward_shaping=a.reward_shaping,
            trainable_agent_ids=SCOPE,
        )
        if not any(row["updated"] for row in rows):
            raise ValueError("eligible samples produced no update")
        candidate.save(a.output / "candidate.npz")
        if any(digest(path) != expected for path, expected in pins.items()):
            raise ValueError("source changed during optimization")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            {
                "status": "PPO_UPDATED_NOT_EVALUATED",
                "samples": counts,
                "sources": source_rows,
                "source_head": a.expected_head,
                "parent_hash": a.parent_hash,
                "protocol_hash": a.protocol_hash,
                "reward_shaping": a.reward_shaping,
                "source_hashes": a.source_hash,
                "declaration_hashes": a.declaration_hash,
                "rows": rows,
                "candidate_hash": digest(a.output / "candidate.npz"),
                "promotion_authorized": False,
                "hardware_authorized": False,
            },
        )
    except Exception as error:
        write_once(
            a.output / "failed.json", {"error_type": type(error).__name__, "error": str(error)}
        )
        raise


if __name__ == "__main__":
    main()
