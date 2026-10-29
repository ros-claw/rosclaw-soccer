"""Single SIM-only focal reception PPO update from verified physical samples."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_native_fresh_common import clean

SCOPE = ("blue.playmaker", "red.finisher")
AUTHORITY_HEAD = "fa931d053caf7d514dfae51f4cc95118f34d00c6"
PILOT_HEAD = "7fe5c019975ee544e10ef06e48125394801b68d1"


def learning_mask(trace, *, agent_ids, agent_id, trials):
    active = np.asarray(trace["residual_active"])
    explored = np.asarray(trace["residual_exploration_mask"])
    injected = np.asarray(trace["residual_pd_target_injected_substeps"])
    overridden = np.asarray(trace["residual_pd_target_override_substeps"])
    if (
        active.ndim != 2
        or active.shape != explored.shape
        or injected.shape != active.shape
        or overridden.shape != active.shape
        or active.dtype != np.bool_
        or explored.dtype != np.bool_
        or injected.dtype.kind not in "iu"
        or overridden.dtype.kind not in "iu"
        or np.any(injected < 0)
        or np.any(overridden < 0)
        or np.any(injected + overridden > 10)
        or active.shape[1] != len(agent_ids)
        or agent_id not in SCOPE
    ):
        raise ValueError("aligned role-scoped physical authority required")
    mask = np.zeros_like(active)
    col = agent_ids.index(agent_id)
    for trial in trials:
        if trial["agent_id"] != agent_id or trial["censored"]:
            continue
        start, frames = trial["start_frame"], trial["frames"]
        if type(start) is not int or frames != 100 or start < 1 or start + frames > len(active):
            raise ValueError("complete receiving-window bounds required")
        mask[start : start + frames, col] = True
    return mask & active & explored & (injected == 10) & (overridden == 0)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def same_arrays(left, right):
    return left.keys() == right.keys() and all(
        np.array_equal(left[k], right[k], equal_nan=left[k].dtype.kind in "fc") for k in left
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("authority", "pilot", "parent", "protocol", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "authority-hash",
        "authority-declaration-hash",
        "pilot-hash",
        "parent-hash",
        "protocol-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    pins = {
        a.authority / "complete.json": a.authority_hash,
        a.authority / "declaration.json": a.authority_declaration_hash,
        a.pilot / "complete.json": a.pilot_hash,
        a.parent: a.parent_hash,
        a.protocol: a.protocol_hash,
    }
    if any(path.is_symlink() or digest(path) != expected for path, expected in pins.items()):
        raise ValueError("receipt, protocol or parent hash mismatch")
    from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.near_ball_residual_ppo import (
        _validate_on_policy,
        update_private_actors,
    )
    from rosclaw_soccer.training.role_receiving_courses import ROSTER
    from rosclaw_soccer.training.team_capture_curriculum import team_capture_trials

    policy = NearBallResidualPolicy.load(a.parent)
    if policy.agent_ids != ROSTER:
        raise ValueError("parent roster mismatch")
    auth = json.loads((a.authority / "complete.json").read_text())
    pilot = json.loads((a.pilot / "complete.json").read_text())
    declaration = json.loads((a.authority / "declaration.json").read_text())
    if any(
        (
            auth.get("source_head") != AUTHORITY_HEAD,
            pilot.get("source_head") != PILOT_HEAD,
            auth.get("status") != "AIRBORNE_RECEIVING_BASELINE_SIM_ONLY",
            pilot.get("status") != "AIRBORNE_RECEIVING_BASELINE_SIM_ONLY",
            auth.get("protocol_hash")
            != "sha256:c4f214fe0023c62e6b8fa46e6aad143fd41ae51284c3116a27cf3d86b398dbe3",
            pilot.get("protocol_hash")
            != "sha256:d96ebcf82baf42aebe8b675efcc147889af7bac1b8e36b265ff6eb3a0c0998df",
            a.parent_hash
            != "sha256:5c508d0f909ef09e8c3d58695557f1e6b55caead11e0eeb0854f1cb05d9b5477",
            auth.get("standard_ball") is not True,
            pilot.get("standard_ball") is not True,
            auth.get("focal_nearest") is not True,
            pilot.get("focal_nearest") is not True,
            auth.get("focal_exploration_seed_base") != 97100,
            pilot.get("focal_exploration_seed_base") != 97100,
            auth.get("capture_residual_target_authority") is not True,
            auth.get("policy_hash") != a.parent_hash,
            pilot.get("policy_hash") != a.parent_hash,
            auth.get("course_bank_hash") != pilot.get("course_bank_hash"),
            len(auth.get("records", [])) != 4,
            len(pilot.get("records", [])) != 4,
        )
    ):
        raise ValueError("unexpected pilot provenance")
    segments, sample_counts = [], {}
    for index, agent in ((0, "red.finisher"), (4, "blue.playmaker")):
        pair, control = [], []
        for root, report, dest in ((a.authority, auth, pair), (a.pilot, pilot, control)):
            for repeat in (0, 1):
                matches = [
                    r for r in report["records"] if r["index"] == index and r["repeat"] == repeat
                ]
                if len(matches) != 1:
                    raise ValueError("exactly one source record required per condition")
                record = matches[0]
                path = root / record["artifact"]["file"]
                expected = record["artifact"]["file_hash"]
                if path.parent != root or path.is_symlink() or digest(path) != expected:
                    raise ValueError("source artifact mismatch")
                pins[path] = expected
                dest.append(load_arrays(path))
            if not same_arrays(dest[0], dest[1]):
                raise ValueError("native replay mismatch")
        original = pair[0]
        if set(original) - set(control[0]) != {
            "residual_pd_target_injected_substeps",
            "residual_pd_target_override_substeps",
        } or any(
            not np.array_equal(
                original[k], control[0][k], equal_nan=control[0][k].dtype.kind in "fc"
            )
            for k in control[0]
        ):
            raise ValueError("telemetry changed physical trajectory")
        course = declaration["courses"][index]
        if (
            course["agent_id"] != agent
            or original["initial_integration_hash"][0]
            != declaration["initial_integration_hashes"][str(course["seed"])]
            or original["initial_physics_hash"][0] != declaration["compiled_model_hash"]
        ):
            raise ValueError("course/model/state mismatch")
        n = len(original["time"])
        trace = {k: v for k, v in original.items() if v.ndim and len(v) == n}
        _validate_on_policy(policy, trace)
        mask = trace["residual_exploration_mask"]
        focal = ROSTER.index(agent)
        if not mask[:, focal].all() or np.delete(mask, focal, axis=1).any():
            raise ValueError("exploration scope does not match focal receiver")
        _, trials = team_capture_trials(trace, agent_ids=ROSTER)
        record = next(r for r in auth["records"] if r["index"] == index and r["repeat"] == 0)
        if hash_json(trials) != hash_json(record["receiving_trials"]):
            raise ValueError("receiving labels mismatch")
        trace["residual_learning_mask"] = learning_mask(
            trace, agent_ids=ROSTER, agent_id=agent, trials=trials
        )
        sample_counts[agent] = int(trace["residual_learning_mask"][:, ROSTER.index(agent)].sum())
        segments.append(trace)
    a.output.mkdir(parents=True)
    try:
        if max(sample_counts.values()) < 32:
            write_once(
                a.output / "complete.json",
                {
                    "status": "NO_ELIGIBLE_PPO_UPDATE",
                    "samples": sample_counts,
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
            reward_shaping="team_capture_v1",
            trainable_agent_ids=SCOPE,
        )
        if not any(row["updated"] for row in rows):
            raise ValueError("eligible samples produced no update")
        candidate.save(a.output / "candidate.npz")
        if any(digest(path) != expected for path, expected in pins.items()):
            raise ValueError("inputs changed during update")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            {
                "status": "PPO_UPDATED_NOT_EVALUATED",
                "samples": sample_counts,
                "source_head": a.expected_head,
                "parent_hash": a.parent_hash,
                "protocol_hash": a.protocol_hash,
                "authority_hash": a.authority_hash,
                "pilot_hash": a.pilot_hash,
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
