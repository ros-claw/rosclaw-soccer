"""Evidence-bound returned-live residual PPO pilot, never policy promotion."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_mjlab_stage1_training_smoke import file_digest
from scripts.rsi_native_fresh_common import clean

SCOPE = ("blue.playmaker", "red.finisher")
SEEDS = (24001, 24002)


def eligible_mask(segment, agent_ids, trials):
    active = np.asarray(segment["residual_active"])
    explored = np.asarray(segment["residual_exploration_mask"])
    if (
        active.ndim != 2
        or active.shape != explored.shape
        or active.dtype != np.bool_
        or explored.dtype != np.bool_
        or active.shape[1] != len(agent_ids)
    ):
        raise ValueError("actual aligned activation and exploration masks required")
    mask = np.zeros_like(active)
    for trial in trials:
        if trial["agent_id"] not in SCOPE or trial["censored"] or trial["frames"] != 100:
            continue
        start = trial["start_frame"]
        if type(start) is not int or start < 1 or start + 100 > len(active):
            raise ValueError("complete in-segment trial bounds required")
        column = agent_ids.index(trial["agent_id"])
        mask[start : start + 100, column] = True
    return mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True)
    p.add_argument("--source-hash", action="append", required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--parent-hash", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    if (
        len(a.source) != 2
        or len(a.source_hash) != 2
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("two sources and unused external output required")
    if file_digest(a.parent) != a.parent_hash or file_digest(a.protocol) != a.protocol_hash:
        raise ValueError("exact parent and protocol hashes required")
    from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
    from rosclaw_soccer.training.near_ball_residual_ppo import (
        _validate_on_policy,
        update_private_actors,
    )
    from rosclaw_soccer.training.returned_ball_learning import returned_ball_segments
    from rosclaw_soccer.training.team_capture_curriculum import team_capture_trials

    policy = NearBallResidualPolicy.load(a.parent)
    segments, source_rows, artifact_pins = [], [], []
    for seed, root, expected in zip(SEEDS, a.source, a.source_hash, strict=True):
        receipt = root / "complete.json"
        if receipt.is_symlink() or file_digest(receipt) != expected:
            raise ValueError("exact source receipt required")
        report = json.loads(receipt.read_text())
        if (
            report.get("source_head") != "cded304098c4330aaee611ebb438a111cf18f947"
            or report.get("exploration_seed") != seed
            or report.get("only_explore_after_return") is not True
            or report.get("policy_hash") != a.parent_hash
            or report.get("exact_replay") is not True
            or report.get("external_ball_state_write") is not True
        ):
            raise ValueError("unexpected source policy, seed, lifecycle or replay")
        traces = []
        for label, artifact in zip(("primary", "replay"), report["artifacts"], strict=True):
            path = root / f"{label}.npz"
            if (
                artifact["file"] != path.name
                or path.is_symlink()
                or file_digest(path) != artifact["file_hash"]
            ):
                raise ValueError("exact local trajectory artifact required")
            artifact_pins.append((path, artifact["file_hash"]))
            with np.load(path, allow_pickle=False) as archive:
                traces.append({k: archive[k] for k in archive.files})
        original, replay = traces
        if original.keys() != replay.keys() or any(
            not np.array_equal(original[k], replay[k], equal_nan=True) for k in original
        ):
            raise ValueError("whole trajectory replay mismatch")
        _validate_on_policy(policy, original)
        code, mask = original["training_return_event_code"], original["residual_exploration_mask"]
        live = False
        for frame, event in enumerate(code):
            if event in (1, 2):
                live = False
            elif event == 3:
                live = True
            if mask[frame].any() != live or any(
                mask[frame, i] != (live and agent in SCOPE)
                for i, agent in enumerate(policy.agent_ids)
            ):
                raise ValueError(
                    "exploration must match observed returned-live lifecycle and scope"
                )
        aligned = {k: v for k, v in original.items() if v.ndim and len(v) == len(code)}
        parts = returned_ball_segments(aligned)
        all_trials, counts = [], np.zeros(8, dtype=np.int64)
        for part in parts:
            _, trials = team_capture_trials(part, agent_ids=policy.agent_ids)
            part["residual_learning_mask"] = eligible_mask(part, policy.agent_ids, trials)
            counts += (
                part["residual_active"]
                & part["residual_exploration_mask"]
                & part["residual_learning_mask"]
            ).sum(axis=0)
            all_trials.extend(trials)
            segments.append(part)
        source_rows.append(
            dict(
                seed=seed,
                receipt_hash=expected,
                live_segment_lengths=[len(part["time"]) for part in parts],
                eligible_samples=counts.tolist(),
                trials=all_trials,
            )
        )
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        total = np.sum([r["eligible_samples"] for r in source_rows], axis=0)
        if not any(total[policy.agent_ids.index(agent)] >= 32 for agent in SCOPE):
            if any(file_digest(path) != expected for path, expected in artifact_pins):
                raise ValueError("source trajectory changed during audit")
            clean(a.expected_head)
            write_once(
                a.output / "complete.json",
                dict(
                    status="NO_ELIGIBLE_PPO_UPDATE",
                    source_head=a.expected_head,
                    parent_hash=a.parent_hash,
                    protocol_hash=a.protocol_hash,
                    sources=source_rows,
                    eligible_samples=total.tolist(),
                    optimizer_steps=0,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
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
            raise ValueError("admitted samples produced no actual actor update")
        path = a.output / "candidate.npz"
        candidate.save(path)
        if (
            file_digest(a.parent) != a.parent_hash
            or file_digest(a.protocol) != a.protocol_hash
            or any(
                file_digest(root / "complete.json") != h
                for root, h in zip(a.source, a.source_hash, strict=True)
            )
            or any(file_digest(source) != expected for source, expected in artifact_pins)
        ):
            raise ValueError("input changed during update")
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="RETURN_LIVE_PPO_UPDATED_NOT_EVALUATED",
                source_head=a.expected_head,
                parent_hash=a.parent_hash,
                protocol_hash=a.protocol_hash,
                sources=source_rows,
                eligible_samples=total.tolist(),
                rows=rows,
                candidate_hash=file_digest(path),
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(error_type=type(error).__name__, error=str(error), hardware_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
