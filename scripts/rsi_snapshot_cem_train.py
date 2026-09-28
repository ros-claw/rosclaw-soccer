"""Resume-safe SIM_ONLY CEM search over a shared G1 first-contact motor actor.

This consumes only an audited closed-loop Isaac snapshot gym. Candidate scores
are development data; a winner still needs untouched-course and full-episode
physics, and this script never promotes a skill or opens a robot interface.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.rsi.snapshot_shared_temporal_policy import candidate_manifest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SEARCH_COORDINATES = tuple((feature, joint) for feature in (0, 1, 4) for joint in range(3))


def score_replay(trace_path: Path, *, max_lanes: int | None = None) -> dict[str, float | int]:
    with np.load(trace_path, allow_pickle=False) as trace:
        force = trace["observed_ball_body_contact_force_peak_n"][:, :max_lanes]
        ball = trace["observed_ball_position_local_m"][:, :max_lanes]
        root = trace["observed_root_pose_local_xyzw_m"][:, :max_lanes]
        if (
            force.ndim != 3
            or force.shape[2] != 6
            or ball.shape != (force.shape[0], force.shape[1], 3)
            or root.shape != (force.shape[0], force.shape[1], 7)
            or not np.isfinite(force).all()
            or not np.isfinite(ball).all()
            or not np.isfinite(root).all()
            or np.any(force < 0)
        ):
            raise ValueError("invalid physical contact score trace")
        reward = []
        clean_count = 0
        first_foot_count = 0
        for lane in range(force.shape[1]):
            active = np.flatnonzero(np.max(force[:, lane], axis=1) > 1.0)
            if len(active) == 0:
                reward.append(-2.0)
                continue
            first = int(active[0])
            first_bodies = set(np.flatnonzero(force[first, lane] > 1.0).tolist())
            all_bodies = set(np.flatnonzero(np.max(force[:, lane], axis=0) > 1.0).tolist())
            first_foot = bool(first_bodies and first_bodies <= {0, 1})
            clean = bool(all_bodies and all_bodies <= {0, 1})
            first_foot_count += first_foot
            clean_count += clean
            later = min(first + 6, force.shape[0] - 1)
            if later == first:
                reward.append(-2.0)
                continue
            speed = float((ball[later, lane, 0] - ball[first, lane, 0]) / ((later - first) * 0.02))
            reward.append(
                float(clean)
                + (0.5 if first_foot else -0.5)
                + 0.2 * float(np.clip(speed, -1.0, 3.0)) / 3.0
            )
        minimum_root_height = float(np.min(root[:, :, 2]))
        if minimum_root_height < 0.65:
            raise ValueError("unsafe G1 root height in contact candidate")
        return {
            "mean_reward": float(np.mean(reward)),
            "clean_foot_count": clean_count,
            "first_foot_count": first_foot_count,
            "sample_count": force.shape[1],
            "minimum_root_height_m": minimum_root_height,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-bank", required=True, type=Path)
    parser.add_argument("--baseline-replay", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--isaac-venv-python", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--sample-count", type=int, default=15)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--max-new-evaluations", type=int)
    args = parser.parse_args()
    if (
        not 2 <= args.population <= 16
        or not 1 <= args.generations <= 20
        or not 2 <= args.sample_count <= 15
        or not 0 <= args.seed < 2**31
        or (args.max_new_evaluations is not None and args.max_new_evaluations < 1)
        or args.cuda_visible_devices not in {"0", "1", "2", "3"}
        or not args.isaac_python.is_file()
        or not args.isaac_venv_python.is_file()
        or not args.g1_usd.is_file()
        or not args.model_root.is_dir()
    ):
        parser.error("bounded SIM_ONLY CEM configuration and local Isaac assets required")
    bank_audit = audit_snapshot_bank(args.snapshot_bank)
    bank = json.loads((args.snapshot_bank / "manifest.json").read_text(encoding="utf-8"))
    base_audit = audit_snapshot_replay(args.baseline_replay, snapshot_bank=args.snapshot_bank)
    base_report = json.loads((args.baseline_replay / "report.json").read_text(encoding="utf-8"))
    if (
        bank.get("fixed_start_frame") != 30
        or args.sample_count > base_report["sample_count"]
        or base_report["start_index"] != 0
        or not base_audit["closed_loop_sonic"]
        or base_audit["intervention_action_audited"]
    ):
        raise ValueError("CEM requires audited unmodified closed-loop development Parent")
    baseline = score_replay(args.baseline_replay / "replay.npz", max_lanes=args.sample_count)
    plan = {
        "schema": "rsi_snapshot_shared_temporal_cem_plan_v1",
        "activation_ceiling": "SIM_ONLY",
        "snapshot_bank_manifest_hash": bank_audit["manifest_hash"],
        "baseline_audit_hash": base_audit["report_hash"],
        "training_selection": {"start_index": 0, "sample_count": args.sample_count},
        "population": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "search_coordinates": [list(item) for item in SEARCH_COORDINATES],
        "runner_source_hash": hash_bytes(
            Path(__file__).with_name("rsi_isaac_snapshot_replay.py").read_bytes()
        ),
        "trainer_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "baseline_score": baseline,
        "learning_authorized": False,
        "promotion_authorized": False,
    }
    plan["plan_hash"] = hash_json(plan)
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("CEM resume plan changed")
    else:
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mean = np.zeros(len(SEARCH_COORDINATES))
    sigma = 0.55
    new_evaluations = 0
    for generation in range(args.generations):
        scores = []
        for member in range(args.population):
            sample = np.clip(
                mean
                + np.random.default_rng(args.seed + generation * 1000 + member).normal(
                    0, sigma, len(SEARCH_COORDINATES)
                ),
                -1.0,
                1.0,
            )
            weights = np.zeros((6, 3))
            for value, (feature, joint) in zip(sample, SEARCH_COORDINATES, strict=True):
                weights[feature, joint] = value
            manifest = candidate_manifest(
                weights, bank_manifest_hash=bank_audit["manifest_hash"], seed=args.seed
            )
            member_root = args.output_root / f"g{generation:02d}-m{member:02d}"
            member_root.mkdir(exist_ok=True)
            candidate_path = member_root / "candidate.json"
            if candidate_path.exists():
                if json.loads(candidate_path.read_text(encoding="utf-8")) != manifest:
                    raise ValueError("CEM candidate changed across resume")
            else:
                candidate_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            replay_dir = member_root / "replay"
            if not replay_dir.exists():
                if (
                    args.max_new_evaluations is not None
                    and new_evaluations >= args.max_new_evaluations
                ):
                    print(
                        json.dumps(
                            {"state": "PAUSED_AFTER_LIMIT", "new_evaluations": new_evaluations}
                        )
                    )
                    return
                env = os.environ.copy()
                env["PYTHONEXE"] = str(args.isaac_venv_python)
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
                command = [
                    str(args.isaac_python),
                    str(Path(__file__).with_name("rsi_isaac_snapshot_replay.py")),
                    "--snapshot-bank",
                    str(args.snapshot_bank),
                    "--g1-usd",
                    str(args.g1_usd),
                    "--model-root",
                    str(args.model_root),
                    "--output-dir",
                    str(replay_dir),
                    "--sample-count",
                    str(args.sample_count),
                    "--start-index",
                    "0",
                    "--closed-loop-sonic",
                    "--shared-candidate",
                    str(candidate_path),
                    "--device",
                    "cuda:0",
                    "--headless",
                ]
                with (member_root / "isaac.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=900,
                        check=False,
                    )
                if completed.returncode != 0 or not (replay_dir / "report.json").is_file():
                    raise RuntimeError(f"CEM Isaac candidate failed: {member_root}")
                new_evaluations += 1
            audit = audit_snapshot_replay(
                replay_dir, snapshot_bank=args.snapshot_bank, candidate_path=candidate_path
            )
            if not audit["intervention_action_audited"] or not audit["closed_loop_sonic"]:
                raise ValueError("CEM action or closed-loop evidence missing")
            score = score_replay(replay_dir / "replay.npz")
            score_record = {
                "schema": "rsi_snapshot_shared_temporal_cem_score_v1",
                "plan_hash": plan["plan_hash"],
                "candidate_hash": manifest["candidate_hash"],
                "audit_hash": audit["report_hash"],
                "score": score,
                "promotion_authorized": False,
            }
            score_record["score_hash"] = hash_json(score_record)
            score_path = member_root / "score.json"
            if (
                score_path.exists()
                and json.loads(score_path.read_text(encoding="utf-8")) != score_record
            ):
                raise ValueError("CEM score changed across resume")
            if not score_path.exists():
                score_path.write_text(
                    json.dumps(score_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            scores.append((float(score["mean_reward"]), sample, score_record))
            print(
                json.dumps(
                    {
                        "generation": generation,
                        "member": member,
                        "score": score,
                        "candidate_hash": manifest["candidate_hash"],
                    }
                ),
                flush=True,
            )
        elite = sorted(scores, key=lambda row: row[0], reverse=True)[: max(1, args.population // 2)]
        mean = np.mean([row[1] for row in elite], axis=0)
        sigma = max(0.08, sigma * 0.7)
        summary = {
            "schema": "rsi_snapshot_shared_temporal_cem_generation_v1",
            "plan_hash": plan["plan_hash"],
            "generation": generation,
            "elite_candidate_hashes": [row[2]["candidate_hash"] for row in elite],
            "elite_mean_reward": float(np.mean([row[0] for row in elite])),
            "next_mean": mean.tolist(),
            "next_sigma": sigma,
            "promotion_authorized": False,
        }
        summary["summary_hash"] = hash_json(summary)
        summary_path = args.output_root / f"generation-{generation:02d}.json"
        if (
            summary_path.exists()
            and json.loads(summary_path.read_text(encoding="utf-8")) != summary
        ):
            raise ValueError("CEM generation changed across resume")
        if not summary_path.exists():
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
