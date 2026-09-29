"""SIM_ONLY advantage-weighted imitation of successful physical contact rollouts."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsi_r1_adaptive_phase_ppo_v229 import _run_task
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_kinematic_advantage_v201 import KinematicActor
from rsi_r1_kinematic_online_ppo_v204 import _save
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_post_contact_awr_v230_helpers import load_success_trajectories
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_post_contact_awr_v230.result.v1"
SEED = 230929
ALPHAS = (0.1, 0.5, 2.0)
HOLDOUT_SEEDS = (223024, 223031)


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v205_dir: Path,
    map_report_path: Path,
    v214_report_path: Path,
    v215_dir: Path,
    v216_report_path: Path,
    v218_dir: Path,
    v220_report_path: Path,
    v223_report_path: Path,
    v224_report_path: Path,
    v229_dir: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY AWR evidence required")
    v223, v224, v229 = (
        _checked(path) for path in (v223_report_path, v224_report_path, v229_dir / "report.json")
    )
    if (
        v229["status"] != "REJECTED_ADAPTIVE_PHASE_PPO_GATE"
        or v229["best_update"] != 1
        or v229["baseline_score"][:2] != [8, 30]
        or v224["v223_report_hash"] != v223["report_hash"]
    ):
        raise ValueError("sealed rejected online post-contact learning required")
    common, (base_weights, _, _), lineage = context(
        asset_root,
        policy_path,
        parent_report,
        right_report,
        refine_report,
        lateral_report,
        v205_dir,
        map_report_path,
        v214_report_path,
        v215_dir,
        v216_report_path,
        v218_dir,
        v220_report_path,
    )
    if lineage["v220_report_hash"] != v224["v220_report_hash"]:
        raise ValueError("sealed precontact lineage required")
    pre_weights = replace(
        base_weights,
        output_bias=tuple(
            float(value)
            for value in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    start = _load_policy(v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    train_states, train_actions, val_states, val_actions, manifest = load_success_trajectories(
        v229_dir, v229, HOLDOUT_SEEDS
    )
    if train_states.shape[0] < 150 or val_states.shape[0] < 60:
        raise ValueError("enough independent successful physical contact frames required")
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    x_train = torch.tensor(train_states, dtype=torch.float32)
    y_train = torch.tensor(train_actions, dtype=torch.float32)
    x_val = torch.tensor(val_states, dtype=torch.float32)
    y_val = torch.tensor(val_actions, dtype=torch.float32)
    reference = KinematicActor(start)
    with torch.no_grad():
        ref_train = reference(x_train)
    output.mkdir(parents=True)
    offline = []
    for index, alpha in enumerate(ALPHAS):
        torch.manual_seed(SEED + index)
        actor = KinematicActor(start)
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
        for _ in range(160):
            predicted = actor(x_train)
            loss = torch.nn.functional.mse_loss(predicted, y_train)
            loss += alpha * torch.nn.functional.mse_loss(predicted, ref_train)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()
        with torch.no_grad():
            train_error = float(torch.nn.functional.mse_loss(actor(x_train), y_train))
            val_error = float(torch.nn.functional.mse_loss(actor(x_val), y_val))
        weights = actor.weights()
        checkpoint_hash = _save(output / f"alpha-{index}.npz", weights)
        offline.append(
            {
                "alpha": alpha,
                "train_error": train_error,
                "validation_error": val_error,
                "checkpoint_hash": checkpoint_hash,
            }
        )
    selected = min(offline, key=lambda row: (row["validation_error"], row["alpha"]))
    selected_index = ALPHAS.index(selected["alpha"])
    candidate_weights = _load_policy(
        output / f"alpha-{selected_index}.npz", selected["checkpoint_hash"]
    )
    old_courses = tuple(ReceivingCourse(**row["course"]) for row in v223["rows"]["baseline"])

    def tasks(weights: Any) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], pre_weights, weights, 0, 0.0)
            for course in old_courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_post_contact_awr_v230.py",
            "scripts/rsi_r1_post_contact_awr_v230_helpers.py",
            "scripts/rsi_r1_adaptive_phase_ppo_v229.py",
            "src/rosclaw_soccer/rsi/receiving_adaptive_phase_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        parent_rows = list(pool.map(_run_task, tasks(start)))
        if any(
            row["physical_trace_hash"] != old["physical_trace_hash"]
            for row, old in zip(parent_rows, v229["history"][0]["deterministic_rows"], strict=True)
        ):
            raise ValueError("AWR parent must physically replay sealed online actor")
        candidate_rows = list(pool.map(_run_task, tasks(candidate_weights)))
    parent_score = _score(parent_rows)
    candidate_score = _score(candidate_rows)
    retained = all(
        not (clean(old) and old["controlled_reception"])
        or (clean(new) and new["controlled_reception"])
        for old, new in zip(parent_rows, candidate_rows, strict=True)
    )
    qualified = (
        retained
        and candidate_score[0] >= parent_score[0] + 2
        and candidate_score[1] >= parent_score[1]
    )
    report = {
        "schema": SCHEMA,
        "v229_report_hash": v229["report_hash"],
        "source_hashes": sources,
        "partition": "SEALED_SUCCESS_TRAJECTORIES_HELDOUT_SEEDS_THEN_CONSUMED_PHYSICS",
        "holdout_seeds": HOLDOUT_SEEDS,
        "trajectory_manifest": manifest,
        "train_frames": int(train_states.shape[0]),
        "holdout_frames": int(val_states.shape[0]),
        "offline": offline,
        "selected_alpha": selected["alpha"],
        "parent_rows": parent_rows,
        "candidate_rows": candidate_rows,
        "parent_score": parent_score,
        "candidate_score": candidate_score,
        "old_success_retained": retained,
        "status": "DEVELOPMENT_POST_CONTACT_AWR_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_POST_CONTACT_AWR_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during post-contact AWR")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "v205-dir",
        "map-report",
        "v214-report",
        "v215-dir",
        "v216-report",
        "v218-dir",
        "v220-report",
        "v223-report",
        "v224-report",
        "v229-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v205_dir,
        args.map_report,
        args.v214_report,
        args.v215_dir,
        args.v216_report,
        args.v218_dir,
        args.v220_report,
        args.v223_report,
        args.v224_report,
        args.v229_dir,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "parent_score",
                    "candidate_score",
                    "selected_alpha",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
