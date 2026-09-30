"""Frozen learned bilateral motor, twelve consumed retention/risk courses.

No fitting during validation, no future-outcome policy choice, no promotion.
The Isaac runner stays on its original immutable v303 source snapshot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.contact_motor_primitive import load_policy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

TRAIN_REPORT_HASH = "sha256:547fffb4a192a6b4ccfe96a287a1040129780a397322e460e0cfb7dc0c170fb5"
POLICY_HASH = "sha256:d170eaa06dd803ca4849d914dc01333477cd1786ec34a45897b40e7f021b734a"
COURSES = (
    (20261177, 0),
    (20261227, 4),
    (20261282, 0),
    (20261360, 0),
    (20261378, 0),
    (20261440, 2),
    (20261446, 0),
    (20261095, 2),
    (20261097, 2),
    (20261146, 4),
    (20261148, 2),
    (20260975, 4),
)


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = len(rows) == 12 and {(r["seed"], r["lane"]) for r in rows} == set(COURSES)
    high = {
        name: sum(r["arms"][name]["high_quality"] for r in rows)
        for name in ("gain_08", "gain_12", "learned")
    }
    old_high_loss = sum(
        r["arms"]["gain_08"]["high_quality"] and not r["arms"]["learned"]["high_quality"]
        for r in rows
    )
    gain_loss = sum(
        r["arms"]["gain_12"]["high_quality"] and not r["arms"]["learned"]["high_quality"]
        for r in rows
        if (r["seed"], r["lane"]) in COURSES[7:]
    )
    foot_loss = sum(
        r["arms"]["gain_08"]["clean_foot_only"] and not r["arms"]["learned"]["clean_foot_only"]
        for r in rows
    )
    new_out = sum(
        r["arms"]["gain_08"]["maximum_lateral_excursion_m"]
        <= 4
        < r["arms"]["learned"]["maximum_lateral_excursion_m"]
        for r in rows
    )
    safe = all(r["arms"]["learned"]["minimum_pelvis_z_m"] >= 0.65 for r in rows)
    passed = bool(
        complete
        and safe
        and old_high_loss == 0
        and gain_loss == 0
        and foot_loss == 0
        and new_out == 0
        and high["learned"] >= max(high["gain_08"], high["gain_12"]) + 2
    )
    return {
        "complete": complete,
        "high_quality_count": high,
        "old_high_quality_loss": old_high_loss,
        "raw_gain_case_loss": gain_loss,
        "clean_foot_loss": foot_loss,
        "new_out_of_play": new_out,
        "safe": safe,
        "consumed_validation_gate_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "runner",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "motor-policy",
        "training-summary",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    collector_root = Path(__file__).resolve().parent.parent
    collector_hash = hash_bytes(Path(__file__).read_bytes())
    collector_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=collector_root, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=collector_root, text=True
    ).strip():
        parser.error("clean frozen collector source required")
    training = json.loads(args.training_summary.read_text())
    policy, _ = load_policy(args.motor_policy)
    protocol = json.loads(
        (collector_root / "docs/rsi/protocols/bilateral-motor-validation-v305.json").read_text()
    )
    if (
        args.output_root.exists()
        or training.get("report_hash") != TRAIN_REPORT_HASH
        or training.get("report_hash")
        != hash_json({k: v for k, v in training.items() if k != "report_hash"})
        or training.get("consumed_development_gate_passed") is not True
        or policy["policy_hash"] != POLICY_HASH
        or policy["training_commitment"] != hash_json(training["commitment"])
        or protocol["courses"] != [list(c) for c in COURSES]
        or hash_bytes(args.runner.read_bytes()) != training["commitment"]["runner_hash"]
    ):
        parser.error("sealed learned model, completed training and original runner required")
    runner_root = args.runner.resolve().parent.parent
    if (
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=runner_root, text=True
        ).strip()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runner_root, text=True).strip()
        != training["commitment"]["source_commit"]
    ):
        raise ValueError("Isaac runner snapshot is not frozen")
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    with (args.output_root / "commitment.json").open("x") as stream:
        json.dump(
            {
                "collector_commit": collector_commit,
                "collector_hash": collector_hash,
                "runner_commit": training["commitment"]["source_commit"],
                "runner_hash": training["commitment"]["runner_hash"],
                "protocol_hash": hash_json(protocol),
                "motor_policy_hash": POLICY_HASH,
                "training_report_hash": TRAIN_REPORT_HASH,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

    def worker(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for index in range(gpu, len(COURSES), 4):
            seed, lane = COURSES[index]
            arms = {}
            for arm, gain in (("gain_08", 0.8), ("gain_12", 1.2), ("learned", 1.2)):
                common = dict(
                    root=args.output_root,
                    runner=args.runner,
                    isaac_python=args.isaac_python,
                    g1_usd=args.g1_usd,
                    model_root=args.model_root,
                    actor=args.late_swing_policy,
                    seed=seed,
                    lane=lane,
                    gpu=gpu,
                    arm=arm,
                    gain=gain,
                    negative_only=True,
                )
                if arm != "learned":
                    parent, _ = _run(**common, kind="parent")
                    parent_path = (
                        args.output_root / f"seed{seed}-lane{lane}-{arm}-parent/report.json"
                    )
                else:
                    parent_path = (
                        args.output_root / f"seed{seed}-lane{lane}-gain_12-parent/report.json"
                    )
                    parent = json.loads(parent_path.read_text())
                report, outcome = _run(
                    **common,
                    kind="actor",
                    motor_policy=args.motor_policy if arm == "learned" else None,
                    parent_report_override=parent_path,
                )
                if (
                    report["source_hash"] != training["commitment"]["runner_hash"]
                    or report["late_swing_actor_hash"]
                    != json.loads(args.late_swing_policy.read_text())["actor_hash"]
                    or report["asset_hash"] != training["commitment"]["asset_hash"]
                ):
                    raise ValueError("frozen validation provenance drift")
                arms[arm] = {
                    **outcome,
                    "high_quality": high_quality(outcome),
                    "report_hash": report["report_hash"],
                    "parent_report_hash": parent["report_hash"],
                }
            row = {"seed": seed, "lane": lane, "gpu": gpu, "arms": arms}
            with (args.output_root / f"seed{seed}-lane{lane}-result.json").open("x") as stream:
                json.dump(row, stream, indent=2, sort_keys=True, allow_nan=False)
            print(
                f"MOTOR_VALIDATION_AUDITED seed={seed} lane={lane} "
                f"high={[arms[a]['high_quality'] for a in arms]}",
                flush=True,
            )
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for batch in pool.map(worker, range(4)) for r in batch]
    if (
        hash_bytes(args.runner.read_bytes()) != training["commitment"]["runner_hash"]
        or load_policy(args.motor_policy)[0]["policy_hash"] != POLICY_HASH
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=runner_root, text=True
        ).strip()
        or hash_bytes(Path(__file__).read_bytes()) != collector_hash
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=collector_root, text=True
        ).strip()
    ):
        raise ValueError("validation source or learned policy drift")
    result = {
        "schema": "rsi_bilateral_motor_consumed_validation_v305",
        "activation_ceiling": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "training_report_hash": TRAIN_REPORT_HASH,
        "motor_policy_hash": POLICY_HASH,
        "protocol_hash": hash_json(protocol),
        "source_commit": training["commitment"]["source_commit"],
        "collector_commit": collector_commit,
        "collector_hash": collector_hash,
        "rows": sorted(rows, key=lambda r: (r["seed"], r["lane"])),
        **score(rows),
        "independent_physical_episode_count": 60,
        "distinct_consumed_course_count": 12,
        "fresh_holdout_open_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    with (args.output_root / "validation_summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                "gate": result["consumed_validation_gate_passed"],
                "report_hash": result["report_hash"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
