"""Fresh-seed SIM_ONLY examination of the audited shadow-contact approach teacher."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.parent_contact_approach_gate import choose_shadow_approach_gain
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

TEACHER_HASH = "sha256:74fe143a2fc03db64c0fb3d093b2776cafad8273467b9e75a3ffd6355c6f5174"
CHALLENGE = (
    (20260983, 2),
    (20260984, 6),
    (20260986, 6),
    (20260987, 0),
    (20260987, 2),
    (20260989, 4),
    (20260990, 4),
    (20260993, 2),
    (20260994, 4),
    (20260995, 2),
    (20260996, 2),
    (20261001, 4),
    (20261002, 2),
    (20261004, 6),
    (20261005, 6),
    (20261007, 2),
    (20261008, 4),
)
PROTECTION = ((20260985, 4), (20260989, 2), (20260990, 0), (20260990, 2))
COURSES = CHALLENGE + PROTECTION


def preflight() -> str:
    challenge = []
    protection = []
    for seed in range(20260983, 20261009):
        for lane, (x, y, vx) in enumerate(sample_training_courses(seed, 16)):
            if lane % 2 or vx >= 0 or not 2.0 <= x <= 2.55:
                continue
            if -0.09 <= y <= -0.02:
                challenge.append((seed, lane))
            if 0.02 <= y <= 0.09:
                protection.append((seed, lane))
    if tuple(challenge) != CHALLENGE or tuple(protection[:4]) != PROTECTION:
        raise ValueError("preregistered outcome-blind fresh catalog drifted")
    return hash_json({"challenge": CHALLENGE, "protection": PROTECTION})


def _course(
    seed: int,
    lane: int,
    gpu: int,
    *,
    root: Path,
    runner: Path,
    isaac_python: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    source_hash: str,
) -> dict[str, Any]:
    arms = {}
    parent_reports = {}
    course_rows = []
    for arm, gain in (("gain_08", 0.8), ("gain_12", 1.2)):
        parent, parent_receipt = _run(
            root,
            runner,
            isaac_python,
            g1_usd,
            model_root,
            actor,
            seed,
            lane,
            gpu,
            arm,
            gain,
            "parent",
            negative_only=True,
        )
        if parent["source_hash"] != source_hash:
            raise ValueError("candidate parent source drifted")
        parent_reports[arm] = parent
        # The option is fixed from the candidate parent before either actor
        # result can be used by the selected-outcome calculation below.
        if arm == "gain_12":
            selected_gain, decision_reason = choose_shadow_approach_gain(parent)
        report, outcome = _run(
            root,
            runner,
            isaac_python,
            g1_usd,
            model_root,
            actor,
            seed,
            lane,
            gpu,
            arm,
            gain,
            "actor",
            negative_only=True,
        )
        if report["source_hash"] != source_hash:
            raise ValueError("candidate actor source drifted")
        course_rows.append(report["environments"][0]["course"])
        arms[arm] = {
            "parent_report_hash": parent["report_hash"],
            "parent_command_audit_hash": parent_receipt["command_audit_hash"],
            "actor_report_hash": report["report_hash"],
            "high_quality": high_quality(outcome),
            **outcome,
        }
    if course_rows[0] != course_rows[1]:
        raise ValueError("arm course drifted")
    if parent_reports["gain_08"]["environments"][0]["course"] != course_rows[0]:
        raise ValueError("baseline parent course drifted")
    selected_arm = "gain_12" if selected_gain == 1.2 else "gain_08"
    print(
        f"SHADOW_FRESH_AUDITED seed={seed} lane={lane} selected={selected_gain} "
        f"reason={decision_reason} gpu={gpu}",
        flush=True,
    )
    return {
        "seed": seed,
        "lane": lane,
        "gpu": gpu,
        "course": course_rows[0],
        "selected_gain": selected_gain,
        "decision_reason": decision_reason,
        "selected_arm": selected_arm,
        "arms": arms,
    }


def _gpu_courses(
    gpu: int, common: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for index, (seed, lane) in enumerate(COURSES):
        if index % 4 != gpu:
            continue
        try:
            rows.append(_course(seed, lane, gpu, **common))
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"SHADOW_FRESH_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def score(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    complete = (
        len(rows) == len(COURSES)
        and {(row["seed"], row["lane"]) for row in rows} == set(COURSES)
        and not failures
    )
    challenge = [row for row in rows if (row["seed"], row["lane"]) in CHALLENGE]
    controls = [row for row in rows if (row["seed"], row["lane"]) in PROTECTION]
    base = [row["arms"]["gain_08"] for row in challenge]
    selected = [row["arms"][row["selected_arm"]] for row in challenge]
    high_gain = sum(c["high_quality"] for c in selected) - sum(b["high_quality"] for b in base)
    mean_reward_gain = sum(
        c["reward"] - b["reward"] for b, c in zip(base, selected, strict=True)
    ) / len(CHALLENGE)
    clean_loss = sum(
        b["clean_foot_only"] and not c["clean_foot_only"]
        for b, c in zip(base, selected, strict=True)
    )
    new_out = sum(
        b["maximum_lateral_excursion_m"] <= 4 and c["maximum_lateral_excursion_m"] > 4
        for b, c in zip(base, selected, strict=True)
    )
    controls_pass = len(controls) == len(PROTECTION) and all(
        (b := row["arms"]["gain_08"])["contact_body_indices"]
        == (c := row["arms"]["gain_12"])["contact_body_indices"]
        and b["first_contact_frame"] == c["first_contact_frame"]
        and b["high_quality"] == c["high_quality"]
        and b["forward_60_m"] is not None
        and c["forward_60_m"] is not None
        and b["lateral_60_m"] is not None
        and c["lateral_60_m"] is not None
        and abs(b["forward_60_m"] - c["forward_60_m"]) <= 0.01
        and abs(b["lateral_60_m"] - c["lateral_60_m"]) <= 0.01
        and c["command_active_frames"] == 0
        for row in controls
    )
    passed = bool(
        complete
        and high_gain >= 2
        and mean_reward_gain >= 0.1
        and clean_loss == 0
        and new_out == 0
        and controls_pass
        and all(c["minimum_pelvis_z_m"] >= 0.65 for c in selected)
    )
    return {
        "complete": complete,
        "challenge_count": len(challenge),
        "protection_count": len(controls),
        "shadow_veto_count": sum(row["selected_gain"] == 0.8 for row in challenge),
        "high_quality_gain": high_gain,
        "mean_reward_gain": mean_reward_gain,
        "clean_foot_loss": clean_loss,
        "new_out_of_play": new_out,
        "positive_control_passed": controls_pass,
        "fresh_gate_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--retrospective-report", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    args = parser.parse_args()
    earlier = json.loads(args.retrospective_report.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        args.output_root.exists()
        or earlier.get("report_hash") != TEACHER_HASH
        or earlier.get("report_hash")
        != hash_json({k: v for k, v in earlier.items() if k != "report_hash"})
        or earlier.get("data_status") != "CONSUMED_DEVELOPMENT_NOT_GENERALIZATION"
        or not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed retrospective teacher and new qualified output required")
    catalog_hash = preflight()
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    source_hash = hash_bytes(runner.read_bytes())
    common = dict(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        source_hash=source_hash,
    )
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_gpu_courses, gpu, common) for gpu in range(4)]
        for future in as_completed(futures):
            gpu_rows, gpu_failures = future.result()
            rows.extend(gpu_rows)
            failures.extend(gpu_failures)
    rows.sort(key=lambda row: COURSES.index((row["seed"], row["lane"])))
    result: dict[str, Any] = {
        "schema": "rsi_isaac_shadow_approach_fresh_v1",
        "activation_ceiling": "SIM_ONLY",
        "retrospective_teacher_hash": TEACHER_HASH,
        "course_catalog_selection_hash": catalog_hash,
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "courses": rows,
        "failures": failures,
        **score(rows, failures),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"SHADOW_APPROACH_FRESH={result['report_hash']}", flush=True)
    print(f"FRESH_GATE_PASSED={result['fresh_gate_passed']}", flush=True)
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
