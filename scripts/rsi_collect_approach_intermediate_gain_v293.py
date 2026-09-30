"""Independent SIM_ONLY replay of the preregistered 0.8/1.0 approach-gain pilot."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_gain_pilot_v292 import COURSES, FAILURES, PROTECTION
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

V292_HASH = "sha256:8172d79326422997eeb5706e7ea554a080441d292b08dbfa48180334d9f086ae"


def gate(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    complete = (
        len(rows) == len(COURSES)
        and {(row["seed"], row["lane"]) for row in rows} == set(COURSES)
        and not failures
    )
    failed = [row for row in rows if (row["seed"], row["lane"]) in FAILURES]
    protected = [row for row in rows if (row["seed"], row["lane"]) in PROTECTION]
    rescued = sum(
        not row["arms"]["gain_08"]["high_quality"] and row["arms"]["gain_10"]["high_quality"]
        for row in failed
    )
    preserved = len(protected) == len(PROTECTION) and all(
        row["arms"]["gain_08"]["high_quality"] and row["arms"]["gain_10"]["high_quality"]
        for row in protected
    )
    base = [row["arms"]["gain_08"] for row in rows]
    candidate = [row["arms"]["gain_10"] for row in rows]
    clean_retained = len(rows) == len(COURSES) and all(
        b["clean_foot_only"] and c["clean_foot_only"] for b, c in zip(base, candidate, strict=True)
    )
    new_out = sum(
        b["maximum_lateral_excursion_m"] <= 4 and c["maximum_lateral_excursion_m"] > 4
        for b, c in zip(base, candidate, strict=True)
    )
    return {
        "complete": complete,
        "previous_failures_rescued": rescued,
        "success_protection_retained": preserved,
        "clean_foot_retained": clean_retained,
        "new_out_of_play": new_out,
        "development_gate_passed": bool(
            complete
            and rescued >= 3
            and preserved
            and clean_retained
            and new_out == 0
            and all(c["minimum_pelvis_z_m"] >= 0.65 for c in candidate)
        ),
    }


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
    courses = []
    for arm, gain in (("gain_08", 0.8), ("gain_10", 1.0)):
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
        if parent["source_hash"] != source_hash or report["source_hash"] != source_hash:
            raise ValueError("pilot runner source drifted")
        courses.append(report["environments"][0]["course"])
        arms[arm] = {
            "parent_report_hash": parent["report_hash"],
            "parent_command_audit_hash": parent_receipt["command_audit_hash"],
            "actor_report_hash": report["report_hash"],
            "high_quality": high_quality(outcome),
            **outcome,
        }
    if courses[0] != courses[1]:
        raise ValueError("course drifted")
    print(f"INTERMEDIATE_GAIN_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
    return {"seed": seed, "lane": lane, "gpu": gpu, "course": courses[0], "arms": arms}


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
            print(f"INTERMEDIATE_GAIN_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--v292-report", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    earlier = json.loads(args.v292_report.read_text(encoding="utf-8"))
    if (
        args.output_root.exists()
        or earlier.get("report_hash") != V292_HASH
        or earlier.get("report_hash")
        != hash_json({key: value for key, value in earlier.items() if key != "report_hash"})
        or earlier.get("development_gate_passed") is not False
        or not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed failed v292 evidence and new qualified output required")
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
        "schema": "rsi_isaac_approach_intermediate_gain_v1",
        "activation_ceiling": "SIM_ONLY",
        "v292_report_hash": V292_HASH,
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "courses": rows,
        "failures": failures,
        **gate(rows, failures),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"INTERMEDIATE_GAIN_PILOT={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
