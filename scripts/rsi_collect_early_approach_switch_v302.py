"""Three-course SIM_ONLY timing ablation after failed frame-30 proprio switch."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

V301_HASH = "sha256:7f4807a285b8ac16ea31661d51d1b0ba40631c738de0bca8edd3beff3d4f4bd9"
COURSES = ((20261227, 4), (20261446, 0), (20261095, 2))
ARMS = (("gain_08", 0.8, False), ("gain_12", 1.2, False), ("early_switch_10", 1.2, True))


def _course(seed: int, lane: int, gpu: int, common: dict[str, Any]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    course = None
    actor_hash = None
    asset_hash = None
    for arm, gain, early in ARMS:
        arguments = dict(
            root=common["root"],
            runner=common["runner"],
            isaac_python=common["isaac_python"],
            g1_usd=common["g1_usd"],
            model_root=common["model_root"],
            actor=common["actor"],
            seed=seed,
            lane=lane,
            gpu=gpu,
            arm=arm,
            gain=gain,
            negative_only=True,
            early_switch=early,
        )
        parent, parent_outcome = _run(**arguments, kind="parent")
        report, outcome = _run(**arguments, kind="actor")
        for item in (parent, report):
            if item["source_hash"] != common["source_hash"]:
                raise ValueError("runner source drift")
            if asset_hash is not None and item["asset_hash"] != asset_hash:
                raise ValueError("asset drift")
            asset_hash = item["asset_hash"]
            if course is not None and item["environments"][0]["course"] != course:
                raise ValueError("course drift")
            course = item["environments"][0]["course"]
        if actor_hash is not None and report["late_swing_actor_hash"] != actor_hash:
            raise ValueError("actor drift")
        actor_hash = report["late_swing_actor_hash"]
        arms[arm] = {
            "parent_report_hash": parent["report_hash"],
            "parent_command_audit_hash": parent_outcome["command_audit_hash"],
            "actor_report_hash": report["report_hash"],
            "high_quality": high_quality(outcome),
            **outcome,
        }

    def trace(arm: str) -> np.lib.npyio.NpzFile:
        return np.load(
            common["root"] / f"seed{seed}-lane{lane}-{arm}-actor" / "body_trace.npz",
            allow_pickle=False,
        )

    with trace("gain_12") as raw, trace("early_switch_10") as switched:
        stop = min(
            299 if arms[name]["first_contact_frame"] is None else arms[name]["first_contact_frame"]
            for name in ("gain_12", "early_switch_10")
        )
        command = (
            raw["navigation_lateral_speed_mps"][: stop + 1, 0]
            - switched["navigation_lateral_speed_mps"][: stop + 1, 0]
        )
        root_delta = (
            raw["root_pose_xyzw_m"][: stop + 1, 0, :2]
            - switched["root_pose_xyzw_m"][: stop + 1, 0, :2]
        )
        target = (
            raw["joint_target_rad"][: stop + 1, 0] - switched["joint_target_rad"][: stop + 1, 0]
        )
        response = {
            "max_command_delta_mps": float(np.max(np.abs(command))),
            "max_root_xy_delta_before_contact_m": float(np.max(np.linalg.norm(root_delta, axis=1))),
            "first_joint_target_difference_frame": int(
                np.flatnonzero(np.max(np.abs(target), axis=1) > 1e-6)[0]
            )
            if np.any(np.max(np.abs(target), axis=1) > 1e-6)
            else None,
        }
    print(f"EARLY_SWITCH_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
    return {
        "seed": seed,
        "lane": lane,
        "gpu": gpu,
        "course": course,
        "arms": arms,
        "response": response,
    }


def score(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    by_course = {(r["seed"], r["lane"]): r for r in rows}
    complete = (
        not failures and len(rows) == 3 and {(r["seed"], r["lane"]) for r in rows} == set(COURSES)
    )
    meaningfully_changed = sum(
        r["response"]["max_root_xy_delta_before_contact_m"] >= 0.02 for r in rows
    )
    rescued = sum(
        r["arms"]["early_switch_10"]["clean_foot_only"]
        and r["arms"]["early_switch_10"]["maximum_lateral_excursion_m"] <= 4
        and (
            not r["arms"]["gain_12"]["clean_foot_only"]
            or r["arms"]["gain_12"]["maximum_lateral_excursion_m"] > 4
        )
        for key, r in by_course.items()
        if key in COURSES[:2]
    )
    clean_loss = sum(
        r["arms"]["gain_08"]["clean_foot_only"]
        and not r["arms"]["early_switch_10"]["clean_foot_only"]
        for r in rows
    )
    new_out = sum(
        r["arms"]["gain_08"]["maximum_lateral_excursion_m"]
        <= 4
        < r["arms"]["early_switch_10"]["maximum_lateral_excursion_m"]
        for r in rows
    )
    gain_high_quality = bool(
        COURSES[2] in by_course and by_course[COURSES[2]]["arms"]["early_switch_10"]["high_quality"]
    )
    passed = bool(
        complete
        and meaningfully_changed >= 2
        and rescued >= 1
        and gain_high_quality
        and clean_loss == 0
        and new_out == 0
    )
    return {
        "complete": complete,
        "meaningful_root_response_count": meaningfully_changed,
        "risk_rescued_count": rescued,
        "gain_case_high_quality": gain_high_quality,
        "clean_foot_loss": clean_loss,
        "new_out_of_play": new_out,
        "mechanism_gate_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "protocol",
        "v301-summary",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    failed = json.loads(args.v301_summary.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        args.output_root.exists()
        or protocol.get("courses") != [list(x) for x in COURSES]
        or protocol.get("source_failed_report_hash") != V301_HASH
        or failed.get("report_hash") != V301_HASH
        or failed.get("report_hash")
        != hash_json({k: v for k, v in failed.items() if k != "report_hash"})
        or failed.get("consumed_gate_passed") is not False
        or not all(
            x.is_file() for x in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed failure, protocol, assets, and new output required")
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    common = dict(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        source_hash=hash_bytes(runner.read_bytes()),
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_course, seed, lane, gpu, common): (seed, lane)
            for gpu, (seed, lane) in enumerate(COURSES)
        }
        for future in as_completed(futures):
            seed, lane = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"seed": seed, "lane": lane, "error": str(exc)})
                print(f"EARLY_SWITCH_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    rows.sort(key=lambda row: COURSES.index((row["seed"], row["lane"])))
    result: dict[str, Any] = {
        "schema": "rsi_isaac_early_approach_switch_evidence_v1",
        "activation_ceiling": "SIM_ONLY",
        "data_status": "CONSUMED_MECHANISM_PROBE",
        "source_failed_report_hash": V301_HASH,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hash": common["source_hash"],
        "courses": rows,
        "failures": failures,
        **score(rows, failures),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "switch_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"EARLY_SWITCH_EVIDENCE={result['report_hash']} GATE={result['mechanism_gate_passed']}",
        flush=True,
    )
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
