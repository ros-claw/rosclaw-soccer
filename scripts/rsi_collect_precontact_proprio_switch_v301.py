"""Audited independent Isaac switch-control experiment on consumed courses."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.precontact_proprio_policy import V300_HASH, load_policy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

POLICY_HASH = "sha256:aa33d1fffcb073dc5d7a1a97fea5de653d182571c14b8e973f4c39c147820cfc"
RISK = (
    (20261177, 0),
    (20261227, 4),
    (20261282, 0),
    (20261360, 0),
    (20261378, 0),
    (20261440, 2),
    (20261446, 0),
)
GAIN = ((20261095, 2), (20261097, 2), (20261146, 4), (20261148, 2), (20260975, 4))
COURSES = RISK + GAIN
ARMS = (("gain_08", 0.8, False), ("gain_12", 1.2, False), ("proprio_switch", 1.2, True))


def _course(seed: int, lane: int, gpu: int, common: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    course = None
    asset_hash = None
    actor_hash = None
    for arm, gain, switched in ARMS:
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
            proprio_policy=common["policy"] if switched else None,
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
                raise ValueError("ball course drift")
            course = item["environments"][0]["course"]
        if actor_hash is not None and report["late_swing_actor_hash"] != actor_hash:
            raise ValueError("late-swing actor drift")
        actor_hash = report["late_swing_actor_hash"]
        rows[arm] = {
            "parent_report_hash": parent["report_hash"],
            "parent_command_audit_hash": parent_outcome["command_audit_hash"],
            "actor_report_hash": report["report_hash"],
            "risk_probability": report.get("navigation_proprio_risk_probability"),
            "risk_vetoed": report.get("navigation_proprio_risk_vetoed"),
            "high_quality": high_quality(outcome),
            **outcome,
        }
    print(f"PROPRIO_SWITCH_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
    return {"seed": seed, "lane": lane, "gpu": gpu, "course": course, "arms": rows}


def _gpu(gpu: int, common: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for index, (seed, lane) in enumerate(COURSES):
        if index % 4 != gpu:
            continue
        try:
            rows.append(_course(seed, lane, gpu, common))
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"PROPRIO_SWITCH_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def score(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    complete = (
        not failures
        and len(rows) == len(COURSES)
        and {(r["seed"], r["lane"]) for r in rows} == set(COURSES)
    )
    base = [r["arms"]["gain_08"] for r in rows]
    raw = [r["arms"]["gain_12"] for r in rows]
    switched = [r["arms"]["proprio_switch"] for r in rows]
    harm_prevented = sum(
        (
            b["clean_foot_only"]
            and not a["clean_foot_only"]
            or b["maximum_lateral_excursion_m"] <= 4 < a["maximum_lateral_excursion_m"]
        )
        and s["clean_foot_only"]
        and s["maximum_lateral_excursion_m"] <= 4
        for b, a, s in zip(base[:7], raw[:7], switched[:7], strict=True)
    )
    clean_loss = sum(
        b["clean_foot_only"] and not s["clean_foot_only"]
        for b, s in zip(base, switched, strict=True)
    )
    new_out = sum(
        b["maximum_lateral_excursion_m"] <= 4 < s["maximum_lateral_excursion_m"]
        for b, s in zip(base, switched, strict=True)
    )
    gain_high = sum(s["high_quality"] for s in switched[7:])
    passed = bool(
        complete
        and harm_prevented >= 6
        and clean_loss == 0
        and new_out == 0
        and gain_high >= 3
        and all(s["minimum_pelvis_z_m"] >= 0.65 for s in switched)
    )
    return {
        "complete": complete,
        "baseline_high_quality": sum(b["high_quality"] for b in base),
        "raw_high_quality": sum(a["high_quality"] for a in raw),
        "switch_high_quality": sum(s["high_quality"] for s in switched),
        "risk_harm_prevented": harm_prevented,
        "gain_cases_switch_high_quality": gain_high,
        "clean_foot_loss": clean_loss,
        "new_out_of_play": new_out,
        "minimum_switch_pelvis_z_m": min((s["minimum_pelvis_z_m"] for s in switched), default=None),
        "switch_veto_count": sum(bool(s["risk_vetoed"][0]) for s in switched),
        "consumed_gate_passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "protocol",
        "policy",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    policy, policy_hash = load_policy(args.policy)
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        args.output_root.exists()
        or protocol.get("risk_cases") != [list(x) for x in RISK]
        or protocol.get("gain_cases") != [list(x) for x in GAIN]
        or protocol.get("policy_hash") != POLICY_HASH
        or policy_hash != POLICY_HASH
        or policy["training_report_hash"] != V300_HASH
        or not all(
            x.is_file() for x in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed protocol and policy, qualified assets, and new output required")
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    common = dict(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        policy=args.policy,
        source_hash=hash_bytes(runner.read_bytes()),
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_gpu, gpu, common) for gpu in range(4)]
        for future in as_completed(futures):
            gpu_rows, gpu_failures = future.result()
            rows.extend(gpu_rows)
            failures.extend(gpu_failures)
    rows.sort(key=lambda row: COURSES.index((row["seed"], row["lane"])))
    result: dict[str, Any] = {
        "schema": "rsi_isaac_precontact_proprio_switch_evidence_v1",
        "activation_ceiling": "SIM_ONLY",
        "data_status": "TRAIN_CONSUMED_NOT_FRESH",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "policy_hash": policy_hash,
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
        f"PROPRIO_SWITCH_EVIDENCE={result['report_hash']} GATE={result['consumed_gate_passed']}",
        flush=True,
    )
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
