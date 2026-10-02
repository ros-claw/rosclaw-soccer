"""Collect preregistered independent G1 lateral-approach causal comparisons."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.conservative_approach_rectangle import load_guarded_approach_policy
from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.physical_report_io import load_physical_report, resolve_physical_report
from rosclaw_soccer.rsi.precontact_proprio_policy import load_policy as load_precontact_policy
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

COURSES = (
    (20260962, 4),
    (20260963, 0),
    (20260964, 2),
    (20260965, 0),
    (20260959, 4),
    (20260961, 8),
    (20260962, 2),
    (20260958, 8),
    (20260959, 10),
    (20260962, 10),
    (20260965, 4),
)
ARMS = {"baseline": 0.0, "lateral_tracking": 0.8}
OUT_CASES = set(COURSES[:4])
PROTECTION_CASES = set(COURSES[7:])


def _run(
    root: Path,
    runner: Path,
    isaac_python: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    seed: int,
    lane: int,
    gpu: int,
    arm: str,
    gain: float,
    kind: str,
    negative_only: bool = False,
    rectangle_policy: Path | None = None,
    proprio_policy: Path | None = None,
    early_switch: bool = False,
    motor_policy: Path | None = None,
    parent_report_override: Path | None = None,
    resume: bool = False,
    motor_bootstrap: Path | None = None,
    motor_online: Path | None = None,
    motor_step: Path | None = None,
    core_root: Path | None = None,
    execution_timeout_s: float | None = None,
    compressed_report: bool = False,
    shared_model_report: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if shared_model_report and not compressed_report:
        raise ValueError("shared model report requires explicit compressed transport")
    if sum(p is not None for p in (motor_policy, motor_bootstrap, motor_online, motor_step)) > 1:
        raise ValueError("one explicit motor proposal backend required")
    folder = root / f"seed{seed}-lane{lane}-{arm}-{kind}"
    parent_folder = root / f"seed{seed}-lane{lane}-{arm}-parent"
    log_path = root / "logs" / f"seed{seed}-lane{lane}-{arm}-{kind}.log"
    existing = folder.exists()
    if existing:
        if not resume:
            raise ValueError(f"new output required: {folder}")
        actual_report = resolve_physical_report(folder / "report.json")
        if actual_report.name != ("report.json.gz" if compressed_report else "report.json"):
            raise ValueError("resume report representation differs")
    command = [
        str(isaac_python),
        str(runner),
        "--g1-usd",
        str(g1_usd),
        "--model-root",
        str(model_root),
        "--output-dir",
        str(folder),
        "--frames",
        "300",
        "--env-count",
        "1",
        "--training-course-seed",
        str(seed),
        "--single-course-lane",
        str(lane),
        "--inference-threads",
        "1",
        "--record-body-trace",
        "--record-foot-geometry",
        "--torch-batch-plan-only",
        "--navigation-lateral-ball-gain",
        str(gain),
        "--headless",
        "--device",
        "cuda:0",
    ]
    if negative_only:
        command.append("--navigation-lateral-negative-only")
    if compressed_report:
        command.append("--compressed-report")
    if shared_model_report:
        command.append("--shared-model-report")
    if rectangle_policy is not None:
        command.extend(("--navigation-rectangle-policy", str(rectangle_policy)))
    if proprio_policy is not None:
        command.extend(("--navigation-proprio-risk-policy", str(proprio_policy)))
    if early_switch:
        command.append("--navigation-lateral-early-switch")
    if kind == "actor":
        command.extend(
            (
                "--late-swing-policy",
                str(actor),
                "--parent-report",
                str(
                    resolve_physical_report(parent_report_override or parent_folder / "report.json")
                ),
                "--revalidate-swing-side",
                "--late-swing-lateral-cap-m",
                "0.15",
                "--late-swing-forward-cap-m",
                "0.08",
                "--late-swing-side-acquisition-gap-m",
                "0.95",
            )
        )
        if motor_policy is not None:
            command.extend(("--contact-motor-policy", str(motor_policy)))
        if motor_bootstrap is not None:
            command.extend(("--contact-motor-bootstrap-model", str(motor_bootstrap)))
        if motor_online is not None:
            command.extend(("--contact-motor-online-model", str(motor_online)))
        if motor_step is not None:
            command.extend(("--contact-motor-step-model", str(motor_step)))
    env = os.environ.copy()
    env.update(
        OMNI_KIT_ACCEPT_EULA="YES",
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=os.pathsep.join(
            [
                str(runner.parent.parent),
                str(runner.parent.parent / "src"),
                str(core_root / "src") if core_root else "/code/rosclaw/rosclaw_test/src",
            ]
        ),
    )
    if not existing:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=runner.parent.parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=execution_timeout_s,
            )
        if completed.returncode:
            raise RuntimeError(f"Isaac failed; inspect {log_path}")
        actual_report = resolve_physical_report(folder / "report.json")
        if actual_report.name != ("report.json.gz" if compressed_report else "report.json"):
            raise ValueError("producer report representation differs")
    report = load_physical_report(folder / "report.json")
    audit = audit_lateral_approach(folder)
    expected_motor_hash = (
        json.loads(motor_policy.read_text(encoding="utf-8"))["policy_hash"]
        if motor_policy is not None and kind == "actor"
        else None
    )
    if motor_bootstrap is not None and kind == "actor":
        model = json.loads(motor_bootstrap.read_text(encoding="utf-8"))
        proof = report.get("contact_motor_policy", {}).get("bootstrap_proof", {})
        if proof.get("model", {}).get("model_hash") != model["model_hash"]:
            raise ValueError("neural preview does not bind requested numerical model")
        expected_motor_hash = report["contact_motor_policy"]["policy_hash"]
    if motor_online is not None and kind == "actor":
        model = json.loads(motor_online.read_text(encoding="utf-8"))
        proof_key = (
            "progressive_motor_proof"
            if model.get("schema") == "soccer.rsi.progressive_contextual_actor_critic.v312"
            else "online_motor_proof"
        )
        proof = report.get("contact_motor_policy", {}).get(proof_key, {})
        if proof.get("model", {}).get("model_hash") != model["model_hash"]:
            raise ValueError("online preview does not bind requested numerical model")
        expected_motor_hash = report["contact_motor_policy"]["policy_hash"]
    if motor_step is not None and kind == "actor":
        from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

        model = load_json_artifact(motor_step)
        proof = report.get("contact_motor_policy", {}).get("step_motor_proof", {})
        if proof.get("model", {}).get("model_hash") != model["model_hash"]:
            raise ValueError("per-frame preview does not bind requested neural model")
        expected_motor_hash = report["contact_motor_policy"]["policy_hash"]
    if (
        report["navigation_lateral_ball_gain"] != gain
        or report.get("navigation_lateral_negative_only") is not negative_only
        or report.get("navigation_rectangle_policy_hash")
        != (load_guarded_approach_policy(rectangle_policy)[1] if rectangle_policy else None)
        or report.get("navigation_proprio_risk_policy_hash")
        != (load_precontact_policy(proprio_policy)[1] if proprio_policy else None)
        or report.get("navigation_lateral_early_switch", False) is not early_switch
        or report.get("contact_motor_policy_hash") != expected_motor_hash
        or report["training_course_seed"] != seed
        or report["single_course_lane"] != lane
    ):
        raise ValueError(f"approach arm report drift: {folder}")
    if kind == "actor":
        parent = load_physical_report(parent_report_override or parent_folder / "report.json")
        if (
            report["parent_report_hash"] != parent["report_hash"]
            or report["source_hash"] != parent["source_hash"]
            or report["asset_hash"] != parent["asset_hash"]
            or report["sonic_qualification_hash"] != parent["sonic_qualification_hash"]
            or report["environments"][0]["course"] != parent["environments"][0]["course"]
        ):
            raise ValueError(f"actor/parent provenance drift: {folder}")
        with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as trace:
            action = audit_taskspace_swing_trace(trace, report, frames=300, count=1)
        row = report["environments"][0]
        with np.load(folder / "trace.npz", allow_pickle=False) as physics:
            displacement = post_contact_displacement(
                physics["ball_position_m"][:, 0], row["first_contact_frame"]
            )
        outcome = {
            "contact_body_indices": row["contact_body_indices"],
            "first_contact_frame": row["first_contact_frame"],
            "minimum_pelvis_z_m": row["minimum_pelvis_z_m"],
            "maximum_lateral_excursion_m": report["single_instance_max_lateral_excursion_m"],
            "clean_foot_only": bool(
                row["contact_body_indices"] and set(row["contact_body_indices"]) <= {0, 1}
            ),
            "action_audit": action,
            **displacement,
        }
        outcome["reward"] = first_touch_reward(
            {**outcome, "max_lateral_excursion_m": outcome["maximum_lateral_excursion_m"]}
        )
    else:
        outcome = {}
    return report, {
        "command_audit_hash": audit["report_hash"],
        "command_active_frames": audit["active_frames"],
        **outcome,
    }


def _course(
    seed: int,
    lane: int,
    *,
    root: Path,
    runner: Path,
    isaac_python: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    source_hash: str,
) -> dict[str, Any]:
    gpu = (lane // 2) % 4
    results = {}
    courses = []
    for arm, gain in ARMS.items():
        parent, parent_outcome = _run(
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
        )
        if parent["source_hash"] != source_hash or report["source_hash"] != source_hash:
            raise ValueError("runner source drifted during collection")
        courses.append(report["environments"][0]["course"])
        results[arm] = {
            "parent_report_hash": parent["report_hash"],
            "parent_command_audit_hash": parent_outcome["command_audit_hash"],
            "actor_report_hash": report["report_hash"],
            **outcome,
        }
    if courses[0] != courses[1]:
        raise ValueError("counterfactual ball course differs")
    print(f"APPROACH_COURSE_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
    return {"seed": seed, "lane": lane, "gpu": gpu, "course": courses[0], "arms": results}


def _gpu_courses(
    gpu: int, common: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for seed, lane in COURSES:
        if (lane // 2) % 4 != gpu:
            continue
        try:
            rows.append(_course(seed, lane, **common))
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"APPROACH_COURSE_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        args.output_root.exists()
        or not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("new output and qualified assets required")
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
    base = [row["arms"]["baseline"] for row in rows]
    candidate = [row["arms"]["lateral_tracking"] for row in rows]
    complete = len(rows) == len(COURSES) and not failures
    out_reduced = (
        sum(
            row["arms"]["lateral_tracking"]["maximum_lateral_excursion_m"] > 4
            for row in rows
            if (row["seed"], row["lane"]) in OUT_CASES
        )
        <= 2
    )
    other_out = any(
        row["arms"]["lateral_tracking"]["maximum_lateral_excursion_m"] > 4
        for row in rows
        if (row["seed"], row["lane"]) not in OUT_CASES
    )
    protected = all(
        (arm := row["arms"]["lateral_tracking"])["clean_foot_only"]
        and arm["forward_60_m"] is not None
        and arm["forward_60_m"] >= 1
        and abs(arm["lateral_60_m"]) / max(arm["forward_60_m"], 0.01) <= 0.3
        and arm["maximum_lateral_excursion_m"] <= 4
        for row in rows
        if (row["seed"], row["lane"]) in PROTECTION_CASES
    )
    safe = sum(item["clean_foot_only"] for item in candidate) >= sum(
        item["clean_foot_only"] for item in base
    ) and all(item["minimum_pelvis_z_m"] >= 0.65 for item in candidate)
    result: dict[str, Any] = {
        "schema": "rsi_isaac_approach_lateral_tracking_bank_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "courses": rows,
        "failures": failures,
        "complete": complete,
        "out_reduced": out_reduced,
        "other_out_of_play": other_out,
        "protection_retained": protected,
        "safety_retained": safe,
        "development_gate_passed": complete
        and out_reduced
        and not other_out
        and protected
        and safe,
        "fresh_test_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"APPROACH_BANK={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
