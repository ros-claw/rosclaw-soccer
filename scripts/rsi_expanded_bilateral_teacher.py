"""Expand only after all16 producer outcomes AND all4 native reviews qualify."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prior-gate", "world", "motion-csv", "model-root", "core-source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-gate-hash"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--expected-preflight-hash")
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source and unused output required")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("clean source required before physics imports")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "scripts"], text=True
    ).splitlines()
    if any(p.endswith(".py") for p in untracked):
        raise ValueError("unsealed Python source")
    if args.preflight_only:
        if any(x is not None for x in (args.preflight, args.expected_preflight_hash, args.shard)):
            raise ValueError("zero-actor admission cannot allocate a shard")
    elif any(x is None for x in (args.preflight, args.expected_preflight_hash, args.shard)):
        raise ValueError("same-source whole40 preflight required before shard")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if digest(args.prior_gate / "complete.json") != args.expected_gate_hash:
        raise ValueError("prior full16 review dispatch identity differs")
    gate = json.loads((args.prior_gate / "complete.json").read_text())
    if (
        gate["study"] != "bilateral_entry"
        or gate["status"] != "REVIEW_CPU_AUTHORIZED"
        or gate["producer_claimed_qualified"] < 7
        or gate["regressions"]
        or gate["added_unsafe"]
        or len(gate["launched_reviews"]) != 4
    ):
        raise ValueError("whole16 bilateral gain required, no single rescue shortcut")
    pins = dict(json.loads((args.prior_gate / "declaration.json").read_text())["input_file_hashes"])
    pins.update({str(p): digest(p) for p in args.prior_gate.iterdir() if p.is_file()})
    clusters, reviewed = set(), []
    for launch in gate["launched_reviews"]:
        terminal = dict(
            line.split("=", 1)
            for line in subprocess.check_output(
                [
                    "systemctl",
                    "--user",
                    "show",
                    launch["unit"],
                    "-p",
                    "MainPID",
                    "-p",
                    "SubState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                ],
                text=True,
            ).splitlines()
        )
        if terminal != dict(MainPID="0", SubState="exited", Result="success", ExecMainStatus="0"):
            raise ValueError("all native review jobs must actually be closed successfully")
        root = Path(launch["output"])
        rd = json.loads((root / "declaration.json").read_text())
        result = json.loads((root / "complete.json").read_text())
        cluster = result["cluster"]
        if (
            cluster not in range(4)
            or cluster in clusters
            or len(result["results"]) != 4
            or result["additional_native_steps"] != 112800
            or rd["expected_additional_native_steps"] != 112800
            or rd["schema"] != "soccer.rsi.bilateral_entry_recorded_review.v1"
            or result["optimizer_updates"] != 0
            or any(root.rglob("failure.json"))
        ):
            raise ValueError("exact closed native bilateral shard required")
        for row in result["results"]:
            if (
                row["candidate_count"] != 9
                or row["actual_and_all_candidate_native_physics_bit_exact"] is not True
                or row["recorded_action_to_target_and_posthandoff_reference_verified"] is not True
                or row["old_teacher_success_lost"]
                or row["added_unsafe_vs_old_teacher"]
            ):
                raise ValueError("all native candidate/endpoint evidence required")
        pins.update(rd["input_file_hashes"])
        pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
        clusters.add(cluster)
        reviewed.extend(result["results"])
    if (
        clusters != set(range(4))
        or sum(r["score"]["qualified"] for r in reviewed) != gate["producer_claimed_qualified"]
    ):
        raise ValueError("full16 independent outcome sum must match dispatch")
    check_pins(pins)
    for path in [args.world, args.motion_csv] + [
        args.model_root / s
        for s in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        if pins.get(str(path)) != digest(path):
            raise ValueError("unchanged native world/motion/NN required")
    runtime_pins = {}
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        runtime_pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    for path in [args.world, args.motion_csv] + [
        args.model_root / s
        for s in (
            "planner_sonic.onnx",
            "low_latency/model_encoder.onnx",
            "low_latency/model_decoder.onnx",
            "low_latency/observation_config.yaml",
        )
    ]:
        runtime_pins[str(path)] = digest(path)
    protocol = Path.cwd() / "docs/rsi/bilateral-expanded40-preregister-20261020.zh-CN.md"
    runtime_pins[str(protocol)] = digest(protocol)
    pins.update(runtime_pins)
    prior_paths = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
    if len(prior_paths) != 1:
        raise ValueError("one unchanged historical prior required")
    runtime_pins[str(prior_paths[0])] = digest(prior_paths[0])
    from rosclaw_soccer.training.expanded_entry_courses import expanded_entry_courses
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
    from scripts.rsi_atomic_artifacts import write_once

    courses = expanded_entry_courses()
    bank_hash = preflight_receiving_courses(courses)
    declaration = dict(
        schema="soccer.rsi.bilateral_expanded40.v1",
        source_head=head,
        input_file_hashes=pins,
        runtime_file_hashes=runtime_pins,
        courses=[asdict(c) for c in courses],
        bank_hash=bank_hash,
        shard=args.shard,
        maximum_candidates_per_course=9,
        full_bank_maximum_native_steps=1240000,
        actual_endpoint_prefix_shared=True,
        prior_gate_hash=args.expected_gate_hash,
        original_rsi_exam=False,
        private_fresh_accessed=False,
        neural_optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if not args.preflight_only:
        if digest(args.preflight / "complete.json") != args.expected_preflight_hash:
            raise ValueError("same-source40 admission identity differs")
        old = json.loads((args.preflight / "declaration.json").read_text())
        proof = json.loads((args.preflight / "complete.json").read_text())
        if (
            old["source_head"] != head
            or old["input_file_hashes"] != pins
            or old["bank_hash"] != bank_hash
            or proof["declaration_hash"] != digest(args.preflight / "declaration.json")
            or proof["new_native_steps"] != 0
            or proof["new_actor_allocations"] != 0
            or proof["new_model_calls"] != 0
        ):
            raise ValueError("exact same-source whole40 zero-actor proof required")
        pins.update({str(p): digest(p) for p in args.preflight.iterdir() if p.is_file()})
    args.output.mkdir(parents=True, exist_ok=False)
    write_once(args.output / "declaration.json", declaration)
    results = []
    try:
        if args.preflight_only:
            check_pins(pins)
            write_once(
                args.output / "complete.json",
                dict(
                    source_head=head,
                    bank_hash=bank_hash,
                    declaration_hash=digest(args.output / "declaration.json"),
                    new_actor_allocations=0,
                    new_native_steps=0,
                    new_model_calls=0,
                    hardware_authorized=False,
                    promotion_authorized=False,
                ),
            )
            print("WHOLE40_BILATERAL_NATIVE_REVIEW_ADMITTED_ZERO_ACTORS", flush=True)
            return
        import mujoco
        import numpy as np

        from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
        from rosclaw_soccer.sim.contracts import hash_json
        from scripts.rsi_expanded_bilateral_course import run_course

        choices = [Path(p) for p in pins if Path(p).name == "entry-affordance.json"]
        if len(choices) != 1:
            raise ValueError("original historical right prior required")
        stored = json.loads(choices[0].read_text())
        affordance = SkillEntryAffordance(
            tuple(stored["ball_offset_xy_m"]),
            stored["historical_frame"],
            stored["historical_forward_speed_mps"],
            stored["historical_artifact_hash"],
        )
        if affordance.contract_hash != stored["contract_hash"]:
            raise ValueError("entry prior identity differs")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        for item in courses:
            if item.course_index % 4 == args.shard:
                results.append(
                    run_course(
                        args, item, model, poses, affordance, hash_json(declaration), runtime_pins
                    )
                )
        if len(results) != 10:
            raise ValueError("all10 ordered shard courses required")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                source_head=head,
                bank_hash=bank_hash,
                shard=args.shard,
                results=results,
                forecast_native_steps=sum(r["forecast_native_steps"] for r in results),
                endpoint_native_steps=sum(r["endpoint_native_steps"] for r in results),
                baseline_qualified=sum(r["baseline"]["score"]["qualified"] for r in results),
                selected_qualified=sum(r["selected"]["score"]["qualified"] for r in results),
                rescues=sum(r["old_teacher_failure_rescued"] for r in results),
                regressions=sum(r["old_teacher_success_lost"] for r in results),
                added_unsafe=sum(r["added_unsafe_vs_old_teacher"] for r in results),
                neural_optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_courses=len(results),
                complete_forecast_steps=sum(r["forecast_native_steps"] for r in results),
                complete_endpoint_steps=sum(r["endpoint_native_steps"] for r in results),
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
